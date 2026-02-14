# app09.py
# Q-Quest 量子神託（app08ベース + STAGE(季節×時間) + QUOTES神託）
# - Excel(pack) 1枚で完結: VOW/CHAR/AXIS/STAGE/QUOTES
# - テキスト→誓願ベクトル自動生成（char n-gram）
# - QUBO風エネルギー（低いほど選ばれやすい）→ 温度(beta)で観測
# - QUOTESを「エネルギー的に近い」ものとして温度付きで選択

import os
import re
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Q-Quest 量子神託 app09", layout="wide")

APP_TITLE = "🔮 Q-Quest 量子神託（app09：STAGE×QUOTES神託）"

# ----------------------------
# Utility
# ----------------------------
def _safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    # temperature > 0, larger => flatter
    t = max(1e-9, float(temperature))
    z = (x - np.max(x)) / t
    e = np.exp(z)
    s = e / (np.sum(e) + 1e-12)
    return s

def ensure_cols(df: pd.DataFrame, required: List[str], sheet_name: str):
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{sheet_name} の列が不足: {miss}\n検出列={df.columns.tolist()}")

def vow_key_to_num(v: str) -> int:
    # "VOW_01" -> 1
    m = re.search(r"VOW_(\d+)", str(v))
    return int(m.group(1)) if m else -1

def pick_one_by_prob(items: List, p: np.ndarray):
    idx = np.random.choice(len(items), p=p)
    return items[idx], idx

def normalize01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

# ----------------------------
# Data model
# ----------------------------
@dataclass
class Pack:
    vow_dict: pd.DataFrame
    axis_dict: pd.DataFrame
    char_master: pd.DataFrame
    char_to_vow: pd.DataFrame
    stage_dict: pd.DataFrame
    stage_to_axis: pd.DataFrame
    quotes: pd.DataFrame

# ----------------------------
# Excel loader (pickle-safe)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_pack_excel_bytes(xlsx_bytes: bytes) -> Pack:
    # 重要: st.cache_data で pickle 可能な戻り値にする（pandas DF はOK）
    xls = pd.ExcelFile(xlsx_bytes)

    required_sheets = [
        "VOW_DICT", "AXIS_DICT", "CHAR_MASTER", "CHAR_TO_VOW",
        "STAGE_DICT", "STAGE_TO_AXIS", "QUOTES"
    ]
    for s in required_sheets:
        if s not in xls.sheet_names:
            raise ValueError(f"統合Excelに必要なシート '{s}' がありません。検出={xls.sheet_names}")

    vow_dict = pd.read_excel(xls, "VOW_DICT")
    axis_dict = pd.read_excel(xls, "AXIS_DICT")
    char_master = pd.read_excel(xls, "CHAR_MASTER")
    char_to_vow = pd.read_excel(xls, "CHAR_TO_VOW")
    stage_dict = pd.read_excel(xls, "STAGE_DICT")
    stage_to_axis = pd.read_excel(xls, "STAGE_TO_AXIS")
    quotes = pd.read_excel(xls, "QUOTES")

    # validate columns (あなたのExcel仕様に合わせる)
    ensure_cols(vow_dict, ["VOW_ID", "TITLE"], "VOW_DICT")
    ensure_cols(char_master, ["CHAR_ID", "公式キャラ名", "AXIS_SEI", "AXIS_RYU", "AXIS_MA", "AXIS_MAKOTO"], "CHAR_MASTER")
    ensure_cols(char_to_vow, ["CHAR_ID", "IMAGE_FILE", "公式キャラ名"], "CHAR_TO_VOW")
    ensure_cols(stage_dict, ["STAGE_ID", "LABEL"], "STAGE_DICT")
    ensure_cols(stage_to_axis, ["STAGE_ID", "AXIS_SEI", "AXIS_RYU", "AXIS_MA", "AXIS_MAKOTO"], "STAGE_TO_AXIS")
    ensure_cols(quotes, ["QUOTE_ID", "QUOTE", "SOURCE", "LANG"], "QUOTES")

    return Pack(
        vow_dict=vow_dict,
        axis_dict=axis_dict,
        char_master=char_master,
        char_to_vow=char_to_vow,
        stage_dict=stage_dict,
        stage_to_axis=stage_to_axis,
        quotes=quotes,
    )

def load_pack_from_uploader(uploaded_file) -> Pack:
    b = uploaded_file.getvalue()
    return load_pack_excel_bytes(b)

# ----------------------------
# Text -> vow auto vector (char n-gram)
# ----------------------------
def char_ngrams(s: str, n: int = 2) -> List[str]:
    s = re.sub(r"\s+", "", str(s))
    if len(s) < n:
        return [s] if s else []
    return [s[i:i+n] for i in range(len(s)-n+1)]

def build_vow_text_corpus(vow_dict: pd.DataFrame) -> Dict[str, str]:
    # vow_text = "LABEL TITLE SUBTITLE DESCRIPTION_LONG UI_HINT" をまとめる
    cols = ["VOW_ID", "LABEL", "TITLE", "SUBTITLE", "DESCRIPTION_LONG", "UI_HINT", "TRAIT_FROM_FILE"]
    exists = [c for c in cols if c in vow_dict.columns]
    corpus = {}
    for _, r in vow_dict[exists].iterrows():
        vid = str(r.get("VOW_ID"))
        texts = []
        for c in exists:
            if c == "VOW_ID":
                continue
            v = r.get(c)
            if pd.notna(v):
                texts.append(str(v))
        corpus[vid] = " ".join(texts)
    return corpus

def text_to_vow_auto(text: str, vow_ids: List[str], vow_corpus: Dict[str, str], ngram_n: int = 2) -> np.ndarray:
    # 単純な n-gram 重なりスコア（軽量・ローカル完結）
    tx = str(text or "")
    tgrams = set(char_ngrams(tx, ngram_n))
    if not tgrams:
        return np.zeros(len(vow_ids), dtype=float)

    scores = np.zeros(len(vow_ids), dtype=float)
    for i, vid in enumerate(vow_ids):
        c = vow_corpus.get(vid, "")
        cgrams = set(char_ngrams(c, ngram_n))
        if not cgrams:
            scores[i] = 0.0
            continue
        inter = len(tgrams & cgrams)
        # 正規化（長さの影響を抑える）
        scores[i] = inter / (math.sqrt(len(tgrams) * len(cgrams)) + 1e-9)

    # 0..5 スケールに寄せる（最大を 5 に）
    mx = float(np.max(scores))
    if mx > 1e-12:
        scores = scores / mx * 5.0
    return scores

def extract_keywords_simple(text: str, topk: int = 8) -> List[str]:
    # 形態素なしの簡易：漢字/ひらがな/カタカナの2-3gram上位
    s = re.sub(r"\s+", "", str(text or ""))
    grams = []
    for n in [2, 3]:
        grams += char_ngrams(s, n)
    # 記号っぽいもの除去
    grams = [g for g in grams if re.search(r"[ぁ-んァ-ン一-龥]", g)]
    if not grams:
        return []
    from collections import Counter
    cnt = Counter(grams)
    return [w for w, _ in cnt.most_common(topk)]

# ----------------------------
# Stage (season × time)
# ----------------------------
def season_from_month(month: int) -> str:
    # 日本の感覚（ざっくり）
    if month in [3, 4, 5]:
        return "SPRING"
    if month in [6, 7, 8]:
        return "SUMMER"
    if month in [9, 10, 11]:
        return "AUTUMN"
    return "WINTER"

def time_slot_from_hour(hour: int) -> str:
    # ざっくり4区分
    if 5 <= hour <= 10:
        return "MORNING"
    if 11 <= hour <= 16:
        return "DAY"
    if 17 <= hour <= 20:
        return "EVENING"
    return "NIGHT"

def build_stage_id(season: str, time_slot: str) -> str:
    return f"{season}_{time_slot}"

def get_stage_axis_weights(pack: Pack, stage_id: str) -> np.ndarray:
    row = pack.stage_to_axis[pack.stage_to_axis["STAGE_ID"].astype(str) == str(stage_id)]
    if row.empty:
        return np.zeros(4, dtype=float)
    r = row.iloc[0]
    return np.array([
        _safe_float(r["AXIS_SEI"]),
        _safe_float(r["AXIS_RYU"]),
        _safe_float(r["AXIS_MA"]),
        _safe_float(r["AXIS_MAKOTO"]),
    ], dtype=float)

# ----------------------------
# Energy model (QUBO "like")
# ----------------------------
def get_vow_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if re.match(r"VOW_\d+", str(c))]
    # sort by number
    cols.sort(key=lambda x: vow_key_to_num(x))
    return cols

def build_char_matrix(pack: Pack) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    c2v = pack.char_to_vow.copy()
    vow_cols = get_vow_cols(c2v)
    if len(vow_cols) == 0:
        raise ValueError("CHAR_TO_VOW に VOW_01.. の列が見つかりません。")
    W = c2v[vow_cols].fillna(0).astype(float).to_numpy()  # (n_char, n_vow)
    return c2v, W, vow_cols

def build_char_axis_matrix(pack: Pack) -> Tuple[np.ndarray, List[str]]:
    cm = pack.char_master.copy()
    axis_cols = ["AXIS_SEI", "AXIS_RYU", "AXIS_MA", "AXIS_MAKOTO"]
    A = cm[axis_cols].fillna(0).astype(float).to_numpy()  # (n_char, 4)
    return A, axis_cols

def compute_energy(
    v_mix: np.ndarray,
    W_char_vow: np.ndarray,
    stage_axis_w: np.ndarray,
    A_char_axis: np.ndarray,
    stage_gain: float,
    eps_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # base score: vowsとの整合（大きいほど良い） => energy は低いほど良いので負にする
    base_score = W_char_vow @ v_mix  # (n_char,)

    # stage bias: (char_axis ⋅ stage_axis_w) を加点（状況に合う軸が高いキャラを押す）
    stage_score = A_char_axis @ stage_axis_w  # (n_char,)

    # energy: 小さいほど選ばれやすい
    noise = rng.normal(0.0, eps_noise, size=base_score.shape[0])
    energy = -(base_score + stage_gain * stage_score) + noise
    return energy

def observe_distribution(energy: np.ndarray, beta: float, n_samples: int, rng: np.random.Generator):
    # p ∝ exp(-beta * energy)
    p = softmax(-beta * energy, temperature=1.0)
    idxs = rng.choice(len(p), size=int(n_samples), replace=True, p=p)
    return p, idxs

# ----------------------------
# QUOTES selection (energy-like)
# ----------------------------
def build_vow_id_list_from_pack(pack: Pack, vow_cols: List[str]) -> List[str]:
    # vow_cols: ["VOW_01",...]
    # pack.vow_dict has VOW_ID like "VOW_01"
    # Keep vow_cols order
    return [str(v) for v in vow_cols]

def quote_score_row(
    r: pd.Series,
    observed_char_id: str,
    top_vow_ids: List[str],
    v_mix_map: Dict[str, float],
    keywords: List[str],
    stage_axis_label: str,
    quote_char_gain: float = 2.0,
    quote_vow_gain: float = 1.2,
    quote_kw_gain: float = 0.25,
    quote_axis_gain: float = 0.5,
) -> float:
    s = 0.0

    # 1) char match
    q_char = str(r.get("CHAR_ID") or "")
    if q_char and q_char == str(observed_char_id):
        s += quote_char_gain

    # 2) vow match
    q_vow = str(r.get("VOW_ID") or "")
    if q_vow:
        s += quote_vow_gain * float(v_mix_map.get(q_vow, 0.0))
        if q_vow in top_vow_ids:
            s += 0.6

    # 3) keyword match (SENSE_TAG or quote text itself)
    q_sense = str(r.get("SENSE_TAG") or "")
    q_text = str(r.get("QUOTE") or "")
    for kw in keywords:
        if kw and (kw in q_sense or kw in q_text):
            s += quote_kw_gain

    # 4) axis tag match (stage axis label is like "静/流/間/誠" のいずれか)
    q_axis = str(r.get("AXIS_TAG") or "")
    if stage_axis_label and q_axis and (stage_axis_label in q_axis):
        s += quote_axis_gain

    return float(s)

def pick_quote_temperature(
    quotes_df: pd.DataFrame,
    lang: str,
    observed_char_id: str,
    top_vow_ids: List[str],
    v_mix_map: Dict[str, float],
    keywords: List[str],
    stage_axis_label: str,
    temperature: float,
    topn: int = 30,
    rng: Optional[np.random.Generator] = None,
):
    if rng is None:
        rng = np.random.default_rng()

    df = quotes_df.copy()
    df["LANG"] = df["LANG"].fillna("").astype(str)

    # language filter (空なら全部)
    if lang:
        cand = df[df["LANG"].str.lower() == lang.lower()].copy()
        if cand.empty:
            cand = df.copy()
    else:
        cand = df.copy()

    if cand.empty:
        return None, cand

    scores = []
    for _, r in cand.iterrows():
        scores.append(
            quote_score_row(
                r,
                observed_char_id=observed_char_id,
                top_vow_ids=top_vow_ids,
                v_mix_map=v_mix_map,
                keywords=keywords,
                stage_axis_label=stage_axis_label,
            )
        )

    cand["SCORE"] = scores

    # 上位候補に絞って温度付き抽選
    cand = cand.sort_values("SCORE", ascending=False)
    cand_top = cand.head(int(topn)).copy()

    # スコア→確率（温度が高いほどランダム）
    # ここは "energy-like": scoreが高いほど選ばれやすい
    p = softmax(cand_top["SCORE"].to_numpy(dtype=float), temperature=max(1e-6, float(temperature)))

    choice, idx = pick_one_by_prob(cand_top.to_dict("records"), p)
    return choice, cand_top

# ----------------------------
# Image loader
# ----------------------------
@st.cache_data(show_spinner=False)
def load_image(path: str) -> Optional[Image.Image]:
    try:
        if not path or not os.path.exists(path):
            return None
        return Image.open(path)
    except Exception:
        return None

# ----------------------------
# Main UI
# ----------------------------
st.title(APP_TITLE)

with st.sidebar:
    st.header("📁 データ")

    pack_file = st.file_uploader(
        "統合Excel（pack）",
        type=["xlsx"],
        help="quantum_shintaku_pack_v3_with_sense_20260213_oposite_modify.xlsx をアップロード",
    )

    img_dir = st.text_input(
        "🖼️ 画像フォルダ（相対/絶対）",
        value="./assets/images/characters",
        help="ローカル実行: ./assets/images/characters でOK。Windows絶対パスでも可。",
    )

    st.divider()
    st.header("🕰️ 季節×時間（Stage）")
    auto_now = st.checkbox("現在時刻から自動推定", value=True)

    if auto_now:
        from datetime import datetime
        now = datetime.now()
        month = now.month
        hour = now.hour
    else:
        month = st.slider("月", 1, 12, 2)
        hour = st.slider("時刻（0-23）", 0, 23, 21)

    season = season_from_month(month)
    time_slot = time_slot_from_hour(hour)
    stage_id_guess = build_stage_id(season, time_slot)

    # stage override (存在しないstage_idを避ける)
    stage_ids = []
    stage_label_map = {}
    if pack_file is not None:
        try:
            tmp_pack = load_pack_from_uploader(pack_file)
            for _, r in tmp_pack.stage_dict.iterrows():
                sid = str(r["STAGE_ID"])
                stage_ids.append(sid)
                stage_label_map[sid] = str(r.get("LABEL") or sid)
        except Exception:
            stage_ids = []
            stage_label_map = {}

    if stage_ids:
        default_idx = stage_ids.index(stage_id_guess) if stage_id_guess in stage_ids else 0
        stage_id = st.selectbox(
            "STAGE_ID（手動上書き可）",
            options=stage_ids,
            index=default_idx,
            format_func=lambda x: f"{x}  |  {stage_label_map.get(x, '')}",
        )
    else:
        stage_id = stage_id_guess
        st.caption(f"STAGE_ID 推定: {stage_id_guess}（pack未読込のため候補一覧は後で出ます）")

    st.divider()
    st.header("🎛️ 揺らぎ（観測のブレ）")
    beta = st.slider("β（大→最小エネルギー寄り / 小→多様）", 0.2, 6.0, 2.2, 0.1)
    eps_noise = st.slider("微小ノイズ ε（エネルギーに加える）", 0.0, 0.30, 0.08, 0.01)
    n_samples = st.slider("サンプル数（観測分布）", 50, 1000, 300, 50)

    st.divider()
    st.header("🧠 テキスト→誓願（自動ベクトル化）")
    ngram_n = st.selectbox("n-gram", [2, 3], index=0)
    mix_alpha = st.slider("mix比率 α（1=スライダーのみ / 0=テキストのみ）", 0.0, 1.0, 0.55, 0.05)

    st.divider()
    st.header("🗣️ QUOTES神託（温度付き選択）")
    quote_lang = st.selectbox("LANG", ["ja", "en", ""], index=0, help="空は全言語")
    quote_temp = st.slider("格言温度（高→ランダム / 低→上位固定）", 0.2, 3.0, 1.2, 0.1)

# Load pack
if pack_file is None:
    st.info("左のサイドバーから **統合Excel（pack）** をアップロードしてください。")
    st.stop()

try:
    pack = load_pack_from_uploader(pack_file)
except Exception as e:
    st.error(f"統合Excelの解析に失敗: {e}")
    st.stop()

# Build matrices
c2v_df, W_char_vow, vow_cols = build_char_matrix(pack)
A_char_axis, axis_cols = build_char_axis_matrix(pack)

# vow ids in order (VOW_01..)
vow_ids = build_vow_id_list_from_pack(pack, vow_cols)

# VOW_DICT map
vow_title_map = {}
vow_desc_map = {}
for _, r in pack.vow_dict.iterrows():
    vid = str(r["VOW_ID"])
    vow_title_map[vid] = str(r.get("TITLE") or vid)
    # 常時表示したい説明（短いもの）
    hint = str(r.get("SUBTITLE") or r.get("UI_HINT") or r.get("LABEL") or "")
    vow_desc_map[vid] = hint

# Stage axis weights + stage axis label (最も効いてる軸)
stage_axis_w = get_stage_axis_weights(pack, stage_id)
axis_labels = ["静", "流", "間", "誠"]
stage_axis_label = axis_labels[int(np.argmax(np.abs(stage_axis_w)))] if np.any(stage_axis_w) else ""

# Main layout
left, right = st.columns([1.05, 1.0], gap="large")

with left:
    st.subheader("Step 1：誓願入力（スライダー）＋ テキスト（自動ベクトル化）")

    user_text = st.text_area(
        "あなたの状況を一文で（例：疲れていて決断ができない / 新しい挑戦が怖い など）",
        value="",
        height=90,
    )

    st.caption("スライダー入力は **TITLEを常時表示** し、テキストからの自動推定と mix します。")

    manual = np.zeros(len(vow_ids), dtype=float)

    for i, vid in enumerate(vow_ids):
        title = vow_title_map.get(vid, vid)
        hint = vow_desc_map.get(vid, "")
        label = f"{vid}｜{title}"
        manual[i] = st.slider(
            label,
            0.0, 5.0, 0.0, 0.5,
            help=hint if hint else None,
            key=f"vow_slider_{vid}",
        )

    # Auto vector
    vow_corpus = build_vow_text_corpus(pack.vow_dict)
    auto = text_to_vow_auto(user_text, vow_ids, vow_corpus, ngram_n=ngram_n)

    # Mix
    v_mix = mix_alpha * manual + (1.0 - mix_alpha) * auto

    # Show vector table
    vec_df = pd.DataFrame({
        "VOW_ID": vow_ids,
        "TITLE": [vow_title_map.get(v, v) for v in vow_ids],
        "manual(0-5)": np.round(manual, 3),
        "auto(0-5)": np.round(auto, 3),
        "mix(0-5)": np.round(v_mix, 3),
    })
    with st.expander("🔎 誓願ベクトル（manual / auto / mix）"):
        st.dataframe(vec_df, use_container_width=True, hide_index=True)

    # Buttons
    observe_btn = st.button("🧪 観測する（QUBOから抽出）", use_container_width=True)

# Compute energies & distribution
rng = np.random.default_rng()

# stage gain (影響量はUI化しても良いが、まず固定)
stage_gain = 0.35

energy = compute_energy(
    v_mix=v_mix,
    W_char_vow=W_char_vow,
    stage_axis_w=stage_axis_w,
    A_char_axis=A_char_axis,
    stage_gain=stage_gain,
    eps_noise=eps_noise,
    rng=rng
)

p_char, sample_idxs = observe_distribution(energy, beta=beta, n_samples=n_samples, rng=rng)

# chars list
char_ids = c2v_df["CHAR_ID"].astype(str).tolist()
char_names = c2v_df["公式キャラ名"].astype(str).tolist()
img_files = c2v_df["IMAGE_FILE"].astype(str).tolist()

# If not pressed, still show “current” best (argmin energy)
best_idx = int(np.argmin(energy))
observed_idx = int(sample_idxs[-1]) if observe_btn else best_idx
observed_char_id = char_ids[observed_idx]
observed_char_name = char_names[observed_idx]
observed_img_file = img_files[observed_idx]

# Contributing vows (Top)
# Use char's vow weights × mix
char_w = W_char_vow[observed_idx, :]
contrib = char_w * v_mix
top_k = 6
top_idx = np.argsort(contrib)[::-1][:top_k]
top_vow_ids = [vow_ids[i] for i in top_idx]

v_mix_map = {vow_ids[i]: float(v_mix[i]) for i in range(len(vow_ids))}

# Keywords from user text
keywords = extract_keywords_simple(user_text, topk=10)

# Pick quote
quote_choice, quote_top = pick_quote_temperature(
    quotes_df=pack.quotes,
    lang=quote_lang,
    observed_char_id=observed_char_id,
    top_vow_ids=top_vow_ids,
    v_mix_map=v_mix_map,
    keywords=keywords,
    stage_axis_label=stage_axis_label,
    temperature=quote_temp,
    topn=30,
    rng=rng
)

# Build oracle text
top_titles = [vow_title_map.get(v, v) for v in top_vow_ids[:3]]
top_titles_txt = "・".join(top_titles) if top_titles else "（未設定）"

quote_text = ""
quote_source = ""
if quote_choice:
    quote_text = str(quote_choice.get("QUOTE") or "").strip()
    quote_source = str(quote_choice.get("SOURCE") or "").strip()

oracle_lines = []
if user_text.strip():
    oracle_lines.append(f"「{user_text.strip()}」の奥に、**{top_titles_txt}** が見えている。")
else:
    oracle_lines.append(f"いまの波は **{top_titles_txt}** に寄っている。")

if stage_axis_label:
    oracle_lines.append(f"季節×時間の気配（Stage）は **{stage_axis_label}** を強める。")

if quote_text:
    oracle_lines.append(f"格言：『{quote_text}』")
    if quote_source:
        oracle_lines.append(f"— {quote_source}")

oracle = "\n".join(oracle_lines)

# Right column outputs
with right:
    st.subheader("Step 3：結果（観測された神＋理由＋QUOTES神託）")

    # Table of top 3 by energy
    rank_idx = np.argsort(energy)[:3]
    rank_df = pd.DataFrame({
        "順位": [1, 2, 3],
        "CHAR_ID": [char_ids[i] for i in rank_idx],
        "神": [char_names[i] for i in rank_idx],
        "energy（低いほど選ばれやすい）": [float(np.round(energy[i], 4)) for i in rank_idx],
        "確率p（softmax）": [float(np.round(p_char[i], 4)) for i in rank_idx],
    })
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

    st.markdown(f"### 🌟 今回“観測”された神：**{observed_char_name}**（{observed_char_id}）")
    st.caption(
        "※ここは「単発の観測（1回抽選）」です。下の📊観測分布（サンプル）は「同条件で何回も観測したらどう出るか」のヒストグラムです。"
        " そのため、分布の最多と単発の観測結果が一致しないことがあります（正常挙動）。"
    )

    # Image
    img_path = os.path.join(img_dir, observed_img_file) if observed_img_file else ""
    img = load_image(img_path)
    if img is not None:
        st.image(img, caption=f"{observed_char_name}（{observed_img_file}）", use_container_width=True)
    else:
        st.warning(f"画像が見つかりません: {img_path}")

    # Oracle
    st.success(oracle)

    # Contrib table
    contrib_df = pd.DataFrame({
        "VOW": top_vow_ids,
        "TITLE": [vow_title_map.get(v, v) for v in top_vow_ids],
        "mix(v)": [float(np.round(v_mix_map.get(v, 0.0), 3)) for v in top_vow_ids],
        "W(char,v)": [float(np.round(char_w[vow_ids.index(v)], 3)) for v in top_vow_ids],
        "寄与(v*w)": [float(np.round(contrib[vow_ids.index(v)], 3)) for v in top_vow_ids],
    })
    st.markdown("#### 🧩 寄与した誓願（Top）")
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    # Quote debug
    st.markdown("#### 🗣️ QUOTES神託（温度付きで選択）")
    if quote_text:
        st.info(f"『{quote_text}』\n\n— {quote_source}")
        with st.expander("🔎 格言候補Top（デバッグ）"):
            show_cols = [c for c in ["QUOTE_ID","QUOTE","SOURCE","LANG","CHAR_ID","VOW_ID","SENSE_TAG","AXIS_TAG","SCORE"] if c in quote_top.columns]
            st.dataframe(quote_top[show_cols].head(10), use_container_width=True, hide_index=True)
    else:
        st.warning("QUOTESから格言が選べませんでした（LANGフィルタやシート内容を確認してください）。")

# Visualizations (bottom)
st.divider()
st.subheader("📊 可視化：テキストの影響・観測分布・エネルギー地形")

colA, colB = st.columns([1, 1], gap="large")

with colA:
    st.markdown("### 1) テキスト→誓願 自動推定の影響（auto vs manual vs mix）")
    plot_df = pd.DataFrame({
        "VOW": vow_ids,
        "manual": manual,
        "auto": auto,
        "mix": v_mix
    })
    st.caption("auto（テキスト由来）と manual（スライダー）と mix の差が見える化されます。")
    st.line_chart(plot_df.set_index("VOW"))

    st.markdown("### 2) エネルギー地形（全候補）")
    land_df = pd.DataFrame({
        "CHAR": char_names,
        "energy": energy,
        "p": p_char
    }).sort_values("energy", ascending=True)
    st.bar_chart(land_df.set_index("CHAR")["energy"])

with colB:
    st.markdown("### 3) 観測分布（サンプル）")
    # histogram of sampled chars
    from collections import Counter
    cnt = Counter(sample_idxs.tolist())
    hist_df = pd.DataFrame({
        "CHAR": [char_names[i] for i in range(len(char_names))],
        "count": [cnt.get(i, 0) for i in range(len(char_names))]
    }).sort_values("count", ascending=False)
    st.bar_chart(hist_df.set_index("CHAR")["count"])

    st.markdown("### 4) テキストのキーワード抽出（簡易）")
    if keywords:
        st.write(" / ".join(keywords))
    else:
        st.caption("（入力テキストが短い/空のため、キーワードが抽出できません）")

st.caption("© Q-Quest / Quantum Shintaku prototype (app09)")
