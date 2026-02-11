# app07_fixed.py  (rev2)
# ------------------------------------------------------------
# 修正点:
# - CHAR_MASTERにIMAGE_FILEが無くても停止しない（警告扱い）
# - 画像ファイル名は CHAR_TO_VOW の IMAGE_FILE を優先利用
# - VOW_DICTの説明列は DESCRIPTION_LONG を優先（無ければDESCRIPTION等）
# ------------------------------------------------------------

from __future__ import annotations
import re
import math
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Utilities
# =========================
def _norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u3000", " ").strip()
    return s


def char_ngrams(text: str, n: int = 3) -> Dict[str, int]:
    t = _norm_text(text)
    t = re.sub(r"\s+", "", t)
    if len(t) < n:
        return {}
    grams: Dict[str, int] = {}
    for i in range(len(t) - n + 1):
        g = t[i : i + n]
        grams[g] = grams.get(g, 0) + 1
    return grams


def cosine_sim(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb:
            dot += va * vb
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def softmax_from_energy(energy: np.ndarray, beta: float = 1.0) -> np.ndarray:
    # p ~ exp(-beta * E)
    z = -beta * energy
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower_map = {str(c).lower(): c for c in cols}
    for c in candidates:
        k = str(c).lower()
        if k in lower_map:
            return lower_map[k]
    return None


# =========================
# Load Excel
# =========================
@st.cache_data(show_spinner=False)
def load_pack_excel_bytes(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    bio = BytesIO(file_bytes)
    xls = pd.ExcelFile(bio)
    sheets = xls.sheet_names

    vow_df = pd.read_excel(bio, sheet_name="VOW_DICT")
    bio.seek(0)
    char_df = pd.read_excel(bio, sheet_name="CHAR_MASTER")
    bio.seek(0)
    ctv_df = pd.read_excel(bio, sheet_name="CHAR_TO_VOW")

    return vow_df, char_df, ctv_df, sheets


@st.cache_data(show_spinner=False)
def load_quotes_excel_bytes(file_bytes: bytes) -> pd.DataFrame:
    bio = BytesIO(file_bytes)
    df = pd.read_excel(bio)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    return df


def validate_pack(vow_df: pd.DataFrame, char_df: pd.DataFrame, ctv_df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    """return (fatal_errors, warnings)"""
    fatal: Dict[str, str] = {}
    warn: Dict[str, str] = {}

    # VOW_DICT
    vow_id = find_col(vow_df, ["VOW_ID", "VOW", "vow_id", "vow"])
    vow_title = find_col(vow_df, ["TITLE", "誓願", "誓願名", "title"])
    # 説明は DESCRIPTION_LONG を優先
    vow_desc = find_col(vow_df, ["DESCRIPTION_LONG", "DESCRIPTION", "DESC", "説明", "description"])

    if vow_id is None or vow_title is None:
        fatal["VOW_DICT"] = f"VOW_DICTの必須列(VOW_ID/TITLE)が見つかりません。列={list(vow_df.columns)}"
    if vow_desc is None:
        warn["VOW_DICT_DESC"] = f"VOW_DICTの説明列(DESCRIPTION_LONG等)が見つかりません（TITLEのみで進みます）。列={list(vow_df.columns)}"

    # CHAR_MASTER
    char_id = find_col(char_df, ["CHAR_ID", "CHAR", "char_id"])
    char_name = find_col(char_df, ["公式キャラ名", "CHAR_NAME", "名前", "キャラ名", "char_name"])
    if char_id is None or char_name is None:
        fatal["CHAR_MASTER"] = f"CHAR_MASTERの必須列(CHAR_ID/公式キャラ名)が見つかりません。列={list(char_df.columns)}"

    # CHAR_TO_VOW
    vow_cols = [c for c in ctv_df.columns if re.match(r"^VOW_\d+$", str(c))]
    if not vow_cols:
        fatal["CHAR_TO_VOW"] = f"CHAR_TO_VOWにVOW_01..列が見つかりません。列={list(ctv_df.columns)}"

    # 画像列は CHAR_TO_VOW の IMAGE_FILE を優先
    ctv_img = find_col(ctv_df, ["IMAGE_FILE", "image_file", "FILE", "ファイル"])
    if ctv_img is None:
        warn["IMAGE_FILE"] = f"CHAR_TO_VOWにIMAGE_FILEが見つかりません（画像表示なしで進みます）。列={list(ctv_df.columns)}"

    return fatal, warn


def build_matrices(
    vow_df: pd.DataFrame, char_df: pd.DataFrame, ctv_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str], np.ndarray]:
    # VOW_DICT columns
    vow_id_col = find_col(vow_df, ["VOW_ID", "VOW", "vow_id", "vow"])
    vow_title_col = find_col(vow_df, ["TITLE", "誓願", "誓願名", "title"])
    vow_desc_col = find_col(vow_df, ["DESCRIPTION_LONG", "DESCRIPTION", "DESC", "説明", "description"])

    vow_tbl = vow_df.copy()
    vow_tbl = vow_tbl.rename(columns={vow_id_col: "VOW_ID", vow_title_col: "TITLE"})
    if vow_desc_col:
        vow_tbl = vow_tbl.rename(columns={vow_desc_col: "DESCRIPTION"})
    else:
        vow_tbl["DESCRIPTION"] = ""

    vow_tbl["VOW_ID"] = vow_tbl["VOW_ID"].astype(str)
    vow_tbl["TITLE"] = vow_tbl["TITLE"].astype(str)
    vow_tbl["DESCRIPTION"] = vow_tbl["DESCRIPTION"].astype(str)
    vow_tbl["VOW_TEXT"] = (vow_tbl["TITLE"].fillna("") + " " + vow_tbl["DESCRIPTION"].fillna("")).map(_norm_text)

    # CHAR_MASTER columns
    char_id_col = find_col(char_df, ["CHAR_ID", "CHAR", "char_id"])
    char_name_col = find_col(char_df, ["公式キャラ名", "CHAR_NAME", "名前", "キャラ名", "char_name"])

    char_tbl = char_df.copy()
    char_tbl = char_tbl.rename(columns={char_id_col: "CHAR_ID", char_name_col: "CHAR_NAME"})
    char_tbl["CHAR_ID"] = char_tbl["CHAR_ID"].astype(str)
    char_tbl["CHAR_NAME"] = char_tbl["CHAR_NAME"].astype(str)

    # CHAR_TO_VOW: weights + image file
    ctv_char_id = find_col(ctv_df, ["CHAR_ID", "CHAR", "char_id"])
    ctv_img_col = find_col(ctv_df, ["IMAGE_FILE", "image_file", "FILE", "ファイル"])

    vow_cols = [c for c in ctv_df.columns if re.match(r"^VOW_\d+$", str(c))]
    vow_cols_sorted = sorted(vow_cols, key=lambda x: int(str(x).split("_")[1]))

    # VOW_01.. と VOW_ID は順番対応
    vow_ids = list(vow_tbl["VOW_ID"].values)
    k = min(len(vow_cols_sorted), len(vow_ids))
    vow_cols_sorted = vow_cols_sorted[:k]
    vow_ids = vow_ids[:k]

    # weight matrix
    use_cols = [ctv_char_id] + ([ctv_img_col] if ctv_img_col else []) + vow_cols_sorted
    tmp = ctv_df[use_cols].copy()
    tmp[ctv_char_id] = tmp[ctv_char_id].astype(str)

    # image file map
    if ctv_img_col:
        img_map = tmp[[ctv_char_id, ctv_img_col]].rename(columns={ctv_char_id: "CHAR_ID", ctv_img_col: "IMAGE_FILE"})
        img_map["CHAR_ID"] = img_map["CHAR_ID"].astype(str)
        img_map["IMAGE_FILE"] = img_map["IMAGE_FILE"].astype(str)
    else:
        img_map = pd.DataFrame({"CHAR_ID": tmp[ctv_char_id].astype(str), "IMAGE_FILE": ""})

    W_tbl = tmp[[ctv_char_id] + vow_cols_sorted].rename(columns={ctv_char_id: "CHAR_ID"}).set_index("CHAR_ID")
    rename_map = {vow_cols_sorted[i]: vow_ids[i] for i in range(k)}
    W_tbl = W_tbl.rename(columns=rename_map)
    W_tbl = W_tbl.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # sync ids
    W_tbl = W_tbl.loc[W_tbl.index.intersection(char_tbl["CHAR_ID"])]
    char_ids = list(W_tbl.index.values)
    W = W_tbl.values.astype(float)

    # build final char table aligned + image file
    char_tbl = char_tbl.set_index("CHAR_ID").loc[char_ids].reset_index()
    char_tbl = char_tbl.merge(img_map, on="CHAR_ID", how="left")
    char_tbl["IMAGE_FILE"] = char_tbl["IMAGE_FILE"].fillna("").astype(str)

    return vow_tbl[["VOW_ID", "TITLE", "DESCRIPTION", "VOW_TEXT"]], char_tbl, W_tbl, vow_ids, char_ids, W


# =========================
# UI
# =========================
st.set_page_config(page_title="Q-Quest 量子神託", layout="wide")
st.title("🔮 Q-Quest 量子神託（app07_fixed：テキスト影響の可視化つき）")

base_dir = Path(__file__).resolve().parent
img_dir_default = base_dir / "assets" / "images" / "characters"

with st.sidebar:
    st.header("📁 データ")
    st.caption("統合Excel（pack）を優先して読み込みます。")

    pack_up = st.file_uploader("統合Excel（quantum_shintaku_pack...）", type=["xlsx"])
    quote_up = st.file_uploader("格言Excel（任意）", type=["xlsx"])

    st.divider()
    st.subheader("🖼️ 画像フォルダ")
    st.caption("ローカル実行では通常ここでOK：./assets/images/characters/")
    img_dir_text = st.text_input("画像フォルダ（相対/絶対）", value=str(img_dir_default))

if pack_up is None:
    st.warning("左のサイドバーから **統合Excel（pack）** をアップロードしてください。")
    st.stop()

try:
    vow_df, char_df, ctv_df, sheet_names = load_pack_excel_bytes(pack_up.getvalue())
except Exception as e:
    st.error(f"統合Excelの読み込みに失敗しました: {e}")
    st.stop()

fatal, warn = validate_pack(vow_df, char_df, ctv_df)
if warn:
    st.warning("⚠️ 一部の列が見つかりませんが、動作は継続します。")
    st.write(warn)

if fatal:
    st.error("列名が想定と合いません。VOW_DICT/CHAR_MASTER/CHAR_TO_VOW の列名を確認してください。")
    st.write("検出したシート名:", sheet_names)
    st.write("VOW_DICT:", list(vow_df.columns))
    st.write("CHAR_MASTER:", list(char_df.columns))
    st.write("CHAR_TO_VOW:", list(ctv_df.columns))
    st.write("詳細:", fatal)
    st.stop()

vow_tbl, char_tbl, W_tbl, vow_ids, char_ids, W = build_matrices(vow_df, char_df, ctv_df)

quotes_df = None
if quote_up is not None:
    try:
        quotes_df = load_quotes_excel_bytes(quote_up.getvalue())
    except Exception as e:
        st.warning(f"格言Excelの読み込みに失敗（格言はスキップ）: {e}")
        quotes_df = None


# =========================
# Step 1: input
# =========================
colL, colR = st.columns([1.25, 1.0], gap="large")

with colL:
    st.subheader("✅ Step 1：誓願入力（スライダー） + テキスト（自動ベクトル化）")

    user_text = st.text_area(
        "あなたの状況を一文〜数文で（例：疲れていて決断できない／新規PJを頑張る など）",
        value="",
        height=90,
        placeholder="ここに入力すると、誓願ベクトル（auto）が変化します",
    )

    st.caption("スライダー（0〜5）で手動入力できます。テキストからの自動推定も混ぜられます。")

    manual = np.zeros(len(vow_ids), dtype=float)
    for i, row in enumerate(vow_tbl.itertuples(index=False)):
        label = f"{row.VOW_ID}｜{row.TITLE}"
        manual[i] = st.slider(label, 0.0, 5.0, 0.0, 0.5)

    st.divider()
    st.subheader("⚙️ 揺らぎ（観測のブレ）パラメータ")
    mix_alpha = st.slider("mix（手動 vs 自動） 0=手動のみ / 1=自動のみ", 0.0, 1.0, 0.45, 0.05)
    temperature = st.slider("温度（格言選択のランダムさ）", 0.05, 2.0, 0.70, 0.05)
    beta = st.slider("分布の尖り具合 β（低いほどランダム）", 0.1, 10.0, 2.2, 0.1)
    sample_n = st.slider("サンプル回数（分布可視化用）", 50, 2000, 400, 50)
    eps = st.slider("微小ゆらぎ ε（エネルギーに加えるノイズ）", 0.0, 0.30, 0.05, 0.01)

    run = st.button("🧿 観測する（QUBOから抽出）", use_container_width=True)

# =========================
# text -> auto vow vector
# =========================
vow_ng = [char_ngrams(t, n=3) for t in vow_tbl["VOW_TEXT"].tolist()]
user_ng = char_ngrams(user_text, n=3)
sims = np.array([cosine_sim(user_ng, v) for v in vow_ng], dtype=float)

if float(np.max(sims)) > 0:
    auto = 5.0 * (sims / float(np.max(sims)))
else:
    auto = np.zeros_like(sims)

mix = (1.0 - mix_alpha) * manual + mix_alpha * auto

# =========================
# Visualization: text influence
# =========================
with colR:
    st.subheader("👁️ テキスト入力の影響（見える化）")

    delta = mix - manual
    df_viz = pd.DataFrame(
        {
            "VOW": vow_ids,
            "TITLE": vow_tbl["TITLE"].values,
            "manual(0-5)": manual,
            "auto(0-5)": auto,
            "mix(0-5)": mix,
            "delta(text)": delta,
            "sim(ngram)": sims,
        }
    )
    st.caption("✅ **delta(text)** が大きいほど、あなたの文章がその誓願を強めています。")
    st.dataframe(df_viz.sort_values("delta(text)", ascending=False).reset_index(drop=True),
                 use_container_width=True, height=260)

    topk = df_viz.sort_values("delta(text)", ascending=False).head(8)
    st.caption("📊 delta(text) 上位（文章が効いている誓願）")
    st.bar_chart(pd.DataFrame({"delta(text)": topk["delta(text)"].values},
                              index=[f"{v}" for v in topk["VOW"]]),
                 use_container_width=True)

    st.caption("📊 manual vs auto vs mix（上位だけ）")
    st.bar_chart(topk.set_index("VOW")[["manual(0-5)", "auto(0-5)", "mix(0-5)"]],
                 use_container_width=True)

# =========================
# Energy / Observe
# =========================
def compute_energy(mix_vec: np.ndarray, W: np.ndarray, eps: float) -> np.ndarray:
    base = -W.dot(mix_vec)  # lower is better
    if eps > 0:
        base = base + np.random.normal(0.0, eps, size=base.shape)
    return base


if run:
    energies = compute_energy(mix, W, eps=eps)
    probs = softmax_from_energy(energies, beta=beta)

    obs_idx = int(np.random.choice(len(char_ids), p=probs))
    obs_char_id = char_ids[obs_idx]
    obs_row = char_tbl.iloc[obs_idx]
    obs_name = obs_row["CHAR_NAME"]
    obs_img = obs_row.get("IMAGE_FILE", "")

    # sampling distribution
    samples = []
    for _ in range(sample_n):
        e = compute_energy(mix, W, eps=eps)
        p = softmax_from_energy(e, beta=beta)
        k = int(np.random.choice(len(char_ids), p=p))
        samples.append(char_ids[k])
    s_counts = pd.Series(samples).value_counts()
    dist_df = pd.DataFrame({"count": s_counts}).reset_index().rename(columns={"index": "CHAR_ID"})
    dist_df = dist_df.merge(char_tbl, on="CHAR_ID", how="left")

    # contributions
    w_obs = W_tbl.loc[obs_char_id].values.astype(float)
    contrib = mix * w_obs
    contrib_df = pd.DataFrame(
        {"VOW": vow_ids, "TITLE": vow_tbl["TITLE"].values, "mix(v)": mix, "W(char,v)": w_obs, "寄与(v*w)": contrib}
    ).sort_values("寄与(v*w)", ascending=False)

    top_titles = contrib_df.head(5)["TITLE"].tolist()
    oracle = f"「{_norm_text(user_text)}」の奥に、" + "・".join(top_titles[:3]) + " が見えている。焦らず、次の一手を小さく刻め。"
    oracle = oracle.replace("の奥に、・", "の奥に、")

    st.divider()
    st.subheader("✅ Step 3：結果（観測された神 + 理由 + 格言）")

    c1, c2 = st.columns([0.9, 1.1], gap="large")

    with c1:
        st.markdown(f"### 🌟 今回“観測”された神：**{obs_name}**（{obs_char_id}）")
        st.caption(f"Energy={energies[obs_idx]:.4f} / p={probs[obs_idx]:.4f} / β={beta:.2f} / ε={eps:.2f}")

        img_dir = Path(img_dir_text)
        img_path = (img_dir / obs_img) if obs_img else None
        if img_path and img_path.exists():
            st.image(str(img_path), use_container_width=True, caption=f"{obs_name} ({obs_img})")
        else:
            st.warning(f"画像が見つかりません: {img_path}")

        st.success(oracle)

    with c2:
        st.markdown("### 🧩 寄与した誓願（Top）")
        view = contrib_df.head(8).copy()
        for c in ["mix(v)", "W(char,v)", "寄与(v*w)"]:
            view[c] = view[c].astype(float).round(3)
        st.dataframe(view.reset_index(drop=True), use_container_width=True, height=260)

        st.markdown("### 🗺️ エネルギー地形（全候補）")
        land = pd.DataFrame({"CHAR_ID": char_ids, "CHAR_NAME": char_tbl["CHAR_NAME"].values, "energy": energies, "p": probs})
        land = land.sort_values("energy", ascending=True).reset_index(drop=True)
        st.dataframe(land.head(12), use_container_width=True, height=260)

    st.markdown("### 📊 観測分布（サンプル）")
    top_dist = dist_df.sort_values("count", ascending=False).head(12).set_index("CHAR_NAME")[["count"]]
    st.bar_chart(top_dist, use_container_width=True)

    # Quotes
    if quotes_df is not None and len(quotes_df) > 0:
        st.markdown("### 🗣️ 格言（温度付きで選択）")

        q_text_col = find_col(quotes_df, ["格言", "QUOTE", "quote", "言葉"])
        q_src_col = find_col(quotes_df, ["出典", "SOURCE", "source", "作者"])
        q_tag_col = find_col(quotes_df, ["感覚タグ", "TAG", "tag"])
        q_char_col = find_col(quotes_df, ["公式キャラ名", "CHAR_NAME", "キャラ名"])

        if q_text_col is None:
            st.warning("格言Excelに「格言」列が見つからないため、格言表示はスキップします。")
        else:
            qdf = quotes_df.copy()
            qdf["_quote"] = qdf[q_text_col].astype(str)
            qdf["_src"] = qdf[q_src_col].astype(str) if q_src_col else ""

            score = np.zeros(len(qdf), dtype=float)

            if q_char_col:
                score += (qdf[q_char_col].astype(str) == obs_name).astype(float) * 1.5

            if q_tag_col:
                tags = qdf[q_tag_col].astype(str).fillna("")
                for j, t in enumerate(tags.values):
                    for tt in top_titles[:5]:
                        if tt and tt in t:
                            score[j] += 0.6

            score += np.random.normal(0, 0.05, size=score.shape)

            # 温度でランダム度調整（scoreが高いほど選ばれやすくしたい）
            temp = max(1e-6, float(temperature))
            z = (score - np.max(score)) / temp
            p = np.exp(z)
            p = p / np.sum(p)
            idx = int(np.random.choice(len(qdf), p=p))
            picked = qdf.iloc[idx]

            st.info(f"“{picked['_quote']}” — {picked['_src']}".strip())

            with st.expander("格言候補 Top10（デバッグ）"):
                dbg = qdf.assign(score=score).sort_values("score", ascending=False).head(10)
                cols = [c for c in [q_text_col, q_src_col, q_tag_col, q_char_col] if c]
                st.dataframe(dbg[cols + ["score"]], use_container_width=True)
else:
    st.caption("※「観測する」を押すと、誓願×キャラ重み（+ノイズ）で観測結果が変化します。")
