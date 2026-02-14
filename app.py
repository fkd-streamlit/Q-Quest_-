# app08.py
# ============================================================
# 🔮 Q-Quest 量子神託 app08（完成版）
# - 統合Excel（pack）を最優先で読み込み（アップロード対応）
# - 誓願スライダー + テキスト入力 → VOWベクトル生成（SENSE辞書 + char n-gram補助）
# - QUBO（12変数）を SA（焼きなまし）でサンプリング → 観測分布（ヒスト）
# - 「今回観測された神」＋「入力の影響（寄与の可視化）」＋「神託文（格言/金言付き）」
#
# 期待する統合Excelの主シート名（同名が望ましい）：
#   VOW_DICT, CHAR_TO_VOW, CHAR_MASTER, SENSE_DICT, SENSE_TO_VOW, (任意: QUOTES)
#
# 既知の列名（あなたの統合Excelに合わせて吸収）：
#   VOW_DICT: VOW_ID, LABEL, TITLE, SUBTITLE, DESCRIPTION_LONG, UI_HINT ...
#   CHAR_TO_VOW: CHAR_ID, IMAGE_FILE, 公式キャラ名, VOW_01..VOW_12
#   CHAR_MASTER: CHAR_ID, 公式キャラ名, 役割, 役割補足説明, 絵馬文字分析, VOW_01..12, AXIS_*
#   SENSE_TO_VOW: (例) SENSE, VOW_ID, WEIGHT など（多少ゆらいでも吸収）
#   QUOTES: TEXT / QUOTE / 格言 / BODY など（多少ゆらいでも吸収）
#
# 画像フォルダ：
#   ./assets/images/characters/ （ローカル実行推奨）
# ============================================================

import os
import re
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# -----------------------------
# 基本設定
# -----------------------------
st.set_page_config(page_title="🔮 Q-Quest 量子神託 app08", layout="wide")


# -----------------------------
# 小道具
# -----------------------------
def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def _safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def _first_existing(col_candidates: List[str], cols: List[str]) -> Optional[str]:
    cols_set = set(cols)
    for c in col_candidates:
        if c in cols_set:
            return c
    return None

def _ensure_vow_cols(df: pd.DataFrame, n_vow: int = 12) -> List[str]:
    """VOW_01..VOW_12 の列名を探して返す（無ければエラー）"""
    needed = [f"VOW_{i:02d}" for i in range(1, n_vow + 1)]
    cols = list(df.columns)
    missing = [c for c in needed if c not in cols]
    if missing:
        raise ValueError(f"VOW列が不足しています: {missing}")
    return needed

def _read_image_maybe(path: str) -> Optional[Image.Image]:
    try:
        if path and os.path.exists(path):
            return Image.open(path)
    except Exception:
        return None
    return None

def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(1e-6, float(temperature))
    z = logits / t
    z = z - np.max(z)
    e = np.exp(z)
    s = e / (np.sum(e) + 1e-12)
    return s

def _tokenize_jp_loose(text: str) -> List[str]:
    """形態素なしの“ゆるい”トークン化（日本語/英語混在でもOK）"""
    if not text:
        return []
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    # 記号で区切る
    parts = re.split(r"[ \t\n\r,，、。．.!！?？:：;；/／()\[\]{}「」『』“”\"'`~\-_=+<>＜＞]+", t)
    return [p for p in parts if p]

def _char_ngrams(text: str, n: int = 3) -> List[str]:
    t = re.sub(r"\s+", "", (text or ""))
    if len(t) < n:
        return [t] if t else []
    return [t[i:i+n] for i in range(len(t) - n + 1)]


# -----------------------------
# データロード（統合Excel優先）
# -----------------------------
@dataclass
class PackData:
    sheets: Dict[str, pd.DataFrame]
    vow_dict: pd.DataFrame
    char_to_vow: pd.DataFrame
    char_master: pd.DataFrame
    sense_dict: Optional[pd.DataFrame]
    sense_to_vow: Optional[pd.DataFrame]
    quotes: Optional[pd.DataFrame]
    sheet_names: List[str]

def load_pack_excel(file) -> PackData:
    # pandasの dict(DataFrame) は普通にpickle可能ですが、環境差で st.cache_data がコケることがあるため、あえてキャッシュしません。
    xls = pd.read_excel(file, sheet_name=None, engine="openpyxl")
    sheets = {str(k): v.copy() for k, v in xls.items()}
    sheet_names = list(sheets.keys())

    # 必須シート
    if "VOW_DICT" not in sheets:
        raise ValueError("統合Excelに VOW_DICT シートが見つかりません。")
    if "CHAR_TO_VOW" not in sheets:
        raise ValueError("統合Excelに CHAR_TO_VOW シートが見つかりません。")
    if "CHAR_MASTER" not in sheets:
        raise ValueError("統合Excelに CHAR_MASTER シートが見つかりません。")

    vow_dict = sheets["VOW_DICT"]
    char_to_vow = sheets["CHAR_TO_VOW"]
    char_master = sheets["CHAR_MASTER"]

    sense_dict = sheets.get("SENSE_DICT", None)
    sense_to_vow = sheets.get("SENSE_TO_VOW", None)
    quotes = sheets.get("QUOTES", None)  # まとめたならここに入っている想定

    return PackData(
        sheets=sheets,
        vow_dict=vow_dict,
        char_to_vow=char_to_vow,
        char_master=char_master,
        sense_dict=sense_dict,
        sense_to_vow=sense_to_vow,
        quotes=quotes,
        sheet_names=sheet_names,
    )


# -----------------------------
# SENSE→VOW マップ構築（列名ゆらぎ吸収）
# -----------------------------
def build_sense_maps(sense_dict: Optional[pd.DataFrame], sense_to_vow: Optional[pd.DataFrame]) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """
    返り値:
      sense_label_map: 正規化した sense_key -> 表示ラベル
      sense2vow_vec:   正規化した sense_key -> vow_vec(12)
    """
    sense_label_map: Dict[str, str] = {}
    sense2vow_vec: Dict[str, np.ndarray] = {}

    if sense_to_vow is None or len(sense_to_vow) == 0:
        return sense_label_map, sense2vow_vec

    df = sense_to_vow.copy()
    cols = list(df.columns)

    # sense列候補
    sense_col = _first_existing(
        ["SENSE", "SENSE_KEY", "SENSE_ID", "KEY", "WORD", "TERM", "キーワード", "言葉", "概念"],
        cols,
    )
    # vow列候補（VOW_IDが理想だが、VOW_01..12 直列の可能性もある）
    vow_id_col = _first_existing(["VOW_ID", "VOW", "VOW_KEY", "誓願ID", "誓願"], cols)
    weight_col = _first_existing(["WEIGHT", "W", "SCORE", "重み", "寄与"], cols)

    # Case A: 行形式（sense, vow_id, weight）
    if sense_col and vow_id_col:
        # VOW_IDが "VOW_01" 形式 / "01" / 1 など揺れるので吸収
        def vow_index(v):
            if pd.isna(v):
                return None
            s = str(v).strip()
            m = re.search(r"(\d+)", s)
            if not m:
                return None
            k = int(m.group(1))
            if 1 <= k <= 12:
                return k - 1
            return None

        for _, r in df.iterrows():
            sk = str(r.get(sense_col, "")).strip()
            if not sk:
                continue
            key = sk.lower()
            idx = vow_index(r.get(vow_id_col))
            if idx is None:
                continue
            w = _safe_float(r.get(weight_col), default=1.0) if weight_col else 1.0
            if key not in sense2vow_vec:
                sense2vow_vec[key] = np.zeros(12, dtype=float)
            sense2vow_vec[key][idx] += w

        # ラベル（SENSE_DICTがあれば優先）
        if sense_dict is not None and len(sense_dict) > 0:
            sdc = list(sense_dict.columns)
            key_col = _first_existing(["SENSE", "SENSE_KEY", "KEY", "SENSE_ID", "ID", "概念"], sdc)
            label_col = _first_existing(["LABEL", "NAME", "表示名", "名称", "ラベル"], sdc)
            if key_col:
                for _, r in sense_dict.iterrows():
                    sk = str(r.get(key_col, "")).strip()
                    if not sk:
                        continue
                    sense_label_map[sk.lower()] = str(r.get(label_col, sk)).strip() if label_col else sk

        # sense_dictが無いなら自前
        for k in list(sense2vow_vec.keys()):
            if k not in sense_label_map:
                sense_label_map[k] = k

        return sense_label_map, sense2vow_vec

    # Case B: 列形式（sense列 + VOW_01..12列）
    if sense_col:
        try:
            vow_cols = _ensure_vow_cols(df, 12)
            for _, r in df.iterrows():
                sk = str(r.get(sense_col, "")).strip()
                if not sk:
                    continue
                key = sk.lower()
                vec = np.array([_safe_float(r.get(c), 0.0) for c in vow_cols], dtype=float)
                if np.allclose(vec, 0):
                    continue
                sense2vow_vec[key] = vec
                sense_label_map[key] = key
            return sense_label_map, sense2vow_vec
        except Exception:
            return sense_label_map, sense2vow_vec

    return sense_label_map, sense2vow_vec


# -----------------------------
# テキスト → VOW ベクトル
# -----------------------------
def text_to_vow_vector(
    text: str,
    sense_label_map: Dict[str, str],
    sense2vow_vec: Dict[str, np.ndarray],
    ngram_n: int = 3,
) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """
    返り値:
      v_text: 12次元
      hits:   [(sense_label, hit_score)] ざっくり説明用
    """
    if not text:
        return np.zeros(12, dtype=float), []

    t = text.strip().lower()
    if not t:
        return np.zeros(12, dtype=float), []

    # 1) トークン（単語）一致
    tokens = _tokenize_jp_loose(text)
    tokens_l = [x.lower() for x in tokens]

    # 2) char n-gram 一致（日本語の「部分一致」補助）
    ngrams = _char_ngrams(t, n=ngram_n)
    ngrams_set = set([g.lower() for g in ngrams if g])

    v = np.zeros(12, dtype=float)
    hits: List[Tuple[str, float]] = []

    # senseキー集合
    sense_keys = list(sense2vow_vec.keys())
    for sk in sense_keys:
        key = sk.lower()
        # ざっくりスコア（単語完全一致 > 部分一致）
        score = 0.0
        if key in tokens_l:
            score += 2.0
        # 部分一致（ngram側）
        if key and (key in ngrams_set):
            score += 1.0
        # 文字列包含（保険）
        if key and (key in t):
            score += 0.5

        if score > 0:
            v += score * sense2vow_vec[sk]
            hits.append((sense_label_map.get(sk, sk), float(score)))

    # 正規化（テキストは暴れやすいので軽く）
    if np.linalg.norm(v) > 1e-12:
        v = v / (np.linalg.norm(v) + 1e-12)

    # ヒット上位を返す
    hits.sort(key=lambda x: x[1], reverse=True)
    return v, hits[:12]


# -----------------------------
# QUBO（12変数） + SAサンプリング
# -----------------------------
@dataclass
class QuboModel:
    h: np.ndarray          # (12,)
    J: np.ndarray          # (12,12) 上三角を使う
    lam: float             # 制約ペナルティ
    target_k: int          # 選択数の目標（基本1）

def build_qubo(
    W_char_vow: np.ndarray,    # (12,12)
    v_total: np.ndarray,       # (12,)
    beta_pair: float = 0.15,
    lam: float = 3.0,
    target_k: int = 1,
) -> QuboModel:
    """
    E(x) = sum_i h_i x_i + sum_{i<j} J_ij x_i x_j + lam*(sum x - target_k)^2
    """
    # 一次項：誓願との整合（低いほど選ばれやすい）
    # score = Wv・v_total
    scores = W_char_vow @ v_total  # (12,)
    h = -scores.copy()

    # 二次項：キャラ間相性（近いほど（同時に立ちやすい／立ちにくい））
    # ここでは「似てる同士は同時に選ぶとエネルギーが上がる」= 競合として +sim
    # → 12神が僅差で揺らぐと、分布が“地形っぽく”なる
    Wn = W_char_vow.copy().astype(float)
    # 正規化
    norms = np.linalg.norm(Wn, axis=1, keepdims=True) + 1e-12
    Wn = Wn / norms
    sim = Wn @ Wn.T  # cosine
    J = beta_pair * sim
    np.fill_diagonal(J, 0.0)

    return QuboModel(h=h, J=J, lam=float(lam), target_k=int(target_k))

def qubo_energy(x: np.ndarray, model: QuboModel) -> float:
    x = x.astype(float)
    # linear
    e = float(np.dot(model.h, x))
    # quadratic (i<j)
    # x^T J x /2 だが対角0にして対称なので半分
    e += 0.5 * float(x @ model.J @ x)
    # constraint
    s = float(np.sum(x))
    e += model.lam * (s - model.target_k) ** 2
    return e

def sa_sample(model: QuboModel, n_steps: int = 300, t0: float = 2.0, t1: float = 0.3) -> np.ndarray:
    """メトロポリスSAで1サンプル（12次元二値）"""
    d = model.h.shape[0]
    x = (np.random.rand(d) < 0.3).astype(int)

    # 0ベクトル対策：最低1個は立てて開始
    if x.sum() == 0:
        x[np.random.randint(0, d)] = 1

    e = qubo_energy(x, model)

    for step in range(n_steps):
        # 温度スケジュール（線形）
        t = t0 + (t1 - t0) * (step / max(1, n_steps - 1))
        i = np.random.randint(0, d)
        x2 = x.copy()
        x2[i] = 1 - x2[i]
        e2 = qubo_energy(x2, model)
        de = e2 - e
        if de <= 0:
            x, e = x2, e2
        else:
            p = math.exp(-de / max(1e-9, t))
            if np.random.rand() < p:
                x, e = x2, e2

    # 最後に「全部0」を防ぐ
    if x.sum() == 0:
        x[np.random.randint(0, d)] = 1
    return x.astype(int)

def sample_distribution(
    model: QuboModel,
    n_samples: int = 200,
    n_steps: int = 300,
    t0: float = 2.0,
    t1: float = 0.3,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    返り値:
      counts: (12,) 各神の出現回数（x_i==1の回数）
      samples: 各サンプルのx
    """
    d = model.h.shape[0]
    counts = np.zeros(d, dtype=int)
    samples: List[np.ndarray] = []
    for _ in range(n_samples):
        x = sa_sample(model, n_steps=n_steps, t0=t0, t1=t1)
        counts += x
        samples.append(x)
    return counts, samples


# -----------------------------
# 神託文（VOW_DICT + CHAR_MASTER + QUOTES）
# -----------------------------
def build_oracle_text(
    char_row: pd.Series,
    vow_dict: pd.DataFrame,
    v_total: np.ndarray,
    v_slider: np.ndarray,
    v_text: np.ndarray,
    quotes_df: Optional[pd.DataFrame],
    temperature: float,
) -> str:
    # VOW辞書
    vcols = [f"VOW_{i:02d}" for i in range(1, 13)]
    # top vows
    top_idx = np.argsort(-v_total)[:3]
    lines = []

    char_name = str(char_row.get("公式キャラ名", char_row.get("CHAR_NAME", char_row.get("NAME", "（不明）"))))
    role = str(char_row.get("役割", "")).strip()
    role_note = str(char_row.get("役割補足説明", "")).strip()
    ema = str(char_row.get("絵馬文字分析", "")).strip()

    lines.append(f"### 🔮 神託：{char_name}")
    if role:
        lines.append(f"- **役割**：{role}")
    if role_note:
        lines.append(f"- **補足**：{role_note}")
    if ema:
        lines.append(f"- **読み**：{ema}")

    lines.append("")
    lines.append("#### 🧭 あなたの誓願の“核心”（上位3つ）")

    # vow_dict の TITLE/SUBTITLE/UI_HINT を添える
    # VOW_ID は 1..12 or VOW_01.. など揺れる可能性 → indexで対応
    for k in top_idx:
        vow_no = k + 1
        # vow_dict の行を探す
        row = None
        if "VOW_ID" in vow_dict.columns:
            # 数字含むかで拾う
            for _, r in vow_dict.iterrows():
                s = str(r.get("VOW_ID", ""))
                m = re.search(r"(\d+)", s)
                if m and int(m.group(1)) == vow_no:
                    row = r
                    break
        if row is None and len(vow_dict) >= vow_no:
            # 行順に入っている場合の保険
            row = vow_dict.iloc[vow_no - 1]

        label = str(row.get("LABEL", f"VOW_{vow_no:02d}")) if row is not None else f"VOW_{vow_no:02d}"
        title = str(row.get("TITLE", "")).strip() if row is not None else ""
        subtitle = str(row.get("SUBTITLE", "")).strip() if row is not None else ""
        hint = str(row.get("UI_HINT", "")).strip() if row is not None else ""

        val = float(v_total[k])
        val_s = float(v_slider[k])
        val_t = float(v_text[k])

        msg = f"- **{label}**（合算 {val:.2f} / slider {val_s:.2f} / text {val_t:.2f}）"
        if title:
            msg += f"：{title}"
        lines.append(msg)
        if subtitle:
            lines.append(f"  - {subtitle}")
        if hint:
            lines.append(f"  - *ヒント*：{hint}")

    # QUOTES（任意）
    quote_line = pick_quote(quotes_df, top_idx=top_idx, temperature=temperature)
    if quote_line:
        lines.append("")
        lines.append("#### 🕯️ 添えられた言葉")
        lines.append(f"> {quote_line}")

    lines.append("")
    lines.append("#### ✅ 今日の一歩")
    lines.append("今は“正解”を探すより、**誓願の上位1つだけ**を小さく実行してみてください。観測は、行動で収束していきます。")

    return "\n".join(lines)

def pick_quote(quotes_df: Optional[pd.DataFrame], top_idx: np.ndarray, temperature: float) -> Optional[str]:
    if quotes_df is None or len(quotes_df) == 0:
        return None

    df = quotes_df.copy()
    cols = list(df.columns)

    text_col = _first_existing(["TEXT", "QUOTE", "格言", "名言", "BODY", "文章", "言葉"], cols)
    author_col = _first_existing(["AUTHOR", "著者", "出典", "SOURCE"], cols)
    tag_col = _first_existing(["TAG", "TAGS", "VOW_ID", "VOW", "カテゴリ", "CATEGORY"], cols)

    if not text_col:
        return None

    # 候補抽出（TAGに VOW番号が含まれるなら寄せる）
    candidates = []
    top_vows = [int(i + 1) for i in top_idx]

    if tag_col:
        for _, r in df.iterrows():
            t = str(r.get(tag_col, "")).strip()
            if not t:
                continue
            hit = False
            for vn in top_vows:
                if re.search(rf"\b{vn}\b", t) or re.search(rf"VOW[_\- ]*0*{vn}\b", t, flags=re.IGNORECASE):
                    hit = True
                    break
            if hit:
                candidates.append(r)

    # 候補が少なければ全体から
    if len(candidates) < 3:
        candidates = [r for _, r in df.iterrows()]

    if not candidates:
        return None

    # 温度で“寄せる度合い”を変える：低温=上位から、 高温=ランダム広め
    # ここではシンプルに、温度が低いほど先頭近くを選びやすくする重み
    n = len(candidates)
    ranks = np.arange(n, dtype=float)
    # 温度が低いほど減衰を強く
    tau = max(0.25, float(temperature))
    weights = np.exp(-ranks / (2.0 * tau))
    weights = weights / (weights.sum() + 1e-12)
    idx = int(np.random.choice(np.arange(n), p=weights))
    r = candidates[idx]

    q = str(r.get(text_col, "")).strip()
    if not q:
        return None
    a = str(r.get(author_col, "")).strip() if author_col else ""
    return f"{q}" + (f"（{a}）" if a else "")


# -----------------------------
# キャラクタテーブル構築（列名ゆらぎ吸収）
# -----------------------------
@dataclass
class CharTable:
    ids: List[str]
    names: List[str]
    images: List[str]         # ファイル名 or パス
    W_char_vow: np.ndarray    # (12,12)
    master_df: pd.DataFrame   # CHAR_MASTER参照用（行引き）

def build_char_table(pack: PackData) -> CharTable:
    ctv = pack.char_to_vow.copy()
    cm = pack.char_master.copy()

    # 列名候補
    ctv_cols = list(ctv.columns)
    id_col = _first_existing(["CHAR_ID", "ID", "キャラID"], ctv_cols)
    name_col = _first_existing(["公式キャラ名", "CHAR_NAME", "NAME", "キャラ名"], ctv_cols)
    img_col = _first_existing(["IMAGE_FILE", "IMAGE", "IMG", "画像", "画像ファイル"], ctv_cols)

    # VOW列
    vow_cols = _ensure_vow_cols(ctv, 12)

    # 必須（id/nameは片方欠けても、最悪 index で補う）
    if id_col is None:
        # CHAR_MASTER側で拾えるなら拾う
        if "CHAR_ID" in cm.columns:
            ctv["CHAR_ID"] = cm["CHAR_ID"].values[:len(ctv)]
            id_col = "CHAR_ID"
        else:
            ctv["CHAR_ID"] = [f"CHAR_{i+1:02d}" for i in range(len(ctv))]
            id_col = "CHAR_ID"

    if name_col is None:
        # CHAR_MASTERの公式キャラ名で補完
        cm_cols = list(cm.columns)
        cm_name_col = _first_existing(["公式キャラ名", "CHAR_NAME", "NAME", "キャラ名"], cm_cols)
        if cm_name_col is not None:
            # CHAR_IDでマージ
            ctv = ctv.merge(cm[[ "CHAR_ID", cm_name_col ]], on="CHAR_ID", how="left", suffixes=("", "_m"))
            ctv["公式キャラ名"] = ctv["公式キャラ名"] if "公式キャラ名" in ctv.columns else ctv[cm_name_col]
            name_col = "公式キャラ名"
        else:
            ctv["公式キャラ名"] = ctv[id_col].astype(str)
            name_col = "公式キャラ名"

    if img_col is None:
        # CHAR_TO_VOWに無い場合は空でOK（画像は任意）
        ctv["IMAGE_FILE"] = ""
        img_col = "IMAGE_FILE"

    ids = [str(x) for x in ctv[id_col].tolist()]
    names = [str(x) for x in ctv[name_col].tolist()]
    images = [str(x) for x in ctv[img_col].fillna("").tolist()]
    W = np.array(ctv[vow_cols].fillna(0.0).astype(float).values, dtype=float)

    # 12神に合わせて切り詰め/補う（念のため）
    if W.shape[0] < 12:
        pad = np.zeros((12 - W.shape[0], 12), dtype=float)
        W = np.vstack([W, pad])
        ids += [f"CHAR_PAD_{i+1}" for i in range(12 - len(ids))]
        names += [f"（未定義）{i+1}" for i in range(12 - len(names))]
        images += [""] * (12 - len(images))
    if W.shape[0] > 12:
        W = W[:12, :]
        ids = ids[:12]
        names = names[:12]
        images = images[:12]

    return CharTable(ids=ids, names=names, images=images, W_char_vow=W, master_df=cm)


# -----------------------------
# UI
# -----------------------------
st.sidebar.title("📁 データ")

pack_file = st.sidebar.file_uploader(
    "統合Excel（pack）をアップロード（最優先）",
    type=["xlsx"],
    help="例：quantum_shintaku_pack_v3_with_sense_*.xlsx",
)

st.sidebar.markdown("---")
img_dir = st.sidebar.text_input(
    "🖼️ 画像フォルダ（相対/絶対）",
    value="./assets/images/characters/",
    help="ローカル実行：./assets/images/characters/  例：C:\\Users\\...\\assets\\images\\characters",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🌡️ サンプリング設定")
temperature = st.sidebar.slider("温度 T（高いほど揺らぐ）", 0.2, 3.0, 1.1, 0.1)
n_samples = st.sidebar.slider("サンプル数（分布用）", 50, 800, 250, 10)
sa_steps = st.sidebar.slider("SAステップ数（1サンプルあたり）", 80, 1200, 350, 10)
beta_pair = st.sidebar.slider("相性（二次項）強さ β", 0.0, 0.6, 0.18, 0.01)
lam = st.sidebar.slider("制約ペナルティ λ（目標=1柱）", 0.5, 12.0, 3.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 入力統合")
w_slider = st.sidebar.slider("w1: スライダー誓願の重み", 0.0, 3.0, 1.0, 0.05)
w_text = st.sidebar.slider("w2: テキスト誓願の重み", 0.0, 3.0, 1.1, 0.05)

st.title("🔮 Q-Quest 量子神託 app08（完成版）")
st.caption("誓願（スライダー＋文章）→ QUBO → サンプリング分布 → 今回“観測”された神と神託を表示します。")

if not pack_file:
    st.info("左のサイドバーから **統合Excel（pack）** をアップロードしてください。")
    st.stop()

# 読み込み
try:
    pack = load_pack_excel(pack_file)
except Exception as e:
    st.error(f"統合Excelの解析に失敗: {e}")
    st.stop()

# 表示（検出情報）
with st.expander("🔍 検出したシート名 / 列名（デバッグ）", expanded=False):
    st.write("検出したシート名:", pack.sheet_names)
    for nm in ["VOW_DICT", "CHAR_MASTER", "CHAR_TO_VOW", "SENSE_TO_VOW", "QUOTES"]:
        df = pack.sheets.get(nm, None)
        if df is not None:
            st.write(f"**{nm} 列名:**", list(df.columns))

# キャラクタテーブル
try:
    char_table = build_char_table(pack)
except Exception as e:
    st.error(f"キャラクタテーブル生成に失敗: {e}")
    st.stop()

# SENSEマップ
sense_label_map, sense2vow_vec = build_sense_maps(pack.sense_dict, pack.sense_to_vow)

# VOW辞書（12個だけ使う）
vow_dict = pack.vow_dict.copy()

# UI: 入力
colL, colR = st.columns([1.05, 0.95], gap="large")

with colL:
    st.subheader("✅ Step 1：誓願入力（スライダー）")
    st.write("各誓願を 0〜5 で入力してください（明示意図）。")

    # VOW表示ラベル
    vow_labels: List[str] = []
    if "LABEL" in vow_dict.columns and len(vow_dict) >= 12:
        vow_labels = [str(vow_dict.iloc[i].get("LABEL", f"VOW_{i+1:02d}")) for i in range(12)]
    else:
        vow_labels = [f"VOW_{i+1:02d}" for i in range(12)]

    sliders = []
    for i in range(12):
        # タイトル、説明、ヒントを取得
        title = ""
        description = ""
        hint = ""
        if len(vow_dict) > i:
            row = vow_dict.iloc[i]
            if "TITLE" in vow_dict.columns:
                title = str(row.get("TITLE", "")).strip()
            if "DESCRIPTION_LONG" in vow_dict.columns:
                description = str(row.get("DESCRIPTION_LONG", "")).strip()
            elif "DESCRIPTION" in vow_dict.columns:
                description = str(row.get("DESCRIPTION", "")).strip()
            if "UI_HINT" in vow_dict.columns:
                hint = str(row.get("UI_HINT", "")).strip()
        
        # スライダーラベルにタイトルを含める（あれば）
        slider_label = vow_labels[i]
        if title:
            slider_label = f"{vow_labels[i]}：{title}"
        
        val = st.slider(slider_label, 0, 5, 0, 1)
        sliders.append(val)
        
        # 説明とヒントを常時表示（あれば）
        if description or hint:
            info_text = []
            if description:
                info_text.append(description)
            if hint:
                info_text.append(f"💡 {hint}")
            if info_text:
                st.caption(" | ".join(info_text))

    v_slider = np.array(sliders, dtype=float)
    # 正規化（0-5→0-1）
    v_slider_n = v_slider / 5.0

    st.subheader("📝 Step 1b：誓願入力（文章）")
    text = st.text_area(
        "あなたの誓願を自由に書いてください（暗黙意図を抽出します）",
        value="",
        height=120,
        placeholder="例：迷いを断ち切って一歩踏み出したい。自分の芯を取り戻したい。"
    )

    v_text, hits = text_to_vow_vector(text, sense_label_map, sense2vow_vec, ngram_n=3)
    # v_text は正規化済み（0..1程度）なので、合わせてスケールを揃える
    v_text_n = _norm01(v_text)

    if hits:
        with st.expander("🧩 テキストから検出したキーワード（SENSE）", expanded=False):
            st.write(pd.DataFrame(hits, columns=["SENSE", "一致スコア"]).head(12))

    # 合算
    v_total = w_slider * v_slider_n + w_text * v_text_n
    # 見栄え用に0-1へ
    v_total_n = _norm01(v_total)

with colR:
    st.subheader("📌 影響の可視化（入力がどう効いたか）")
    df_vis = pd.DataFrame({
        "VOW": vow_labels,
        "slider": v_slider_n,
        "text": v_text_n,
        "total": v_total_n
    }).set_index("VOW")

    st.write("**VOWベクトル（slider / text / total）**")
    st.bar_chart(df_vis[["slider", "text", "total"]])

    # Top寄与
    top_df = df_vis.copy()
    top_df["total_rank"] = (-top_df["total"]).rank(method="first")
    top_df = top_df.sort_values("total", ascending=False).head(6)
    st.write("**寄与Top（上位6）**")
    st.dataframe(top_df[["slider", "text", "total"]], use_container_width=True)

# QUBO構築
W = char_table.W_char_vow  # (12,12)
model = build_qubo(W_char_vow=W, v_total=v_total_n, beta_pair=beta_pair, lam=lam, target_k=1)

# キャラ一次スコア（参考）
char_scores = W @ v_total_n  # 高いほど合う
char_energies_unary = -char_scores  # 小さいほど合う

# 観測
st.markdown("---")
st.subheader("✅ Step 2：QUBOサンプリング（観測）")

obs_col1, obs_col2 = st.columns([0.55, 0.45], gap="large")
with obs_col1:
    st.write("**観測ボタン**を押すたびに、同じ条件でも“揺らぎ”により結果が変わり得ます。")
    observe = st.button("👁️ 観測する（QUBOをサンプルして神を観測）", use_container_width=True)

with obs_col2:
    st.info(
        "ℹ️ **『今回観測された神』と『分布ヒストグラムTop』がズレることがあります。**\n\n"
        "- 観測：**1回サンプル**（乱数/温度の影響を強く受ける）\n"
        "- ヒスト：**多数回サンプルの統計**\n\n"
        "上位候補が拮抗しているほど、1回観測はTop以外にも飛びます。"
    )

# セッション保持（最後に観測された神）
if "last_obs_idx" not in st.session_state:
    st.session_state.last_obs_idx = None
if "last_samples_counts" not in st.session_state:
    st.session_state.last_samples_counts = None

if observe:
    counts, samples = sample_distribution(
        model=model,
        n_samples=int(n_samples),
        n_steps=int(sa_steps),
        t0=float(temperature * 2.0),
        t1=float(temperature * 0.35),
    )
    st.session_state.last_samples_counts = counts

    # “今回の観測”：分布から確率的に1柱を選ぶ（マージナルを利用）
    # counts は「その神が立った回数」なので確率にして抽選
    probs = counts.astype(float)
    if probs.sum() <= 0:
        probs = np.ones(12, dtype=float)
    probs = probs / probs.sum()

    obs_idx = int(np.random.choice(np.arange(12), p=probs))
    st.session_state.last_obs_idx = obs_idx

# 表示
counts = st.session_state.last_samples_counts
obs_idx = st.session_state.last_obs_idx

if counts is None or obs_idx is None:
    st.warning("まだ観測していません。上の **👁️ 観測する** を押してください。")
    st.stop()

# 分布（ヒスト）
st.subheader("📊 観測分布（サンプル）")
df_hist = pd.DataFrame({"神": char_table.names, "count": counts}).set_index("神")
st.bar_chart(df_hist)

# 観測結果（キャラ表示）
st.markdown("---")
st.subheader("🧿 今回“観測”された神（基板曼荼羅）")

# CHAR_MASTERから該当行を探す（CHAR_ID一致）
cm = char_table.master_df.copy()
cm_id_col = "CHAR_ID" if "CHAR_ID" in cm.columns else None
row = None
if cm_id_col:
    # CHAR_TO_VOW側のIDで突合
    cid = char_table.ids[obs_idx]
    m = cm[cm[cm_id_col].astype(str) == str(cid)]
    if len(m) > 0:
        row = m.iloc[0]
if row is None:
    # fallback: 名前一致
    name = char_table.names[obs_idx]
    name_col = _first_existing(["公式キャラ名", "CHAR_NAME", "NAME", "キャラ名"], list(cm.columns))
    if name_col:
        m = cm[cm[name_col].astype(str) == str(name)]
        if len(m) > 0:
            row = m.iloc[0]
if row is None:
    # 最後の手段：空のSeries
    row = pd.Series({"公式キャラ名": char_table.names[obs_idx]})

# 画像
img_file = char_table.images[obs_idx]
img_path = ""
if img_file:
    # すでにフルパスならそのまま、ファイル名ならディレクトリ結合
    if os.path.isabs(img_file) and os.path.exists(img_file):
        img_path = img_file
    else:
        img_path = os.path.join(img_dir, img_file)

img = _read_image_maybe(img_path)

left, right = st.columns([0.42, 0.58], gap="large")
with left:
    if img is not None:
        st.image(img, caption=f"{char_table.names[obs_idx]}（{img_file}）", use_container_width=True)
    else:
        st.warning(
            "画像が見つかりません。\n\n"
            f"- 期待パス: `{img_path}`\n"
            "- サイドバーの画像フォルダ設定を確認してください。\n"
            "- CHAR_TO_VOW の IMAGE_FILE が空の場合はファイル名を入れてください。"
        )

with right:
    # 神託文
    oracle = build_oracle_text(
        char_row=row,
        vow_dict=vow_dict,
        v_total=v_total_n,
        v_slider=v_slider_n,
        v_text=v_text_n,
        quotes_df=pack.quotes,
        temperature=float(temperature),
    )
    st.markdown(oracle)

# 追加：上位候補（エネルギー順位）
st.markdown("---")
st.subheader("🏁 エネルギー順位（参考：一次項ベース）")
rank_idx = np.argsort(char_energies_unary)[:5]
rank_df = pd.DataFrame({
    "順位": np.arange(1, len(rank_idx) + 1),
    "神": [char_table.names[i] for i in rank_idx],
    "energy(unary)": [float(char_energies_unary[i]) for i in rank_idx],
    "score": [float(char_scores[i]) for i in rank_idx],
})
st.dataframe(rank_df, use_container_width=True)

st.caption(
    "※ 上の順位は主に『誓願ベクトルとの整合（一次項）』の参考です。"
    " 実際の観測は QUBO + SA の揺らぎ（温度T、相性β、制約λ）により変動します。"
)
