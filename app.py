"""
Q-Quest 量子神託 - Streamlitアプリケーション
Human-Centric Quantum Philosophy
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
from dataclasses import dataclass
import itertools
import math
import random
import re
import time
from collections import Counter

# ページ設定
st.set_page_config(
    page_title="Q-Quest 量子神託",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 日本語フォント設定（Streamlit用）
import matplotlib
matplotlib.use('Agg')  # Streamlitでのバックエンド設定

# -------------------------
# データ定義
# -------------------------
# 気持ちを整えるための格言（VARIABLES）と引用元
VARIABLES = [
    "止まることで、流れが見える。動の中に静がある。",
    "水は、争わない。形にこだわらず、流れるがままに。",
    "間こそが答えである。余白にこそ本質がある。",
    "己に誠実であること。それが自由への道である。",
]

# 格言の引用元
MAXIM_SOURCES = {
    "止まることで、流れが見える。動の中に静がある。": {
        "source": "禅の思想",
        "origin": "禅宗の教えから",
        "reference": "動と静の調和を説く禅の教義に基づく"
    },
    "水は、争わない。形にこだわらず、流れるがままに。": {
        "source": "老子『道徳経』",
        "origin": "第八章「上善若水」",
        "reference": "「上善は水の若し。水は善く万物を利して争わず」"
    },
    "間こそが答えである。余白にこそ本質がある。": {
        "source": "日本の美学思想",
        "origin": "「間（Ma）」の概念",
        "reference": "能楽、茶道、俳句などに通底する日本の美意識"
    },
    "己に誠実であること。それが自由への道である。": {
        "source": "エピクテトス『語録』",
        "origin": "ストア派哲学",
        "reference": "「自分自身に対して誠実であることこそ、真の自由につながる」という思想"
    },
}

GLOBAL_WORDS_DATABASE = [
    # 願い・目標
    "世界平和", "貢献", "成長", "学び", "挑戦", "夢", "希望", "未来",
    # 感情・状態
    "感謝", "愛", "幸せ", "喜び", "安心", "充実", "満足", "平和",
    # 行動・姿勢
    "努力", "継続", "忍耐", "誠実", "正直", "優しさ", "思いやり", "共感",
    # 哲学・概念
    "調和", "バランス", "自然", "美", "真実", "自由", "正義", "道",
    # 関係性
    "絆", "つながり", "家族", "友人", "仲間", "信頼", "尊敬", "協力",
    # 時間・流れ
    "今", "瞬間", "過程", "変化", "進化", "発展", "循環", "流れ",
    # 内的状態
    "静けさ", "集中", "覚悟", "決意", "勇気", "強さ", "柔軟性", "寛容",
]

FAMOUS_QUOTES = [
    {
        "keywords": ["平和", "世界", "貢献", "希望"], 
        "quote": "雪の下で種は春を待っている。焦るべからず、時満ちるを待て。",
        "source": "日本の古語・ことわざ",
        "origin": "自然の摂理を説く伝統的な教え",
        "reference": "季節の循環と忍耐の重要性を表現"
    },
    {
        "keywords": ["成長", "努力", "継続", "挑戦"], 
        "quote": "千里の道も一歩から。歩みを止めず、続けることに意味がある。",
        "source": "老子『道徳経』",
        "origin": "第六十四章",
        "reference": "「千里の行も足下に始まる」に基づく"
    },
    {
        "keywords": ["感謝", "愛", "絆", "つながり"], 
        "quote": "一期一会。今この瞬間を大切に。すべては縁で繋がっている。",
        "source": "千利休の茶道精神",
        "origin": "「一期一会」の思想",
        "reference": "千利休に連なる茶道の教え「一期一会」と縁の概念"
    },
    {
        "keywords": ["自然", "調和", "バランス", "流れ"], 
        "quote": "水は、争わない。形にこだわらず、流れるがままに。",
        "source": "老子『道徳経』",
        "origin": "第八章「上善若水」",
        "reference": "「上善は水の若し。水は善く万物を利して争わず」"
    },
    {
        "keywords": ["静けさ", "集中", "今", "瞬間"], 
        "quote": "止まることで、流れが見える。動の中に静がある。",
        "source": "禅の思想",
        "origin": "禅宗の教えから",
        "reference": "動と静の調和を説く禅の教義に基づく"
    },
    {
        "keywords": ["勇気", "決意", "挑戦", "道"], 
        "quote": "道が分れていたら、念がない方へ行け。",
        "source": "柳のことば（創作）",
        "origin": "本プロジェクトの創作",
        "reference": "禅的思考に基づく創作格言"
    },
    {
        "keywords": ["思いやり", "優しさ", "共感", "信頼"], 
        "quote": "人の心に寄り添う。それが真の強さである。",
        "source": "日本の伝統的価値観",
        "origin": "和の精神",
        "reference": "他者への共感と寄り添いを重視する日本の文化"
    },
    {
        "keywords": ["変化", "進化", "発展", "未来"], 
        "quote": "無為にして為す。動くことが静である。",
        "source": "老子『道徳経』",
        "origin": "第三十七章",
        "reference": "「道は常に無為にして為さず」に基づく無為自然の思想"
    },
    {
        "keywords": ["美", "真実", "自然", "調和"], 
        "quote": "間こそが答えである。余白にこそ本質がある。",
        "source": "日本の美学思想",
        "origin": "「間（Ma）」の概念",
        "reference": "能楽、茶道、俳句などに通底する日本の美意識"
    },
    {
        "keywords": ["自由", "正義", "道", "誠実"], 
        "quote": "己に誠実であること。それが自由への道である。",
        "source": "エピクテトス『語録』",
        "origin": "ストア派哲学",
        "reference": "「自分自身に対して誠実であることこそ、真の自由につながる」という思想"
    },
]

SEASONS = ["薄氷", "立春", "春霞", "若葉", "夕立", "秋声", "木枯らし", "雪明り"]

# Moodに応じた次の一歩の提案
NEXT_STEPS_BY_MOOD = {
    "fatigue": [
        "一つだけ、今日やることを減らしなさい。",
        "遠回りを選びなさい。答えは道の途中にある。",
        "決めなくてよい。保留は、立派な選択である。",
    ],
    "anxiety": [
        "話すなら「結論」より「気配」を渡しなさい。",
        "境界（しきい）を越えるのは、静かな一歩でよい。",
        "水のように流れるがままに。形にこだわらない。",
    ],
    "curiosity": [
        "千里の道も一歩から。歩みを止めず、続けることに意味がある。",
        "成長は過程にあり。今この瞬間を大切に。",
        "挑戦する勇気こそが、未来を開く鍵である。",
    ],
    "loneliness": [
        "一期一会。今この瞬間を大切に。すべては縁で繋がっている。",
        "人の心に寄り添う。それが真の強さである。",
        "絆は見えなくても、そこにある。",
    ],
    "decisiveness": [
        "決めなくてよい。保留は、立派な選択である。",
        "己に誠実であること。それが自由への道である。",
        "道が分れていたら、念がない方へ行け。",
    ],
    "default": [
        "一つだけ、今日やることを減らしなさい。",
        "遠回りを選びなさい。答えは道の途中にある。",
        "話すなら「結論」より「気配」を渡しなさい。",
        "決めなくてよい。保留は、立派な選択である。",
        "境界（しきい）を越えるのは、静かな一歩でよい。"
    ]
}

# 後方互換性のため（古いコードで使用されている場合）
NEXT_STEPS = NEXT_STEPS_BY_MOOD["default"]

# -------------------------
# QUBO関連関数
# -------------------------
def qubo_energy(x: np.ndarray, Q: Dict[Tuple[int,int], float]) -> float:
    """QUBOエネルギーを計算"""
    e = 0.0
    n = len(x)
    for i in range(n):
        e += Q.get((i,i), 0.0) * x[i]
    for i in range(n):
        for j in range(i+1, n):
            e += Q.get((i,j), 0.0) * x[i] * x[j]
    return float(e)

def bitstring(x: np.ndarray) -> str:
    return "".join(str(int(v)) for v in x)

# -------------------------
# Mood推定
# -------------------------
@dataclass
class Mood:
    fatigue: float
    anxiety: float
    curiosity: float
    loneliness: float
    decisiveness: float

KEYWORDS = {
    "fatigue": ["疲", "しんど", "眠", "だる", "消耗", "限界", "体調", "重", "動けない"],
    "anxiety": [
        "不安", "焦", "怖", "心配", "迷", "落ち着か", "緊張", "気になる", 
        "自信", "持てない", "言葉", "他者", "評価", "目", "周り", "どう思",
        "失敗", "間違い", "否定", "批判", "不安", "恐", "怯"
    ],
    "curiosity": [
        "やってみ", "興味", "面白", "学び", "試", "挑戦", "ワクワク", "知りたい", "探索",
        "成長", "向上", "進化", "高め", "伸ばす", "改善", "発展", "進歩", "前進", "常に"
    ],
    "loneliness": ["孤独", "一人", "寂", "誰にも", "分かって", "話せ", "孤立", "疎外"],
    "decisiveness": [
        "決め", "結論", "選", "判断", "断", "方針", "期限", "決断",
        "自信", "持てない", "迷う", "悩む", "躊躇", "ためら", "優柔不断"
    ],
}

def score_from_text(text: str, keys: List[str]) -> float:
    """テキストからキーワードを検索してスコアを計算（改善版）"""
    s = 0.0
    text_lower = text.lower()  # 一度だけ小文字化
    
    for k in keys:
        k_lower = k.lower()
        # 部分マッチで検索
        matches = len(re.findall(re.escape(k_lower), text_lower))
        if matches > 0:
            # マッチした回数に基づいてスコアを加算
            base_score = matches * 0.5  # 基本スコア
            # 長いキーワードには追加の重み
            if len(k) >= 3:
                base_score += 0.5
            if len(k) >= 4:
                base_score += 0.3
            s += base_score
    
    return float(s)

def infer_mood(text: str) -> Mood:
    """テキストから心の傾き（Mood）を推定（改善版：より敏感で多様な検出）"""
    t = text.strip()
    if not t:
        # 空のテキストの場合、全て0.0を返す
        return Mood(0.0, 0.0, 0.0, 0.0, 0.0)
    
    raw = {k: score_from_text(t, v) for k, v in KEYWORDS.items()}
    
    # 全てのスコアの最大値を計算（相対的な正規化のため）
    max_raw = max(raw.values()) if max(raw.values()) > 0 else 1.0
    
    # 正規化関数（相対的な正規化と絶対的なスケーリングの組み合わせ）
    def norm(x: float, scale: float = 1.5, use_relative: bool = True) -> float:
        if x == 0.0:
            return 0.0
        
        # 相対的な正規化（他のMoodとの比較）
        if use_relative and max_raw > 0:
            relative = x / max_raw
        else:
            relative = 1.0
        
        # 絶対的なスケーリング（キーワード数に基づく）
        absolute = min(1.0, x / scale)
        
        # 両方を組み合わせて、より豊かな表現を実現
        combined = (relative * 0.6 + absolute * 0.4)
        
        # 最小値の確保（0.0でない限り、ある程度の値を保証）
        if x > 0:
            combined = max(0.15, min(1.0, combined))
        
        return combined
    
    # 各Mood値のスケールを個別に調整（より敏感な検出）
    return Mood(
        fatigue=norm(raw["fatigue"], scale=1.2),  # 疲れは敏感に検出
        anxiety=norm(raw["anxiety"], scale=1.0),  # 不安は最も敏感に
        curiosity=norm(raw["curiosity"], scale=1.3),
        loneliness=norm(raw["loneliness"], scale=1.2),
        decisiveness=norm(raw["decisiveness"], scale=1.1),  # 決断力も敏感に
    )

# -------------------------
# QUBO生成
# -------------------------
# QUBOパラメータ（格言同士の関係性）
# 負の値 = 相乗効果（一緒に選ばれやすい）
# 正の値 = 抑制（同時に選ばれにくい）
Q_BASE: Dict[Tuple[int,int], float] = {
    # 線形項（各格言の基本エネルギー）
    # 負の値が大きいほど選ばれやすい
    (0,0): -0.5,  # 静けさの格言
    (1,1): -0.5,  # 流れの格言
    (2,2): -0.5,  # 余白の格言
    (3,3): -0.5,  # 誠実さの格言
    # 相互作用項（格言同士の関係）
    # 負の値 = 相乗効果、正の値 = 抑制効果
    (0,1): -0.3,  # 静けさ × 流れ = 軽い相乗効果
    (0,2): -0.4,  # 静けさ × 余白 = 相乗効果
    (1,2): -0.3,  # 流れ × 余白 = 軽い相乗効果
    (0,3): +0.2,  # 静けさ × 誠実 = 少し抑制（多様性のため）
    (1,3): -0.2,  # 流れ × 誠実 = 軽い相乗効果
    (2,3): +0.1,  # 余白 × 誠実 = わずかな抑制
}

def clamp(v: float, lo: float=-3.0, hi: float=3.0) -> float:
    return max(lo, min(hi, v))

def build_qubo_from_mood(m: Mood) -> Dict[Tuple[int,int], float]:
    """Moodに基づいてQUBOパラメータを調整（改善版：連続的で多様な変化）"""
    Q = dict(Q_BASE)
    
    # 閾値を下げて、より小さなMood値でも反応するようにする
    # また、Mood値に比例して連続的に調整する
    
    # === 線形項の調整（各格言の選択されやすさ） ===
    
    # 疲れ → 静けさ(0)と余白(2)の格言が選ばれやすく、流れ(1)は少し抑制
    fatigue_effect = m.fatigue * 1.5  # 効果を強化
    Q[(0,0)] = clamp(Q[(0,0)] - fatigue_effect)  # 静けさ
    Q[(2,2)] = clamp(Q[(2,2)] - fatigue_effect * 0.9)  # 余白
    Q[(1,1)] = clamp(Q[(1,1)] + m.fatigue * 0.3)  # 流れは少し抑制
    
    # 不安 → 流れ(1)と誠実(3)の格言が選ばれやすく、静けさ(0)も支援
    anxiety_effect = m.anxiety * 1.4
    Q[(1,1)] = clamp(Q[(1,1)] - anxiety_effect)  # 流れ
    Q[(3,3)] = clamp(Q[(3,3)] - anxiety_effect * 0.8)  # 誠実
    Q[(0,0)] = clamp(Q[(0,0)] - m.anxiety * 0.5)  # 静けさも支援
    
    # 好奇心 → 流れ(1)と余白(2)の格言が選ばれやすく、誠実(3)も支援
    curiosity_effect = m.curiosity * 1.3
    Q[(1,1)] = clamp(Q[(1,1)] - curiosity_effect * 0.9)  # 流れ
    Q[(2,2)] = clamp(Q[(2,2)] - curiosity_effect)  # 余白
    Q[(3,3)] = clamp(Q[(3,3)] - m.curiosity * 0.4)  # 誠実も支援
    
    # 孤独感 → 静けさ(0)と誠実(3)の格言が選ばれやすく、余白(2)も支援
    loneliness_effect = m.loneliness * 1.2
    Q[(0,0)] = clamp(Q[(0,0)] - loneliness_effect)  # 静けさ
    Q[(3,3)] = clamp(Q[(3,3)] - loneliness_effect * 0.7)  # 誠実
    Q[(2,2)] = clamp(Q[(2,2)] - m.loneliness * 0.4)  # 余白も支援
    
    # 決断力 → 低い場合は誠実(3)と静けさ(0)、高い場合は流れ(1)と余白(2)
    decisiveness_factor = (1.0 - m.decisiveness) * 1.2  # 低いほど効果大
    Q[(3,3)] = clamp(Q[(3,3)] - decisiveness_factor)  # 決断力が低い→誠実を強調
    Q[(0,0)] = clamp(Q[(0,0)] - decisiveness_factor * 0.6)  # 静けさも
    
    if m.decisiveness > 0.5:  # 決断力が高い場合
        Q[(1,1)] = clamp(Q[(1,1)] - (m.decisiveness - 0.5) * 1.0)  # 流れ
        Q[(2,2)] = clamp(Q[(2,2)] - (m.decisiveness - 0.5) * 0.8)  # 余白
    
    # === 相互作用項の動的調整（組み合わせの相乗効果） ===
    
    # 疲れ×不安 → 静けさ×余白の相乗効果を強化
    if m.fatigue > 0.2 and m.anxiety > 0.2:
        synergy = (m.fatigue + m.anxiety) / 2 * 0.6
        Q[(0,2)] = clamp(Q[(0,2)] - synergy)  # 静けさ × 余白
    
    # 不安×好奇心 → 流れ×余白の相乗効果
    if m.anxiety > 0.2 and m.curiosity > 0.2:
        synergy = (m.anxiety + m.curiosity) / 2 * 0.5
        Q[(1,2)] = clamp(Q[(1,2)] - synergy)  # 流れ × 余白
    
    # 不安×決断力(高) → 流れ×誠実の相乗効果
    if m.anxiety > 0.2 and m.decisiveness > 0.5:
        synergy = m.anxiety * (m.decisiveness - 0.5) * 0.7
        Q[(1,3)] = clamp(Q[(1,3)] - synergy)  # 流れ × 誠実
    
    # 孤独感×疲れ → 静けさ×誠実の相乗効果
    if m.loneliness > 0.2 and m.fatigue > 0.2:
        synergy = (m.loneliness + m.fatigue) / 2 * 0.5
        Q[(0,3)] = clamp(Q[(0,3)] - synergy)  # 静けさ × 誠実（元々は抑制だったが、Moodに応じて変化）
    
    # 好奇心×決断力(高) → 流れ×余白の相乗効果を強化
    if m.curiosity > 0.3 and m.decisiveness > 0.5:
        synergy = m.curiosity * (m.decisiveness - 0.5) * 0.6
        Q[(1,2)] = clamp(Q[(1,2)] - synergy)  # 流れ × 余白
    
    return Q

# -------------------------
# 解探索
# -------------------------
def solve_all(Q: Dict[Tuple[int,int], float]) -> List[Tuple[float, np.ndarray]]:
    n = len(VARIABLES)
    sols = []
    for bits in itertools.product([0,1], repeat=n):
        x = np.array(bits, dtype=int)
        e = qubo_energy(x, Q)
        sols.append((e, x))
    sols.sort(key=lambda t: t[0])
    return sols

# -------------------------
# ボルツマンサンプリング
# -------------------------
def boltzmann_sample(cands: List[Tuple[float, np.ndarray]], T: float) -> Tuple[float, np.ndarray]:
    es = np.array([e for e,_ in cands], dtype=float)
    es0 = es - es.min()
    weights = np.exp(-es0 / max(T, 1e-9))
    weights = weights / weights.sum()
    idx = np.random.choice(len(cands), p=weights)
    return cands[idx]

def temperature_from_mood(m: Mood) -> float:
    """Moodに基づいてボルツマンサンプリングの温度を調整（改善版）"""
    # ベース温度（好奇心が高いと揺らぎが大きくなる）
    T = 0.4 + 0.3 * m.curiosity
    
    # 決断力が高いと、より確定的に（温度を下げる）
    T *= (1.0 - 0.3 * m.decisiveness)
    
    # 不安が高いと、より探索的になる（温度を上げる）
    T *= (1.0 + 0.2 * m.anxiety)
    
    # 疲れが高いと、少し揺らぎを増やす（多様な選択肢を提示）
    T *= (1.0 + 0.15 * m.fatigue)
    
    # 孤独感が高いと、より確定的に（温度を下げる）
    T *= (1.0 - 0.2 * m.loneliness)
    
    # 温度の範囲を制限（揺らぎすぎない、収束しすぎない）
    return max(0.2, min(0.9, T))

# -------------------------
# おみくじ生成
# -------------------------
def picks_from_x(x: np.ndarray) -> List[str]:
    """選ばれた格言を返す"""
    p = [VARIABLES[i] for i,v in enumerate(x) if v==1]
    return p if p else ["今この瞬間を大切に。すべては縁で繋がっている。"]

def get_maxim_source(maxim: str) -> Dict:
    """格言の引用元情報を取得"""
    if maxim in MAXIM_SOURCES:
        return MAXIM_SOURCES[maxim]
    return {
        "source": "伝統的な教え",
        "origin": "古来より伝わる智慧",
        "reference": "長い年月をかけて受け継がれてきた知恵"
    }

def oracle_card(e: float, x: np.ndarray, mood: Mood = None) -> Dict:
    """格言ベースのおみくじカードを生成（Moodに応じて変化）"""
    picks = picks_from_x(x)
    season = random.choice(SEASONS)
    
    # Moodに応じて「次の一歩」を選択
    if mood is not None:
        # 最も高いMood値を基準に選択
        mood_scores = {
            "fatigue": mood.fatigue,
            "anxiety": mood.anxiety,
            "curiosity": mood.curiosity,
            "loneliness": mood.loneliness,
            "decisiveness": mood.decisiveness,
        }
        max_mood = max(mood_scores.items(), key=lambda x: x[1])
        
        if max_mood[1] > 0.3:  # 0.3以上の場合のみMoodに応じた提案
            hints = NEXT_STEPS_BY_MOOD.get(max_mood[0], NEXT_STEPS_BY_MOOD["default"])
        else:
            hints = NEXT_STEPS_BY_MOOD["default"]
    else:
        hints = NEXT_STEPS_BY_MOOD["default"]
    
    hint = random.choice(hints)
    
    # 選ばれた格言を俳句風に表現（選ばれた格言に応じて季節も調整）
    if len(picks) > 0:
        # 選ばれた格言の内容に応じて季節を調整（オプション）
        poem = f"{season}／{picks[0]}"
    else:
        poem = f"{season}／今この瞬間を大切に"
    
    return {
        "energy": e,
        "picks": picks,
        "poem": poem,
        "hint": hint
    }

# -------------------------
# キーワード抽出とネットワーク構築（Cell 4用）
# -------------------------
def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    text_clean = re.sub(r'[0-9０-９\W]+', ' ', text)
    found_keywords = []
    for word in GLOBAL_WORDS_DATABASE:
        if word in text_clean:
            found_keywords.append(word)
    if not found_keywords:
        words = text_clean.split()
        found_keywords = [w for w in words if len(w) >= 2][:top_n]
    return found_keywords[:top_n]

def calculate_energy_between_words(word1: str, word2: str) -> float:
    energy = 0.0
    common_chars = set(word1) & set(word2)
    if common_chars:
        energy -= len(common_chars) * 0.3
    
    categories = {
        "願い": ["世界平和", "貢献", "成長", "夢", "希望"],
        "感情": ["感謝", "愛", "幸せ", "喜び", "安心"],
        "行動": ["努力", "継続", "忍耐", "誠実", "正直"],
        "哲学": ["調和", "バランス", "自然", "美", "道"],
        "関係": ["絆", "つながり", "家族", "友人", "信頼"],
        "内的": ["静けさ", "集中", "覚悟", "決意", "勇気"],
    }
    
    for category, words in categories.items():
        if word1 in words and word2 in words:
            energy -= 0.5
    
    energy += np.random.normal(0, 0.1)
    return energy

def build_word_network(center_words: List[str], database: List[str], n_neighbors: int = 15) -> Dict:
    all_words = list(set(center_words + database))
    word_energies = {}
    for word in all_words:
        if word in center_words:
            energy = -2.0
        else:
            energies = [calculate_energy_between_words(cw, word) for cw in center_words]
            energy = np.mean(energies)
        word_energies[word] = energy
    
    sorted_words = sorted(word_energies.items(), key=lambda x: x[1])
    selected_words = center_words.copy()
    for word, energy in sorted_words:
        if word not in center_words and len(selected_words) < n_neighbors:
            selected_words.append(word)
    
    network = {
        'words': selected_words,
        'energies': {word: word_energies.get(word, 0) for word in selected_words},
        'edges': []
    }
    
    for i, word1 in enumerate(selected_words):
        for j, word2 in enumerate(selected_words[i+1:], start=i+1):
            energy = calculate_energy_between_words(word1, word2)
            if energy < -0.3:
                network['edges'].append((i, j, energy))
    
    return network

def place_words_on_sphere(n_words: int, center_indices: List[int]) -> np.ndarray:
    positions = np.zeros((n_words, 3))
    golden_angle = np.pi * (3 - np.sqrt(5))
    
    for i in range(n_words):
        if i in center_indices:
            r = 0.3 + np.random.rand() * 0.2
        else:
            r = 0.8 + np.random.rand() * 0.4
        
        theta = golden_angle * i
        y = 1 - (i / float(n_words - 1)) * 2
        radius_at_y = np.sqrt(1 - y * y)
        
        x = np.cos(theta) * radius_at_y * r
        z = np.sin(theta) * radius_at_y * r
        
        positions[i] = [x, y, z]
    
    return positions

def select_relevant_quote(keywords: List[str]) -> str:
    keyword_set = set(keywords)
    best_match = None
    best_score = 0
    
    for quote_data in FAMOUS_QUOTES:
        quote_keywords = set(quote_data["keywords"])
        score = len(keyword_set & quote_keywords)
        if score > best_score:
            best_score = score
            best_match = quote_data["quote"]
    
    if best_match is None:
        best_match = "あなたの観測が、この世界線を確定させました。"
    
    return best_match

# -------------------------
# Plotly 3D可視化
# -------------------------
def create_3d_network_plot(network: Dict, positions: np.ndarray, center_indices: List[int]) -> go.Figure:
    fig = go.Figure()
    
    # エッジを描画
    for i, j, energy in network['edges']:
        x_coords = [positions[i, 0], positions[j, 0]]
        y_coords = [positions[i, 1], positions[j, 1]]
        z_coords = [positions[i, 2], positions[j, 2]]
        
        alpha = 0.2 + abs(energy) * 0.3
        linewidth = 0.5 + abs(energy) * 1.5
        color = '#4a9eff' if energy < -0.5 else '#ff6b6b'
        
        fig.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='lines',
            line=dict(color=color, width=linewidth),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # ノードを描画
    for i, word in enumerate(network['words']):
        x, y, z = positions[i]
        is_center = i in center_indices
        
        if is_center:
            size = 15
            color = '#ffd700'
        else:
            size = 8
            color = '#ffffff'
        
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(size=size, color=color, line=dict(width=1, color='white')),
            text=[word],
            textposition="middle center",
            textfont=dict(
                size=14 if is_center else 10, 
                color=color
            ),
            name=word,
            hovertemplate=f'<b>{word}</b><extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='量子神託 - Quantum Oracle',
            font=dict(size=20, color='#ffffff', family='Arial, sans-serif'),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(backgroundcolor='#0a0a1a', showgrid=False, showticklabels=False, title=''),
            yaxis=dict(backgroundcolor='#0a0a1a', showgrid=False, showticklabels=False, title=''),
            zaxis=dict(backgroundcolor='#0a0a1a', showgrid=False, showticklabels=False, title=''),
            bgcolor='#0a0a1a',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        plot_bgcolor='#0a0a1a',
        paper_bgcolor='#0a0a1a',
        margin=dict(l=0, r=0, t=50, b=0),
        height=700
    )
    
    return fig

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.title("🔮 Q-Quest 量子神託")
    st.markdown("### Human-Centric Quantum Philosophy")
    st.markdown("---")
    
    # サイドバー
    st.sidebar.title("機能選択")
    app_mode = st.sidebar.selectbox(
        "実行モードを選択",
        ["基本デモ", "対話型量子神託", "言葉のエネルギー球体視覚化", "絵馬納め"]
    )
    
    if app_mode == "基本デモ":
        st.header("QUBO × 縁：基本デモ")
        st.markdown("基本的なQUBOモデルを使用した「縁」のデモンストレーション")
        
        if st.button("実行"):
            Q = Q_BASE
            sols = solve_all(Q)
            
            # 結果表示
            st.subheader("低エネルギー上位（選ばれた格言の重なり）")
            for rank, (e, x) in enumerate(sols[:8], start=1):
                picks = [VARIABLES[i] for i,v in enumerate(x) if v==1]
                if picks:
                    picks_str = " | ".join(picks[:2])  # 長いので最大2つまで
                    if len(picks) > 2:
                        picks_str += f" ...（他{len(picks)-2}つ）"
                else:
                    picks_str = "今この瞬間を大切に"
                st.write(f"{rank}. E={e:>6.3f}  x={bitstring(x)}")
                st.caption(f"   格言: {picks_str}")
            
            # エネルギー地形の可視化
            labels = [bitstring(x) for _, x in sols]
            energies = [e for e, _ in sols]
            
            fig_bar = px.bar(
                x=labels,
                y=energies,
                labels={'x': '状態', 'y': 'エネルギー'},
                title="Energy landscape（低いほど「縁が結ばれやすい候補」）"
            )
            fig_bar.update_xaxes(tickangle=-90)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # おみくじ（基本デモではmoodなし）
            oracle_pool = sols[:6]
            e_pick, x_pick = boltzmann_sample(oracle_pool, T=0.45)
            card = oracle_card(e_pick, x_pick, mood=None)
            
            st.markdown("---")
            st.subheader("量子おみくじ（Quantum Oracle）")
            st.write(f"**エネルギー**: {card['energy']:.3f}")
            
            # 選ばれた格言と引用元を表示
            picks_display = []
            for pick in card['picks']:
                source_info = get_maxim_source(pick)
                picks_display.append(f"{pick} *（{source_info['source']}）*")
            
            st.write(f"**選ばれた縁**:")
            for pick_text in picks_display:
                st.markdown(f"   - {pick_text}")
            
            st.write(f"**ことば**: 「{card['poem']}」")
            st.write(f"**次の一歩**: {card['hint']}")
    
    elif app_mode == "対話型量子神託":
        st.header("対話型量子神託")
        st.markdown("あなたの悩み・気持ちを入力すると、パーソナライズされた「縁」を提示します")
        
        user_text = st.text_area(
            "今日の悩み・気持ちを一文でどうぞ",
            placeholder="例：疲れていて決断ができない…",
            height=100
        )
        
        if st.button("神託を求める"):
            if not user_text.strip():
                st.warning("テキストを入力してください")
            else:
                m = infer_mood(user_text)
                Q_today = build_qubo_from_mood(m)
                sols = solve_all(Q_today)
                
                # 心の傾きを表示
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("疲れ", f"{m.fatigue:.2f}")
                with col2:
                    st.metric("不安/焦り", f"{m.anxiety:.2f}")
                with col3:
                    st.metric("好奇心", f"{m.curiosity:.2f}")
                with col4:
                    st.metric("孤独", f"{m.loneliness:.2f}")
                with col5:
                    st.metric("決断", f"{m.decisiveness:.2f}")
                
                # 候補Top3
                st.subheader("低エネルギー候補（選ばれた格言の重ね合わせ）Top3")
                for rank, (e, x) in enumerate(sols[:3], start=1):
                    picks = picks_from_x(x)
                    st.write(f"**{rank}. E={e:.3f}**")
                    for pick in picks:
                        source_info = get_maxim_source(pick)
                        st.write(f"   • {pick}")
                        st.caption(f"     *出典: {source_info['source']} - {source_info['origin']}*")
                
                # おみくじ（Moodに応じて変化）
                pool = sols[:6]
                T = temperature_from_mood(m)
                e_pick, x_pick = boltzmann_sample(pool, T=T)
                card = oracle_card(e_pick, x_pick, mood=m)  # Moodを渡す
                
                st.markdown("---")
                st.subheader("量子おみくじ（Quantum Oracle）")
                
                # 選ばれた格言の引用元情報を収集
                sources_text = []
                for pick in card['picks']:
                    source_info = get_maxim_source(pick)
                    sources_text.append(f"- {pick}\n  *出典: {source_info['source']} - {source_info['origin']}*")
                
                st.info(f"""
**エネルギー**: {card['energy']:.3f}

**選ばれた縁**:
{chr(10).join(sources_text)}

**ことば**:
「{card['poem']}」

**次の一歩**:
{card['hint']}
""")
                st.caption(f"※揺らぎ(T)={T:.2f}（大きいほど偶然性が増えます）")
                
                # === エネルギー球体視覚化を統合 ===
                st.markdown("---")
                st.subheader("言葉のエネルギー球体視覚化")
                st.markdown("入力した言葉から抽出されたキーワードと、それに関連する言葉のエネルギー関係を可視化します")
                
                # キーワード抽出（同じテキストを使用）
                keywords = extract_keywords(user_text)
                if keywords:
                    st.write(f"**抽出されたキーワード**: {', '.join(keywords)}")
                    
                    # ネットワーク構築
                    network = build_word_network(keywords, GLOBAL_WORDS_DATABASE, n_neighbors=20)
                    
                    # 3D配置
                    center_indices = [i for i, word in enumerate(network['words']) if word in keywords]
                    positions = place_words_on_sphere(len(network['words']), center_indices)
                    
                    # 3D可視化
                    fig = create_3d_network_plot(network, positions, center_indices)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 選ばれた格言と関連する格言を表示
                    st.markdown("---")
                    st.subheader("関連する格言")
                    
                    # キーワードに基づいて格言を選択
                    quote_text = select_relevant_quote(keywords)
                    st.success(f"「{quote_text}」")
                    
                    # 引用元を表示
                    quote_obj = None
                    for q in FAMOUS_QUOTES:
                        if q['quote'] == quote_text:
                            quote_obj = q
                            break
                    
                    if quote_obj:
                        st.caption(f"*出典: {quote_obj['source']} - {quote_obj['origin']}*")
                        with st.expander("詳細な引用情報"):
                            st.write(f"**出典**: {quote_obj['source']}")
                            st.write(f"**由来**: {quote_obj['origin']}")
                            st.write(f"**参考**: {quote_obj['reference']}")
                else:
                    st.info("キーワードが抽出できませんでした。テキストを詳しく入力してください。")
    
    elif app_mode == "言葉のエネルギー球体視覚化":
        st.header("言葉のエネルギーで繋がる球体視覚化")
        st.markdown("入力した言葉を中心に、QUBOエネルギーで関連する言葉が繋がります")
        
        user_input = st.text_input(
            "願いを入力してください",
            placeholder="例：世界平和に貢献できる人間になる",
            value="世界平和に貢献できる人間になる"
        )
        
        if st.button("可視化"):
            if not user_input.strip():
                st.warning("テキストを入力してください")
            else:
                # キーワード抽出
                keywords = extract_keywords(user_input)
                st.write(f"**抽出されたキーワード**: {', '.join(keywords)}")
                
                # ネットワーク構築
                network = build_word_network(keywords, GLOBAL_WORDS_DATABASE, n_neighbors=20)
                
                # 3D配置
                center_indices = [i for i, word in enumerate(network['words']) if word in keywords]
                positions = place_words_on_sphere(len(network['words']), center_indices)
                
                # 3D可視化
                fig = create_3d_network_plot(network, positions, center_indices)
                st.plotly_chart(fig, use_container_width=True)
                
                # 格言を表示
                quote_text = select_relevant_quote(keywords)
                st.markdown("---")
                st.subheader("神託（Oracle）")
                st.success(f"「{quote_text}」")
                
                # 引用元を表示
                quote_obj = None
                for q in FAMOUS_QUOTES:
                    if q['quote'] == quote_text:
                        quote_obj = q
                        break
                
                if quote_obj:
                    st.caption(f"*出典: {quote_obj['source']} - {quote_obj['origin']}*")
                    with st.expander("詳細な引用情報"):
                        st.write(f"**出典**: {quote_obj['source']}")
                        st.write(f"**由来**: {quote_obj['origin']}")
                        st.write(f"**参考**: {quote_obj['reference']}")
                
                st.markdown("**あなたの観測が、この世界線を確定させました。**")
    
    elif app_mode == "絵馬納め":
        st.header("🎋 絵馬納め")
        st.markdown("願いを絵馬に書いて納めると、秋葉三尺坊大権現が現れて神託を授けてくださいます")
        
        # 絵馬の説明
        st.info("""
        **絵馬とは**: 神社や寺院に願い事を書いて奉納する木の板です。
        願いを書いて納めることで、神様に願いが届くとされています。
        """)
        
        # 絵馬入力
        ema_text = st.text_area(
            "絵馬に願いを書いてください",
            placeholder="例：健康で過ごせますように、仕事がうまくいきますように、家族が幸せでありますように...",
            height=150,
            help="あなたの願いや悩みを自由に書いてください"
        )
        
        if st.button("🎋 絵馬を納める", type="primary", use_container_width=True):
            if not ema_text.strip():
                st.warning("願いを書いてから納めてください")
            else:
                # セッション状態でアニメーション制御
                if 'show_character' not in st.session_state:
                    st.session_state.show_character = False
                
                st.session_state.show_character = True
                
                # 絵馬が納められる演出
                st.success("✨ 絵馬が納められました...")
                
                # 待機演出
                with st.spinner("秋葉三尺坊大権現が現れています..."):
                    time.sleep(1.0)
                
                # キャラクターアニメーション（HTML/CSS/JavaScript）
                character_html = """
                <div id="character-container" style="
                    position: relative;
                    width: 100%;
                    height: 450px;
                    background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 50%, #0a0a1a 100%);
                    border-radius: 15px;
                    overflow: hidden;
                    margin: 20px 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 0 30px rgba(255, 215, 0, 0.2);
                ">
                    <style>
                        @keyframes fadeIn {
                            from { opacity: 0; transform: translateY(30px) scale(0.9); }
                            to { opacity: 1; transform: translateY(0) scale(1); }
                        }
                        
                        @keyframes float {
                            0%, 100% { transform: translateY(0px) rotate(0deg); }
                            25% { transform: translateY(-15px) rotate(-2deg); }
                            50% { transform: translateY(-25px) rotate(0deg); }
                            75% { transform: translateY(-15px) rotate(2deg); }
                        }
                        
                        @keyframes glow {
                            0%, 100% { 
                                text-shadow: 0 0 10px rgba(255, 215, 0, 0.5), 
                                            0 0 20px rgba(255, 215, 0, 0.3),
                                            0 0 30px rgba(255, 215, 0, 0.1); 
                            }
                            50% { 
                                text-shadow: 0 0 20px rgba(255, 215, 0, 0.8), 
                                            0 0 30px rgba(255, 215, 0, 0.5),
                                            0 0 40px rgba(255, 215, 0, 0.3); 
                            }
                        }
                        
                        @keyframes sparkle {
                            0%, 100% { opacity: 0; transform: scale(0) rotate(0deg); }
                            50% { opacity: 1; transform: scale(1.2) rotate(180deg); }
                        }
                        
                        @keyframes shimmer {
                            0% { background-position: -1000px 0; }
                            100% { background-position: 1000px 0; }
                        }
                        
                        .character {
                            animation: fadeIn 2s ease-out, float 4s ease-in-out infinite;
                            font-size: 140px;
                            text-align: center;
                            color: #ffffff;
                            filter: drop-shadow(0 0 20px rgba(255, 255, 255, 0.4));
                            display: inline-block;
                        }
                        
                        .title {
                            animation: fadeIn 2s ease-out 0.5s both, glow 3s ease-in-out infinite;
                            font-size: 28px;
                            color: #ffd700;
                            text-align: center;
                            margin-top: 20px;
                            font-weight: bold;
                            font-family: 'Yu Gothic', 'Meiryo', 'MS Gothic', sans-serif;
                            letter-spacing: 2px;
                        }
                        
                        .message {
                            animation: fadeIn 2s ease-out 1s both;
                            color: #ffffff;
                            margin-top: 15px;
                            font-size: 18px;
                            font-family: 'Yu Gothic', 'Meiryo', 'MS Gothic', sans-serif;
                        }
                        
                        .sparkle {
                            position: absolute;
                            color: #ffd700;
                            font-size: 24px;
                            animation: sparkle 2s ease-in-out infinite;
                            pointer-events: none;
                        }
                        
                        .background-shimmer {
                            position: absolute;
                            top: 0;
                            left: 0;
                            width: 100%;
                            height: 100%;
                            background: linear-gradient(
                                90deg,
                                transparent 0%,
                                rgba(255, 215, 0, 0.1) 50%,
                                transparent 100%
                            );
                            background-size: 200% 100%;
                            animation: shimmer 3s linear infinite;
                            pointer-events: none;
                        }
                    </style>
                    
                    <div class="background-shimmer"></div>
                    
                    <div style="position: relative; text-align: center; z-index: 1;">
                        <div class="character">🦊✨</div>
                        <div class="title">秋葉三尺坊大権現</div>
                        <div class="message">あなたの願いを聞き届けました</div>
                    </div>
                    
                    <div class="sparkle" style="top: 15%; left: 15%; animation-delay: 0s;">✨</div>
                    <div class="sparkle" style="top: 25%; right: 20%; animation-delay: 0.7s;">✨</div>
                    <div class="sparkle" style="bottom: 30%; left: 25%; animation-delay: 1.4s;">✨</div>
                    <div class="sparkle" style="bottom: 40%; right: 15%; animation-delay: 2.1s;">✨</div>
                    <div class="sparkle" style="top: 50%; left: 10%; animation-delay: 0.3s;">✨</div>
                    <div class="sparkle" style="top: 60%; right: 10%; animation-delay: 1.0s;">✨</div>
                </div>
                
                <script>
                    // 追加のインタラクティブ効果（オプション）
                    setTimeout(function() {
                        var container = document.getElementById('character-container');
                        if (container) {
                            container.style.transition = 'all 0.3s ease';
                        }
                    }, 100);
                </script>
                """
                
                st.components.v1.html(character_html, height=450)
                
                # 願いを分析して神託を生成
                m = infer_mood(ema_text)
                Q_today = build_qubo_from_mood(m)
                sols = solve_all(Q_today)
                
                # おみくじ（Moodに応じて変化）
                pool = sols[:6]
                T = temperature_from_mood(m)
                e_pick, x_pick = boltzmann_sample(pool, T=T)
                card = oracle_card(e_pick, x_pick, mood=m)
                
                st.markdown("---")
                st.subheader("🔮 秋葉三尺坊大権現からの神託")
                
                # 選ばれた格言の引用元情報を収集
                sources_text = []
                for pick in card['picks']:
                    source_info = get_maxim_source(pick)
                    sources_text.append(f"- {pick}\n  *出典: {source_info['source']} - {source_info['origin']}*")
                
                # 神託カードを美しく表示
                st.info(f"""
**エネルギー**: {card['energy']:.3f}

**選ばれた縁**:
{chr(10).join(sources_text)}

**ことば**:
「{card['poem']}」

**次の一歩**:
{card['hint']}
""")
                
                st.caption(f"※揺らぎ(T)={T:.2f}（大きいほど偶然性が増えます）")
                
                # 心の傾きを表示
                st.markdown("---")
                st.subheader("📊 あなたの心の傾き")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("疲れ", f"{m.fatigue:.2f}")
                with col2:
                    st.metric("不安/焦り", f"{m.anxiety:.2f}")
                with col3:
                    st.metric("好奇心", f"{m.curiosity:.2f}")
                with col4:
                    st.metric("孤独", f"{m.loneliness:.2f}")
                with col5:
                    st.metric("決断", f"{m.decisiveness:.2f}")
                
                # 感謝のメッセージ
                st.markdown("---")
                st.success("""
                **🎋 絵馬が納められました**
                
                秋葉三尺坊大権現があなたの願いを聞き届け、神託を授けました。
                この神託を胸に、一歩ずつ進んでいきましょう。
                """)

if __name__ == "__main__":
    main()
