"""
Q-Quest 量子神託 - Streamlitアプリケーション
Human-Centric Quantum Philosophy
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import itertools
import math
import random
import re
import time
from collections import Counter
import pandas as pd
import io
import os
import requests
import json

# Janome for Japanese morphological analysis (長期的改善)
try:
    from janome.tokenizer import Tokenizer
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False
    Tokenizer = None

# Optuna for QUBO optimization visualization
try:
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_contour,
        plot_slice,
        plot_timeline
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    # 可視化関数もNoneに設定
    plot_optimization_history = None
    plot_param_importances = None
    plot_parallel_coordinate = None
    plot_contour = None
    plot_slice = None
    plot_timeline = None

# -------------------------
# 文字列ユーティリティ
# -------------------------
def _split_multi_text(cell_value: str) -> List[str]:
    """Excelセル内の複数テキストを分割（改行 / '||' 区切り対応）"""
    if cell_value is None:
        return []
    s = str(cell_value).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    # '||' と改行を同一視して分割
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    parts: List[str] = []
    for chunk in s.split("||"):
        parts.extend([p.strip() for p in chunk.split("\n") if p.strip()])
    return [p for p in parts if p]

def _parse_tagged_quote(line: str) -> Dict[str, object]:
    """'タグ1,タグ2::本文' 形式をパース。タグがなければ tags=[]"""
    raw = (line or "").strip()
    if "::" in raw:
        tag_part, quote_part = raw.split("::", 1)
        tags = [t.strip() for t in tag_part.split(",") if t.strip()]
        quote = quote_part.strip()
        return {"text": quote, "tags": tags}
    return {"text": raw, "tags": []}

def extract_keywords_safe(text: str, top_n: int = 6, use_llm: bool = False, llm_type: str = "huggingface") -> List[str]:
    """UI/最適化用のキーワード抽出（失敗しても落とさない + LLMオプション）"""
    try:
        keywords = extract_keywords(text, top_n=top_n, use_llm=use_llm, llm_type=llm_type)  # 既存関数を利用（後方で定義される）
        # キーワードが抽出されない場合、フォールバック処理
        if not keywords:
            t = (text or "").strip()
            if not t:
                return []
            # 簡易: 2文字以上の連続を上位
            import re
            text_clean = re.sub(r'[0-9０-９\W]+', ' ', t)
            words = text_clean.split()
            keywords = [w for w in words if len(w) >= 2][:top_n]
        return keywords
    except Exception:
        # extract_keywords 定義前に呼ばれた等の保険
        t = (text or "").strip()
        if not t:
            return []
        # 簡易: 2文字以上の連続を上位
        tokens = [w for w in re.split(r"[\s、。,.!！?？]+", t) if len(w) >= 2]
        return tokens[:top_n]

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
# 12神の定義と属性マッピング（秋葉原テーマ）
# 各神は12個の誓願（誓願01～12）と4つの役割属性（静、流、間、誠）を持つ
TWELVE_GODS = [
    {
        "id": 0,
        "name": "秋葉三尺坊",
        "name_en": "Akiba Sanjakubo",
        "attribute": "火",
        "emoji": "🔥",
        # 誓願01～12の数値配置（添付資料より）
        "vows": {
            "vow01": -0.4, "vow02": 0.2, "vow03": -0.2, "vow04": 0.0, "vow05": 0.0,
            "vow06": 0.0, "vow07": 0.0, "vow08": -0.4, "vow09": 0.0, "vow10": 0.0,
            "vow11": 0.0, "vow12": -0.2
        },
        # 役割属性（静、流、間、誠）
        "roles": {"stillness": 0.0, "flow": -0.2, "ma": 0.0, "sincerity": -0.4},
        "maxim": "勢いMAX: 情熱的な筆致に降臨。",
        "description": "秋葉原の守護神。火伏せ=「炎上回避」の神。"
    },
    {
        "id": 1,
        "name": "真空管大将軍",
        "name_en": "Vacuum Tube General",
        "attribute": "電",
        "emoji": "⚡",
        "vows": {
            "vow01": -0.2, "vow02": 0.2, "vow03": 0.0, "vow04": -0.4, "vow05": -0.2,
            "vow06": 0.0, "vow07": 0.0, "vow08": 0.0, "vow09": 0.0, "vow10": 0.0,
            "vow11": 0.0, "vow12": -0.4
        },
        "roles": {"stillness": 0.0, "flow": -0.4, "ma": 0.0, "sincerity": -0.2},
        "maxim": "線の太さ: 力強く、太い線に反応。",
        "description": "秋葉原の原点。増幅=「才能開花」の神。"
    },
    {
        "id": 2,
        "name": "LED弁財天",
        "name_en": "LED Benzaiten",
        "attribute": "光",
        "emoji": "💡",
        "vows": {
            "vow01": 0.0, "vow02": 0.2, "vow03": 0.0, "vow04": -0.4, "vow05": 0.0,
            "vow06": 0.0, "vow07": 0.0, "vow08": 0.0, "vow09": -0.4, "vow10": 0.0,
            "vow11": -0.2, "vow12": -0.2
        },
        "roles": {"stillness": 0.0, "flow": -0.4, "ma": -0.2, "sincerity": 0.0},
        "maxim": "丸み: 華やかで曲線的な筆跡。",
        "description": "イルミネーションと発光。「自己表現」の神。"
    },
    {
        "id": 3,
        "name": "磁気記録黒龍",
        "name_en": "Magnetic Recording Black Dragon",
        "attribute": "磁",
        "emoji": "🐉",
        "vows": {
            "vow01": 0.0, "vow02": 0.0, "vow03": -0.4, "vow04": 0.0, "vow05": -0.2,
            "vow06": 0.0, "vow07": 0.0, "vow08": 0.0, "vow09": 0.0, "vow10": -0.4,
            "vow11": -0.2, "vow12": 0.2
        },
        "roles": {"stillness": -0.2, "flow": 0.0, "ma": 0.0, "sincerity": -0.4},
        "maxim": "緻密さ: 細かく丁寧な書き込み。",
        "description": "HDDやテープ。記憶=「温故知新」の守護龍。"
    },
    {
        "id": 4,
        "name": "無線傍受観音",
        "name_en": "Wireless Interception Kannon",
        "attribute": "波",
        "emoji": "📡",
        "vows": {
            "vow01": -0.4, "vow02": 0.2, "vow03": 0.0, "vow04": -0.2, "vow05": -0.4,
            "vow06": 0.0, "vow07": 0.0, "vow08": 0.0, "vow09": 0.0, "vow10": 0.0,
            "vow11": 0.0, "vow12": -0.2
        },
        "roles": {"stillness": 0.0, "flow": -0.4, "ma": -0.2, "sincerity": 0.0},
        "maxim": "ゆらぎ: 震えや迷いがある筆跡に寄り添う。",
        "description": "電波と通信。縁結び=「マッチング」の神。"
    },
    {
        "id": 5,
        "name": "基板曼荼羅",
        "name_en": "Circuit Board Mandala",
        "attribute": "基",
        "emoji": "🔌",
        "vows": {
            "vow01": 0.0, "vow02": -0.2, "vow03": 0.0, "vow04": 0.0, "vow05": 0.0,
            "vow06": -0.4, "vow07": -0.4, "vow08": 0.0, "vow09": 0.2, "vow10": -0.2,
            "vow11": 0.0, "vow12": 0.0
        },
        "roles": {"stillness": -0.4, "flow": 0.0, "ma": 0.0, "sincerity": -0.2},
        "maxim": "直線的: 迷いのない、カクカクした線。",
        "description": "回路設計。秩序=「論理的思考」の神。"
    },
    {
        "id": 6,
        "name": "絶対零度明王",
        "name_en": "Absolute Zero Myo-o",
        "attribute": "冷",
        "emoji": "❄️",
        "vows": {
            "vow01": 0.0, "vow02": -0.4, "vow03": -0.2, "vow04": 0.0, "vow05": 0.0,
            "vow06": 0.0, "vow07": -0.4, "vow08": 0.0, "vow09": 0.0, "vow10": 0.0,
            "vow11": -0.2, "vow12": 0.2
        },
        "roles": {"stillness": -0.4, "flow": 0.0, "ma": -0.2, "sincerity": 0.0},
        "maxim": "筆圧弱め: クールで淡々とした筆跡。",
        "description": "冷却ファン・超電導。冷静=「沈着冷静」の神。"
    },
    {
        "id": 7,
        "name": "ジャンク再生童子",
        "name_en": "Junk Regeneration Child",
        "attribute": "壊",
        "emoji": "🔧",
        "vows": {
            "vow01": -0.2, "vow02": 0.0, "vow03": 0.0, "vow04": -0.2, "vow05": 0.0,
            "vow06": 0.0, "vow07": 0.2, "vow08": -0.4, "vow09": 0.0, "vow10": 0.0,
            "vow11": 0.0, "vow12": -0.4
        },
        "roles": {"stillness": 0.0, "flow": -0.4, "ma": 0.0, "sincerity": -0.2},
        "maxim": "かすれ: 荒々しい、または掠れた線。",
        "description": "秋葉原のジャンク品。復活=「再起・リトヲ」の神。"
    },
    {
        "id": 8,
        "name": "真空オーディオ如来",
        "name_en": "Vacuum Audio Nyorai",
        "attribute": "音",
        "emoji": "🎧",
        "vows": {
            "vow01": 0.0, "vow02": 0.0, "vow03": 0.0, "vow04": -0.2, "vow05": -0.4,
            "vow06": 0.0, "vow07": 0.2, "vow08": 0.0, "vow09": -0.2, "vow10": 0.0,
            "vow11": -0.4, "vow12": 0.0
        },
        "roles": {"stillness": 0.0, "flow": -0.4, "ma": -0.2, "sincerity": 0.0},
        "maxim": "調和: 文字全体のバランスが良い。",
        "description": "高音質・共鳴。「本質を見極める」神。"
    },
    {
        "id": 9,
        "name": "ハンダ付け結び神",
        "name_en": "Soldering Connection Deity",
        "attribute": "結",
        "emoji": "🔗",
        "vows": {
            "vow01": 0.0, "vow02": -0.4, "vow03": -0.2, "vow04": 0.0, "vow05": -0.4,
            "vow06": 0.0, "vow07": -0.2, "vow08": 0.0, "vow09": 0.0, "vow10": 0.0,
            "vow11": 0.0, "vow12": 0.2
        },
        "roles": {"stillness": -0.2, "flow": 0.0, "ma": -0.4, "sincerity": 0.0},
        "maxim": "トメ・ハネ: 繋ぎ部分がしっかりしている。",
        "description": "接点と結合。協力=「チームワーク」の神。"
    },
    {
        "id": 10,
        "name": "光速通信韋駄天",
        "name_en": "Light-speed Communication Idaten",
        "attribute": "速",
        "emoji": "🚀",
        "vows": {
            "vow01": 0.0, "vow02": 0.2, "vow03": 0.0, "vow04": -0.2, "vow05": -0.4,
            "vow06": 0.0, "vow07": 0.0, "vow08": 0.0, "vow09": -0.2, "vow10": 0.0,
            "vow11": 0.0, "vow12": -0.4
        },
        "roles": {"stillness": 0.0, "flow": -0.4, "ma": 0.0, "sincerity": -0.2},
        "maxim": "書き速度: サッと短時間で書いた線。",
        "description": "5G・光回線。爆速=「即断即決」の神。"
    },
    {
        "id": 11,
        "name": "半導体文殊",
        "name_en": "Semiconductor Manjushri",
        "attribute": "智",
        "emoji": "🧠",
        "vows": {
            "vow01": 0.0, "vow02": 0.0, "vow03": -0.2, "vow04": 0.0, "vow05": 0.0,
            "vow06": -0.4, "vow07": -0.2, "vow08": 0.0, "vow09": 0.0, "vow10": -0.4,
            "vow11": 0.0, "vow12": 0.2
        },
        "roles": {"stillness": -0.4, "flow": 0.0, "ma": 0.0, "sincerity": -0.2},
        "maxim": "規則性: 等間隔で整理された筆跡。",
        "description": "CPU・AI。計算=「合格・知略」の神。"
    },
]

# 気持ちを整えるための格言（VARIABLES）を12神の格言に更新
VARIABLES = [god["maxim"] for god in TWELVE_GODS]

# 感覚層の変数定義（添付資料より）
SENSATION_VARIABLES = [
    "迷い",      # x1: Hesitation/Confusion
    "焦り",      # x2: Impatience/Anxiety
    "静けさ",    # x3: Stillness/Calmness
    "内省",      # x4: Introspection
    "行動",      # x5: Action
    "つながり",  # x6: Connection
    "挑戦",      # x7: Challenge
    "待つ",      # x8: Wait
]

# 格言の引用元（12神の格言に対応）
MAXIM_SOURCES = {god["maxim"]: {
    "source": god["name"],
    "origin": god["name_en"],
    "reference": god["description"]
} for god in TWELVE_GODS}

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
    "絆", "つながり", "家族", "友人", "仲間", "信頼", "尊敬", "協力", "夫婦", "生活", "円満",
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
    # 【追加】願い・祈り・希望のカテゴリ
    "wish": [
        "願い", "祈り", "希望", "願う", "祈る", "望む", "願望", "切望",
        "でありますように", "ように", "であります", "ありますように",
        "できますように", "なりますように", "過ごせますように"
    ],
    # 【追加】家族・関係性のカテゴリ
    "family": [
        "家族", "夫婦", "親", "子", "兄弟", "姉妹", "祖父母", "親戚",
        "家庭", "生活", "円満", "仲良く", "幸せ", "平和", "調和",
        "絆", "つながり", "愛情", "思いやり", "支え", "協力"
    ],
    # 【追加】健康・体調のカテゴリ
    "health": [
        "健康", "体調", "身体", "体", "病気", "治療", "回復", "元気",
        "過ごしたい", "過ごせますように", "健やか", "丈夫", "強く"
    ],
    # 【追加】仕事・キャリアのカテゴリ
    "work": [
        "仕事", "職場", "キャリア", "働く", "就職", "転職", "昇進",
        "成功", "成果", "達成", "目標", "プロジェクト", "業務"
    ],
    # 【追加】学び・成長のカテゴリ
    "learning": [
        "学び", "学習", "勉強", "教育", "知識", "スキル", "向上",
        "成長", "発展", "進歩", "習得", "理解", "覚える"
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

def mood_to_sensation_vector(m: Mood, binary: bool = False, scale: float = 5.0) -> np.ndarray:
    """Moodから感覚ベクトル（x1～x8）を生成
    
    Args:
        m: Moodオブジェクト
        binary: Trueの場合、バイナリ化（0.3以上で1、それ以下で0）
        scale: 感覚ベクトルのスケール（0〜scaleの範囲に正規化、デフォルト5.0）
    
    Returns:
        感覚ベクトル（8次元）
        - x0: 迷い, x1: 焦り, x2: 静けさ, x3: 内省, x4: 行動, x5: つながり, x6: 挑戦, x7: 待つ
    """
    # 感覚変数のマッピング
    x = np.zeros(8)
    
    # 迷い（x0）: 不安と決断力の低さから
    x[0] = m.anxiety * (1.0 - m.decisiveness)
    
    # 焦り（x1）: 不安から
    x[1] = m.anxiety
    
    # 静けさ（x2）: 疲れと孤独から
    x[2] = (m.fatigue + m.loneliness) / 2.0
    
    # 内省（x3）: 孤独と疲れから
    x[3] = (m.loneliness + m.fatigue) / 2.0
    
    # 行動（x4）: 好奇心と決断力から
    x[4] = (m.curiosity + m.decisiveness) / 2.0
    
    # つながり（x5）: 孤独の逆と好奇心から
    x[5] = (1.0 - m.loneliness) * m.curiosity
    
    # 挑戦（x6）: 好奇心と決断力から
    x[6] = m.curiosity * m.decisiveness
    
    # 待つ（x7）: 疲れと決断力の低さから
    x[7] = m.fatigue * (1.0 - m.decisiveness)
    
    # スケール調整（0〜scaleの範囲に正規化）
    x = x * scale
    
    if binary:
        # バイナリ化（閾値0.3*scale以上で1、それ以下で0）
        x_binary = (x >= 0.3 * scale).astype(float)
        return x_binary
    else:
        # 連続値のまま返す（0〜scaleの範囲）
        return x

# -------------------------
# Excelファイル読み込み機能
# -------------------------
# グローバル変数：Excelから読み込んだデータを保持
SENSE_TO_VOW_MATRIX: Optional[np.ndarray] = None  # sense_to_vow行列（8x12：感覚 × 誓願）
K_MATRIX: Optional[np.ndarray] = None  # k行列（12x12：キャラクター × 誓願）
L_MATRIX: Optional[np.ndarray] = None  # l行列（12x4：キャラクター × 世界観軸）
LOADED_GODS: Optional[List[Dict]] = None  # Excelから読み込んだ12神の情報
CHAR_MASTER: Optional[pd.DataFrame] = None  # CHAR_MASTERシートのデータ
SELECTED_ATTRIBUTE: Optional[str] = None  # ユーザーが選択した属性
SELECTED_CHARACTER: Optional[str] = None  # ユーザーが選択したキャラクター（公式キャラ名）
MAXIMS_DATABASE: Optional[List[Dict]] = None  # 格言ファイルから読み込んだ格言データベース

def rebuild_globals_from_gods(gods_list: List[Dict]) -> None:
    """TWELVE_GODS 変更後に、VARIABLES / MAXIM_SOURCES を再生成"""
    global VARIABLES, MAXIM_SOURCES
    VARIABLES = [god.get("maxim", "") for god in gods_list]
    # すべての格言（複数）も出典に載せる
    maxim_sources: Dict[str, Dict] = {}
    for god in gods_list:
        # 単一格言
        if god.get("maxim"):
            maxim_sources[god["maxim"]] = {
                "source": god.get("name", "神託"),
                "origin": god.get("name_en", ""),
                "reference": god.get("description", ""),
            }
        # 複数格言
        for item in god.get("maxims", []) or []:
            text = (item.get("text") if isinstance(item, dict) else str(item)).strip()
            if text:
                maxim_sources[text] = {
                    "source": god.get("name", "神託"),
                    "origin": god.get("name_en", ""),
                    "reference": god.get("description", ""),
                }
    MAXIM_SOURCES = maxim_sources

def load_gods_from_separate_files(
    character_file: io.BytesIO = None,
    k_matrix_file: io.BytesIO = None,
    l_matrix_file: io.BytesIO = None
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """3つの別々のExcelファイルから12神の情報、k行列、l行列を読み込む
    
    Args:
        character_file: 12神基本情報のExcelファイル（akiba12_character_list.xlsx）
        k_matrix_file: k行列のExcelファイル（akiba12_character_to_vow_K.xlsx）
        l_matrix_file: l行列のExcelファイル（akiba12_character_to_axis_L.xlsx）
    
    Returns:
        (gods_list, k_matrix, l_matrix)
    """
    try:
        # k行列を読み込む（キャラクター名を行インデックスとして使用）
        if k_matrix_file is not None:
            k_matrix_file.seek(0)
            df_k = pd.read_excel(k_matrix_file, engine="openpyxl", header=0, index_col=0)
            # 先頭12列・12行に正規化（余分があってもOK）
            df_k = df_k.iloc[:12, :12]
            k_matrix = df_k.values.astype(float)
            character_names_from_k = df_k.index.tolist()
        else:
            raise ValueError("k行列ファイルが必要です")
        
        # l行列を読み込む（キャラクター名を行インデックスとして使用）
        if l_matrix_file is not None:
            l_matrix_file.seek(0)
            df_l = pd.read_excel(l_matrix_file, engine="openpyxl", header=0, index_col=0)
            df_l = df_l.iloc[:12, :4]
            l_matrix = df_l.values.astype(float)
            character_names_from_l = df_l.index.tolist()
        else:
            raise ValueError("l行列ファイルが必要です")
        
        # 12神基本情報を読み込む
        if character_file is not None:
            character_file.seek(0)
            df_gods = pd.read_excel(character_file, engine="openpyxl")
        else:
            # 基本情報ファイルがない場合、k行列とl行列からキャラクター名を取得
            # キャラクター名の一致を確認
            common_names = [name for name in character_names_from_k if name in character_names_from_l]
            if len(common_names) != 12:
                raise ValueError(f"キャラクター名が一致しません。k行列: {len(character_names_from_k)}個, l行列: {len(character_names_from_l)}個")
            
            # ダミーの基本情報を作成
            df_gods = pd.DataFrame({
                "ID": range(12),
                "名前": common_names,
                "名前(英語)": [f"God {i+1}" for i in range(12)],
                "属性": [""] * 12,
                "絵文字": ["🔮"] * 12,
                "説明": [""] * 12,
                "格言": [""] * 12
            })
        
        # キャラクター名のマッピングを作成（k行列とl行列の行インデックスと基本情報の名前を対応）
        name_to_id = {}
        for idx, row in df_gods.iterrows():
            god_name = str(row.get("名前", ""))
            if god_name:
                name_to_id[god_name] = int(row.get("ID", idx))
        
        # 12神の情報を構築
        gods_list = []
        for idx, row in df_gods.iterrows():
            god_id = int(row.get("ID", idx))
            god_name = str(row.get("名前", ""))
            god_name_en = str(row.get("名前(英語)", ""))
            god_attribute = str(row.get("属性", ""))
            god_emoji = str(row.get("絵文字", "🔮"))
            god_description = str(row.get("説明", ""))
            # 複数格言対応：格言 / 格言1.. / 改行 / '||' / 'タグ::本文'
            maxim_cells: List[str] = []
            # 列名が "格言" だけのケース
            maxim_cells.extend(_split_multi_text(row.get("格言", "")))
            # 列名が "格言1","格言2"... のケース
            for col in row.index:
                if isinstance(col, str) and col.startswith("格言") and col != "格言":
                    maxim_cells.extend(_split_multi_text(row.get(col, "")))
            maxims_parsed = [_parse_tagged_quote(m) for m in maxim_cells if str(m).strip()]
            # 互換性のため先頭を maxim に入れる
            god_maxim = maxims_parsed[0]["text"] if maxims_parsed else ""
            
            # k行列から誓願値を取得（キャラクター名で検索）
            vows = {}
            if god_name in df_k.index:
                k_row_idx = df_k.index.get_loc(god_name)
                for j in range(min(12, len(df_k.columns))):
                    vow_key = f"vow{j+1:02d}"
                    col_name = df_k.columns[j]
                    vows[vow_key] = float(k_matrix[k_row_idx, j])
            else:
                # キャラクター名が見つからない場合、IDで検索
                if god_id < len(k_matrix):
                    for j in range(min(12, len(df_k.columns))):
                        vow_key = f"vow{j+1:02d}"
                        vows[vow_key] = float(k_matrix[god_id, j])
            
            # l行列から役割属性を取得（キャラクター名で検索）
            role_names = ["stillness", "flow", "ma", "sincerity"]
            roles = {}
            if god_name in df_l.index:
                l_row_idx = df_l.index.get_loc(god_name)
                for j, role_name in enumerate(role_names):
                    if j < len(df_l.columns):
                        roles[role_name] = float(l_matrix[l_row_idx, j])
                    else:
                        roles[role_name] = 0.0
            else:
                # キャラクター名が見つからない場合、IDで検索
                if god_id < len(l_matrix):
                    for j, role_name in enumerate(role_names):
                        if j < len(l_matrix[god_id]):
                            roles[role_name] = float(l_matrix[god_id, j])
                        else:
                            roles[role_name] = 0.0
            
            god_dict = {
                "id": god_id,
                "name": god_name,
                "name_en": god_name_en,
                "attribute": god_attribute,
                "emoji": god_emoji,
                "vows": vows,
                "roles": roles,
                "maxim": god_maxim,
                "maxims": maxims_parsed,  # 複数格言
                "description": god_description,
            }
            gods_list.append(god_dict)
        
        return gods_list, k_matrix, l_matrix
    
    except Exception as e:
        st.error(f"Excelファイルの読み込みエラー: {str(e)}")
        import traceback
        st.error(f"詳細: {traceback.format_exc()}")
        raise

def get_excel_sheet_names(excel_file: io.BytesIO) -> List[str]:
    """Excelファイルのシート名一覧を取得"""
    try:
        excel_file.seek(0)
        xl_file = pd.ExcelFile(excel_file, engine="openpyxl")
        return xl_file.sheet_names
    except Exception:
        return []

def find_sheet_by_keywords(excel_file: io.BytesIO, keywords: List[str]) -> Optional[str]:
    """キーワードに基づいてシート名を検索"""
    sheet_names = get_excel_sheet_names(excel_file)
    for sheet_name in sheet_names:
        for keyword in keywords:
            if keyword in sheet_name:
                return sheet_name
    return None

def load_gods_from_excel(excel_file: io.BytesIO) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """1つのExcelファイルから12神の情報、k行列、l行列を読み込む（後方互換性のため）
    
    Args:
        excel_file: Excelファイル（BytesIO）- 複数のシートを含む
    
    Returns:
        (gods_list, k_matrix, l_matrix)
    """
    try:
        # ファイルポインタをリセット（複数シートを読み込むため）
        excel_file.seek(0)
        
        # シート名を自動検出
        sheet_names = get_excel_sheet_names(excel_file)
        
        # 12神基本情報のシートを検索
        gods_sheet = find_sheet_by_keywords(excel_file, ["CHAR_MASTER", "12神", "基本情報", "character", "CHAR"])
        if gods_sheet is None:
            # デフォルトのシート名を試す
            try:
                excel_file.seek(0)
                df_gods = pd.read_excel(excel_file, sheet_name=0, engine="openpyxl")  # 最初のシート
            except:
                raise ValueError(f"キャラクター情報のシートが見つかりません。利用可能なシート: {sheet_names}")
        else:
            excel_file.seek(0)
            df_gods = pd.read_excel(excel_file, sheet_name=gods_sheet, engine="openpyxl")
        
        # CHAR_MASTERシートの場合、すべての情報が含まれている
        is_char_master = gods_sheet and "CHAR_MASTER" in gods_sheet.upper()
        
        # k行列の読み込み
        if is_char_master:
            # CHAR_MASTERシートにVOW_01～VOW_12が含まれている
            vow_columns = [f"VOW_{i:02d}" for i in range(1, 13)]
            if all(col in df_gods.columns for col in vow_columns):
                # CHAR_MASTERからk行列を構築
                df_k = df_gods.set_index("公式キャラ名")[vow_columns]
                k_matrix = df_k.values.astype(float)
            else:
                # VOW列が見つからない場合、CHAR_TO_VOWシートを探す
                excel_file.seek(0)
                k_sheet = find_sheet_by_keywords(excel_file, ["CHAR_TO_VOW", "k行列", "K"])
                if k_sheet:
                    excel_file.seek(0)
                    df_k = pd.read_excel(excel_file, sheet_name=k_sheet, engine="openpyxl", header=0)
                    df_k = df_k.set_index("公式キャラ名")
                    vow_columns = [col for col in df_k.columns if str(col).startswith("VOW_")]
                    df_k = df_k[vow_columns[:12]]
                    k_matrix = df_k.values.astype(float)
                else:
                    raise ValueError(f"k行列が見つかりません。CHAR_MASTERにVOW列がないか、CHAR_TO_VOWシートが見つかりません。")
        else:
            # CHAR_TO_VOWシートから読み込む
            excel_file.seek(0)
            k_sheet = find_sheet_by_keywords(excel_file, ["CHAR_TO_VOW", "k行列", "K"])
            if k_sheet is None:
                raise ValueError(f"k行列のシートが見つかりません。利用可能なシート: {sheet_names}")
            
            excel_file.seek(0)
            df_k = pd.read_excel(excel_file, sheet_name=k_sheet, engine="openpyxl", header=0)
            
            # 行インデックスを設定（公式キャラ名またはCHAR_ID）
            index_col = None
            if "公式キャラ名" in df_k.columns:
                index_col = "公式キャラ名"
            elif "CHAR_ID" in df_k.columns:
                index_col = "CHAR_ID"
            
            if index_col:
                df_k = df_k.set_index(index_col)
            
            # VOW_01～VOW_12の列のみを選択（数値列のみ）
            vow_columns = [col for col in df_k.columns if str(col).startswith("VOW_")]
            if len(vow_columns) >= 12:
                df_k = df_k[vow_columns[:12]]
            else:
                raise ValueError(f"VOW列が12個見つかりません。見つかった列: {vow_columns}")
            
            df_k = df_k.iloc[:12, :12]
            k_matrix = df_k.values.astype(float)
        
        # l行列の読み込み
        if is_char_master:
            # CHAR_MASTERシートにAXIS_SEI, AXIS_RYU, AXIS_MA, AXIS_MAKOTOが含まれている
            axis_columns = ["AXIS_SEI", "AXIS_RYU", "AXIS_MA", "AXIS_MAKOTO"]
            if all(col in df_gods.columns for col in axis_columns):
                # CHAR_MASTERからl行列を構築
                df_l = df_gods.set_index("公式キャラ名")[axis_columns]
                l_matrix = df_l.values.astype(float)
            else:
                # AXIS列が見つからない場合、CHAR_TO_AXISシートを探す
                excel_file.seek(0)
                l_sheet = find_sheet_by_keywords(excel_file, ["CHAR_TO_AXIS", "l行列", "L"])
                if l_sheet:
                    excel_file.seek(0)
                    df_l = pd.read_excel(excel_file, sheet_name=l_sheet, engine="openpyxl", header=0)
                    df_l = df_l.set_index("公式キャラ名")
                    axis_columns = ["AXIS_SEI", "AXIS_RYU", "AXIS_MA", "AXIS_MAKOTO"]
                    df_l = df_l[axis_columns]
                    l_matrix = df_l.values.astype(float)
                else:
                    raise ValueError(f"l行列が見つかりません。CHAR_MASTERにAXIS列がないか、CHAR_TO_AXISシートが見つかりません。")
        else:
            # CHAR_TO_AXISシートから読み込む
            excel_file.seek(0)
            l_sheet = find_sheet_by_keywords(excel_file, ["CHAR_TO_AXIS", "l行列", "L"])
            if l_sheet is None:
                raise ValueError(f"l行列のシートが見つかりません。利用可能なシート: {sheet_names}")
            
            excel_file.seek(0)
            df_l = pd.read_excel(excel_file, sheet_name=l_sheet, engine="openpyxl", header=0)
            
            # 行インデックスを設定（公式キャラ名）
            if "公式キャラ名" in df_l.columns:
                df_l = df_l.set_index("公式キャラ名")
            
            # AXIS_SEI, AXIS_RYU, AXIS_MA, AXIS_MAKOTOの列のみを選択
            axis_columns = ["AXIS_SEI", "AXIS_RYU", "AXIS_MA", "AXIS_MAKOTO"]
            if all(col in df_l.columns for col in axis_columns):
                df_l = df_l[axis_columns]
            else:
                raise ValueError(f"AXIS列が見つかりません。必要な列: {axis_columns}")
            
            df_l = df_l.iloc[:12, :4]
            l_matrix = df_l.values.astype(float)
        
        # 12神の情報を構築
        gods_list = []
        for idx, row in df_gods.iterrows():
            # CHAR_IDからIDを取得（CHAR_01 → 0, CHAR_02 → 1, ...）
            char_id = str(row.get("CHAR_ID", "")).strip()
            if char_id and char_id.startswith("CHAR_"):
                try:
                    god_id = int(char_id.replace("CHAR_", "")) - 1  # CHAR_01 → 0
                except:
                    god_id = int(row.get("ID", idx))
            else:
                god_id = int(row.get("ID", idx))
            
            # 名前の取得（CHAR_MASTERの場合、公式キャラ名を使用）
            if "公式キャラ名" in row.index:
                god_name = str(row.get("公式キャラ名", "")).strip()
            else:
                god_name = str(row.get("名前", ""))
            
            god_name_en = str(row.get("名前(英語)", ""))
            god_attribute = str(row.get("属性", ""))
            god_emoji = str(row.get("絵文字", "🔮"))
            
            # 説明の取得（役割補足説明または説明）
            if "役割補足説明" in row.index:
                god_description = str(row.get("役割補足説明", ""))
            else:
                god_description = str(row.get("説明", ""))
            
            # 公式キャラ名を取得（先に取得）
            official_name = str(row.get("公式キャラ名", "")).strip()
            
            # IMAGE_FILEを取得（CHAR_TO_VOWシートから取得する場合もある）
            image_file = str(row.get("IMAGE_FILE", "")).strip()
            # CHAR_TO_VOWシートからIMAGE_FILEを取得（CHAR_MASTERにない場合）
            if not image_file and is_char_master:
                excel_file.seek(0)
                k_sheet = find_sheet_by_keywords(excel_file, ["CHAR_TO_VOW"])
                if k_sheet:
                    try:
                        excel_file.seek(0)
                        df_char_to_vow = pd.read_excel(excel_file, sheet_name=k_sheet, engine="openpyxl", header=0)
                        # 公式キャラ名でマッチング
                        if official_name and "公式キャラ名" in df_char_to_vow.columns:
                            matched_row = df_char_to_vow[df_char_to_vow["公式キャラ名"] == official_name]
                            if not matched_row.empty and "IMAGE_FILE" in matched_row.columns:
                                image_file = str(matched_row.iloc[0]["IMAGE_FILE"]).strip()
                    except:
                        pass
            
            maxim_cells: List[str] = []
            maxim_cells.extend(_split_multi_text(row.get("格言", "")))
            for col in row.index:
                if isinstance(col, str) and col.startswith("格言") and col != "格言":
                    maxim_cells.extend(_split_multi_text(row.get(col, "")))
            maxims_parsed = [_parse_tagged_quote(m) for m in maxim_cells if str(m).strip()]
            god_maxim = maxims_parsed[0]["text"] if maxims_parsed else str(row.get("格言", ""))
            
            # k行列から誓願値を取得（vow01～vow12）
            vows = {}
            # CHAR_MASTERシートにVOW_01～VOW_12が含まれている場合は直接取得
            if is_char_master and all(f"VOW_{i:02d}" in row.index for i in range(1, 13)):
                for j in range(1, 13):
                    vow_key = f"vow{j:02d}"
                    vows[vow_key] = float(row.get(f"VOW_{j:02d}", 0.0))
            else:
                # k_matrixから取得
                if god_id < len(k_matrix):
                    for j in range(12):
                        vow_key = f"vow{j+1:02d}"
                        vows[vow_key] = float(k_matrix[god_id, j])
            
            # l行列から役割属性を取得
            role_names = ["stillness", "flow", "ma", "sincerity"]
            roles = {}
            # CHAR_MASTERシートにAXIS列が含まれている場合は直接取得
            if is_char_master and all(col in row.index for col in ["AXIS_SEI", "AXIS_RYU", "AXIS_MA", "AXIS_MAKOTO"]):
                roles["stillness"] = float(row.get("AXIS_SEI", 0.0))
                roles["flow"] = float(row.get("AXIS_RYU", 0.0))
                roles["ma"] = float(row.get("AXIS_MA", 0.0))
                roles["sincerity"] = float(row.get("AXIS_MAKOTO", 0.0))
            else:
                # l_matrixから取得
                if god_id < len(l_matrix):
                    for j, role_name in enumerate(role_names):
                        roles[role_name] = float(l_matrix[god_id, j])
            
            god_dict = {
                "id": god_id,
                "name": god_name,
                "name_en": god_name_en,
                "attribute": god_attribute,
                "emoji": god_emoji,
                "vows": vows,
                "roles": roles,
                "maxim": god_maxim,
                "maxims": maxims_parsed,
                "description": god_description,
                "char_id": char_id if char_id else None,  # CHAR_IDを追加
                "image_file": image_file if image_file else None,  # IMAGE_FILEを追加
                "official_name": official_name if official_name else None,  # 公式キャラ名を追加
            }
            gods_list.append(god_dict)
        
        return gods_list, k_matrix, l_matrix
    
    except Exception as e:
        st.error(f"Excelファイルの読み込みエラー: {str(e)}")
        raise

def load_sense_to_vow_matrix(sense_to_vow_file: io.BytesIO) -> np.ndarray:
    """sense_to_vow行列を読み込む（8感覚 × 12誓願）
    
    Args:
        sense_to_vow_file: sense_to_vow行列のExcelファイル
    
    Returns:
        sense_to_vow行列（8x12）
    """
    try:
        sense_to_vow_file.seek(0)
        df_sv = pd.read_excel(sense_to_vow_file, engine="openpyxl", header=0, index_col=0)
        # 8行×12列に正規化
        df_sv = df_sv.iloc[:8, :12]
        sense_to_vow_matrix = df_sv.values.astype(float)
        return sense_to_vow_matrix
    except Exception as e:
        st.error(f"sense_to_vow行列の読み込みエラー: {str(e)}")
        raise

def load_maxims_from_excel(maxim_file: io.BytesIO) -> List[Dict]:
    """格言ファイル（Excel）を読み込む
    
    Args:
        maxim_file: 格言ファイル（Excel）
    
    Returns:
        格言のリスト（各要素は {"text": "格言", "source": "出典", "tags": ["タグ1", "タグ2"]}）
    """
    global MAXIMS_DATABASE
    try:
        maxim_file.seek(0)
        df = pd.read_excel(maxim_file, engine="openpyxl", header=0)
        
        maxims_list = []
        for idx, row in df.iterrows():
            maxim_text = str(row.get("格言", "")).strip()
            source = str(row.get("出典", "")).strip()
            
            if not maxim_text or maxim_text.lower() in ("nan", "none", ""):
                continue
            
            # タグの処理（タグ列がある場合）
            tags = []
            if "タグ" in df.columns:
                tag_str = str(row.get("タグ", "")).strip()
                if tag_str and tag_str.lower() not in ("nan", "none"):
                    tags = [t.strip() for t in tag_str.split(",") if t.strip()]
            
            maxims_list.append({
                "text": maxim_text,
                "source": source if source else "伝統的な教え",
                "tags": tags
            })
        
        MAXIMS_DATABASE = maxims_list
        return maxims_list
    except Exception as e:
        st.error(f"格言ファイルの読み込みエラー: {str(e)}")
        return []

def load_all_excel_files(
    character_file: io.BytesIO = None,
    maxim_file: io.BytesIO = None,
    k_matrix_file: io.BytesIO = None,
    l_matrix_file: io.BytesIO = None,
    sense_to_vow_file: io.BytesIO = None
) -> bool:
    """5つのExcelファイルをまとめて読み込む
    
    Args:
        character_file: 12神基本情報 (akiba12_character_list.xlsx)
        maxim_file: 格言ファイル (格言.xlsx) - オプション
        k_matrix_file: k行列 (akiba12_character_to_vow_K.xlsx)
        l_matrix_file: l行列 (akiba12_character_to_axis_L.xlsx)
        sense_to_vow_file: sense_to_vow行列 (sense_to_vow_initial_filled_from_user.xlsx)
    
    Returns:
        True: 成功, False: 失敗
    """
    result = load_excel_config(
        character_file=character_file,
        k_matrix_file=k_matrix_file,
        l_matrix_file=l_matrix_file,
        sense_to_vow_file=sense_to_vow_file
    )
    
    # 格言ファイルを読み込む（オプション）
    if maxim_file is not None and result:
        maxims = load_maxims_from_excel(maxim_file)
        if maxims:
            st.success(f"✅ 格言ファイルを読み込みました（{len(maxims)}件）")
    
    return result

def load_excel_config(
    excel_file: io.BytesIO = None,
    character_file: io.BytesIO = None,
    k_matrix_file: io.BytesIO = None,
    l_matrix_file: io.BytesIO = None,
    sense_to_vow_file: io.BytesIO = None
) -> bool:
    """Excelファイルを読み込んでグローバル変数を更新
    
    Args:
        excel_file: 1つのExcelファイル（3つのシートを含む）- 後方互換性のため
        character_file: 12神基本情報のExcelファイル（別ファイルの場合）
        k_matrix_file: k行列のExcelファイル（別ファイルの場合）
        l_matrix_file: l行列のExcelファイル（別ファイルの場合）
        sense_to_vow_file: sense_to_vow行列のExcelファイル（8感覚 × 12誓願）
    
    Returns:
        True: 成功, False: 失敗
    """
    global SENSE_TO_VOW_MATRIX, K_MATRIX, L_MATRIX, LOADED_GODS, TWELVE_GODS
    
    try:
        # 1つのファイルの場合（後方互換性）
        if excel_file is not None:
            gods_list, k_matrix, l_matrix = load_gods_from_excel(excel_file)
        # 3つの別々のファイルの場合
        elif k_matrix_file is not None and l_matrix_file is not None:
            gods_list, k_matrix, l_matrix = load_gods_from_separate_files(
                character_file=character_file,
                k_matrix_file=k_matrix_file,
                l_matrix_file=l_matrix_file
            )
        else:
            raise ValueError("Excelファイルが指定されていません")
        
        # sense_to_vow行列を読み込む（オプション）
        if sense_to_vow_file is not None:
            sense_to_vow_matrix = load_sense_to_vow_matrix(sense_to_vow_file)
            SENSE_TO_VOW_MATRIX = sense_to_vow_matrix
        else:
            # sense_to_vow行列がない場合、デフォルト値を設定（感覚と誓願の基本的な対応）
            # 後でユーザーがアップロードできるように、Noneのままにしておく
            SENSE_TO_VOW_MATRIX = None
        
        # グローバル変数を更新
        K_MATRIX = k_matrix
        L_MATRIX = l_matrix
        LOADED_GODS = gods_list
        TWELVE_GODS = gods_list  # 既存のコードとの互換性のため
        rebuild_globals_from_gods(gods_list)
        
        return True
    except Exception as e:
        st.error(f"設定の読み込みに失敗しました: {str(e)}")
        import traceback
        st.error(f"詳細: {traceback.format_exc()}")
        return False

# -------------------------
# QUBO生成（12神ベース）
# -------------------------
# QUBOパラメータ（12神同士の関係性）
# 負の値 = 相乗効果（一緒に選ばれやすい）
# 正の値 = 抑制（同時に選ばれにくい）

def calculate_god_similarity(god1: Dict, god2: Dict) -> float:
    """2つの神の属性の類似度を計算（-1.0 ～ 1.0）
    誓願属性（vows）と役割属性（roles）の両方を考慮"""
    # 誓願属性の類似度（vow01～vow12）
    vow_keys = [f"vow{i:02d}" for i in range(1, 13)]
    vow_diff_sum = 0.0
    for key in vow_keys:
        diff = abs(god1["vows"][key] - god2["vows"][key])
        vow_diff_sum += diff
    vow_similarity = 1.0 - (vow_diff_sum / len(vow_keys))
    
    # 役割属性の類似度
    role_attrs = ["stillness", "flow", "ma", "sincerity"]
    role_diff_sum = 0.0
    for attr in role_attrs:
        diff = abs(god1["roles"][attr] - god2["roles"][attr])
        role_diff_sum += diff
    role_similarity = 1.0 - (role_diff_sum / len(role_attrs))
    
    # 両方の類似度を重み付けして統合（誓願:0.6、役割:0.4）
    similarity = vow_similarity * 0.6 + role_similarity * 0.4
    return similarity

def build_qubo_hierarchical(x: np.ndarray, lambda_v: float = 5.0, lambda_c: float = 5.0, 
                            lambda_neg: float = 2.0, lambda_conf: float = 2.0,
                            sense_to_vow_matrix: Optional[np.ndarray] = None,
                            k_matrix: Optional[np.ndarray] = None,
                            l_matrix: Optional[np.ndarray] = None,
                            x_continuous: Optional[np.ndarray] = None,
                            selected_attribute: Optional[str] = None,
                            selected_character: Optional[str] = None,
                            char_master: Optional[pd.DataFrame] = None) -> Dict[Tuple[int,int], float]:
    """多層バイナリ構造QUBOを生成（添付資料の設計に基づく）
    
    Args:
        x: 感覚ベクトル（x1～x8、バイナリ）
        lambda_v: 誓願層のone-hot制約の強度
        lambda_c: キャラクター層のone-hot制約の強度
        lambda_neg: 矛盾制約の強度（迷い×行動）
        lambda_conf: 矛盾制約の強度（焦り×待つ）
        sense_to_vow_matrix: sense_to_vow行列（8x12：感覚 × 誓願）。Noneの場合はデフォルト値を使用
        k_matrix: k行列（12x12：キャラクター × 誓願）。Noneの場合はTWELVE_GODSから生成
        l_matrix: l行列（12x4：キャラクター × 世界観軸）。Noneの場合はTWELVE_GODSから生成
        x_continuous: 感覚ベクトルの連続値（0〜5）。Noneの場合はxを使用
    
    Returns:
        QUBO辞書（(i,j) -> エネルギー係数）
    
    設計の流れ:
    1. ユーザー入力 → Mood → 感覚ベクトル x（8次元、連続値0〜5）
    2. x（感覚）→ v（誓願）を引き寄せる（sense_to_vow_matrixを使用）
       - H_sense-vow = Σ_{i,j} W_{ij} x_i v_j
       - W_{ij}: sense_to_vow_matrix[i, j] = 感覚iが誓願jを引き寄せる強さ
    3. v（誓願）→ c（キャラ）を引き寄せる（k_matrixを使用）
    4. QUBOでone-hot制約により、誓願1つ、キャラ1体が選ばれる
    """
    Q: Dict[Tuple[int,int], float] = {}
    
    n_sense = len(x)  # 8（感覚変数）
    n_vows = 12  # 12誓願
    n_chars = 12  # 12神
    
    # 変数のインデックス定義
    # x: 0～7（感覚変数）
    # v: 8～19（誓願変数）
    # c: 20～31（キャラクター変数）
    v_start = n_sense
    c_start = n_sense + n_vows
    
    # k行列とl行列を取得（Excelから読み込んだ場合はそれを使用、そうでなければTWELVE_GODSから生成）
    if k_matrix is None:
        # TWELVE_GODSからk行列を生成
        k_matrix = np.zeros((n_chars, n_vows))
        for k, god in enumerate(TWELVE_GODS):
            for j in range(n_vows):
                vow_key = f"vow{j+1:02d}"
                k_matrix[k, j] = god["vows"][vow_key]
    
    if l_matrix is None:
        # TWELVE_GODSからl行列を生成
        l_matrix = np.zeros((n_chars, 4))
        role_names = ["stillness", "flow", "ma", "sincerity"]
        for k, god in enumerate(TWELVE_GODS):
            for j, role_name in enumerate(role_names):
                l_matrix[k, j] = god["roles"][role_name]
    
    # === H_sense: 感覚エネルギー項 ===
    # H_sense = Σ_i a_i x_i
    # 感覚が強いほど選ばれやすい（負の値でエネルギーを下げる）
    for i in range(n_sense):
        # 連続値の場合、強さに応じてエネルギーを下げる
        # バイナリの場合、立ち上がっている場合のみ
        if x[i] > 0:
            # 感覚の強さに比例してエネルギーを下げる（負の値）
            # ただし、感覚変数自体はバイナリなので、立ち上がっている場合のみ
            Q[(i, i)] = -0.5 * min(x[i], 1.0)  # 最大1.0に制限
    
    # === H_vow: 誓願選択項（one-hot制約） ===
    # H_vow = λ_v (Σ_j v_j - 1)^2 = λ_v (Σ_j v_j^2 - 2Σ_j v_j + 1)
    # = λ_v (Σ_j v_j - 2Σ_j v_j + 1) = λ_v (1 - Σ_j v_j)
    # 展開すると: λ_v * Σ_j v_j^2 - 2λ_v * Σ_j v_j + λ_v
    # 線形項: -2λ_v * v_j
    # 二次項: λ_v * v_j^2 (j=jの場合) + λ_v * 2 * v_i * v_j (i≠jの場合)
    for j in range(n_vows):
        v_idx = v_start + j
        # 線形項
        Q[(v_idx, v_idx)] = -2.0 * lambda_v
        # 二次項（誓願同士の相互作用）
        for k in range(j+1, n_vows):
            v_idx2 = v_start + k
            Q[(v_idx, v_idx2)] = 2.0 * lambda_v
    
    # 定数項（λ_v）は無視（エネルギー差のみが重要）
    
    # === H_char: キャラクター選択項（one-hot制約） ===
    # H_char = λ_c (Σ_k c_k - 1)^2
    for k in range(n_chars):
        c_idx = c_start + k
        # 線形項
        Q[(c_idx, c_idx)] = -2.0 * lambda_c
        # 二次項（キャラクター同士の相互作用）
        for l in range(k+1, n_chars):
            c_idx2 = c_start + l
            Q[(c_idx, c_idx2)] = 2.0 * lambda_c
    
    # === H_interaction: 相互作用項 ===
    # H_interaction = Σ_{i,j} S_{ij} x_i v_j + Σ_{j,k} K_{jk} v_j c_k + Σ_{i,k} L_{ik} x_i c_k
    
    # 感覚 × 誓願: S_{ij} x_i v_j（sense_to_vow_matrixを使用）
    # H_sense-vow = Σ_{i,j} W_{ij} x_i v_j
    # W_{ij}: sense_to_vow_matrix[i, j] = 感覚iが誓願jを引き寄せる強さ
    # 負の値（例：-0.4）= 引き寄せ（相性が良い）→ エネルギーを下げる
    # 正の値（例：+0.2）= 離す（相性が悪い）→ エネルギーを上げる
    if sense_to_vow_matrix is not None:
        # sense_to_vow行列を使用（中核データ）
        for i in range(n_sense):
            if x[i] > 0:  # 感覚が立ち上がっている場合のみ
                for j in range(n_vows):
                    v_idx = v_start + j
                    # sense_to_vow行列の値を直接使用
                    if i < sense_to_vow_matrix.shape[0] and j < sense_to_vow_matrix.shape[1]:
                        W_ij = sense_to_vow_matrix[i, j]  # 感覚iが誓願jを引き寄せる強さ
                        # QUBOの相互作用項: W_{ij} * x_i * v_j
                        # x_iはバイナリ変数だが、連続値の強さを重みとして使用
                        if x_continuous is not None and i < len(x_continuous):
                            x_strength = min(x_continuous[i], 5.0) / 5.0  # 0〜1に正規化
                        else:
                            x_strength = 1.0 if x[i] > 0 else 0.0
                        # W_{ij} * x_i * v_j の係数
                        # 負の値 = 引き寄せ（エネルギーを下げる）、正の値 = 離す（エネルギーを上げる）
                        Q[(i, v_idx)] = W_ij * x_strength
    else:
        # sense_to_vow行列がない場合、デフォルトの対応関係を使用
        # 迷いが強い → 誓願05/07/10が呼ばれやすい、など
        default_mapping = {
            0: [4, 6, 9],  # 迷い → 誓願05, 07, 10
            1: [0, 1, 3],  # 焦り → 誓願01, 02, 04
            2: [1, 10],    # 静けさ → 誓願02, 11
            3: [2, 8],     # 内省 → 誓願03, 09
            4: [3, 5],     # 行動 → 誓願04, 06
            5: [7, 11],    # つながり → 誓願08, 12
            6: [4, 6],     # 挑戦 → 誓願05, 07
            7: [2, 8],     # 待つ → 誓願03, 09
        }
        for i in range(n_sense):
            if x[i] > 0:
                for j in range(n_vows):
                    v_idx = v_start + j
                    # デフォルトマッピングを使用
                    if i in default_mapping and j in default_mapping[i]:
                        Q[(i, v_idx)] = -0.3 * x[i]  # 負の値で引き寄せる
                    else:
                        Q[(i, v_idx)] = 0.1 * x[i]  # 正の値で少し抑制
    
    # 誓願 × キャラクター: K_{jk} v_j c_k = k_matrix[k, j] v_j c_k
    # k行列を使用して誓願とキャラクターの相互作用を定義
    # この誓願なら、この神が「語り手として自然」という関係を数値で持っている
    for j in range(n_vows):
        v_idx = v_start + j
        for k in range(n_chars):
            c_idx = c_start + k
            # k行列の値を使用（キャラクターkの誓願jの値）
            # 負の値 = その誓願が選ばれやすい、正の値 = 選ばれにくい
            if k < k_matrix.shape[0] and j < k_matrix.shape[1]:
                Q[(v_idx, c_idx)] = k_matrix[k, j]
    
    # 感覚 × キャラクター: L_{ik} x_i c_k = l_matrix[k, role_i] x_i c_k
    # l行列を使用して感覚とキャラクターの相互作用を定義
    # 感覚変数と役割属性のマッピング
    role_mapping = {
        0: 0,  # 迷い → stillness (l_matrixの列0)
        1: 1,  # 焦り → flow (l_matrixの列1)
        2: 0,  # 静けさ → stillness (l_matrixの列0)
        3: 2,  # 内省 → ma (l_matrixの列2)
        4: 1,  # 行動 → flow (l_matrixの列1)
        5: 2,  # つながり → ma (l_matrixの列2)
        6: 1,  # 挑戦 → flow (l_matrixの列1)
        7: 3,  # 待つ → sincerity (l_matrixの列3)
    }
    
    for i in range(n_sense):
        if x[i] > 0:
            role_col = role_mapping.get(i, 0)
            for k in range(n_chars):
                c_idx = c_start + k
                # l行列の値を使用（キャラクターkの役割属性role_colの値）
                Q[(i, c_idx)] = l_matrix[k, role_col] * x[i]
    
    # === H_constraint: 制約項 ===
    # H_constraint = λ_neg (x_迷い・x_行動) + λ_conf (x_焦り・x_待つ)
    # 迷い（x0）と行動（x4）の矛盾
    if x[0] > 0 and x[4] > 0:
        Q[(0, 4)] = lambda_neg
    
    # 焦り（x1）と待つ（x7）の矛盾
    if x[1] > 0 and x[7] > 0:
        Q[(1, 7)] = lambda_conf
    
    # === 選択されたキャラクター/属性の調整 ===
    # ユーザーが選択したキャラクターまたは属性を持つキャラクターのエネルギーを下げる
    if selected_character or selected_attribute:
        gods_list = LOADED_GODS if LOADED_GODS else TWELVE_GODS
        for k, god in enumerate(gods_list):
            c_idx = c_start + k
            should_boost = False
            
            # キャラクターの直接選択
            if selected_character:
                god_name = god.get("name", "")
                official_name = god.get("official_name", "")
                if selected_character == god_name or selected_character == official_name:
                    should_boost = True
            
            # 属性の選択
            if selected_attribute and not should_boost:
                god_attribute = god.get("attribute", "")
                if selected_attribute == god_attribute:
                    should_boost = True
            
            # 選択されたキャラクター/属性のエネルギーを下げる（選ばれやすくする）
            if should_boost:
                # キャラクター変数のエネルギーを下げる（負の値で引き寄せる）
                current_energy = Q.get((c_idx, c_idx), 0)
                Q[(c_idx, c_idx)] = current_energy - 3.0  # 大幅にエネルギーを下げる
    
    return Q

def build_qubo_base() -> Dict[Tuple[int,int], float]:
    """従来のQUBOベース（後方互換性のため）"""
    # デフォルトの感覚ベクトル（全て0）でQUBOを生成
    x_default = np.zeros(8)
    return build_qubo_hierarchical(x_default)

# 基本QUBOパラメータを生成
Q_BASE = build_qubo_base()

def clamp(v: float, lo: float=-3.0, hi: float=3.0) -> float:
    return max(lo, min(hi, v))

def select_god_from_mood(m: Mood) -> Dict:
    """Moodに基づいて最も適した12神の1つを選択
    役割属性（roles）を主に考慮（新しい誓願構造に対応）"""
    best_god = None
    best_score = -float('inf')
    
    for god in TWELVE_GODS:
        # Moodと神の役割属性の類似度を計算
        score = 0.0
        
        # 疲れが高い → 静（stillness）が高い神を選ぶ
        if m.fatigue > 0.3:
            score += abs(god["roles"]["stillness"]) * m.fatigue * 0.3
        
        # 不安が高い → 流（flow）が高い神を選ぶ
        if m.anxiety > 0.3:
            score += abs(god["roles"]["flow"]) * m.anxiety * 0.3
        
        # 好奇心が高い → 間（ma）が高い神を選ぶ
        if m.curiosity > 0.3:
            score += abs(god["roles"]["ma"]) * m.curiosity * 0.3
        
        # 決断力が高い → 誠（sincerity）が高い神を選ぶ
        if m.decisiveness > 0.3:
            score += abs(god["roles"]["sincerity"]) * m.decisiveness * 0.3
        
        # 孤独感が高い → 間（ma）と静（stillness）が高い神を選ぶ
        if m.loneliness > 0.3:
            score += (abs(god["roles"]["ma"]) + abs(god["roles"]["stillness"])) * m.loneliness * 0.2
        
        if score > best_score:
            best_score = score
            best_god = god
    
    return best_god if best_god else TWELVE_GODS[0]

def build_qubo_from_mood(m: Mood, 
                         sense_to_vow_matrix: Optional[np.ndarray] = None,
                         k_matrix: Optional[np.ndarray] = None,
                         l_matrix: Optional[np.ndarray] = None,
                         selected_attribute: Optional[str] = None,
                         selected_character: Optional[str] = None,
                         char_master: Optional[pd.DataFrame] = None) -> Dict[Tuple[int,int], float]:
    """Moodに基づいて多層バイナリ構造QUBOを生成
    
    設計の流れ:
    1. ユーザー入力 → Mood → 感覚ベクトル x（8次元、連続値0〜5）
    2. x（感覚）→ v（誓願）を引き寄せる（sense_to_vow_matrixを使用）
       - H_sense-vow = Σ_{i,j} W_{ij} x_i v_j
       - W_{ij}: sense_to_vow_matrix[i, j] = 感覚iが誓願jを引き寄せる強さ
    3. v（誓願）→ c（キャラ）を引き寄せる（k_matrixを使用）
    4. QUBOでone-hot制約により、誓願1つ、キャラ1体が選ばれる
    
    Args:
        m: Moodオブジェクト
        sense_to_vow_matrix: sense_to_vow行列（8x12：感覚 × 誓願）
        k_matrix: k行列（12x12：キャラクター × 誓願）
        l_matrix: l行列（12x4：キャラクター × 世界観軸）
    
    Returns:
        QUBO辞書（(i,j) -> エネルギー係数）
    """
    # Moodから感覚ベクトルを生成（連続値0〜5として扱う）
    # ただし、QUBO変数はバイナリなので、感覚の強さは重みとして使用
    x_continuous = mood_to_sensation_vector(m, binary=False, scale=5.0)
    # QUBO構築時は、感覚が立ち上がっているかどうかをバイナリで判定
    x = (x_continuous > 0.3).astype(float)  # 閾値0.3*5=1.5以上で1
    
    # グローバル変数から行列を取得（指定されていない場合）
    if sense_to_vow_matrix is None:
        sense_to_vow_matrix = SENSE_TO_VOW_MATRIX
    if k_matrix is None:
        k_matrix = K_MATRIX
    if l_matrix is None:
        l_matrix = L_MATRIX
    
    # グローバル変数から選択されたキャラクター/属性を取得
    global SELECTED_ATTRIBUTE, SELECTED_CHARACTER, CHAR_MASTER
    if selected_attribute is None:
        selected_attribute = SELECTED_ATTRIBUTE
    if selected_character is None:
        selected_character = SELECTED_CHARACTER
    if char_master is None:
        char_master = CHAR_MASTER
    
    # 多層バイナリ構造QUBOを生成
    # x_continuousを渡して、感覚の強さを重みとして使用
    Q = build_qubo_hierarchical(x, 
                                 sense_to_vow_matrix=sense_to_vow_matrix,
                                 k_matrix=k_matrix, 
                                 l_matrix=l_matrix,
                                 x_continuous=x_continuous,
                                 selected_attribute=selected_attribute,
                                 selected_character=selected_character,
                                 char_master=char_master)
    
    return Q

# -------------------------
# 解探索
# -------------------------
def solve_all_with_optuna(Q: Dict[Tuple[int,int], float], use_hierarchical: bool = False, 
                          progress_container=None, n_trials: int = 100):
    """Optunaを使ったQUBO最適化（進捗表示付き）
    
    Args:
        Q: QUBO辞書
        use_hierarchical: 多層バイナリ構造を使用する場合True
        progress_container: Streamlitのコンテナ（進捗表示用）
        n_trials: 試行回数
    
    Returns:
        (解のリスト, Optuna Study)
    """
    if not OPTUNA_AVAILABLE:
        # Optunaが使えない場合は通常のsolve_allを使用
        if progress_container is not None:
            with progress_container:
                st.info("ℹ️ Optunaが利用できません。通常の最適化を使用します。")
        return solve_all(Q, use_hierarchical), None
    
    if use_hierarchical:
        n = 32
        v_start = 8
        c_start = 20
    else:
        n = len(VARIABLES)
        v_start = None
        c_start = None
    
    # 毎回異なる結果を得るため、タイムスタンプベースのシードを使用
    import time
    random_seed = int(time.time() * 1000) % 1000000
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Optuna Studyを作成（in-memory database、ランダムシードを設定）
    study = optuna.create_study(
        direction='minimize', 
        study_name='qubo_optimization',
        sampler=optuna.samplers.TPESampler(seed=random_seed) if OPTUNA_AVAILABLE else None
    )
    
    def objective(trial):
        # バイナリ変数を生成
        if use_hierarchical:
            # one-hot制約を満たすように生成
            # 誓願変数: 12個のうち1つだけ1
            vow_idx = trial.suggest_int('vow_idx', 0, 11)
            # キャラクター変数: 12個のうち1つだけ1
            char_idx = trial.suggest_int('char_idx', 0, 11)
            
            x = np.zeros(n, dtype=int)
            x[v_start + vow_idx] = 1
            x[c_start + char_idx] = 1
            
            # 感覚変数はランダムに設定
            for i in range(8):
                x[i] = trial.suggest_int(f'sense_{i}', 0, 1)
        else:
            x = np.zeros(n, dtype=int)
            for i in range(n):
                x[i] = trial.suggest_int(f'x_{i}', 0, 1)
        
        # エネルギーを計算
        energy = qubo_energy(x, Q)
        
        # 進捗表示
        if progress_container is not None:
            with progress_container:
                st.write(f"試行 {trial.number + 1}/{n_trials}: エネルギー = {energy:.3f}")
        
        return energy
    
    # 最適化実行
    if progress_container is not None:
        with progress_container:
            st.info("🔮 QUBO最適化を実行中...")
            progress_bar = st.progress(0)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    # 最適解を取得
    best_x = np.zeros(n, dtype=int)
    if use_hierarchical:
        best_vow = study.best_params['vow_idx']
        best_char = study.best_params['char_idx']
        best_x[v_start + best_vow] = 1
        best_x[c_start + best_char] = 1
        for i in range(8):
            best_x[i] = study.best_params[f'sense_{i}']
    else:
        for i in range(n):
            best_x[i] = study.best_params[f'x_{i}']
    
    # 解のリストを作成（最適解とその周辺）
    sols = [(study.best_value, best_x)]
    
    # 追加の解を生成（トライアルから）
    for trial in study.trials[:min(100, len(study.trials))]:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            x = np.zeros(n, dtype=int)
            if use_hierarchical:
                x[v_start + trial.params['vow_idx']] = 1
                x[c_start + trial.params['char_idx']] = 1
                for i in range(8):
                    x[i] = trial.params[f'sense_{i}']
            else:
                for i in range(n):
                    x[i] = trial.params[f'x_{i}']
            sols.append((trial.value, x))
    
    sols.sort(key=lambda t: t[0])
    
    # 同エネルギーの解をランダムにシャッフルして多様性を確保
    grouped_sols = []
    current_energy = None
    current_group = []
    for e, x in sols:
        if current_energy is None or abs(e - current_energy) < 0.001:
            current_group.append((e, x))
            current_energy = e
        else:
            random.shuffle(current_group)
            grouped_sols.extend(current_group)
            current_group = [(e, x)]
            current_energy = e
    if current_group:
        random.shuffle(current_group)
        grouped_sols.extend(current_group)
    
    if progress_container is not None:
        with progress_container:
            st.success(f"✅ 最適化完了！最適エネルギー: {study.best_value:.3f}")
    
    return grouped_sols, study

def solve_all(Q: Dict[Tuple[int,int], float], use_hierarchical: bool = False) -> List[Tuple[float, np.ndarray]]:
    """QUBOの全解を探索（毎回異なる結果を得るため、ランダム要素を追加）
    
    Args:
        Q: QUBO辞書
        use_hierarchical: 多層バイナリ構造を使用する場合True
    """
    # 毎回異なる結果を得るため、タイムスタンプベースのシードを使用
    import time
    random_seed = int(time.time() * 1000) % 1000000
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    if use_hierarchical:
        # 多層バイナリ構造の場合
        # 変数の総数: 8（感覚）+ 12（誓願）+ 12（キャラクター）= 32
        n = 32
        v_start = 8  # 誓願変数の開始インデックス
        c_start = 20  # キャラクター変数の開始インデックス
    else:
        # 従来の構造の場合
        n = len(VARIABLES)
        v_start = None
        c_start = None
    
    sols = []
    # 全探索は計算量が大きいため、ランダムサンプリングまたはヒューリスティックを使用
    # ここでは簡易的に全探索を実装（実際の運用では最適化が必要）
    max_samples = 2**min(n, 16)  # 2^16 = 65536まで
    if n <= 16:
        # 全探索の場合でも、結果をランダムにシャッフルして多様性を確保
        all_sols = []
        for bits in itertools.product([0,1], repeat=n):
            x = np.array(bits, dtype=int)
            # one-hot制約をチェック（階層構造の場合）
            if use_hierarchical:
                # 誓願変数（8〜19）のone-hot制約
                vow_sum = np.sum(x[v_start:v_start+12])
                # キャラクター変数（20〜31）のone-hot制約
                char_sum = np.sum(x[c_start:c_start+12])
                # one-hot制約を満たす解のみを追加（厳密に1つだけ選ばれている）
                if vow_sum == 1 and char_sum == 1:
                    e = qubo_energy(x, Q)
                    all_sols.append((e, x))
            else:
                e = qubo_energy(x, Q)
                all_sols.append((e, x))
        
        # エネルギーでソート後、同エネルギーの解をランダムにシャッフル
        all_sols.sort(key=lambda t: t[0])
        # 同エネルギーのグループごとにランダムにシャッフル
        grouped_sols = []
        current_energy = None
        current_group = []
        for e, x in all_sols:
            if current_energy is None or abs(e - current_energy) < 0.001:
                current_group.append((e, x))
                current_energy = e
            else:
                random.shuffle(current_group)
                grouped_sols.extend(current_group)
                current_group = [(e, x)]
                current_energy = e
        if current_group:
            random.shuffle(current_group)
            grouped_sols.extend(current_group)
        sols = grouped_sols
    else:
        # 大きい場合はランダムサンプリング（one-hot制約を満たす解のみ）
        valid_samples = 0
        max_attempts = 50000  # 最大試行回数
        attempts = 0
        
        while valid_samples < min(10000, max_samples) and attempts < max_attempts:
            attempts += 1
            x = np.random.randint(0, 2, size=n, dtype=int)
            
            # one-hot制約をチェック（階層構造の場合）
            if use_hierarchical:
                # 誓願変数（8〜19）のone-hot制約
                vow_sum = np.sum(x[v_start:v_start+12])
                # キャラクター変数（20〜31）のone-hot制約
                char_sum = np.sum(x[c_start:c_start+12])
                # one-hot制約を満たす解のみを追加（厳密に1つだけ選ばれている）
                if vow_sum == 1 and char_sum == 1:
                    e = qubo_energy(x, Q)
                    sols.append((e, x))
                    valid_samples += 1
            else:
                e = qubo_energy(x, Q)
                sols.append((e, x))
                valid_samples += 1
        
        # one-hot制約を満たす解が少ない場合、制約を緩和
        if len(sols) < 10 and use_hierarchical:
            # 制約を緩和して追加の解を生成
            for _ in range(min(1000, max_samples - len(sols))):
                x = np.random.randint(0, 2, size=n, dtype=int)
                # 誓願とキャラクターのどちらかが選ばれていればOK（緩和）
                vow_sum = np.sum(x[v_start:v_start+12])
                char_sum = np.sum(x[c_start:c_start+12])
                if vow_sum >= 1 and char_sum >= 1:
                    e = qubo_energy(x, Q)
                    sols.append((e, x))
    
    # エネルギーでソート後、同エネルギーの解をランダムにシャッフル
    sols.sort(key=lambda t: t[0])
    # 同エネルギーのグループごとにランダムにシャッフル
    grouped_sols = []
    current_energy = None
    current_group = []
    for e, x in sols:
        if current_energy is None or abs(e - current_energy) < 0.001:
            current_group.append((e, x))
            current_energy = e
        else:
            random.shuffle(current_group)
            grouped_sols.extend(current_group)
            current_group = [(e, x)]
            current_energy = e
    if current_group:
        random.shuffle(current_group)
        grouped_sols.extend(current_group)
    
    return grouped_sols

# -------------------------
# ボルツマンサンプリング
# -------------------------
def boltzmann_sample(cands: List[Tuple[float, np.ndarray]], T: float) -> Tuple[float, np.ndarray]:
    """ボルツマンサンプリングで候補から1つを選択
    
    Args:
        cands: 候補リスト（(エネルギー, 解ベクトル)のタプルのリスト）
        T: 温度パラメータ
    
    Returns:
        選ばれた候補（(エネルギー, 解ベクトル)のタプル）
    """
    if not cands:
        raise ValueError("候補が空です")
    
    if len(cands) == 1:
        return cands[0]
    
    # エネルギー値を取得
    es = np.array([e for e,_ in cands], dtype=float)
    
    # NaNやInfをチェック
    if np.any(np.isnan(es)) or np.any(np.isinf(es)):
        # NaNやInfがある場合、最初の候補を返す
        return cands[0]
    
    # 温度の最小値を確保
    T = max(T, 1e-6)
    
    # エネルギーを正規化（最小値を0に）
    es_min = es.min()
    es0 = es - es_min
    
    # 重みを計算（ボルツマン分布）
    # エネルギーが大きすぎる場合を防ぐため、最大値を制限
    es0_clamped = np.clip(es0, 0, 100)  # 最大100に制限
    weights = np.exp(-es0_clamped / T)
    
    # NaNやInfをチェック
    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        # 均等な重みを使用
        weights = np.ones(len(cands)) / len(cands)
    else:
        # 正規化
        weights_sum = weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum) or np.isinf(weights_sum):
            # 合計が0またはNaN/Infの場合、均等な重みを使用
            weights = np.ones(len(cands)) / len(cands)
        else:
            weights = weights / weights_sum
    
    # 最終的なNaNチェック
    if np.any(np.isnan(weights)):
        weights = np.ones(len(cands)) / len(cands)
    
    # サンプリング
    try:
        idx = np.random.choice(len(cands), p=weights)
    except ValueError as e:
        # 重みの合計が1でない場合など、均等サンプリングにフォールバック
        idx = np.random.randint(0, len(cands))
    
    return cands[idx]

def temperature_from_mood(m: Mood, selected_character: Optional[str] = None) -> float:
    """Moodに基づいてボルツマンサンプリングの温度を調整（改善版）
    
    Args:
        m: Moodオブジェクト
        selected_character: 選択されたキャラクター（オプション）
    
    Returns:
        温度パラメータ
    """
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
    
    # キャラクターが選択されている場合、最低温度を確保して多様性を維持
    if selected_character:
        T = max(0.35, T)  # 最低温度0.35を確保
    
    # 温度の範囲を制限（揺らぎすぎない、収束しすぎない）
    return max(0.2, min(0.9, T))

# -------------------------
# おみくじ生成
# -------------------------
def picks_from_x(x: np.ndarray, use_hierarchical: bool = False, selected_god: Dict = None) -> List[str]:
    """選ばれた格言を返す
    
    Args:
        x: 解ベクトル
        use_hierarchical: 多層バイナリ構造を使用する場合True
        selected_god: 既に選ばれた神の情報（オプション、階層構造の場合に使用）
    """
    if use_hierarchical:
        # 階層構造の場合、キャラクター変数から選ばれた神を取得
        # 解ベクトルのサイズを確認（32である必要がある）
        if len(x) < 32:
            # サイズが足りない場合、selected_godから取得を試みる
            if selected_god and selected_god.get("maxim"):
                return [selected_god["maxim"]]
            return ["今この瞬間を大切に。すべては縁で繋がっている。"]
        
        c_start = 20
        selected_god_ids = [i - c_start for i in range(c_start, min(c_start + 12, len(x))) if i < len(x) and x[i] == 1]
        
        # キャラクター変数から選ばれた神を取得
        if selected_god_ids and 0 <= selected_god_ids[0] < len(TWELVE_GODS):
            god = TWELVE_GODS[selected_god_ids[0]]
            if god.get("maxim"):
                return [god["maxim"]]
            elif god.get("description"):
                return [god["description"]]
            else:
                return ["今この瞬間を大切に。すべては縁で繋がっている。"]
        elif selected_god:
            # キャラクター変数から選ばれていない場合、selected_godから取得
            if selected_god.get("maxim"):
                return [selected_god["maxim"]]
            elif selected_god.get("description"):
                return [selected_god["description"]]
        
        # デフォルトの格言を返す
        return ["今この瞬間を大切に。すべては縁で繋がっている。"]
    else:
        # 従来の構造の場合
        max_idx = min(len(x), len(VARIABLES))
        p = [VARIABLES[i] for i in range(max_idx) if x[i] == 1]
        return p if p else ["今この瞬間を大切に。すべては縁で繋がっている。"]

def get_maxim_source(maxim: str) -> Dict:
    """格言の引用元情報を取得"""
    if maxim in MAXIM_SOURCES:
        return MAXIM_SOURCES[maxim]
    # 有名引用（FAMOUS_QUOTES）も参照
    try:
        for q in FAMOUS_QUOTES:
            if q.get("quote") == maxim:
                return {
                    "source": q.get("source", "引用"),
                    "origin": q.get("origin", ""),
                    "reference": q.get("reference", ""),
                }
    except Exception:
        pass
    return {
        "source": "伝統的な教え",
        "origin": "古来より伝わる智慧",
        "reference": "長い年月をかけて受け継がれてきた知恵"
    }

def select_maxims_for_god(
    god: Dict,
    context_text: str,
    top_k: int = 2,
    include_famous_quote: bool = True,
    exclude_maxims: Optional[List[str]] = None,
    selected_vow_index: Optional[int] = None
) -> List[str]:
    """ユーザー入力（context_text）とQUBOで選ばれた誓願（VOW）に応じて、神（キャラクター）の複数格言を選ぶ
    
    Args:
        god: 神の情報
        context_text: ユーザーの入力テキスト
        top_k: 選択する格言の数
        include_famous_quote: 有名名言を含めるか
        exclude_maxims: 除外する格言のリスト（重複を避けるため）
        selected_vow_index: QUBOで選ばれた誓願（VOW）のインデックス（0-11、VOW_01～VOW_12に対応）
    """
    if not god:
        return ["今この瞬間を大切に。すべては縁で繋がっている。"]

    exclude_set = set(exclude_maxims or [])
    ctx = (context_text or "").strip()
    keywords = extract_keywords_safe(ctx, top_n=8) if ctx else []  # より多くのキーワードを抽出
    
    # キーワードが抽出されない場合、コンテキストテキストから直接単語を抽出
    if not keywords and ctx:
        # テキストを単語に分割して、2文字以上の単語を抽出
        import re
        text_clean = re.sub(r'[0-9０-９\W]+', ' ', ctx)
        words = text_clean.split()
        keywords = [w for w in words if len(w) >= 2][:8]

    # 候補（maxims があればそれを、なければ maxim/description）
    maxims = god.get("maxims") or []
    items: List[Dict[str, object]] = []
    for it in maxims:
        if isinstance(it, dict) and it.get("text"):
            text = str(it["text"]).strip()
            # 除外リストに含まれていない場合のみ追加
            if text and text not in exclude_set:
                items.append({"text": text, "tags": it.get("tags") or []})

    if not items:
        base = (god.get("maxim") or "").strip()
        if base and base not in exclude_set:
            items = [{"text": base, "tags": []}]
        else:
            desc = (god.get("description") or "").strip()
            default_text = desc or "今この瞬間を大切に。すべては縁で繋がっている。"
            if default_text not in exclude_set:
                items = [{"text": default_text, "tags": []}]
    
    # キーワードがある場合、MAXIMS_DATABASEからキーワードに基づいて格言を追加
    if keywords and MAXIMS_DATABASE:
        # キーワードに基づいてMAXIMS_DATABASEから格言を選択
        db_maxims = select_maxims_from_database(keywords, top_k=5, exclude_maxims=list(exclude_set))
        for maxim in db_maxims:
            maxim_text = maxim.get("text", "")
            if maxim_text and maxim_text not in [it.get("text", "") for it in items]:
                items.append({"text": maxim_text, "tags": maxim.get("tags", [])})
    
    # キーワードがない場合、格言データベースからも追加で候補を取得（多様性を確保）
    if not keywords and MAXIMS_DATABASE:
        # ランダムにいくつかの格言を追加候補として取得
        import time
        random.seed(int(time.time() * 1000) % 1000000)
        available_maxims = [m for m in MAXIMS_DATABASE if m.get("text") and m.get("text") not in exclude_set]
        random.shuffle(available_maxims)
        # 最大3つまで追加
        for maxim in available_maxims[:3]:
            maxim_text = maxim.get("text", "")
            if maxim_text and maxim_text not in [it.get("text", "") for it in items]:
                items.append({"text": maxim_text, "tags": maxim.get("tags", [])})

    def score_item(item: Dict[str, object], item_index: int) -> float:
        text = str(item.get("text", "") or "")
        tags = [str(t) for t in (item.get("tags") or [])]
        text_lower = text.lower()
        tags_lower = [str(t).lower() for t in tags]
        s = 0.0
        
        # キーワードベースのスコアリング（最優先：ユーザー入力の分析結果）
        if keywords:
            matched_keywords = 0
            # タグ一致を最優先（ユーザー入力の分析結果を反映）
            for kw in keywords:
                kw_lower = kw.lower()
                # タグ完全一致
                if kw_lower in tags_lower:
                    s += 10.0  # タグ一致は最高スコア（ユーザー入力の分析結果を最優先）
                    matched_keywords += 1
                # タグ部分一致
                elif any(kw_lower in tag_lower for tag_lower in tags_lower):
                    s += 8.0  # タグ部分一致も高スコア
                    matched_keywords += 1
                # テキスト完全一致（最優先）
                elif kw_lower in text_lower:
                    s += 15.0  # テキスト内のキーワード完全一致は最高スコア
                    matched_keywords += 1
                # テキスト部分一致（日本語対応：2文字以上の部分一致）
                elif len(kw) >= 2 and kw[:2] in text_lower:
                    s += 8.0  # 部分一致も高スコア
                    matched_keywords += 1
                # さらに柔軟なマッチング：キーワードの文字が含まれているか
                elif any(c in text_lower for c in kw_lower if len(c) >= 1):
                    s += 3.0  # 文字レベルでの一致も考慮（スコアを上げる）
                    matched_keywords += 1
            
            # キーワードが複数一致する場合、大幅なボーナス（重要）
            if matched_keywords >= 2:
                s += 10.0 * matched_keywords  # 複数キーワード一致は大幅なボーナス
            elif matched_keywords == 1:
                s += 5.0  # 単一キーワード一致でもボーナス
        
        # QUBOで選ばれた誓願（VOW）に基づくスコアリング（キーワードがない場合の補助）
        if selected_vow_index is not None:
            # 選ばれた誓願に対応するVOW値が高い場合、その神の格言を優先
            vows = god.get("vows", {})
            vow_key = f"vow{selected_vow_index+1:02d}"
            if vow_key in vows:
                vow_value = float(vows[vow_key])
                # VOW値が負（強い関連性）の場合、スコアを上げる（キーワードがない場合の補助）
                if vow_value < 0:
                    s += abs(vow_value) * 3.0  # VOW値に基づく優先度（キーワードがない場合の補助）
                elif vow_value > 0:
                    s += vow_value * 1.0  # 正の値でも少し優先
        
        # キーワードもVOWスコアもない場合
        if s == 0.0:
            # ランダム要素を追加して多様性を確保
            import time
            random.seed(int(time.time() * 1000) % 1000000 + item_index)
            s = random.uniform(0.01, 0.5)  # ランダムなスコアを設定
        
        # 文章が短すぎる場合は少し減点
        if len(text) < 6:
            s -= 0.5
        return s

    scored = [(score_item(it, idx), it["text"]) for idx, it in enumerate(items) if it.get("text")]
    
    # スコアが同点ならランダムに揺らぐ（各呼び出しで異なる結果を得るため）
    import time
    # 関数の呼び出しごとに異なるシードを使用（godのIDと時間を組み合わせ）
    god_id = god.get("id", 0) if isinstance(god.get("id"), int) else hash(str(god.get("name", ""))) % 1000
    random.seed(int(time.time() * 1000) % 1000000 + god_id * 100 + len(items))
    random.shuffle(scored)
    scored.sort(key=lambda t: t[0], reverse=True)

    picks: List[str] = []
    for s, t in scored:
        if t and t not in picks and t not in exclude_set:
            picks.append(t)
        if len(picks) >= max(1, top_k):
            break

    # 全部スコアが低い（=キーワードに引っかからない）なら、ランダムに複数提示
    # ただし、キーワードがある場合は、スコアが低くてもキーワードに基づいて選択
    if scored:
        if keywords and scored[0][0] < 5.0:  # スコアが低い場合（5.0未満）
            # キーワードがあるがスコアが低い場合、キーワードに基づいて再スコアリング
            # 部分一致や類似語も考慮して、より柔軟にマッチング
            rescored = []
            for s, t in scored:
                text_lower = t.lower()
                new_score = s
                matched_kw_count = 0
                # キーワードの部分一致をチェック（日本語対応）
                for kw in keywords:
                    kw_lower = kw.lower()
                    # 完全一致
                    if kw_lower in text_lower:
                        new_score += 10.0  # 完全一致は大幅なスコアアップ
                        matched_kw_count += 1
                    # 部分一致（2文字以上）
                    elif len(kw) >= 2 and kw[:2] in text_lower:
                        new_score += 5.0  # 部分一致も高スコア
                        matched_kw_count += 1
                    # 文字レベルでの一致
                    elif any(c in text_lower for c in kw_lower if len(c) >= 1):
                        new_score += 2.0  # 文字レベルでの一致も考慮
                        matched_kw_count += 1
                
                # 複数キーワード一致のボーナス
                if matched_kw_count >= 2:
                    new_score += matched_kw_count * 5.0
                
                rescored.append((new_score, t))
            rescored.sort(key=lambda t: t[0], reverse=True)
            scored = rescored
        
        # キーワードがある場合、スコアが低くてもキーワードに基づいて選択
        if keywords and scored and scored[0][0] < 3.0:
            # キーワードに基づいて再選択（MAXIMS_DATABASEからも追加で取得）
            if MAXIMS_DATABASE:
                db_maxims = select_maxims_from_database(keywords, top_k=top_k * 2, exclude_maxims=list(exclude_set))
                for maxim in db_maxims:
                    maxim_text = maxim.get("text", "")
                    if maxim_text and maxim_text not in picks and maxim_text not in exclude_set:
                        picks.append(maxim_text)
                        if len(picks) >= top_k:
                            break
        
        # キーワードがない場合、またはスコアが非常に低い場合のみランダム選択
        if not keywords and scored and scored[0][0] < 1.0:
            all_texts = [t for _, t in scored if t and t not in exclude_set]
            # 再度ランダムシードを設定して多様性を確保
            random.seed(int(time.time() * 1000) % 1000000 + god_id * 200 + len(all_texts))
            random.shuffle(all_texts)
            picks = list(dict.fromkeys(all_texts))[:max(1, top_k)]

    # 有名名言も1つ混ぜる（任意）
    if include_famous_quote and keywords:
        famous = select_relevant_quote(keywords, exclude_quotes=exclude_set)
        if famous and famous not in picks and famous not in exclude_set:
            picks.append(famous)

    return picks if picks else ["今この瞬間を大切に。すべては縁で繋がっている。"]

def get_selected_vow_from_x(x: np.ndarray, use_hierarchical: bool = False) -> Optional[int]:
    """QUBOの解ベクトルから選ばれた誓願（VOW）のインデックスを取得
    
    Args:
        x: 解ベクトル
        use_hierarchical: 多層バイナリ構造を使用する場合True
    
    Returns:
        選ばれた誓願のインデックス（0-11、VOW_01～VOW_12に対応）、選ばれていない場合はNone
    """
    if use_hierarchical:
        # 多層バイナリ構造の場合
        if len(x) < 32:
            return None
        
        # 誓願変数のインデックス: 8～19
        v_start = 8
        selected_vow_ids = [i - v_start for i in range(v_start, min(v_start + 12, len(x))) if i < len(x) and x[i] == 1]
        
        if selected_vow_ids:
            return selected_vow_ids[0]  # 最初に選ばれた誓願を返す
    else:
        # 従来の構造では誓願変数がない
        return None
    
    return None

def get_selected_god_from_x(x: np.ndarray, mood: Mood = None, use_hierarchical: bool = False) -> Dict:
    """選ばれた神を取得
    
    Args:
        x: 解ベクトル
        mood: Moodオブジェクト（オプション）
        use_hierarchical: 多層バイナリ構造を使用する場合True
    """
    if use_hierarchical:
        # 多層バイナリ構造の場合
        # 解ベクトルのサイズを確認（32である必要がある）
        if len(x) < 32:
            # サイズが足りない場合、Moodから選択するかデフォルトを返す
            if mood is not None:
                return select_god_from_mood(mood)
            else:
                return TWELVE_GODS[0]  # デフォルト
        
        # キャラクター変数のインデックス: 20～31
        c_start = 20
        selected_god_ids = [i - c_start for i in range(c_start, min(c_start + 12, len(x))) if i < len(x) and x[i] == 1]
    else:
        # 従来の構造の場合
        max_idx = min(len(x), len(TWELVE_GODS))
        selected_god_ids = [i for i in range(max_idx) if x[i] == 1]
    
    if not selected_god_ids:
        # 何も選ばれていない場合、Moodから最も適した神を選択
        if mood is not None:
            return select_god_from_mood(mood)
        else:
            return TWELVE_GODS[0]  # デフォルト
    
    # 選ばれた神の中で、Moodに最も近い神を選択
    if mood is not None:
        best_god = None
        best_score = -float('inf')
        for god_id in selected_god_ids:
            if 0 <= god_id < len(TWELVE_GODS):
                god = TWELVE_GODS[god_id]
                # 新しい誓願構造（vow01～vow12）では、Moodとの直接比較が難しいため
                # 役割属性（roles）を使用して類似度を計算
                score = 0.0
                # 役割属性の類似度
                if mood.fatigue > 0.5:
                    score += abs(god["roles"]["stillness"]) * 0.25
                if mood.anxiety > 0.5:
                    score += abs(god["roles"]["flow"]) * 0.25
                if mood.curiosity > 0.5:
                    score += abs(god["roles"]["ma"]) * 0.25
                if mood.decisiveness > 0.5:
                    score += abs(god["roles"]["sincerity"]) * 0.25
                
                if score > best_score:
                    best_score = score
                    best_god = god
        return best_god if best_god else TWELVE_GODS[selected_god_ids[0]]
    else:
        # Moodがない場合、最初に選ばれた神を返す
        return TWELVE_GODS[selected_god_ids[0]] if selected_god_ids[0] < len(TWELVE_GODS) else TWELVE_GODS[0]

def oracle_card(
    e: float,
    x: np.ndarray,
    mood: Mood = None,
    use_hierarchical: bool = False,
    context_text: str = "",
    use_llm: bool = False,
    llm_type: str = "huggingface"
) -> Dict:
    """格言ベースのおみくじカードを生成（Moodに応じて変化、12神対応）
    
    Args:
        e: エネルギー値
        x: 解ベクトル
        mood: Moodオブジェクト
        use_hierarchical: 階層構造を使用するか
        context_text: ユーザーの入力テキスト
        use_llm: LLMを使用するか
        llm_type: LLMの種類（"ollama" or "huggingface"）
    """
    # 選ばれた神を取得（先に取得して、格言も取得）
    selected_god = get_selected_god_from_x(x, mood, use_hierarchical=use_hierarchical)
    
    # QUBOの解ベクトルから選ばれた誓願（VOW）を取得
    selected_vow_index = None
    if use_hierarchical:
        selected_vow_index = get_selected_vow_from_x(x, use_hierarchical=use_hierarchical)
    
    # 最近使用した格言を取得（重複を避けるため）
    if 'recent_maxims' not in st.session_state:
        st.session_state.recent_maxims = []
    exclude_maxims = st.session_state.recent_maxims[-10:]  # 直近10件を除外
    
    # 格言を取得（階層構造の場合はユーザー文面で複数選ぶ）
    # QUBOで選ばれた誓願（VOW）の情報を使用
    # 【神託寄りにシフト】優先順位を変更：VOWベースの格言生成 > 神の格言 > キーワードベースの格言
    if use_hierarchical:
        picks = []
        
        # 【最優先】QUBOで選ばれた誓願（VOW）のベクトルに基づいてオリジナルな格言を生成（神託らしい）
        if selected_vow_index is not None:
            original_maxim = create_original_maxim_from_vow(
                selected_vow_index=selected_vow_index,
                god=selected_god,
                top_k=3
            )
            if original_maxim and original_maxim not in picks and original_maxim not in exclude_maxims:
                # オリジナルな格言を最初に追加（最優先表示：神託らしい）
                picks.insert(0, original_maxim)
        
        # 【第2優先】選ばれた神の格言を優先的に選択（神託の核心）
        # 神の格言は常に含める（入力がない場合も神託として機能）
        top_k_for_selection = 4 if context_text else 3
        god_picks = select_maxims_for_god(
            selected_god, 
            context_text=context_text,  # ユーザー入力を使用（あれば）
            top_k=top_k_for_selection, 
            include_famous_quote=False,  # 有名名言は除外（神託らしさを優先）
            exclude_maxims=exclude_maxims,
            selected_vow_index=selected_vow_index
        )
        for pick in god_picks:
            if pick and pick not in picks and pick not in exclude_maxims:
                picks.append(pick)
                if len(picks) >= 6:  # 神の格言を優先的に多く選択
                    break
        
        # 【第3優先】ユーザー入力がある場合、キーワードに基づいて格言を選択（補助的）
        if context_text:
            # キーワード抽出（より多くのキーワードを抽出）
            keywords = extract_keywords_safe(context_text, top_n=12)
            
            # キーワードに基づいて格言を生成（ユーザー入力に直接応える）
            if keywords and len(picks) < 4:  # 神の格言が少ない場合のみ
                generated_maxim = generate_maxim_from_keywords(keywords, context_text)
                if generated_maxim and generated_maxim not in picks and generated_maxim not in exclude_maxims:
                    picks.append(generated_maxim)  # 生成された格言を追加（優先度は低め）
            
            # MAXIMS_DATABASEからキーワードに基づいて格言を選択（補助的）
            if MAXIMS_DATABASE and keywords and len(picks) < 5:
                # キーワードに基づいて格言データベースから選択
                db_maxims = select_maxims_from_database(keywords, top_k=5, exclude_maxims=exclude_maxims)
                for db_maxim in db_maxims:
                    maxim_text = db_maxim.get("text", "")
                    if maxim_text and maxim_text not in picks and maxim_text not in exclude_maxims:
                        picks.append(maxim_text)
                        if len(picks) >= 6:  # 最大6つまで
                            break
        
        # 最大4つまでに制限（神託らしさを保つため、多すぎないように）
        if len(picks) > 4:
            picks = picks[:4]
    
    # 選択した格言を履歴に追加（重複を避ける）
    for pick in picks:
        if pick and pick not in st.session_state.recent_maxims:
            st.session_state.recent_maxims.append(pick)
            # 履歴は最大20件に制限
            if len(st.session_state.recent_maxims) > 20:
                st.session_state.recent_maxims.pop(0)
    else:
        picks = picks_from_x(x, use_hierarchical=use_hierarchical, selected_god=selected_god)
    
    # 格言が空またはデフォルトの場合、選ばれた神の格言を使用
    # ただし、context_textがある場合は、キーワードに基づいて再試行
    if not picks or (len(picks) == 1 and picks[0] == "今この瞬間を大切に。すべては縁で繋がっている。"):
        # context_textがある場合、キーワードに基づいて再試行
        if context_text and MAXIMS_DATABASE:
            keywords = extract_keywords_safe(context_text, top_n=8)
            if keywords:
                db_maxims = select_maxims_from_database(keywords, top_k=3, exclude_maxims=exclude_maxims)
                if db_maxims:
                    picks = [m.get("text", "") for m in db_maxims if m.get("text")]
        
        # それでも格言がない場合、選ばれた神の格言を使用
        if not picks or (len(picks) == 1 and picks[0] == "今この瞬間を大切に。すべては縁で繋がっている。"):
            if selected_god and selected_god.get("maxim"):
                picks = [selected_god["maxim"]]
            elif selected_god and selected_god.get("description"):
                picks = [selected_god["description"]]
            else:
                picks = ["今この瞬間を大切に。すべては縁で繋がっている。"]
    
    season = random.choice(SEASONS)
    
    # LLMを使用してパーソナライズされた神託を生成（オプション）
    llm_oracle = None
    if use_llm and context_text and mood:
        try:
            llm_oracle = generate_oracle_with_llm(
                user_text=context_text,
                selected_god=selected_god,
                selected_maxims=picks,
                mood=mood,
                llm_type=llm_type
            )
        except Exception as e:
            # LLM生成に失敗した場合、通常の格言を使用
            pass
    
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
    
    # 選ばれた格言を俳句風に表現（神託らしく、季節と格言を組み合わせる）
    if len(picks) > 0:
        # 神託らしい表現：季節と格言を組み合わせる
        # 格言が長すぎる場合は短縮して、神託らしくする
        maxim_text = picks[0]
        if len(maxim_text) > 30:
            # 長い格言は最初の部分を取るか、要約
            maxim_text = maxim_text[:30] + "..."
        poem = f"{season}／{maxim_text}"
    else:
        # デフォルトの神託
        poem = f"{season}／今この瞬間を大切に"
    
    return {
        "energy": e,
        "picks": picks,
        "poem": poem,
        "hint": hint,
        "god": selected_god,  # 選ばれた神の情報を追加
        "llm_oracle": llm_oracle  # LLM生成の神託（オプション）
    }

# -------------------------
# LLM統合（無償で使用可能）
# -------------------------
def generate_oracle_with_llm(
    user_text: str,
    selected_god: Dict,
    selected_maxims: List[str],
    mood: Mood,
    llm_type: str = "huggingface"  # "ollama" or "huggingface"
) -> str:
    """LLMを使用してパーソナライズされた神託を生成
    
    Args:
        user_text: ユーザーの入力テキスト
        selected_god: 選ばれたキャラクター
        selected_maxims: 選ばれた格言リスト
        mood: ユーザーの感情状態
        llm_type: LLMの種類（"ollama" or "huggingface"）
    
    Returns:
        生成された神託テキスト
    """
    # プロンプトを構築
    god_name = selected_god.get("name", "神")
    god_description = selected_god.get("description", "")
    maxims_text = "\n".join([f"- {m}" for m in selected_maxims[:3]])
    
    prompt = f"""あなたは{god_name}です。{god_description}

ユーザーの悩みや気持ち：
「{user_text}」

ユーザーの感情状態：
- 疲れ: {mood.fatigue:.2f}
- 不安/焦り: {mood.anxiety:.2f}
- 好奇心: {mood.curiosity:.2f}
- 孤独: {mood.loneliness:.2f}
- 決断力: {mood.decisiveness:.2f}

関連する格言：
{maxims_text}

上記の情報を基に、ユーザーに寄り添う温かみのある神託（50-100文字程度）を生成してください。
日本の伝統的な「おみくじ」のスタイルで、希望と励ましを含む内容にしてください。
"""
    
    if llm_type == "ollama":
        return generate_with_ollama(prompt)
    elif llm_type == "huggingface":
        return generate_with_huggingface(prompt)
    else:
        # LLMが使用できない場合、格言ベースの神託を返す
        return f"{maxims_text}\n\nあなたの観測が、この世界線を確定させました。"

def generate_with_ollama(prompt: str, model: str = "llama3.2") -> str:
    """Ollamaを使用してLLMでテキストを生成（ローカル実行）
    
    Args:
        prompt: プロンプト
        model: 使用するモデル名（デフォルト: llama3.2）
    
    Returns:
        生成されたテキスト
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 200
                }
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            return ""
    except Exception as e:
        # Ollamaが起動していない場合など
        return ""

def extract_keywords_with_llm(text: str, llm_type: str = "huggingface") -> List[str]:
    """LLMを使用してキーワードと意図を抽出（中期的改善）
    
    Args:
        text: 入力テキスト
        llm_type: LLMの種類（"ollama" or "huggingface"）
    
    Returns:
        抽出されたキーワードのリスト
    """
    prompt = f"""以下の日本語の文章から、重要なキーワードとユーザーの意図を抽出してください。

文章：「{text}」

以下の形式で回答してください：
- 重要なキーワード（名詞や重要な概念）を3-5個、カンマ区切りで
- ユーザーの意図（願い、悩み、希望など）を1つ

例：
キーワード: 夫婦, 生活, 円満, 家族
意図: 家族の幸せを願う

回答："""
    
    try:
        if llm_type == "ollama":
            response = generate_with_ollama(prompt, model="llama3.2")
        elif llm_type == "huggingface":
            response = generate_with_huggingface(prompt, model="microsoft/DialoGPT-medium")
        else:
            return []
        
        if not response:
            return []
        
        # レスポンスからキーワードを抽出
        keywords = []
        lines = response.strip().split('\n')
        for line in lines:
            if 'キーワード' in line or 'keyword' in line.lower():
                # 「キーワード: 夫婦, 生活, 円満」のような形式から抽出
                parts = line.split(':')
                if len(parts) >= 2:
                    kw_text = parts[1].strip()
                    # カンマ区切りで分割
                    kw_list = [kw.strip() for kw in kw_text.split(',')]
                    keywords.extend(kw_list)
        
        # キーワードが見つからない場合、レスポンス全体から抽出を試みる
        if not keywords:
            # レスポンスから日本語の単語を抽出（2文字以上）
            import re
            japanese_words = re.findall(r'[ひらがなカタカナ一-龠]{2,}', response)
            keywords = [w for w in japanese_words if len(w) <= 8][:5]
        
        return keywords
    except Exception as e:
        # LLMが使用できない場合、空のリストを返す
        return []

def generate_with_huggingface(prompt: str, model: str = "microsoft/DialoGPT-medium") -> str:
    """Hugging Face Inference APIを使用してテキストを生成（無料枠）
    
    Args:
        prompt: プロンプト
        model: 使用するモデル名（デフォルト: Mistral-7B-Instruct）
    
    Returns:
        生成されたテキスト
    """
    try:
        # Hugging Face Inference API（無料枠）
        # 注意: 実際の使用時は、Hugging FaceのAPIキーが必要な場合があります
        # 無料枠では、公開モデルを使用できます
        
        # Hugging Face Inference APIを使用（無料枠）
        # APIキーが設定されている場合は使用、なければ公開エンドポイントを使用
        api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # 無料で使用可能なモデル（テキスト生成用）
        # 注意: モデルによってはAPIキーが必要な場合があります
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_p": 0.9,
                "return_full_text": False
            }
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            # レスポンス形式に応じてテキストを抽出
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    return result[0]["generated_text"].strip()
                elif "text" in result[0]:
                    return result[0]["text"].strip()
            elif isinstance(result, dict):
                if "generated_text" in result:
                    return result["generated_text"].strip()
                elif "text" in result:
                    return result["text"].strip()
        
        # APIエラーの場合、空文字を返す（フォールバック）
        return ""
    except Exception as e:
        # エラーが発生した場合、空文字を返す（フォールバック）
        return ""

def select_maxims_from_database(
    keywords: List[str], 
    top_k: int = 3,
    exclude_maxims: Optional[List[str]] = None
) -> List[Dict]:
    """格言データベースからキーワードに基づいて格言を選択
    
    Args:
        keywords: キーワードリスト
        top_k: 選択する格言の数
        exclude_maxims: 除外する格言のリスト（重複を避けるため）
    
    Returns:
        選択された格言のリスト
    """
    global MAXIMS_DATABASE
    if not MAXIMS_DATABASE:
        return []
    
    exclude_set = set(exclude_maxims or [])
    # キーワードに基づいてスコアリング
    scored_maxims = []
    keyword_set = set([kw.lower() for kw in keywords])
    
    for maxim in MAXIMS_DATABASE:
        maxim_text = maxim.get("text", "")
        # 除外リストに含まれている場合はスキップ
        if maxim_text in exclude_set:
            continue
            
        score = 0.0
        maxim_text_lower = maxim_text.lower()
        maxim_tags = [tag.lower() for tag in maxim.get("tags", [])]
        
        # タグ一致を優先（完全一致）
        for tag in maxim_tags:
            if tag in keyword_set:
                score += 5.0  # タグ一致は高スコア
            # タグ部分一致も考慮
            elif any(kw in tag for kw in keyword_set):
                score += 3.0
        
        # テキスト内のキーワード一致（日本語対応：部分一致も考慮）
        matched_count = 0
        for kw in keyword_set:
            # 完全一致（最優先）
            if kw in maxim_text_lower:
                score += 10.0  # テキスト内のキーワード完全一致は最高スコア
                matched_count += 1
            # 部分一致（日本語の場合、単語の境界が明確でないため）
            elif len(kw) >= 2 and kw[:2] in maxim_text_lower:
                score += 5.0  # 部分一致も高スコア
                matched_count += 1
            # 文字レベルでの一致
            elif any(c in maxim_text_lower for c in kw if len(c) >= 1):
                score += 2.0  # 文字レベルでの一致も考慮
                matched_count += 1
        
        # 複数キーワード一致のボーナス（重要）
        if matched_count >= 2:
            score += matched_count * 5.0  # 複数キーワード一致は大幅なボーナス
        elif matched_count == 1:
            score += 3.0  # 単一キーワード一致でもボーナス
        
        if score > 0:
            scored_maxims.append((score, maxim))
    
    # スコアでソート
    scored_maxims.sort(key=lambda x: x[0], reverse=True)
    
    # 上位k個を選択（ランダム要素を追加して多様性を確保）
    if scored_maxims:
        # 上位10個からランダムに選択
        top_candidates = scored_maxims[:min(10, len(scored_maxims))]
        random.shuffle(top_candidates)
        selected = []
        for _, maxim in top_candidates:
            if len(selected) >= top_k:
                break
            if maxim.get("text") not in exclude_set:
                selected.append(maxim)
        return selected
    
    return []

def generate_maxim_from_keywords(keywords: List[str], context_text: str) -> Optional[str]:
    """キーワードに基づいて格言を生成
    
    Args:
        keywords: キーワードリスト
        context_text: ユーザー入力テキスト
    
    Returns:
        生成された格言（文字列）、生成できない場合はNone
    """
    if not keywords or not context_text:
        return None
    
    # キーワードから意味を推測して格言を生成
    # 例：「疲れ」「決断」→「疲れていても、決断する勇気を持て。一歩ずつ進めば道は開ける。」
    
    # キーワードの意味に基づくテンプレート
    maxim_templates = {
        "健康": ["健康は最大の財産。日々の積み重ねが、未来の自分を創る。", "健康な体に、健康な心が宿る。自分を大切に、今日も一歩ずつ。", "健康は贈り物。感謝して、大切に守っていこう。"],
        "疲": ["疲れていても、一歩ずつ進めば道は開ける。", "疲れは休息の合図。無理をせず、今を大切に。", "疲れた時こそ、自分を労わる時。休息も成長の一部。"],
        "決断": ["決断は勇気。迷う時間も、選択の一部。", "決断できない時は、時間をかけて考えてもよい。", "決断は一瞬、その結果は一生。慎重に、しかし恐れずに。"],
        "不安": ["不安は未来への準備。今できることを大切に。", "不安は成長の証。一歩ずつ進めば、道は見えてくる。", "不安があっても、前に進む勇気を持て。"],
        "迷": ["迷うことは、真剣に考えている証。時間をかけて答えを見つけよう。", "迷いは選択の余地がある証。焦らず、自分を信じて。", "迷う時は、心に問いかけてみよう。答えは必ず見つかる。"],
        "孤独": ["孤独は自分と向き合う時間。大切な気づきが生まれる。", "一人の時間も、成長の糧。自分を大切に。", "孤独は一時的なもの。必ずつながりは見つかる。"],
        "挑戦": ["挑戦は成長の種。失敗を恐れず、一歩を踏み出そう。", "挑戦する勇気が、新しい道を開く。", "挑戦は自分を変える力。恐れずに進もう。"],
        "仕事": ["仕事は人生の一部。バランスを保ちながら、一歩ずつ進もう。", "仕事を通じて、自分を成長させよう。", "仕事は貢献の形。誠実に、丁寧に取り組もう。"],
        "家族": ["家族は絆。大切な人を思いやり、共に歩もう。", "家族の幸せは、自分の幸せ。共に笑い、共に支え合おう。", "家族は宝物。感謝の気持ちを忘れずに。"],
        "幸せ": ["幸せは今この瞬間にある。小さな喜びを大切に。", "幸せは自分で創るもの。感謝の心を持って、一歩ずつ。", "幸せは分かち合うもの。周りの人と共に喜びを。"],
    }
    
    # キーワードに基づいてテンプレートを選択
    selected_template = None
    for kw in keywords:
        kw_lower = kw.lower()
        for key, templates in maxim_templates.items():
            if key in kw_lower or kw_lower in key:
                import random
                selected_template = random.choice(templates)
                break
        if selected_template:
            break
    
    # テンプレートがない場合、キーワードから直接生成
    if not selected_template:
        # キーワードを組み合わせて格言を生成
        if len(keywords) >= 2:
            # 例：「疲れ」「決断」→「疲れていても、決断する勇気を持て。」
            key_phrases = {
                "健康": "健康を大切に",
                "疲": "疲れていても",
                "決断": "決断する勇気を持て",
                "不安": "不安があっても",
                "迷": "迷う時は",
                "孤独": "一人でも",
                "挑戦": "挑戦する勇気が",
                "仕事": "仕事に誠実に",
                "家族": "家族を大切に",
                "幸せ": "幸せを願って",
            }
            
            phrases = []
            for kw in keywords[:3]:  # 最大3つのキーワードを使用
                kw_lower = kw.lower()
                for key, phrase in key_phrases.items():
                    if key in kw_lower or kw_lower in key:
                        phrases.append(phrase)
                        break
            
            if phrases:
                if len(phrases) >= 2:
                    selected_template = f"{phrases[0]}、{phrases[1]}。一歩ずつ進めば道は開ける。"
                else:
                    selected_template = f"{phrases[0]}。今を大切に、一歩ずつ進もう。"
    
    # それでも生成できない場合、汎用的な格言を生成
    if not selected_template:
        if "健康" in context_text or "体" in context_text or "身体" in context_text:
            selected_template = "健康は最大の財産。日々の積み重ねが、未来の自分を創る。"
        elif "疲" in context_text or "だる" in context_text:
            selected_template = "疲れていても、一歩ずつ進めば道は開ける。休息も大切な選択。"
        elif "決断" in context_text or "決め" in context_text:
            selected_template = "決断は勇気。迷う時間も、選択の一部。焦らず、自分を信じて。"
        elif "不安" in context_text or "心配" in context_text:
            selected_template = "不安は未来への準備。今できることを大切に、一歩ずつ進もう。"
        elif "仕事" in context_text:
            selected_template = "仕事は人生の一部。バランスを保ちながら、一歩ずつ進もう。"
        elif "家族" in context_text:
            selected_template = "家族は絆。大切な人を思いやり、共に歩もう。"
        elif "幸せ" in context_text or "幸福" in context_text:
            selected_template = "幸せは今この瞬間にある。小さな喜びを大切に。"
        else:
            # 汎用的な格言（キーワードを使用）
            if keywords:
                selected_template = f"{keywords[0]}を大切に。一歩ずつ進めば道は開ける。"
            else:
                selected_template = "今を大切に。一歩ずつ進めば道は開ける。"
    
    return selected_template

def create_original_maxim_from_vow(
    selected_vow_index: Optional[int],
    god: Dict,
    top_k: int = 3
) -> Optional[str]:
    """選ばれた誓願（VOW）のベクトルに基づいて、格言データベースから部分的に組み合わせてオリジナルな格言を生成
    
    Args:
        selected_vow_index: 選ばれた誓願（VOW）のインデックス（0-11、Noneの場合は生成しない）
        god: 選ばれた神の情報
        top_k: 組み合わせに使用する格言の数
    
    Returns:
        生成されたオリジナルな格言（生成できない場合はNone）
    """
    global MAXIMS_DATABASE
    if not MAXIMS_DATABASE or selected_vow_index is None:
        return None
    
    # 選ばれた誓願に対応するVOW値を取得
    vows = god.get("vows", {})
    vow_key = f"vow{selected_vow_index+1:02d}"
    selected_vow_value = vows.get(vow_key, 0.0) if vow_key in vows else 0.0
    
    # VOW値が非常に小さい（関連性が弱い）場合は、オリジナル格言を生成しない
    if abs(selected_vow_value) < 0.1:
        return None
    
    # 神のVOW値から関連する誓願を取得（VOW値の絶対値が大きい順）
    vow_scores = []
    for i in range(12):
        v_key = f"vow{i+1:02d}"
        if v_key in vows:
            vow_scores.append((i, float(vows[v_key])))
    vow_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # 格言データベースから、選ばれた誓願と関連する誓願に対応する格言を選択
    selected_maxims = []
    
    # 選ばれた誓願に対応する格言を優先的に選択
    # VOW値が負（強い関連性）の場合、より多くの格言を選択
    num_maxims_to_select = max(2, min(top_k + 1, int(abs(selected_vow_value) * 5) + 2))
    
    # ランダムに格言を選択（多様性を確保）
    import time
    random.seed(int(time.time() * 1000) % 1000000)
    available_maxims = [m for m in MAXIMS_DATABASE if m.get("text")]
    random.shuffle(available_maxims)
    
    # 選ばれた誓願に関連する格言を選択
    for maxim in available_maxims:
        if len(selected_maxims) >= num_maxims_to_select:
            break
        maxim_text = maxim.get("text", "")
        if maxim_text and maxim_text not in [m.get("text", "") for m in selected_maxims]:
            selected_maxims.append(maxim)
    
    # 格言が少ない場合、神のデフォルト格言を追加
    if len(selected_maxims) < 2:
        god_maxim = god.get("maxim", "")
        if god_maxim and god_maxim not in [m.get("text", "") for m in selected_maxims]:
            selected_maxims.append({"text": god_maxim, "source": god.get("name", "神託"), "tags": []})
    
    # 格言を部分的に組み合わせてオリジナルな格言を生成
    if len(selected_maxims) >= 2:
        # 複数の格言から重要なフレーズを抽出して組み合わせ
        phrases = []
        for maxim in selected_maxims[:min(3, len(selected_maxims))]:  # 最大3つの格言を使用
            text = maxim.get("text", "")
            if text:
                # 句点や読点で分割
                parts = re.split(r'[。、，]', text)
                # 空でない部分を追加（3文字以上）
                phrases.extend([p.strip() for p in parts if p.strip() and len(p.strip()) >= 3])
        
        if len(phrases) >= 2:
            # ランダムに2-3つのフレーズを選択して組み合わせ
            random.shuffle(phrases)
            selected_phrases = phrases[:min(3, len(phrases))]
            
            # フレーズを組み合わせ（句点で区切る）
            combined = "。".join(selected_phrases)
            if not combined.endswith("。"):
                combined += "。"
            
            # 長すぎる場合は短縮
            if len(combined) > 100:
                combined = combined[:100] + "..."
            
            return combined
    
    # 組み合わせができない場合、Noneを返す（通常の格言選択にフォールバック）
    return None

# -------------------------
# キーワード抽出とネットワーク構築（Cell 4用）
# -------------------------
def extract_keywords(text: str, top_n: int = 5, use_llm: bool = False, llm_type: str = "huggingface") -> List[str]:
    """テキストからキーワードを抽出（改善版：日本語対応強化 + LLMオプション）
    
    Args:
        text: 入力テキスト
        top_n: 抽出するキーワードの最大数
        use_llm: LLMを使用してキーワード抽出を行うか
        llm_type: LLMの種類（"ollama" or "huggingface"）
    """
    if not text or not text.strip():
        return []
    
    # 【中期的改善】LLMを使用してキーワード抽出を行う場合
    if use_llm:
        llm_keywords = extract_keywords_with_llm(text, llm_type=llm_type)
        if llm_keywords:
            return llm_keywords[:top_n]
    
    found_keywords = []
    text_original = text.strip()
    text_lower = text_original.lower()
    
    # 【短期的改善】文脈パターンの検出（願い・祈りのパターン）
    context_patterns = {
        "wish": [
            r"でありますように", r"ように", r"であります", r"ありますように",
            r"できますように", r"なりますように", r"過ごせますように",
            r"願い", r"祈り", r"希望", r"願う", r"祈る", r"望む"
        ],
        "family": [
            r"家族", r"夫婦", r"親", r"子", r"家庭", r"生活", r"円満"
        ],
        "health": [
            r"健康", r"体調", r"身体", r"過ごしたい", r"過ごせますように"
        ]
    }
    
    # 文脈パターンに基づいてカテゴリを検出
    detected_categories = []
    for category, patterns in context_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_original):
                detected_categories.append(category)
                # カテゴリに対応するキーワードを追加
                if category in KEYWORDS:
                    for kw in KEYWORDS[category]:
                        if kw not in found_keywords:
                            found_keywords.append(kw)
                break
    
    # 1. KEYWORDS辞書から関連キーワードを抽出（最優先：ユーザー入力の分析）
    # 例：「疲れていて決断が出来ない」→「疲」「決断」を抽出
    for category, keywords in KEYWORDS.items():
        for kw in keywords:
            # 部分一致で検索（「疲れ」に「疲」が含まれる）
            if kw in text_lower or text_lower in kw:
                if kw not in found_keywords:
                    found_keywords.append(kw)
            # より柔軟なマッチング：キーワードの文字が含まれているか
            elif any(c in text_lower for c in kw if len(c) >= 1):
                # 「疲れ」に「疲」が含まれる場合
                if len(kw) >= 2 and kw[:2] in text_lower:
                    if kw not in found_keywords:
                        found_keywords.append(kw)
    
    # 2. GLOBAL_WORDS_DATABASEから一致するキーワードを抽出
    for word in GLOBAL_WORDS_DATABASE:
        if word in text_original or word in text_lower:
            if word not in found_keywords:
                found_keywords.append(word)
    
    # 3. 日本語の形態素解析的なアプローチ：文字列から意味のある単語を抽出
    # 「世界平和に貢献できる人間になる」→「世界平和」「貢献」「人間」を抽出
    import re
    # まず、GLOBAL_WORDS_DATABASEに含まれる長い単語から抽出（優先）
    # 長い単語から順にチェック（「世界平和」が「世界」や「平和」より優先される）
    text_for_extraction = text_original  # 抽出用のテキスト（元のテキストは保持）
    for word in sorted(GLOBAL_WORDS_DATABASE, key=len, reverse=True):
        if word in text_for_extraction and word not in found_keywords:
            found_keywords.append(word)
            # 抽出した単語をテキストから削除（重複抽出を避ける）
            text_for_extraction = text_for_extraction.replace(word, " ", 1)
    
    # 助詞・助動詞で文章を分割してから、個別の単語を抽出
    # 「夫婦生活が円満でありますように」→「夫婦生活」「円満」を抽出
    import re
    
    # 助詞・助動詞のパターン（分割用：より包括的）
    # 助詞：が、を、に、で、と、から、まで、より、ので、のに、でも、など、とか、だけ、ばかり、くらい、ほど、しか
    # 助動詞：である、です、ます、れる、られる、せる、させる、ない、ぬ、ん、う、よう、まい
    # その他：て、で、た、だ、あります、ように、であります
    particle_pattern = r'[がをにでとからまでよりのでのにでもなどとかだけばかりくらいほどしかてでただであるですますれるられるせるさせるないぬんうようまいありますように]|であります|ありますように'
    
    # 助詞・助動詞で分割
    # 例：「夫婦生活が円満でありますように」→「夫婦生活」「円満」に分割
    split_text = re.split(particle_pattern, text_for_extraction)
    
    # 分割後の各単語を抽出
    japanese_words = []
    for segment in split_text:
        segment = segment.strip()
        if not segment:
            continue
        
        # セグメント全体を単語として追加（複合語の場合：例「夫婦生活」）
        # ただし、長すぎる場合は除外
        if len(segment) >= 2 and len(segment) <= 8:
            # 助詞・助動詞を含まない場合のみ追加
            if not re.search(particle_pattern, segment):
                if segment not in japanese_words:
                    japanese_words.append(segment)
        
        # セグメントから個別の単語も抽出（例：「夫婦生活」→「夫婦」「生活」）
        # 漢字・ひらがな・カタカナの連続を抽出（2文字以上、最大6文字まで）
        words_in_segment = re.findall(r'[ひらがなカタカナ一-龠]{2,6}', segment)
        for word in words_in_segment:
            if len(word) >= 2 and len(word) <= 8 and word not in japanese_words:
                # 助詞・助動詞を含まない場合のみ追加
                if not re.search(particle_pattern, word):
                    japanese_words.append(word)
    
    # 助詞・助動詞のリスト（除外用）
    stop_words = [
        'こと', 'もの', 'とき', 'ため', 'から', 'まで', 'より', 'ので', 'のに', 
        'でも', 'など', 'とか', 'だけ', 'ばかり', 'くらい', 'ほど', 'しか',
        'ていて', 'が', 'を', 'に', 'で', 'と', 'から', 'まで', 'より', 'ので',
        '出来ない', 'できない', '出来る', 'できる', 'である', 'です', 'ます',
        'なる', 'する', 'れる', 'られる', 'させる', 'させられる', 'て', 'で', 'た', 'だ',
        'れる', 'られる', 'せる', 'させる', 'ない', 'ぬ', 'ん', 'う', 'よう', 'まい',
        'あります', 'ように', 'であります', 'でありますように'
    ]
    
    for word in japanese_words:
        # 長すぎる単語（文章全体）を除外（最大8文字まで）
        if len(word) > 8:
            continue
        
        if len(word) >= 2 and word not in found_keywords:
            # 助詞や助動詞を除外（完全一致と部分一致の両方をチェック）
            is_stop_word = word in stop_words or any(sw in word for sw in stop_words if len(sw) >= 2)
            if not is_stop_word:
                # 既存のキーワードの一部でないかチェック（長いキーワードを優先）
                is_substring = any(word in kw for kw in found_keywords if len(kw) > len(word))
                if not is_substring:
                    # 短すぎる単語（1-2文字）は除外（ただし、GLOBAL_WORDS_DATABASEに含まれる場合はOK）
                    if len(word) >= 2 or word in GLOBAL_WORDS_DATABASE:
                        found_keywords.append(word)
    
    # 【長期的改善】Janomeを使用した形態素解析（より正確な分割）
    if JANOME_AVAILABLE:
        try:
            tokenizer = Tokenizer()
            tokens = tokenizer.tokenize(text_original)
            for token in tokens:
                # 名詞、動詞、形容詞のみを抽出
                pos = token.part_of_speech.split(',')[0]
                if pos in ['名詞', '動詞', '形容詞']:
                    surface = token.surface
                    # 長すぎる単語を除外
                    if 2 <= len(surface) <= 8 and surface not in found_keywords:
                        # 助詞・助動詞を除外
                        stop_words_list = [
                            'こと', 'もの', 'とき', 'ため', 'から', 'まで', 'より', 'ので', 'のに', 
                            'でも', 'など', 'とか', 'だけ', 'ばかり', 'くらい', 'ほど', 'しか',
                            'が', 'を', 'に', 'で', 'と', 'て', 'た', 'だ'
                        ]
                        if surface not in stop_words_list:
                            found_keywords.append(surface)
        except Exception:
            # Janomeが使用できない場合、従来の方法を使用
            pass
    
    # 4. テキストを単語に分割して、2文字以上の単語を抽出（英語やスペース区切りの場合）
    text_clean = re.sub(r'[0-9０-９\W]+', ' ', text_original)
    words = text_clean.split()
    for word in words:
        # 長すぎる単語（文章全体）を除外（最大8文字まで）
        if len(word) > 8:
            continue
        
        if len(word) >= 2 and word not in found_keywords:
            if word not in ['こと', 'もの', 'とき', 'ため', 'から', 'まで', 'より', 'ので', 'のに', 
                           'でも', 'でも', 'など', 'とか', 'だけ', 'ばかり', 'くらい', 'ほど', 'しか']:
                found_keywords.append(word)
    
    # 5. 重複を除去し、上位N個を返す（優先順位：GLOBAL_WORDS_DATABASE > KEYWORDS辞書 > その他）
    unique_keywords = list(dict.fromkeys(found_keywords))  # 順序を保持しながら重複除去
    
    # 文章全体（長すぎる単語）を除外（最大8文字まで）
    filtered_keywords = [kw for kw in unique_keywords if len(kw) <= 8]
    
    # 優先順位でソート：
    # 1. GLOBAL_WORDS_DATABASEに含まれるキーワード（長い順）
    # 2. KEYWORDS辞書からの抽出
    # 3. その他のキーワード（長い順）
    keywords_from_global = [kw for kw in filtered_keywords if kw in GLOBAL_WORDS_DATABASE]
    keywords_from_global.sort(key=lambda x: (GLOBAL_WORDS_DATABASE.index(x) if x in GLOBAL_WORDS_DATABASE else 999, -len(x)))
    
    keywords_from_dict = [kw for kw in filtered_keywords if kw not in keywords_from_global and any(kw in kws or any(k in kw for k in kws) for kws in KEYWORDS.values() for k in kws)]
    
    other_keywords = [kw for kw in filtered_keywords if kw not in keywords_from_global and kw not in keywords_from_dict]
    other_keywords.sort(key=lambda x: -len(x))  # 長い順
    
    sorted_keywords = keywords_from_global + keywords_from_dict + other_keywords
    
    return sorted_keywords[:top_n]

def calculate_energy_between_words(
    word1: str, 
    word2: str,
    selected_character: Optional[str] = None,
    selected_attribute: Optional[str] = None,
    char_master: Optional[pd.DataFrame] = None
) -> float:
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
    
    # キャラクター選択を反映
    if selected_character and char_master is not None:
        try:
            # 選択されたキャラクターの行を取得
            char_row = None
            if "公式キャラ名" in char_master.columns:
                char_row = char_master[char_master["公式キャラ名"] == selected_character]
            elif "CHAR_ID" in char_master.columns:
                char_row = char_master[char_master["CHAR_ID"] == selected_character]
            
            if char_row is not None and not char_row.empty:
                # VOW値が高い単語を優先（単語とVOWの対応は簡易的に実装）
                # 単語がキャラクターの特徴と関連する場合、エネルギーを下げる（近づける）
                vow_values = []
                for i in range(1, 13):
                    vow_col = f"VOW_{i:02d}"
                    if vow_col in char_row.columns:
                        vow_val = char_row[vow_col].iloc[0]
                        if pd.notna(vow_val):
                            vow_values.append(float(vow_val))
                
                if vow_values:
                    avg_vow = np.mean(vow_values)
                    # キャラクターの特徴が強い場合、単語間のエネルギーを下げる（近づける）
                    energy -= avg_vow * 0.2
                
                # 属性選択を反映
                if selected_attribute and "属性" in char_row.columns:
                    char_attribute = char_row["属性"].iloc[0]
                    if pd.notna(char_attribute) and str(char_attribute) == selected_attribute:
                        # 属性が一致する場合、さらにエネルギーを下げる
                        energy -= 0.3
        except Exception:
            # エラーが発生した場合は無視（デフォルトの計算を続行）
            pass
    
    # 毎回異なる結果を得るため、タイムスタンプベースのランダム要素を追加
    energy += np.random.normal(0, 0.15)
    return energy

def build_word_network(
    center_words: List[str], 
    database: List[str], 
    n_neighbors: int = 15,
    selected_character: Optional[str] = None,
    selected_attribute: Optional[str] = None,
    char_master: Optional[pd.DataFrame] = None
) -> Dict:
    """単語ネットワークを構築（毎回異なる結果を得るため、ランダムシードを追加）
    
    Args:
        center_words: 中心となる単語のリスト
        database: 単語データベース
        n_neighbors: 選択する近傍単語の数
        selected_character: 選択されたキャラクター（公式キャラ名）
        selected_attribute: 選択された属性
        char_master: CHAR_MASTERシートのデータ
    """
    # 毎回異なる結果を得るため、タイムスタンプベースのシードを使用
    import time
    random_seed = int(time.time() * 1000) % 1000000
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    all_words = list(set(center_words + database))
    word_energies = {}
    for word in all_words:
        if word in center_words:
            energy = -2.0
        else:
            # ランダム要素を追加して毎回異なる結果を得る
            energies = [
                calculate_energy_between_words(
                    cw, word, 
                    selected_character=selected_character,
                    selected_attribute=selected_attribute,
                    char_master=char_master
                ) 
                for cw in center_words
            ]
            energy = np.mean(energies) + np.random.normal(0, 0.1)  # ランダム要素を追加
        word_energies[word] = energy
    
    sorted_words = sorted(word_energies.items(), key=lambda x: (x[1], np.random.random()))  # 同点の場合はランダム
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
            energy = calculate_energy_between_words(
                word1, word2,
                selected_character=selected_character,
                selected_attribute=selected_attribute,
                char_master=char_master
            )
            # 閾値を少し緩和して、より多くのエッジを表示
            if energy < -0.25:
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

def select_relevant_quote(
    keywords: List[str],
    exclude_quotes: Optional[set] = None
) -> str:
    """キーワードに基づいて関連する格言を選択（毎回異なる結果を得るため、ランダム要素を追加）
    
    Args:
        keywords: キーワードリスト
        exclude_quotes: 除外する格言のセット（重複を避けるため）
    """
    # 毎回異なる結果を得るため、タイムスタンプベースのシードを使用
    import time
    random.seed(int(time.time() * 1000) % 1000000)
    
    exclude_set = exclude_quotes or set()
    keyword_set = set(keywords)
    scored_quotes = []
    
    for quote_data in FAMOUS_QUOTES:
        quote_text = quote_data["quote"]
        # 除外リストに含まれている場合はスキップ
        if quote_text in exclude_set:
            continue
            
        quote_keywords = set(quote_data["keywords"])
        score = len(keyword_set & quote_keywords)
        # スコアに小さなランダム要素を追加して、毎回異なる結果を得る
        score += random.uniform(-0.3, 0.3)
        scored_quotes.append((score, quote_text))
    
    # スコアでソート
    scored_quotes.sort(key=lambda x: x[0], reverse=True)
    
    # 上位10個からランダムに選択（多様性を確保）
    if scored_quotes:
        top_quotes = [q for _, q in scored_quotes[:min(10, len(scored_quotes))]]
        if top_quotes:
            return random.choice(top_quotes)
    
    return "あなたの観測が、この世界線を確定させました。"

# -------------------------
# キャラクター表示
# -------------------------
def get_character_image_path(god: Dict, gods_list: Optional[List[Dict]] = None) -> Optional[str]:
    """キャラクター画像のパスを取得"""
    # IMAGE_FILEから取得（実際のファイル名を使用）
    image_file = god.get("image_file")
    if image_file:
        # ファイル名をそのまま使用（CHAR_p1.pngなど）
        if image_file.endswith(".png"):
            image_path = f"assets/images/characters/{image_file}"
            if os.path.exists(image_path):
                return image_path
        # CHAR_p1.png形式の場合
        if image_file.startswith("CHAR_p"):
            image_path = f"assets/images/characters/{image_file}"
            if os.path.exists(image_path):
                return image_path
    
    # CHAR_IDから取得
    char_id = god.get("char_id")
    if char_id and char_id.startswith("CHAR_"):
        try:
            char_num = int(char_id.replace("CHAR_", ""))
            # CHAR_01 → CHAR_p1.png
            image_path = f"assets/images/characters/CHAR_p{char_num}.png"
            if os.path.exists(image_path):
                return image_path
        except:
            pass
    
    # IDから取得（CHAR_p1.png形式を試す）
    god_id = god.get("id", 0)
    image_path = f"assets/images/characters/CHAR_p{god_id+1}.png"
    if os.path.exists(image_path):
        return image_path
    
    # character_01.png形式も試す（後方互換性）
    image_path = f"assets/images/characters/character_{god_id+1:02d}.png"
    if os.path.exists(image_path):
        return image_path
    
    return None

def render_god_character(god: Dict, gods_list: Optional[List[Dict]] = None) -> str:
    """選ばれた神のキャラクターをHTMLで表示"""
    god_name = god["name"]
    god_emoji = god["emoji"]
    god_description = god["description"]
    
    # 画像パスを取得
    image_path = get_character_image_path(god, gods_list)
    image_html = ""
    if image_path and os.path.exists(image_path):
        try:
            import base64
            with open(image_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
                image_html = f'<img src="data:image/png;base64,{img_data}" style="max-width: 300px; max-height: 300px; border-radius: 10px; margin-bottom: 20px;" />'
        except Exception:
            pass
    
    # f-stringを使わず、通常の文字列でHTMLを生成（CSSの{}をエスケープする必要がない）
    character_html = """
    <div id="god-character-container" style="
        position: relative;
        width: 100%;
        height: 400px;
        background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 50%, #0a0a1a 100%);
        border-radius: 15px;
        overflow: hidden;
        margin: 20px 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.3);
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
            
            .god-emoji {
                animation: fadeIn 2s ease-out, float 4s ease-in-out infinite;
                font-size: 120px;
                text-align: center;
                filter: drop-shadow(0 0 20px rgba(255, 255, 255, 0.4));
                display: inline-block;
            }
            
            .god-name {
                animation: fadeIn 2s ease-out 0.5s both, glow 3s ease-in-out infinite;
                font-size: 32px;
                color: #ffd700;
                text-align: center;
                margin-top: 20px;
                font-weight: bold;
                font-family: 'Yu Gothic', 'Meiryo', 'MS Gothic', sans-serif;
                letter-spacing: 2px;
            }
            
            .god-description {
                animation: fadeIn 2s ease-out 1s both;
                color: #ffffff;
                margin-top: 15px;
                font-size: 18px;
                font-family: 'Yu Gothic', 'Meiryo', 'MS Gothic', sans-serif;
                text-align: center;
                padding: 0 20px;
            }
            
            .sparkle {
                position: absolute;
                color: #ffd700;
                font-size: 24px;
                animation: sparkle 2s ease-in-out infinite;
                pointer-events: none;
            }
        </style>
        
        <div style="position: relative; text-align: center; z-index: 1;">
            """ + (image_html if image_html else f'<div class="god-emoji">{god_emoji}</div>') + """
            <div class="god-name">""" + god_name + """</div>
            <div class="god-description">""" + god_description + """</div>
        </div>
        
        <div class="sparkle" style="top: 15%; left: 15%; animation-delay: 0s;">✨</div>
        <div class="sparkle" style="top: 25%; right: 20%; animation-delay: 0.7s;">✨</div>
        <div class="sparkle" style="bottom: 30%; left: 25%; animation-delay: 1.4s;">✨</div>
        <div class="sparkle" style="bottom: 40%; right: 15%; animation-delay: 2.1s;">✨</div>
        <div class="sparkle" style="top: 50%; left: 10%; animation-delay: 0.3s;">✨</div>
        <div class="sparkle" style="top: 60%; right: 10%; animation-delay: 1.0s;">✨</div>
    </div>
    """
    return character_html

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
            marker=dict(
                size=size, 
                color=color, 
                line=dict(width=2, color='white'),
                opacity=0.6 if not is_center else 0.9
            ),
            text=[word],
            textposition="middle center",
            textfont=dict(
                size=20 if is_center else 16, 
                color='#ffd700' if is_center else '#ffffff',  # 中心語は金色、その他は白色
                family='Arial, sans-serif',
                weight='bold'  # すべて太字で見やすく
            ),
            name=word,
            hovertemplate=f'<b>{word}</b><br>エネルギー: {network["energies"].get(word, 0):.2f}<extra></extra>'
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
    
    # Excelファイルアップロード機能
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 設定ファイル")
    
    # アップロード方法を選択
    upload_mode = st.sidebar.radio(
        "アップロード方法",
        ["5つのファイル（推奨）", "1つのファイル（3シート）", "4つの別ファイル"],
        help="5つのExcelファイルをまとめて読み込むか、個別に読み込むかを選択"
    )
    
    if upload_mode == "5つのファイル（推奨）":
        st.sidebar.markdown("**5つのExcelファイルをアップロード:**")
        
        character_file = st.sidebar.file_uploader(
            "1. 12神基本情報 (akiba12_character_list.xlsx)",
            type=['xlsx', 'xls'],
            key="char_file_all",
            help="12神の基本情報（ID、名前、属性、絵文字、説明、格言）"
        )
        
        maxim_file = st.sidebar.file_uploader(
            "2. 格言ファイル (格言.xlsx)",
            type=['xlsx', 'xls'],
            key="maxim_file_all",
            help="格言データ（オプション）"
        )
        
        sense_to_vow_file = st.sidebar.file_uploader(
            "3. sense_to_vow行列 (sense_to_vow_initial_filled_from_user.xlsx)",
            type=['xlsx', 'xls'],
            key="sense_to_vow_file_all",
            help="感覚 × 誓願（8x12の行列）"
        )
        
        k_matrix_file = st.sidebar.file_uploader(
            "4. k行列 (akiba12_character_to_vow_K.xlsx)",
            type=['xlsx', 'xls'],
            key="k_matrix_file_all",
            help="キャラクター × 誓願（12x12の行列）"
        )
        
        l_matrix_file = st.sidebar.file_uploader(
            "5. l行列 (akiba12_character_to_axis_L.xlsx)",
            type=['xlsx', 'xls'],
            key="l_matrix_file_all",
            help="キャラクター × 世界観軸（12x4の行列）"
        )
        
        if k_matrix_file is not None and l_matrix_file is not None:
            if load_all_excel_files(
                character_file=character_file,
                maxim_file=maxim_file,
                k_matrix_file=k_matrix_file,
                l_matrix_file=l_matrix_file,
                sense_to_vow_file=sense_to_vow_file
            ):
                st.sidebar.success("✅ 設定ファイルを読み込みました")
                if LOADED_GODS:
                    st.sidebar.info(f"読み込まれた神の数: {len(LOADED_GODS)}")
                    with st.sidebar.expander("📋 読み込んだ設定の詳細"):
                        st.write("**12神のリスト:**")
                        for god in LOADED_GODS[:3]:
                            st.write(f"- {god['emoji']} {god['name']}")
                        if len(LOADED_GODS) > 3:
                            st.write(f"... 他 {len(LOADED_GODS) - 3} 神")
                        
                        if SENSE_TO_VOW_MATRIX is not None:
                            st.write(f"**sense_to_vow行列サイズ:** {SENSE_TO_VOW_MATRIX.shape}")
                        if K_MATRIX is not None:
                            st.write(f"**k行列サイズ:** {K_MATRIX.shape}")
                        if L_MATRIX is not None:
                            st.write(f"**l行列サイズ:** {L_MATRIX.shape}")
            else:
                st.sidebar.error("❌ 設定ファイルの読み込みに失敗しました")
        elif k_matrix_file is not None or l_matrix_file is not None:
            st.sidebar.warning("⚠️ k行列とl行列の両方が必要です")
        else:
            st.sidebar.info("💡 デフォルト設定を使用中")
    
    elif upload_mode == "1つのファイル（3シート）":
        uploaded_file = st.sidebar.file_uploader(
            "Excel設定ファイルをアップロード",
            type=['xlsx', 'xls'],
            help="12神の設定、k行列、l行列を含むExcelファイル（3つのシート）"
        )
        
        # 格言ファイルのアップロード（オプション）
        maxim_file = st.sidebar.file_uploader(
            "格言ファイル (格言.xlsx) - オプション",
            type=['xlsx', 'xls'],
            key="maxim_file_single",
            help="格言データベース（格言、出典、タグを含むExcelファイル）"
        )
        
        if uploaded_file is not None:
            if load_excel_config(excel_file=uploaded_file):
                st.sidebar.success("✅ 設定ファイルを読み込みました")
                
                # 格言ファイルを読み込む（オプション）
                if maxim_file is not None:
                    maxims = load_maxims_from_excel(maxim_file)
                    if maxims:
                        st.sidebar.success(f"✅ 格言ファイルを読み込みました（{len(maxims)}件）")
                if LOADED_GODS:
                    st.sidebar.info(f"読み込まれた神の数: {len(LOADED_GODS)}")
                    # 読み込んだ設定の詳細を表示（展開可能）
                    with st.sidebar.expander("📋 読み込んだ設定の詳細"):
                        st.write("**12神のリスト:**")
                        for god in LOADED_GODS[:3]:  # 最初の3つだけ表示
                            st.write(f"- {god['emoji']} {god['name']}")
                        if len(LOADED_GODS) > 3:
                            st.write(f"... 他 {len(LOADED_GODS) - 3} 神")
                        
                        if K_MATRIX is not None:
                            st.write(f"**k行列サイズ:** {K_MATRIX.shape}")
                        if L_MATRIX is not None:
                            st.write(f"**l行列サイズ:** {L_MATRIX.shape}")
            else:
                st.sidebar.error("❌ 設定ファイルの読み込みに失敗しました")
        else:
            st.sidebar.info("💡 デフォルト設定を使用中")
    
    else:  # 4つの別ファイル
        st.sidebar.markdown("**4つのファイルをアップロード:**")
        
        character_file = st.sidebar.file_uploader(
            "1. 12神基本情報 (akiba12_character_list.xlsx)",
            type=['xlsx', 'xls'],
            key="character_file",
            help="12神の基本情報（ID、名前、属性、絵文字、説明、格言）"
        )
        
        sense_to_vow_file = st.sidebar.file_uploader(
            "2. sense_to_vow行列 (sense_to_vow_initial_filled_from_user.xlsx)",
            type=['xlsx', 'xls'],
            key="sense_to_vow_file",
            help="感覚 × 誓願（8x12の行列：迷い/焦り/静けさ/内省/行動/つながり/挑戦/待つ → 12誓願）"
        )
        
        k_matrix_file = st.sidebar.file_uploader(
            "3. k行列 (akiba12_character_to_vow_K.xlsx)",
            type=['xlsx', 'xls'],
            key="k_matrix_file",
            help="キャラクター × 誓願（12x12の行列）"
        )
        
        l_matrix_file = st.sidebar.file_uploader(
            "4. l行列 (akiba12_character_to_axis_L.xlsx)",
            type=['xlsx', 'xls'],
            key="l_matrix_file",
            help="キャラクター × 世界観軸（12x4の行列：静、流、間、誠）"
        )
        
        # 格言ファイルのアップロード（オプション）
        maxim_file = st.sidebar.file_uploader(
            "5. 格言ファイル (格言.xlsx) - オプション",
            type=['xlsx', 'xls'],
            key="maxim_file_separate",
            help="格言データベース（格言、出典、タグを含むExcelファイル）"
        )
        
        if k_matrix_file is not None and l_matrix_file is not None:
            if load_excel_config(
                character_file=character_file,
                sense_to_vow_file=sense_to_vow_file,
                k_matrix_file=k_matrix_file,
                l_matrix_file=l_matrix_file
            ):
                st.sidebar.success("✅ 設定ファイルを読み込みました")
                
                # 格言ファイルを読み込む（オプション）
                if maxim_file is not None:
                    maxims = load_maxims_from_excel(maxim_file)
                    if maxims:
                        st.sidebar.success(f"✅ 格言ファイルを読み込みました（{len(maxims)}件）")
                if LOADED_GODS:
                    st.sidebar.info(f"読み込まれた神の数: {len(LOADED_GODS)}")
                    # 読み込んだ設定の詳細を表示（展開可能）
                    with st.sidebar.expander("📋 読み込んだ設定の詳細"):
                        st.write("**12神のリスト:**")
                        for god in LOADED_GODS[:3]:  # 最初の3つだけ表示
                            st.write(f"- {god['emoji']} {god['name']}")
                        if len(LOADED_GODS) > 3:
                            st.write(f"... 他 {len(LOADED_GODS) - 3} 神")
                        
                        if K_MATRIX is not None:
                            st.write(f"**k行列サイズ:** {K_MATRIX.shape}")
                        if L_MATRIX is not None:
                            st.write(f"**l行列サイズ:** {L_MATRIX.shape}")
            else:
                st.sidebar.error("❌ 設定ファイルの読み込みに失敗しました")
        elif k_matrix_file is not None or l_matrix_file is not None:
            st.sidebar.warning("⚠️ k行列とl行列の両方が必要です")
        else:
            st.sidebar.info("💡 デフォルト設定を使用中")
    
    # キャラクターと属性の選択（量子重ねの効果を出すため）
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎭 キャラクター選択（オプション）")
    
    global SELECTED_ATTRIBUTE, SELECTED_CHARACTER, CHAR_MASTER
    
    # 属性の選択
    if LOADED_GODS:
        # 属性の一覧を取得
        attributes = set()
        for god in LOADED_GODS:
            attr = god.get("attribute", "")
            if attr:
                attributes.add(attr)
        
        if attributes:
            selected_attribute = st.sidebar.selectbox(
                "属性を選択（オプション）",
                ["選択しない"] + sorted(list(attributes)),
                help="属性を選択すると、その属性を持つキャラクターが選ばれやすくなります（量子重ねの効果）"
            )
            SELECTED_ATTRIBUTE = selected_attribute if selected_attribute != "選択しない" else None
            
            # 選択された属性のキャラクターを表示
            if SELECTED_ATTRIBUTE:
                matching_gods = [god for god in LOADED_GODS if god.get("attribute") == SELECTED_ATTRIBUTE]
                if matching_gods:
                    st.sidebar.info(f"**{SELECTED_ATTRIBUTE}**属性のキャラクター: {len(matching_gods)}体")
                    with st.sidebar.expander(f"📋 {SELECTED_ATTRIBUTE}属性のキャラクター一覧"):
                        for god in matching_gods:
                            st.write(f"- {god.get('emoji', '🔮')} {god.get('name', '')}")
        
        # キャラクターの直接選択
        character_names = [god.get("name", "") for god in LOADED_GODS if god.get("name")]
        if character_names:
            selected_character = st.sidebar.selectbox(
                "キャラクターを直接選択（オプション）",
                ["選択しない"] + character_names,
                help="キャラクターを直接選択すると、そのキャラクターが選ばれやすくなります（量子重ねの効果）"
            )
            SELECTED_CHARACTER = selected_character if selected_character != "選択しない" else None
    
    st.sidebar.markdown("---")
    app_mode = st.sidebar.selectbox(
        "実行モードを選択",
        ["基本デモ", "対話型量子神託", "言葉のエネルギー球体視覚化", "絵馬納め"]
    )
    
    if app_mode == "基本デモ":
        st.header("QUBO × 縁：基本デモ")
        st.markdown("基本的なQUBOモデルを使用した「縁」のデモンストレーション")
        
        # 基本デモでもキャラクター選択を反映
        st.info("💡 サイドバーでキャラクターや属性を選択すると、QUBOに反映されます")
        
        # ユーザー入力を受け付ける（オプション）
        user_input_basic = st.text_area(
            "今日の悩み・気持ちを入力してください（オプション）",
            placeholder="例：疲れていて決断ができない…",
            height=100,
            help="入力した文面を分析して、エネルギーが近い格言を選択します"
        )
        
        if st.button("実行"):
            # ユーザー入力からMoodを推定（入力がある場合）
            if user_input_basic and user_input_basic.strip():
                user_mood = infer_mood(user_input_basic.strip())
                context_text_for_basic = user_input_basic.strip()
            else:
                # 入力がない場合、デフォルトのMoodを使用
                user_mood = Mood(
                    fatigue=0.5,
                    anxiety=0.5,
                    curiosity=0.5,
                    loneliness=0.5,
                    decisiveness=0.5
                )
                context_text_for_basic = ""
            
            # 選択されたキャラクター/属性を反映したQUBOを生成
            Q = build_qubo_from_mood(
                user_mood,
                selected_attribute=SELECTED_ATTRIBUTE,
                selected_character=SELECTED_CHARACTER,
                char_master=CHAR_MASTER
            )
            
            # 階層構造を使用
            sols = solve_all(Q, use_hierarchical=True)
            
            # 結果表示
            st.subheader("低エネルギー上位（選ばれた格言の重なり）")
            if context_text_for_basic:
                st.caption(f"📝 入力文面: 「{context_text_for_basic}」")
                # 抽出されたキーワードを表示
                keywords_basic = extract_keywords_safe(context_text_for_basic, top_n=8)
                if keywords_basic:
                    st.caption(f"🔑 抽出されたキーワード: {', '.join(keywords_basic)}")
                else:
                    st.warning("⚠️ キーワードが抽出できませんでした。より詳しい文面を入力してください。")
            
            displayed_maxims_basic = []  # 既に表示した格言を記録（重複を避ける）
            for rank, (e, x) in enumerate(sols[:8], start=1):
                # 階層構造の場合、キャラクターと格言を取得
                if len(x) >= 32:
                    selected_god = get_selected_god_from_x(x, user_mood, use_hierarchical=True)
                    selected_vow_idx = get_selected_vow_from_x(x, use_hierarchical=True)
                    
                    # ユーザー入力文面を分析して、エネルギーが近い格言を選択
                    # キーワードを事前に抽出して、より効果的な選択を行う
                    context_for_selection = context_text_for_basic if context_text_for_basic else ""
                    picks = select_maxims_for_god(
                        selected_god, 
                        context_text=context_for_selection,  # ユーザー入力を使用
                        top_k=8,  # より多くの候補から選択（キーワードに基づいて絞り込む）
                        include_famous_quote=False,
                        selected_vow_index=selected_vow_idx,
                        exclude_maxims=displayed_maxims_basic  # 既に表示した格言を除外
                    )
                    
                    # 既に表示した格言を除外して、新しい格言のみを表示
                    new_picks = [p for p in picks if p not in displayed_maxims_basic]
                    if not new_picks and picks:
                        # 新しい格言がない場合、最初の格言を使用（重複を許容）
                        new_picks = picks[:1]
                    
                    # 表示する格言を記録
                    for pick in new_picks:
                        if pick not in displayed_maxims_basic:
                            displayed_maxims_basic.append(pick)
                    
                    if new_picks:
                        picks_str = " | ".join(new_picks[:2])
                    else:
                        picks_str = selected_god.get("maxim", "今この瞬間を大切に")
                else:
                    picks = [VARIABLES[i] for i,v in enumerate(x) if v==1]
                    if picks:
                        picks_str = " | ".join(picks[:2])  # 長いので最大2つまで
                        if len(picks) > 2:
                            picks_str += f" ...（他{len(picks)-2}つ）"
                    else:
                        picks_str = "今この瞬間を大切に"
                
                st.write(f"{rank}. E={e:>6.3f}")
                st.caption(f"   格言: {picks_str}")
            
            # エネルギー地形の可視化（簡略版）
            if len(sols) > 0:
                energies = [e for e, _ in sols[:20]]  # 上位20個のみ表示
                labels = [f"解{i+1}" for i in range(len(energies))]
                
                fig_bar = px.bar(
                    x=labels,
                    y=energies,
                    labels={'x': '解', 'y': 'エネルギー'},
                    title="Energy landscape（低いほど「縁が結ばれやすい候補」）"
                )
                fig_bar.update_xaxes(tickangle=-90)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # おみくじ（基本デモでも階層構造を使用）
            # より多くの候補から選択（ユーザー入力がある場合、より多様な結果を得るため）
            pool_size = 10 if context_text_for_basic else 6
            oracle_pool = sols[:pool_size]
            
            # 毎回異なる結果を得るため、ランダム要素を追加
            import time
            random.seed(int(time.time() * 1000) % 1000000)
            # プールを少しシャッフルして多様性を確保
            if len(oracle_pool) > 1:
                shuffled_pool = list(oracle_pool)
                random.shuffle(shuffled_pool[:min(5, len(shuffled_pool))])
                oracle_pool = shuffled_pool
            
            T = temperature_from_mood(user_mood, SELECTED_CHARACTER)
            e_pick, x_pick = boltzmann_sample(oracle_pool, T=T)
            card = oracle_card(e_pick, x_pick, mood=user_mood, use_hierarchical=True, context_text=context_text_for_basic)
            
            st.markdown("---")
            st.subheader("量子おみくじ（Quantum Oracle）")
            
            # 選ばれた神のキャラクターを表示
            # プルダウンで選択されたキャラクターを優先
            display_god = None
            if SELECTED_CHARACTER and LOADED_GODS:
                # 選択されたキャラクターをLOADED_GODSから検索
                for god in LOADED_GODS:
                    god_official_name = god.get("公式キャラ名", "")
                    god_name = god.get("name", "")
                    if SELECTED_CHARACTER == god_official_name or SELECTED_CHARACTER == god_name:
                        display_god = god
                        break
            
            # 選択されたキャラクターが見つからない場合、QUBOの結果から選ばれた神を使用
            if display_god is None and 'god' in card and card['god']:
                display_god = card['god']
            
            # キャラクターを表示
            if display_god:
                character_html = render_god_character(display_god, LOADED_GODS)
                st.components.v1.html(character_html, height=400)
            
            st.write(f"**エネルギー**: {card['energy']:.3f}")
            
            # 選ばれた格言と引用元を表示
            picks_display = []
            if card.get('picks') and len(card['picks']) > 0:
                for pick in card['picks']:
                    source_info = get_maxim_source(pick)
                    picks_display.append(f"{pick} *（{source_info['source']}）*")
            else:
                # 格言が空の場合、選ばれた神の格言を使用
                selected_god_from_card = card.get('god')
                if selected_god_from_card:
                    if selected_god_from_card.get("maxim"):
                        maxim = selected_god_from_card["maxim"]
                        source_info = get_maxim_source(maxim)
                        picks_display.append(f"{maxim} *（{source_info['source']}）*")
                    elif selected_god_from_card.get("description"):
                        desc = selected_god_from_card["description"]
                        picks_display.append(f"{desc} *（{selected_god_from_card.get('name', '神託')}）*")
            
            if not picks_display:
                picks_display.append("今この瞬間を大切に。すべては縁で繋がっている。 *（伝統的な教え）*")
            
            st.write(f"**選ばれた縁**:")
            for pick_text in picks_display:
                st.markdown(f"   - {pick_text}")
            
            st.write(f"**ことば**: 「{card.get('poem', '今この瞬間を大切に。')}」")
            st.write(f"**次の一歩**: {card.get('hint', '一歩ずつ進んでいきましょう。')}")
    
    elif app_mode == "対話型量子神託":
        st.header("対話型量子神託")
        st.markdown("あなたの悩み・気持ちを入力すると、パーソナライズされた「縁」を提示します")
        
        # LLM使用オプション
        col1, col2 = st.columns(2)
        with col1:
            use_llm = st.checkbox(
                "🤖 LLMでパーソナライズされた神託を生成",
                value=False,
                help="LLMを使用して、よりパーソナライズされた神託を生成します（無償で使用可能）"
            )
        with col2:
            if use_llm:
                llm_type = st.selectbox(
                    "LLMの種類",
                    ["huggingface", "ollama"],
                    help="Hugging Face: 無料API（推奨） / Ollama: ローカル実行（要インストール）"
                )
            else:
                llm_type = "huggingface"
        
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
                
                # Optunaを使った最適化（進捗表示付き）
                optuna_container = st.empty()
                sols, study = solve_all_with_optuna(Q_today, use_hierarchical=True, 
                                                     progress_container=optuna_container, n_trials=50)
                
                # Optunaの可視化（全ての可視化を表示）
                if study is not None and OPTUNA_AVAILABLE:
                    with st.expander("📊 QUBO最適化の詳細", expanded=False):
                        # タブで可視化を整理
                        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                            "📈 最適化履歴", 
                            "🎯 パラメータ重要度", 
                            "🔄 パラレルコーディネート",
                            "🗺️ 等高線",
                            "📊 スライス",
                            "⏱️ タイムライン"
                        ])
                        
                        with tab1:
                            try:
                                fig_history = plot_optimization_history(study)
                                st.plotly_chart(fig_history, use_container_width=True)
                                st.caption("最適化の進行状況を表示します。")
                            except Exception as e:
                                st.write(f"最適化履歴の可視化に失敗しました: {str(e)}")
                        
                        with tab2:
                            try:
                                fig_importance = plot_param_importances(study)
                                st.plotly_chart(fig_importance, use_container_width=True)
                                st.caption("各パラメータの重要度を表示します。")
                            except Exception as e:
                                st.write(f"パラメータ重要度の可視化に失敗しました: {str(e)}")
                        
                        with tab3:
                            try:
                                if len(study.trials) > 0:
                                    fig_parallel = plot_parallel_coordinate(study)
                                    st.plotly_chart(fig_parallel, use_container_width=True)
                                    st.caption("パラメータ間の関係を可視化します。")
                                else:
                                    st.info("パラレルコーディネートを表示するには、複数のトライアルが必要です。")
                            except Exception as e:
                                st.write(f"パラレルコーディネートの可視化に失敗しました: {str(e)}")
                        
                        with tab4:
                            try:
                                if len(study.trials) > 0:
                                    # 最初の2つのパラメータで等高線を表示
                                    params = list(study.trials[0].params.keys()) if study.trials else []
                                    if len(params) >= 2:
                                        fig_contour = plot_contour(study, params=[params[0], params[1]])
                                        st.plotly_chart(fig_contour, use_container_width=True)
                                        st.caption(f"パラメータ「{params[0]}」と「{params[1]}」の関係を等高線で表示します。")
                                    else:
                                        st.info("等高線を表示するには、少なくとも2つのパラメータが必要です。")
                                else:
                                    st.info("等高線を表示するには、複数のトライアルが必要です。")
                            except Exception as e:
                                st.write(f"等高線の可視化に失敗しました: {str(e)}")
                        
                        with tab5:
                            try:
                                if len(study.trials) > 0:
                                    fig_slice = plot_slice(study)
                                    st.plotly_chart(fig_slice, use_container_width=True)
                                    st.caption("各パラメータのスライスプロットを表示します。")
                                else:
                                    st.info("スライスプロットを表示するには、複数のトライアルが必要です。")
                            except Exception as e:
                                st.write(f"スライスプロットの可視化に失敗しました: {str(e)}")
                        
                        with tab6:
                            try:
                                if len(study.trials) > 0:
                                    fig_timeline = plot_timeline(study)
                                    st.plotly_chart(fig_timeline, use_container_width=True)
                                    st.caption("最適化のタイムラインを表示します。")
                                else:
                                    st.info("タイムラインを表示するには、複数のトライアルが必要です。")
                            except Exception as e:
                                st.write(f"タイムラインの可視化に失敗しました: {str(e)}")
                
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
                displayed_maxims = []  # 既に表示した格言を記録（重複を避ける）
                for rank, (e, x) in enumerate(sols[:3], start=1):
                    god_for_candidate = get_selected_god_from_x(x, m, use_hierarchical=True)
                    selected_vow_idx = get_selected_vow_from_x(x, use_hierarchical=True)
                    
                    # 既に表示した格言を除外
                    picks = select_maxims_for_god(
                        god_for_candidate, 
                        context_text=user_text, 
                        top_k=3,  # より多くの候補から選択
                        include_famous_quote=False,
                        selected_vow_index=selected_vow_idx,
                        exclude_maxims=displayed_maxims  # 既に表示した格言を除外
                    )
                    
                    # 既に表示した格言を除外して、新しい格言のみを表示
                    new_picks = [p for p in picks if p not in displayed_maxims]
                    if not new_picks and picks:
                        # 新しい格言がない場合、最初の格言を使用（重複を許容）
                        new_picks = picks[:1]
                    
                    # 表示する格言を記録
                    for pick in new_picks:
                        if pick not in displayed_maxims:
                            displayed_maxims.append(pick)
                    
                    st.write(f"**{rank}. E={e:.3f}**")
                    for pick in new_picks[:2]:  # 最大2つまで表示
                        source_info = get_maxim_source(pick)
                        st.write(f"   • {pick}")
                        st.caption(f"     *出典: {source_info['source']} - {source_info['origin']}*")
                
                # おみくじ（Moodに応じて変化）
                pool = sols[:6]
                T = temperature_from_mood(m, SELECTED_CHARACTER)
                e_pick, x_pick = boltzmann_sample(pool, T=T)
                
                # デバッグ: 解ベクトルの内容を確認
                if len(x_pick) >= 32:
                    # キャラクター変数（20〜31）が選ばれているか確認
                    c_start = 20
                    selected_char_indices = [i for i in range(c_start, min(c_start + 12, len(x_pick))) if i < len(x_pick) and x_pick[i] == 1]
                    if not selected_char_indices:
                        # キャラクターが選ばれていない場合、Moodから選択
                        st.warning("⚠️ キャラクター変数が選ばれていません。Moodから選択します。")
                
                card = oracle_card(e_pick, x_pick, mood=m, use_hierarchical=True, context_text=user_text, use_llm=use_llm, llm_type=llm_type)  # Moodを渡す
                
                st.markdown("---")
                st.subheader("量子おみくじ（Quantum Oracle）")
                
                # 選ばれた神のキャラクターを表示
                if 'god' in card and card['god']:
                    selected_god = card['god']
                    character_html = render_god_character(selected_god, LOADED_GODS)
                    st.components.v1.html(character_html, height=400)
                
                # 選ばれた格言の引用元情報を収集
                sources_text = []
                if card.get('picks') and len(card['picks']) > 0:
                    for pick in card['picks']:
                        source_info = get_maxim_source(pick)
                        sources_text.append(f"- {pick}\n  *出典: {source_info['source']} - {source_info['origin']}*")
                else:
                    # 格言が空の場合、選ばれた神の格言を使用
                    selected_god_from_card = card.get('god')
                    if selected_god_from_card and selected_god_from_card.get("maxim"):
                        maxim = selected_god_from_card["maxim"]
                        source_info = get_maxim_source(maxim)
                        sources_text.append(f"- {maxim}\n  *出典: {source_info['source']} - {source_info['origin']}*")
                    elif selected_god_from_card and selected_god_from_card.get("description"):
                        desc = selected_god_from_card["description"]
                        sources_text.append(f"- {desc}\n  *出典: {selected_god_from_card.get('name', '神託')}*")
                    else:
                        sources_text.append("- 今この瞬間を大切に。すべては縁で繋がっている。\n  *出典: 伝統的な教え*")
                
                            # LLM生成の神託を「選ばれた縁」に統合
                if card.get('llm_oracle') and use_llm and card['llm_oracle'].strip():
                    # LLM生成の神託を最初に追加
                    llm_text = card['llm_oracle'].strip()
                    sources_text.insert(0, f"🤖 {llm_text}\n  *出典: LLM生成 - パーソナライズされた神託*")
                elif use_llm:
                    # LLM生成に失敗した場合、フォールバックメッセージを追加
                    sources_text.insert(0, f"💭 LLM生成は現在利用できません。格言ベースの神託を表示しています。\n  *出典: 伝統的な教え*")
                
                st.info(f"""
**エネルギー**: {card['energy']:.3f}

**選ばれた縁**:
{chr(10).join(sources_text) if sources_text else "- 今この瞬間を大切に。すべては縁で繋がっている。"}

**ことば**:
「{card.get('poem', '今この瞬間を大切に。すべては縁で繋がっている。')}」

**次の一歩**:
{card.get('hint', '一歩ずつ進んでいきましょう。')}
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
                    
                    # キーワードに基づいて格言を選択（重複を避ける）
                    if 'recent_maxims' not in st.session_state:
                        st.session_state.recent_maxims = []
                    exclude_quotes = set(st.session_state.recent_maxims[-10:])
                    quote_text = select_relevant_quote(keywords, exclude_quotes=exclude_quotes)
                    
                    # 選択した格言を履歴に追加
                    if quote_text and quote_text not in st.session_state.recent_maxims:
                        st.session_state.recent_maxims.append(quote_text)
                        if len(st.session_state.recent_maxims) > 20:
                            st.session_state.recent_maxims.pop(0)
                    
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
                
                # ネットワーク構築（キャラクター選択を反映）
                network = build_word_network(
                    keywords, 
                    GLOBAL_WORDS_DATABASE, 
                    n_neighbors=20,
                    selected_character=SELECTED_CHARACTER,
                    selected_attribute=SELECTED_ATTRIBUTE,
                    char_master=CHAR_MASTER
                )
                
                # 3D配置
                center_indices = [i for i, word in enumerate(network['words']) if word in keywords]
                positions = place_words_on_sphere(len(network['words']), center_indices)
                
                # 3D可視化
                fig = create_3d_network_plot(network, positions, center_indices)
                st.plotly_chart(fig, use_container_width=True)
                
                # 格言を表示（重複を避ける）
                if 'recent_maxims' not in st.session_state:
                    st.session_state.recent_maxims = []
                exclude_quotes = set(st.session_state.recent_maxims[-10:])
                quote_text = select_relevant_quote(keywords, exclude_quotes=exclude_quotes)
                
                # 選択した格言を履歴に追加
                if quote_text and quote_text not in st.session_state.recent_maxims:
                    st.session_state.recent_maxims.append(quote_text)
                    if len(st.session_state.recent_maxims) > 20:
                        st.session_state.recent_maxims.pop(0)
                
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
        
        # LLM使用オプション
        col1, col2 = st.columns(2)
        with col1:
            use_llm_ema = st.checkbox(
                "🤖 LLMでパーソナライズされた神託を生成",
                value=False,
                help="LLMを使用して、よりパーソナライズされた神託を生成します（無償で使用可能）"
            )
        with col2:
            if use_llm_ema:
                llm_type_ema = st.selectbox(
                    "LLMの種類",
                    ["huggingface", "ollama"],
                    index=0,
                    help="Hugging Face（無償）またはOllama（ローカル）を選択"
                )
            else:
                llm_type_ema = "huggingface"
        
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
                
                # 願いを分析して神託を生成
                m = infer_mood(ema_text)
                Q_today = build_qubo_from_mood(m)
                
                # Optunaを使った最適化（進捗表示付き）
                optuna_container = st.empty()
                sols, study = solve_all_with_optuna(Q_today, use_hierarchical=True, 
                                                     progress_container=optuna_container, n_trials=50)
                
                # Optunaの可視化（全ての可視化を表示）
                if study is not None and OPTUNA_AVAILABLE:
                    with st.expander("📊 QUBO最適化の詳細", expanded=False):
                        # タブで可視化を整理
                        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                            "📈 最適化履歴", 
                            "🎯 パラメータ重要度", 
                            "🔄 パラレルコーディネート",
                            "🗺️ 等高線",
                            "📊 スライス",
                            "⏱️ タイムライン"
                        ])
                        
                        with tab1:
                            try:
                                fig_history = plot_optimization_history(study)
                                st.plotly_chart(fig_history, use_container_width=True)
                                st.caption("最適化の進行状況を表示します。")
                            except Exception as e:
                                st.write(f"最適化履歴の可視化に失敗しました: {str(e)}")
                        
                        with tab2:
                            try:
                                fig_importance = plot_param_importances(study)
                                st.plotly_chart(fig_importance, use_container_width=True)
                                st.caption("各パラメータの重要度を表示します。")
                            except Exception as e:
                                st.write(f"パラメータ重要度の可視化に失敗しました: {str(e)}")
                        
                        with tab3:
                            try:
                                if len(study.trials) > 0:
                                    fig_parallel = plot_parallel_coordinate(study)
                                    st.plotly_chart(fig_parallel, use_container_width=True)
                                    st.caption("パラメータ間の関係を可視化します。")
                                else:
                                    st.info("パラレルコーディネートを表示するには、複数のトライアルが必要です。")
                            except Exception as e:
                                st.write(f"パラレルコーディネートの可視化に失敗しました: {str(e)}")
                        
                        with tab4:
                            try:
                                if len(study.trials) > 0:
                                    # 最初の2つのパラメータで等高線を表示
                                    params = list(study.trials[0].params.keys()) if study.trials else []
                                    if len(params) >= 2:
                                        fig_contour = plot_contour(study, params=[params[0], params[1]])
                                        st.plotly_chart(fig_contour, use_container_width=True)
                                        st.caption(f"パラメータ「{params[0]}」と「{params[1]}」の関係を等高線で表示します。")
                                    else:
                                        st.info("等高線を表示するには、少なくとも2つのパラメータが必要です。")
                                else:
                                    st.info("等高線を表示するには、複数のトライアルが必要です。")
                            except Exception as e:
                                st.write(f"等高線の可視化に失敗しました: {str(e)}")
                        
                        with tab5:
                            try:
                                if len(study.trials) > 0:
                                    fig_slice = plot_slice(study)
                                    st.plotly_chart(fig_slice, use_container_width=True)
                                    st.caption("各パラメータのスライスプロットを表示します。")
                                else:
                                    st.info("スライスプロットを表示するには、複数のトライアルが必要です。")
                            except Exception as e:
                                st.write(f"スライスプロットの可視化に失敗しました: {str(e)}")
                        
                        with tab6:
                            try:
                                if len(study.trials) > 0:
                                    fig_timeline = plot_timeline(study)
                                    st.plotly_chart(fig_timeline, use_container_width=True)
                                    st.caption("最適化のタイムラインを表示します。")
                                else:
                                    st.info("タイムラインを表示するには、複数のトライアルが必要です。")
                            except Exception as e:
                                st.write(f"タイムラインの可視化に失敗しました: {str(e)}")
                
                # おみくじ（Moodに応じて変化）
                pool = sols[:6]
                T = temperature_from_mood(m, SELECTED_CHARACTER)
                e_pick, x_pick = boltzmann_sample(pool, T=T)
                
                # デバッグ: 解ベクトルの内容を確認
                if len(x_pick) >= 32:
                    # キャラクター変数（20〜31）が選ばれているか確認
                    c_start = 20
                    selected_char_indices = [i for i in range(c_start, min(c_start + 12, len(x_pick))) if i < len(x_pick) and x_pick[i] == 1]
                    if not selected_char_indices:
                        # キャラクターが選ばれていない場合、Moodから選択
                        st.warning("⚠️ キャラクター変数が選ばれていません。Moodから選択します。")
                
                # キーワード抽出の結果を表示（デバッグ用）
                keywords_ema = extract_keywords_safe(ema_text, top_n=10)
                if keywords_ema:
                    with st.expander("🔑 抽出されたキーワード", expanded=False):
                        st.write(f"**キーワード**: {', '.join(keywords_ema)}")
                
                card = oracle_card(e_pick, x_pick, mood=m, use_hierarchical=True, context_text=ema_text, use_llm=use_llm_ema, llm_type=llm_type_ema)
                
                # 選ばれた神を取得
                selected_god = card['god'] if 'god' in card else select_god_from_mood(m)
                
                # 待機演出（Streamlitではtime.sleep()は非推奨のため、spinnerのみ使用）
                # time.sleep()は削除（Streamlitの非同期処理と競合するため）
                
                # 選ばれた神のキャラクターを表示
                character_html = render_god_character(selected_god, LOADED_GODS)
                st.components.v1.html(character_html, height=400)
                
                st.markdown("---")
                st.subheader(f"🔮 {selected_god['name']}からの神託")
                
                # 選ばれた格言の引用元情報を収集
                sources_text = []
                if card.get('picks') and len(card['picks']) > 0:
                    for pick in card['picks']:
                        source_info = get_maxim_source(pick)
                        sources_text.append(f"- {pick}\n  *出典: {source_info['source']} - {source_info['origin']}*")
                else:
                    # 格言が空の場合、選ばれた神の格言を使用
                    if selected_god and selected_god.get("maxim"):
                        maxim = selected_god["maxim"]
                        source_info = get_maxim_source(maxim)
                        sources_text.append(f"- {maxim}\n  *出典: {source_info['source']} - {source_info['origin']}*")
                    elif selected_god and selected_god.get("description"):
                        desc = selected_god["description"]
                        sources_text.append(f"- {desc}\n  *出典: {selected_god.get('name', '神託')}*")
                    else:
                        sources_text.append("- 今この瞬間を大切に。すべては縁で繋がっている。\n  *出典: 伝統的な教え*")
                
                # LLM生成の神託を「選ばれた縁」に統合
                if card.get('llm_oracle') and use_llm_ema and card['llm_oracle'].strip():
                    # LLM生成の神託を最初に追加
                    llm_text = card['llm_oracle'].strip()
                    sources_text.insert(0, f"🤖 {llm_text}\n  *出典: LLM生成 - パーソナライズされた神託*")
                elif use_llm_ema:
                    # LLM生成に失敗した場合、フォールバックメッセージを追加
                    sources_text.insert(0, f"💭 LLM生成は現在利用できません。格言ベースの神託を表示しています。\n  *出典: 伝統的な教え*")
                
                # 神託カードを美しく表示
                st.info(f"""
**エネルギー**: {card['energy']:.3f}

**選ばれた縁**:
{chr(10).join(sources_text) if sources_text else "- 今この瞬間を大切に。すべては縁で繋がっている。"}

**ことば**:
「{card.get('poem', '今この瞬間を大切に。すべては縁で繋がっている。')}」

**次の一歩**:
{card.get('hint', '一歩ずつ進んでいきましょう。')}
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
