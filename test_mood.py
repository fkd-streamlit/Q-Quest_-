"""Mood推定のテストスクリプト"""
import sys
sys.path.insert(0, '.')

from app import infer_mood, KEYWORDS

# テストケース
test_text = "人の言葉が気になり、自信が持てない。"

print("=" * 50)
print("Mood推定テスト")
print("=" * 50)
print(f"入力テキスト: 「{test_text}」")
print()

m = infer_mood(test_text)

print("推定結果:")
print(f"  疲れ: {m.fatigue:.2f}")
print(f"  不安/焦り: {m.anxiety:.2f}")
print(f"  好奇心: {m.curiosity:.2f}")
print(f"  孤独: {m.loneliness:.2f}")
print(f"  決断: {m.decisiveness:.2f}")
print()

# キーワードマッチングの詳細を表示
print("検出されたキーワード:")
for category, keywords in KEYWORDS.items():
    matches = [kw for kw in keywords if kw in test_text]
    if matches:
        print(f"  {category}: {matches}")
