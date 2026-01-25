# クイックスタートガイド

## プロジェクトのセットアップ

### 1. Notebookファイルのコピー

現在、Notebookファイル（`Q_QUEST_量子神託.ipynb`）は `C:\Users\FMV\Downloads\` にあります。
このファイルをワークスペース（`C:\Users\FMV\Desktop\Q-Quest_量子神託\`）にコピーしてください。

**方法1: エクスプローラーでコピー**
1. エクスプローラーで `C:\Users\FMV\Downloads\Q_QUEST_量子神託.ipynb` を開く
2. ファイルをコピー（Ctrl+C）
3. `C:\Users\FMV\Desktop\Q-Quest_量子神託\` にペースト（Ctrl+V）

**方法2: Pythonスクリプトを使用**
```bash
python setup_notebook.py
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

必要なパッケージ:
- numpy>=1.20.0
- matplotlib>=3.3.0
- jupyter>=1.0.0
- ipykernel>=6.0.0

### 3. Jupyter Notebookの起動

```bash
jupyter notebook Q_QUEST_量子神託.ipynb
```

または

```bash
jupyter notebook
```
（起動後、ブラウザで `Q_QUEST_量子神託.ipynb` を開く）

## Notebookの使い方

### Cell 0: 基本デモ

- QUBOによる「縁」のモデリング
- エネルギー地形の可視化
- 縁のネットワーク表示
- 量子おみくじの生成

**実行方法**: Cell 0を選択してShift+Enter

### Cell 2: 対話型量子神託

- ユーザーの悩み・気持ちをテキスト入力
- 心の傾き（Mood）の自動推定
- パーソナライズされた「縁」の提示

**実行方法**: 
1. Cell 2を選択してShift+Enter
2. プロンプトが表示されたら、悩みや気持ちを入力
3. 例: 「疲れていて決断ができない」「眠い。休みたい。」

## トラブルシューティング

### 日本語フォントの警告

Colab環境では日本語フォントが自動インストールされますが、
ローカル環境では警告が出る場合があります。

**対処法**:
- 警告は表示に影響しませんが、日本語を正しく表示したい場合は
  日本語フォントをインストールしてください

### モジュールが見つからないエラー

```bash
pip install -r requirements.txt
```
を実行してください。

## 次のステップ

1. **README.md**を読んでプロジェクトの概要を理解
2. **docs/PROJECT_VISION.md**でビジョンを確認
3. **docs/CULTURAL_ELEMENTS.md**で文化的背景を学習
4. **docs/DEVELOPMENT_ROADMAP.md**で今後の展開を確認

## プロジェクト構造

```
Q-Quest_量子神託/
├── README.md                      # プロジェクト概要（日本語）
├── README_EN.md                   # Project Overview (English)
├── CONTRIBUTING.md                # 貢献ガイドライン
├── QUICK_START.md                 # このファイル
├── requirements.txt               # 依存パッケージ
├── setup_notebook.py              # Notebookコピー用スクリプト
├── Q_QUEST_量子神託.ipynb         # メインのNotebook（手動でコピー）
└── docs/
    ├── PROJECT_VISION.md          # プロジェクトビジョン
    ├── CULTURAL_ELEMENTS.md       # 文化的要素の説明
    ├── DEVELOPMENT_ROADMAP.md     # 開発ロードマップ
    └── README_EN.md               # 英語版README
```

---

**Q-Quest 量子神託** - 量子技術でつなぐ「縁」

