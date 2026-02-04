# Q-Quest 量子神託 - GitHub用README

## クイックスタート

### 1. リポジトリのクローン

```bash
git clone https://github.com/YOUR_USERNAME/Q-Quest_量子神託.git
cd Q-Quest_量子神託
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 3. 設定ファイルの準備（オプション）

`config/` フォルダーに以下のExcelファイルを配置してください：

- `akiba12_character_list.xlsx` - 12神の基本情報
- `格言.xlsx` - 格言データ（オプション）
- `akiba12_character_to_vow_K.xlsx` - k行列（12×12）
- `akiba12_character_to_axis_L.xlsx` - l行列（12×4）
- `sense_to_vow_initial_filled_from_user.xlsx` - sense_to_vow行列（8×12）

**注意**: 設定ファイルがない場合、デフォルト設定が使用されます。

### 4. アプリの起動

```bash
streamlit run app.py
```

または

```bash
python scripts/start_app.bat  # Windows
```

## 主な機能

### 1. Optunaを使ったQUBO最適化の可視化

QUBO最適化の実行中に、Optunaを使って進捗を可視化します：

- **最適化履歴**: 試行回数とエネルギーの推移
- **パラメータ重要度**: どのパラメータが最適化に重要か

### 2. 5つのExcelファイルをまとめて読み込み

サイドバーから「5つのファイル（推奨）」を選択すると、すべての設定ファイルを一度に読み込めます。

### 3. 階層構造QUBO

- **感覚層** (8変数): 迷い/焦り/静けさ/内省/行動/つながり/挑戦/待つ
- **誓願層** (12変数): 12の誓願（one-hot制約）
- **キャラクター層** (12変数): 12神（one-hot制約）

## フォルダー構造

```
Q-Quest_量子神託/
├── app.py                    # メインアプリケーション
├── requirements.txt          # 依存パッケージ
├── README.md                 # プロジェクト説明
├── .gitignore               # Git除外設定
│
├── config/                   # 設定ファイル（Excel）
│   └── *.xlsx
│
├── scripts/                  # スクリプト類
│   ├── start_app.bat
│   ├── organize_files.py
│   └── ...
│
└── docs/                     # ドキュメント
    └── ...
```

## トラブルシューティング

### Optunaがインストールされていない

```bash
pip install optuna>=3.0.0
```

### Excelファイルが読み込めない

- ファイルパスを確認してください
- ファイル名が正しいか確認してください
- Excelファイルが開かれていないか確認してください

## ライセンス

[ライセンス情報を追加してください]

## 貢献

貢献を歓迎します！詳細は `CONTRIBUTING.md` を参照してください。