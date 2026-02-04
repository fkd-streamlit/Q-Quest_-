# GitHub用セットアップガイド

## フォルダー整理

プロジェクトをGitHubにアップロードする前に、フォルダーを整理します。

### 推奨フォルダー構造

```
Q-Quest_量子神託/
├── app.py                    # メインアプリケーション
├── requirements.txt          # 依存パッケージ
├── README.md                 # プロジェクト説明
├── .gitignore               # Git除外設定
│
├── config/                   # 設定ファイル（Excel）
│   ├── akiba12_character_list.xlsx
│   ├── 格言.xlsx
│   ├── akiba12_character_to_vow_K.xlsx
│   ├── akiba12_character_to_axis_L.xlsx
│   └── sense_to_vow_initial_filled_from_user.xlsx
│
├── scripts/                  # スクリプト類
│   ├── start_app.bat
│   ├── install_requirements.bat
│   ├── deploy_to_github.bat
│   ├── organize_files.py
│   └── ...
│
├── docs/                     # ドキュメント
│   ├── PROJECT_VISION.md
│   ├── DEVELOPMENT_ROADMAP.md
│   └── ...
│
└── .streamlit/              # Streamlit設定
    └── config.toml
```

## ファイル整理手順

### 1. フォルダー整理スクリプトを実行

```bash
python scripts/organize_files.py
```

このスクリプトは以下を実行します：
- `config/` フォルダーにExcelファイルをコピー
- `scripts/` フォルダーにバッチファイルとPythonスクリプトをコピー

### 2. 手動で整理するファイル

以下のファイルは手動で整理してください：

- **ドキュメント**: `docs/` フォルダーに移動
  - `EXCEL_SETUP_GUIDE.md`
  - `格言追加ガイド.md`
  - `設計の流れ.md`
  - `sense_to_vow行列の説明.md`
  - その他のガイドファイル

- **一時ファイル**: 削除または `.gitignore` に追加
  - `~$*.xlsx` (Excel一時ファイル)
  - `*.tmp`
  - `*.log`

## .gitignore の設定

以下のファイルはGitHubにアップロードしないように設定されています：

- Excel一時ファイル (`~$*.xlsx`)
- Optunaデータベース (`*.db`)
- 個人用設定ファイル (`config/*.xlsx` - ただし例ファイルは含める)
- 一時ファイル (`*.tmp`, `*.log`, `*.bak`)

## GitHubへのアップロード

### 1. Gitリポジトリの初期化（初回のみ）

```bash
git init
git add .
git commit -m "Initial commit: Q-Quest 量子神託"
```

### 2. GitHubリポジトリを作成

1. GitHubで新しいリポジトリを作成
2. リモートリポジトリを追加：

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### 3. 更新をプッシュ

```bash
git add .
git commit -m "Update: フォルダー整理とOptuna統合"
git push
```

## 設定ファイルの管理

### 個人用設定ファイル

`config/` フォルダー内のExcelファイルは個人用設定のため、GitHubには含めません。

代わりに、以下のテンプレートファイルを用意することを推奨します：

- `config/akiba12_character_list.xlsx.example`
- `config/akiba12_character_to_vow_K.xlsx.example`
- など

### デフォルト設定

アプリは、設定ファイルがアップロードされていない場合、デフォルト設定を使用します。

## 注意事項

1. **機密情報**: APIキーや個人情報を含むファイルは `.gitignore` に追加してください
2. **大容量ファイル**: Excelファイルが大きい場合は、Git LFSを使用することを検討してください
3. **ライセンス**: `LICENSE` ファイルを追加することを推奨します

## トラブルシューティング

### Excelファイルが読み込めない

- ファイルパスを確認してください
- `config/` フォルダーにファイルが存在するか確認してください
- ファイル名が正しいか確認してください

### Optunaがインストールされていない

```bash
pip install optuna>=3.0.0
```

### その他の問題

`TROUBLESHOOTING.md` を参照してください。