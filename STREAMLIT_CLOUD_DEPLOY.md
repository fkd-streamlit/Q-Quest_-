# Streamlit Cloud デプロイガイド

## 必要なフォルダー構造

Streamlit Cloudで公開するために、以下のフォルダー構造を推奨します：

```
Q-Quest_量子神託/
├── app.py                    # メインアプリケーション（必須）
├── requirements.txt          # 依存パッケージ（必須）
├── README.md                 # プロジェクト説明（推奨）
├── .gitignore               # Git除外設定（推奨）
│
├── .streamlit/              # Streamlit設定（推奨）
│   └── config.toml         # テーマ設定など
│
├── config/                  # 設定ファイル（オプション）
│   └── README.md           # 設定ファイルの説明
│
└── docs/                    # ドキュメント（オプション）
    ├── PROJECT_VISION.md
    ├── DEVELOPMENT_ROADMAP.md
    └── ...
```

## 必須ファイル

### 1. app.py
メインアプリケーション。Streamlit Cloudはこのファイルを実行します。

### 2. requirements.txt
依存パッケージのリスト。Streamlit Cloudは自動的にインストールします。

**現在の内容：**
```
numpy>=1.20.0
matplotlib>=3.3.0
jupyter>=1.0.0
ipykernel>=6.0.0
streamlit>=1.28.0
plotly>=5.17.0
pandas>=1.3.0
openpyxl>=3.0.0
optuna>=3.0.0
```

### 3. .streamlit/config.toml（推奨）
Streamlitの設定ファイル。テーマやページ設定など。

## オプションファイル

### config/ フォルダー
- Excel設定ファイルは**GitHubには含めません**（個人用設定のため）
- 代わりに、ユーザーがStreamlitアプリからアップロードできるようにします
- `config/README.md` に設定ファイルの説明を記載

### docs/ フォルダー
- プロジェクトのドキュメント
- 技術的な詳細や設計思想など

## GitHubに含めないファイル

以下のファイルは `.gitignore` で除外します：

- Excelファイル（`*.xlsx`, `*.xls`）- 個人用設定のため
- Excel一時ファイル（`~$*.xlsx`）
- バッチファイル（`*.bat`）- Windows専用のため
- Optunaデータベース（`*.db`）
- 一時ファイル（`*.tmp`, `*.log`, `*.bak`）

## Streamlit Cloudでの動作

1. **設定ファイルのアップロード**: ユーザーがStreamlitアプリのサイドバーからExcelファイルをアップロード
2. **デフォルト設定**: 設定ファイルがない場合、アプリ内のデフォルト設定を使用
3. **データの永続化**: Streamlit Cloudでは、アップロードされたファイルはセッションごとに保持されます

## デプロイ手順

### 1. GitHubリポジトリの準備

```bash
# リポジトリを初期化
git init

# ファイルを追加
git add app.py requirements.txt README.md .gitignore .streamlit/

# コミット
git commit -m "Initial commit: Q-Quest 量子神託"

# GitHubリポジトリを作成してプッシュ
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### 2. Streamlit Cloudでデプロイ

1. [Streamlit Cloud](https://streamlit.io/cloud) にアクセス
2. GitHubアカウントでログイン
3. 「New app」をクリック
4. リポジトリを選択
5. メインファイル: `app.py`
6. ブランチ: `main`
7. 「Deploy」をクリック

### 3. デプロイ後の確認

- アプリが正常に起動するか確認
- サイドバーからExcelファイルをアップロードして動作確認
- エラーがないか確認

## トラブルシューティング

### 依存パッケージのエラー

`requirements.txt` に必要なパッケージがすべて含まれているか確認してください。

### Excelファイルの読み込みエラー

- ファイルサイズが200MB以下であることを確認
- ファイル形式が正しいか確認（.xlsx または .xls）

### Optunaのエラー

Optunaがインストールされていない場合、通常の最適化にフォールバックします。エラーは表示されません。

## 注意事項

1. **機密情報**: APIキーや個人情報を含むファイルは絶対にGitHubにアップロードしないでください
2. **大容量ファイル**: Excelファイルが大きい場合は、Git LFSを使用することを検討してください
3. **ライセンス**: `LICENSE` ファイルを追加することを推奨します