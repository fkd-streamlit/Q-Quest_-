# フォルダー構造ガイド

## GitHub用推奨フォルダー構造

```
Q-Quest_量子神託/
│
├── app.py                    # ⭐必須: メインアプリケーション
├── requirements.txt          # ⭐必須: 依存パッケージ
├── README.md                 # ⭐推奨: プロジェクト説明
├── README_GITHUB.md          # GitHub用README
├── .gitignore               # ⭐推奨: Git除外設定
│
├── .streamlit/              # ⭐推奨: Streamlit設定
│   └── config.toml         # テーマ設定など
│
├── config/                  # 設定ファイル（オプション）
│   └── README.md           # 設定ファイルの説明
│
├── docs/                    # ドキュメント（オプション）
│   ├── PROJECT_VISION.md
│   ├── DEVELOPMENT_ROADMAP.md
│   ├── README_EN.md
│   └── CULTURAL_ELEMENTS.md
│
└── [その他のドキュメント]    # ガイドファイルなど
    ├── EXCEL_SETUP_GUIDE.md
    ├── 格言追加ガイド.md
    ├── 設計の流れ.md
    ├── GITHUB_SETUP.md
    └── STREAMLIT_CLOUD_DEPLOY.md
```

## 必須ファイル（Streamlit Cloud）

### 1. app.py
メインアプリケーション。Streamlit Cloudはこのファイルを実行します。

### 2. requirements.txt
依存パッケージのリスト。Streamlit Cloudが自動的にインストールします。

### 3. README.md
プロジェクトの説明。GitHubで最初に表示されるファイル。

## 推奨ファイル

### .streamlit/config.toml
Streamlitの設定ファイル（テーマ、ページ設定など）。

### .gitignore
Gitで管理しないファイルを指定。

## オプションフォルダー

### config/
- **用途**: 設定ファイル（Excel）の説明
- **GitHubに含める**: `README.md` のみ
- **含めない**: Excelファイル（個人用設定のため）

### docs/
- **用途**: プロジェクトのドキュメント
- **GitHubに含める**: すべてのドキュメント

## GitHubに含めないファイル

以下のファイルは `.gitignore` で除外されます：

- **Excelファイル** (`*.xlsx`, `*.xls`) - 個人用設定
- **Excel一時ファイル** (`~$*.xlsx`) - 自動生成
- **バッチファイル** (`*.bat`) - Windows専用
- **Optunaデータベース** (`*.db`) - 自動生成
- **一時ファイル** (`*.tmp`, `*.log`, `*.bak`)
- **Jupyter Notebook** (`*.ipynb`) - オプション（含めたい場合は `.gitignore` を編集）

## Streamlit Cloudでの動作

1. **設定ファイル**: ユーザーがアプリのサイドバーからExcelファイルをアップロード
2. **デフォルト設定**: 設定ファイルがない場合、アプリ内のデフォルト設定を使用
3. **データの永続化**: アップロードされたファイルはセッションごとに保持

## 整理手順

### 1. フォルダー整理スクリプトを実行

```bash
python scripts/organize_files.py
```

### 2. 手動で整理

以下のファイルを適切な場所に移動：

- **ドキュメント**: `docs/` フォルダーに移動（既に存在する場合はそのまま）
- **設定ファイル**: `config/` フォルダーに移動（ただしGitHubには含めない）

### 3. .gitignoreを確認

`.gitignore` に必要な除外設定が含まれているか確認してください。

## 最小構成（Streamlit Cloud）

Streamlit Cloudで公開する最小構成：

```
Q-Quest_量子神託/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

これだけで動作します！設定ファイルはアプリからアップロードできます。