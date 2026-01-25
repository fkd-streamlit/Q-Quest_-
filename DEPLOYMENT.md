# デプロイメントガイド

## GitHub + Streamlit Cloud で公開する方法

### 1. GitHubリポジトリの作成

1. [GitHub](https://github.com) にログイン
2. 右上の「+」→「New repository」をクリック
3. リポジトリ名を入力（例：`Q-Quest-Quantum-Oracle`）
4. 「Public」または「Private」を選択
5. 「Create repository」をクリック

### 2. ローカルでGitを初期化

```bash
cd C:\Users\FMV\Desktop\Q-Quest_量子神託

# Gitリポジトリを初期化
git init

# すべてのファイルを追加
git add .

# コミット
git commit -m "Initial commit: Q-Quest 量子神託 with Streamlit"

# リモートリポジトリを追加（<YOUR_REPO_URL>を実際のURLに置き換え）
git remote add origin <YOUR_REPO_URL>

# メインブランチにプッシュ
git branch -M main
git push -u origin main
```

### 3. Streamlit Cloudでデプロイ

1. [Streamlit Cloud](https://streamlit.io/cloud) にアクセス
2. 「Sign in」をクリックしてGitHubアカウントでログイン
3. 「New app」ボタンをクリック
4. 以下を設定：
   - **Repository**: 作成したGitHubリポジトリを選択
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. 「Deploy」をクリック

数分でデプロイが完了し、公開URLが生成されます！

### 4. URLの共有

デプロイが完了すると、以下のようなURLが生成されます：
```
https://your-username-your-app-name.streamlit.app
```

このURLを共有するだけで、世界中の誰でもアプリにアクセスできます。

## ローカルでの実行

### 方法1: Streamlitアプリ

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 方法2: Jupyter Notebook

```bash
jupyter notebook Q_QUEST_量子神託.ipynb
```

## ファイル構成

デプロイに必要なファイル：

```
Q-Quest_量子神託/
├── app.py                    # Streamlitアプリ（メインファイル）
├── requirements.txt          # 依存パッケージ
├── .streamlit/
│   └── config.toml          # Streamlit設定
├── Q_QUEST_量子神託.ipynb    # Jupyter Notebook（開発用）
└── README.md                # プロジェクト説明
```

## トラブルシューティング

### デプロイが失敗する場合

1. `requirements.txt` にすべての依存パッケージが含まれているか確認
2. `app.py` に構文エラーがないか確認
3. Streamlit Cloudのログを確認（エラーメッセージが表示されます）

### 日本語が表示されない場合

- Streamlitアプリでは、Plotlyが自動的に日本語フォントを使用します
- ブラウザのフォント設定を確認してください

## 次のステップ

- カスタムドメインの設定
- 認証機能の追加（Streamlit Authenticatorなど）
- データベースとの連携
- API化

詳しくはStreamlitのドキュメントを参照：
https://docs.streamlit.io/
