# GitHub + Streamlit Cloud 公開ガイド

このガイドでは、Q-Quest 量子神託アプリを**無料で**GitHubに公開し、Streamlit Cloudでデプロイする手順を説明します。

## ✅ Streamlit Cloudの無料プランについて

- **完全無料**で利用可能
- **制限なし**でアプリをデプロイ可能
- GitHubリポジトリと連携（自動デプロイ）
- 公開URLを無制限に共有可能

## 📋 事前準備

### 必要なもの
- GitHubアカウント（無料）
- 本プロジェクトのファイル

## 🚀 デプロイ手順

### ステップ1: GitHubリポジトリを作成

1. [GitHub](https://github.com) にログイン
2. 右上の「**+**」→「**New repository**」をクリック
3. 以下を設定：
   - **Repository name**: `Q-Quest-Quantum-Oracle`（お好きな名前でOK）
   - **Description**: `Human-Centric Quantum Philosophy - QUBOベースの量子神託アプリ`
   - **Public** または **Private** を選択（チームメンバーに共有する場合はPrivateでも可）
   - **⚠️ 重要**: 「Add a README file」「Add .gitignore」「Choose a license」は**チェックを外す**（既にファイルがあるため）
4. 「**Create repository**」をクリック

### ステップ2: ローカルでGitを初期化（初回のみ）

PowerShellで以下のコマンドを実行：

```powershell
# プロジェクトディレクトリに移動
cd "C:\Users\FMV\Desktop\Q-Quest_量子神託"

# Gitリポジトリを初期化
git init

# すべてのファイルをステージング
git add .

# 初回コミット
git commit -m "Initial commit: Q-Quest 量子神託 with Streamlit"

# メインブランチに設定
git branch -M main

# リモートリポジトリを追加（<YOUR_USERNAME>と<YOUR_REPO_NAME>を置き換え）
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git

# プッシュ（GitHubの認証が求められます）
git push -u origin main
```

**例**:
```powershell
git remote add origin https://github.com/your-username/Q-Quest-Quantum-Oracle.git
```

### ステップ3: Streamlit Cloudでデプロイ

1. [Streamlit Cloud](https://streamlit.io/cloud) にアクセス
2. 「**Sign in**」をクリックしてGitHubアカウントでログイン（初回のみ認証が必要）
3. 「**New app**」ボタンをクリック
4. 以下を設定：
   - **Repository**: 先ほど作成したリポジトリを選択（例：`your-username/Q-Quest-Quantum-Oracle`）
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. 「**Deploy**」をクリック

### ステップ4: デプロイ完了！

数分でデプロイが完了すると、以下のようなURLが生成されます：

```
https://your-username-your-app-name.streamlit.app
```

例：`https://q-quest-quantum-oracle.streamlit.app`

このURLをチームメンバーに共有するだけで、誰でもアプリにアクセスできます！

## 🔄 今後の更新方法

コードを更新したら、以下のコマンドでGitHubにプッシュします：

```powershell
cd "C:\Users\FMV\Desktop\Q-Quest_量子神託"

# 変更を確認
git status

# 変更をステージング
git add .

# コミット（更新内容を説明）
git commit -m "説明: 何を更新したか"

# GitHubにプッシュ
git push
```

**Streamlit Cloudは自動的に再デプロイされます**（通常30秒〜2分程度）

## 📁 デプロイに必要なファイル

以下のファイルがリポジトリに含まれていることを確認してください：

- ✅ `app.py` - Streamlitアプリのメインファイル
- ✅ `requirements.txt` - 依存パッケージ一覧
- ✅ `README.md` - プロジェクト説明
- ✅ `.gitignore` - Git除外設定

## 🛠️ トラブルシューティング

### デプロイが失敗する場合

1. **`requirements.txt`の確認**
   - すべての依存パッケージが含まれているか確認
   - バージョン番号が正しいか確認

2. **`app.py`の構文エラー**
   - ローカルで `streamlit run app.py` を実行してエラーがないか確認

3. **Streamlit Cloudのログ確認**
   - Streamlit Cloudのダッシュボードで「Logs」タブを確認
   - エラーメッセージが表示されます

### GitHubへのプッシュでエラーが出る場合

```powershell
# リモートリポジトリの確認
git remote -v

# リモートリポジトリのURLを変更する場合
git remote set-url origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git

# 強制プッシュ（注意：既存の履歴を上書きします）
git push -u origin main --force
```

### 日本語が正しく表示されない場合

- Streamlitアプリでは、Plotlyが自動的にブラウザのフォントを使用します
- ブラウザの設定を確認してください
- 通常は問題なく表示されます

## 🔒 セキュリティ（Privateリポジトリの場合）

チームメンバーにのみ共有したい場合：

1. GitHubリポジトリを**Private**に設定
2. 「Settings」→「Collaborators」からメンバーを追加
3. Streamlit Cloudアプリも自動的にPrivateになります
4. URLを知っている人だけがアクセス可能

## 📊 Streamlit Cloudダッシュボード

Streamlit Cloudのダッシュボードでは以下を確認できます：

- アプリの実行状況
- アクセスログ
- エラーログ
- アプリの設定変更

## 🎉 完了！

これで、チームメンバーは以下の方法でアプリにアクセスできます：

1. **Streamlit CloudのURL**を直接開く（最も簡単）
2. **GitHubリポジトリ**からコードをクローンしてローカル実行

---

**質問や問題があれば、GitHubのIssuesに投稿してください！**
