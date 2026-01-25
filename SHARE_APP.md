# 🚀 アプリ共有のクイックガイド

## 最も簡単な方法：GitHub + Streamlit Cloud（完全無料）

### ステップ1: GitHubリポジトリを作成

1. [GitHub](https://github.com) にログイン（アカウントがない場合は作成）
2. 右上の「**+**」→「**New repository**」をクリック
3. 以下を設定：
   - **Repository name**: `Q-Quest-Quantum-Oracle`（お好きな名前でOK）
   - **Description**: `Human-Centric Quantum Philosophy - QUBOベースの量子神託アプリ`
   - **Public** または **Private** を選択
   - ⚠️ 「Add a README file」などは**チェックを外す**
4. 「**Create repository**」をクリック

### ステップ2: コードをGitHubにプッシュ

#### 方法A: 自動スクリプトを使用（推奨）

1. `deploy_to_github.bat` をダブルクリック
2. 指示に従って進める

#### 方法B: 手動で実行

PowerShellで以下を実行：

```powershell
cd "C:\Users\FMV\Desktop\Q-Quest_量子神託"

# Gitリポジトリを初期化（初回のみ）
git init

# すべてのファイルを追加
git add .

# コミット
git commit -m "Initial commit: Q-Quest 量子神託"

# メインブランチに設定
git branch -M main

# リモートリポジトリを追加（<YOUR_REPO_URL>を実際のURLに置き換え）
git remote add origin https://github.com/あなたのユーザー名/リポジトリ名.git

# GitHubにプッシュ
git push -u origin main
```

**例**:
```powershell
git remote add origin https://github.com/your-username/Q-Quest-Quantum-Oracle.git
```

### ステップ3: Streamlit Cloudでデプロイ

1. [Streamlit Cloud](https://streamlit.io/cloud) にアクセス
2. 「**Sign in**」をクリックしてGitHubアカウントでログイン
3. 「**New app**」ボタンをクリック
4. 以下を設定：
   - **Repository**: 作成したリポジトリを選択
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. 「**Deploy**」をクリック

### ステップ4: 完了！

数分でデプロイが完了し、以下のようなURLが生成されます：

```
https://your-username-your-app-name.streamlit.app
```

**このURLを共有するだけで、誰でもアプリにアクセスできます！** 🎉

## 共有方法

### チームメンバーに共有

1. **Streamlit CloudのURL**を送る（最も簡単）
2. **GitHubリポジトリのURL**を送る（コードも見たい場合）

### プライベートに共有したい場合

1. GitHubリポジトリを**Private**に設定
2. 「Settings」→「Collaborators」からメンバーを追加
3. Streamlit Cloudアプリも自動的にPrivateになります

## 今後の更新方法

コードを更新したら：

```powershell
cd "C:\Users\FMV\Desktop\Q-Quest_量子神託"
git add .
git commit -m "更新内容の説明"
git push
```

**Streamlit Cloudは自動的に再デプロイされます**（30秒〜2分程度）

## トラブルシューティング

### GitHubへのプッシュでエラーが出る場合

- GitHubの認証が必要な場合、ブラウザで認証が求められます
- リモートリポジトリのURLが正しいか確認してください

### Streamlit Cloudでデプロイが失敗する場合

- `requirements.txt`にすべての依存パッケージが含まれているか確認
- `app.py`に構文エラーがないか確認
- Streamlit Cloudのログを確認（エラーメッセージが表示されます）

## 詳細情報

- **詳細な手順**: `GITHUB_DEPLOY.md`
- **クイックデプロイ**: `QUICK_DEPLOY.md`

---

**🎉 アプリを共有して、みんなで量子神託を体験しましょう！**
