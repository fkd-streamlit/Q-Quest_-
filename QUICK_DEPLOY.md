# 🚀 クイックデプロイ（3ステップ）

## 1️⃣ GitHubリポジトリを作成
- https://github.com → 「New repository」
- 名前を入力 → 「Create repository」

## 2️⃣ ローカルからプッシュ

PowerShellで実行：

```powershell
cd "C:\Users\FMV\Desktop\Q-Quest_量子神託"
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/あなたのユーザー名/リポジトリ名.git
git push -u origin main
```

## 3️⃣ Streamlit Cloudでデプロイ
- https://streamlit.io/cloud → 「Sign in」→「New app」
- Repository: 作成したリポジトリを選択
- Branch: `main`
- Main file: `app.py`
- 「Deploy」をクリック

**完了！** 数分で公開URLが生成されます 🎉

---

詳細は `GITHUB_DEPLOY.md` を参照してください。
