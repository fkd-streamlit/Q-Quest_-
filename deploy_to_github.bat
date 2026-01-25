@echo off
chcp 65001 >nul
echo ========================================
echo Q-Quest 量子神託 - GitHub公開スクリプト
echo ========================================
echo.

cd /d "%~dp0"

echo [1/5] Gitリポジトリの状態を確認...
git status >nul 2>&1
if %errorlevel% neq 0 (
    echo Gitリポジトリが初期化されていません。初期化します...
    git init
    echo ✓ Gitリポジトリを初期化しました
) else (
    echo ✓ Gitリポジトリは既に初期化されています
)
echo.

echo [2/5] すべてのファイルをステージング...
git add .
echo ✓ ファイルを追加しました
echo.

echo [3/5] コミット...
git commit -m "Update: Q-Quest 量子神託 with 絵馬納め機能" 2>nul
if %errorlevel% neq 0 (
    echo 変更がないか、既にコミット済みです
) else (
    echo ✓ コミットしました
)
echo.

echo [4/5] リモートリポジトリの確認...
git remote -v >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ⚠ リモートリポジトリが設定されていません
    echo.
    echo 以下の手順でGitHubリポジトリを作成してください:
    echo 1. https://github.com にアクセス
    echo 2. 右上の「+」→「New repository」をクリック
    echo 3. リポジトリ名を入力（例: Q-Quest-Quantum-Oracle）
    echo 4. 「Create repository」をクリック
    echo.
    set /p REPO_URL="GitHubリポジトリのURLを入力してください（例: https://github.com/ユーザー名/リポジトリ名.git）: "
    if not "%REPO_URL%"=="" (
        git remote add origin "%REPO_URL%"
        echo ✓ リモートリポジトリを追加しました
    )
) else (
    echo ✓ リモートリポジトリが設定されています
    git remote -v
)
echo.

echo [5/5] GitHubにプッシュ...
git branch -M main 2>nul
git push -u origin main
if %errorlevel% neq 0 (
    echo.
    echo ⚠ プッシュに失敗しました
    echo 以下の可能性があります:
    echo - GitHubの認証が必要（ブラウザで認証が求められます）
    echo - リモートリポジトリが存在しない
    echo.
    echo 手動でプッシュする場合:
    echo   git push -u origin main
) else (
    echo.
    echo ========================================
    echo ✓ GitHubへのプッシュが完了しました！
    echo ========================================
    echo.
    echo 次のステップ:
    echo 1. https://streamlit.io/cloud にアクセス
    echo 2. 「Sign in」→「New app」をクリック
    echo 3. リポジトリを選択、Branch: main、Main file: app.py
    echo 4. 「Deploy」をクリック
    echo.
    echo 詳細は GITHUB_DEPLOY.md を参照してください
)
echo.
pause
