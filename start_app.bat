@echo off
chcp 65001 > nul
title Q-Quest 量子神託 - Streamlit
echo ========================================
echo Q-Quest 量子神託 - Streamlitアプリ起動
echo ========================================
echo.

REM 既存のStreamlitプロセスを終了（オプション）
taskkill /F /IM streamlit.exe > nul 2>&1
timeout /t 1 > nul

REM 依存パッケージの確認とインストール
echo 依存パッケージを確認中...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo.
    echo ⚠ Streamlitがインストールされていません。
    echo 依存パッケージをインストール中...
    echo.
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    echo.
    if errorlevel 1 (
        echo ❌ インストールに失敗しました。
        echo 手動で以下を実行してください:
        echo   pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo ✅ インストール完了！
    echo.
)

REM Streamlitアプリを起動
echo.
echo ========================================
echo Streamlitアプリを起動しています...
echo ブラウザが自動的に開きます。
echo.
echo 閉じるには、このウィンドウで Ctrl+C を押してください
echo ========================================
echo.

streamlit run app.py

pause
