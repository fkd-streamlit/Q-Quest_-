@echo off
chcp 65001 > nul
title Q-Quest 量子神託 - 依存パッケージインストール
echo ========================================
echo Q-Quest 量子神託
echo 依存パッケージのインストール
echo ========================================
echo.

echo 依存パッケージをインストール中...
echo.

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ========================================
if %ERRORLEVEL% EQU 0 (
    echo ✅ インストール完了！
    echo.
    echo 次に、start_app.batを実行してStreamlitアプリを起動してください。
) else (
    echo ❌ インストール中にエラーが発生しました。
    echo エラーメッセージを確認してください。
)

echo ========================================
pause
