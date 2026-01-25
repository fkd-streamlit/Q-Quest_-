@echo off
chcp 65001 > nul
echo ========================================
echo Q-Quest 量子神託 - Streamlitアプリ起動
echo ========================================
echo.

REM 依存パッケージのインストール確認
python -m pip install -q -r requirements.txt

REM Streamlitアプリを起動
echo Streamlitアプリを起動しています...
echo ブラウザが自動的に開きます。
echo.
streamlit run app.py

pause
