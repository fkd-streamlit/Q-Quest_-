# エラー報告ガイド

エラーが発生した場合は、以下の情報を提供してください：

## 必要な情報

1. **エラーメッセージの全文**（コピー&ペーストしてください）

2. **実行したコマンド**
   ```
   例: streamlit run app.py
   ```

3. **環境情報**
   - OS: Windows 10 / macOS / Linux
   - Pythonバージョン: `python --version` の結果
   - Streamlitバージョン: `streamlit --version` の結果

4. **エラーが発生した操作**
   - Streamlitアプリを起動した時
   - 特定のボタンをクリックした時
   - テキストを入力した時
   - など

5. **依存パッケージの確認結果**
   ```bash
   python test_imports.py
   ```
   の結果

## よくあるエラー例

### 1. インポートエラー
```
ModuleNotFoundError: No module named 'streamlit'
```
→ `pip install -r requirements.txt` を実行

### 2. ポートエラー
```
Port 8501 is already in use
```
→ 別のポートを指定: `streamlit run app.py --server.port 8502`

### 3. エンコーディングエラー
```
UnicodeDecodeError: 'cp932' codec can't decode byte...
```
→ UTF-8エンコーディングで保存されているか確認

### 4. Plotlyエラー
```
AttributeError: 'Figure' object has no attribute 'add_trace'
```
→ `pip install --upgrade plotly` を実行

## エラーメッセージの取得方法

### ターミナル/コマンドプロンプトから実行

```bash
streamlit run app.py
```

エラーメッセージがターミナルに表示されます。その全文をコピーしてください。

### ブラウザの開発者ツール

1. ブラウザでF12キーを押す
2. 「Console」タブを開く
3. エラーメッセージを確認
