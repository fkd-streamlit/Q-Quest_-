# Streamlit接続エラーの解決方法

## エラー: "Connection error - Is Streamlit still running?"

このエラーは、Streamlitサーバーが停止しているか、ブラウザとの接続が切れた場合に発生します。

## 解決方法

### 方法1: Streamlitアプリを再起動（最も一般的）

1. **ターミナル/コマンドプロンプトを開く**
2. **プロジェクトディレクトリに移動**
   ```bash
   cd C:\Users\FMV\Desktop\Q-Quest_量子神託
   ```

3. **Streamlitアプリを起動**
   ```bash
   streamlit run app.py
   ```

4. **ブラウザをリロード**
   - ブラウザでF5キーを押す
   - または、ターミナルに表示されたURLをクリック

### 方法2: 別のポートを使用

ポート8501が使用中の場合は、別のポートを指定：

```bash
streamlit run app.py --server.port 8502
```

### 方法3: 既存のStreamlitプロセスを終了

Windowsの場合：
```bash
# PowerShellで実行
Get-Process | Where-Object {$_.ProcessName -like "*streamlit*"} | Stop-Process
```

または、タスクマネージャーで`streamlit`プロセスを終了

### 方法4: バッチファイルを使用

`run_streamlit.bat`をダブルクリックして実行

## トラブルシューティング

### ポートが既に使用されている場合

```bash
# 別のポートを指定
streamlit run app.py --server.port 8502
```

### ブラウザが自動的に開かない場合

ターミナルに表示されたURLを手動でブラウザにコピー&ペースト：
```
http://localhost:8501
```

### 複数のStreamlitインスタンスが起動している場合

すべてのStreamlitプロセスを終了してから、再度起動してください。

## 正常に起動すると...

ターミナルに以下のようなメッセージが表示されます：

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

このURLをクリックするか、ブラウザで開いてください。
