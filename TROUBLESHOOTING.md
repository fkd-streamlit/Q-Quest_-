# トラブルシューティングガイド

## Streamlitアプリのエラー解決

### 1. 依存パッケージの確認

まず、依存パッケージがインストールされているか確認してください：

```bash
python test_imports.py
```

不足しているパッケージがある場合は、以下を実行：

```bash
pip install -r requirements.txt
```

### 2. Streamlitが起動しない場合

```bash
# Streamlitの再インストール
pip install --upgrade streamlit

# Streamlitアプリを起動
streamlit run app.py
```

### 3. よくあるエラーと解決法

#### エラー: `ModuleNotFoundError: No module named 'streamlit'`

**解決法**:
```bash
pip install streamlit
```

#### エラー: `ModuleNotFoundError: No module named 'plotly'`

**解決法**:
```bash
pip install plotly
```

#### エラー: `ImportError: cannot import name 'go' from 'plotly.graph_objects'`

**解決法**:
```bash
pip install --upgrade plotly
```

#### エラー: `UnicodeDecodeError` または日本語の文字化け

**解決法**:
- Windowsの場合、コマンドプロンプトのエンコーディングをUTF-8に設定：
```bash
chcp 65001
streamlit run app.py
```

#### エラー: `Port already in use`

**解決法**:
```bash
# 別のポートを指定
streamlit run app.py --server.port 8502
```

### 4. 実行時のエラー

#### エラーが発生した場合の確認事項

1. **Pythonのバージョン**: Python 3.8以上が必要
   ```bash
   python --version
   ```

2. **仮想環境の使用**: 仮想環境を使用することを推奨
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. **ファイルの文字エンコーディング**: UTF-8で保存されているか確認

### 5. ブラウザでエラーが表示される場合

1. ブラウザのコンソールを開く（F12キー）
2. エラーメッセージを確認
3. 以下の場合、ページをリロード（F5キー）

### 6. 日本語が表示されない場合

- Streamlitアプリでは、Plotlyが自動的に日本語フォントを使用します
- ブラウザのフォント設定を確認してください
- Windowsの場合、日本語フォントがインストールされているか確認

## エラーログの確認方法

### Streamlitのログ

Streamlitアプリを実行すると、ターミナルにログが表示されます。エラーメッセージを確認してください。

### 詳細なエラー情報を取得

```bash
# 詳細モードで実行
streamlit run app.py --logger.level=debug
```

## サポート

問題が解決しない場合は、以下の情報を含めて報告してください：

1. Pythonのバージョン: `python --version`
2. エラーメッセージの全文
3. 実行したコマンド
4. オペレーティングシステム
