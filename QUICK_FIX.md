# クイックフィックス: ModuleNotFoundError

## エラー: `ModuleNotFoundError: No module named 'plotly'`

このエラーは、必要なパッケージがインストールされていない場合に発生します。

## 解決方法（3つの方法）

### 方法1: 自動インストールスクリプトを使用（推奨）

1. `install_requirements.bat` をダブルクリック
2. インストールが完了したら、`start_app.bat` を実行

### 方法2: コマンドプロンプトから実行

1. **コマンドプロンプト**（cmd.exe）を開く
   - Windowsキー + R → `cmd` と入力 → Enter

2. **プロジェクトディレクトリに移動**
   ```
   cd C:\Users\FMV\Desktop\Q-Quest_量子神託
   ```

3. **依存パッケージをインストール**
   ```
   pip install -r requirements.txt
   ```

4. **Streamlitアプリを起動**
   ```
   streamlit run app.py
   ```

### 方法3: 個別にインストール

```bash
pip install streamlit plotly numpy matplotlib
```

## インストールが必要なパッケージ

- streamlit (>=1.28.0)
- plotly (>=5.17.0)
- numpy (>=1.20.0)
- matplotlib (>=3.3.0)

## 確認方法

インストールが成功したか確認：

```bash
python -c "import streamlit; import plotly; print('OK')"
```

`OK` と表示されれば成功です。

## トラブルシューティング

### pipが見つからない場合

```bash
python -m pip install -r requirements.txt
```

### 権限エラーが発生する場合

管理者権限でコマンドプロンプトを開いてから実行してください。

### 仮想環境を使用する場合

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
