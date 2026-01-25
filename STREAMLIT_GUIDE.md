# Streamlitアプリケーション ガイド

## Streamlitとは

Streamlitは、Pythonで簡単にWebアプリケーションを作成できるフレームワークです。このプロジェクトをStreamlitアプリとして動かすことで、より多くの人に簡単に体験してもらえます。

## メリット

### 1. **簡単にWebアプリとして公開できる**
- GitHubにプッシュするだけで、Streamlit Cloudで自動デプロイ
- URLを共有するだけで誰でもアクセス可能
- サーバー管理不要

### 2. **インタラクティブなUI**
- テキスト入力、ボタン、スライダーなどが簡単に実装可能
- リアルタイムで結果が更新される

### 3. **3D可視化に最適**
- Plotlyを使用することで、回転・ズーム可能なインタラクティブな3Dグラフを表示
- 日本語フォントの問題も解決しやすい

### 4. **GitHubとの統合**
- コード変更が自動的にアプリに反映される（CI/CD）
- バージョン管理が容易

## セットアップ方法

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

これでStreamlitとPlotlyがインストールされます。

### 2. ローカルで実行

```bash
streamlit run app.py
```

ブラウザが自動的に開き、`http://localhost:8501` でアプリが表示されます。

### 3. GitHubにプッシュ

```bash
git init
git add .
git commit -m "Initial commit: Streamlit app"
git branch -M main
git remote add origin <あなたのGitHubリポジトリURL>
git push -u origin main
```

### 4. Streamlit Cloudでデプロイ

1. [Streamlit Cloud](https://streamlit.io/cloud) にアクセス
2. GitHubアカウントでログイン
3. 「New app」をクリック
4. リポジトリを選択
5. Main file pathに `app.py` を指定
6. 「Deploy」をクリック

数分でアプリが公開されます！

## アプリの機能

### 1. 基本デモ
- QUBOモデルの基本動作を確認
- エネルギー地形の可視化
- 量子おみくじの生成

### 2. 対話型量子神託
- 悩み・気持ちをテキスト入力
- 心の傾き（Mood）の推定
- パーソナライズされたおみくじ

### 3. 言葉のエネルギー球体視覚化
- 願いや問いを入力
- 3D球体上にキーワードと関連語が配置
- インタラクティブに回転・ズーム可能
- 適切な格言を表示

## カスタマイズ

### テーマの変更

`.streamlit/config.toml` ファイルで色やフォントを変更できます。

### 機能の追加

`app.py` の `main()` 関数内に新しい機能を追加できます。Streamlitのドキュメントを参照してください：
- https://docs.streamlit.io/

## トラブルシューティング

### 日本語が表示されない場合

Streamlitはブラウザのフォントを使用するため、システムに日本語フォントがインストールされている必要があります。Plotlyでは日本語フォントが正しく設定されています。

### 3Dグラフが表示されない場合

Plotlyが正しくインストールされているか確認してください：
```bash
pip install plotly --upgrade
```

## 次のステップ

1. GitHubにプッシュ
2. Streamlit Cloudでデプロイ
3. URLを共有して世界中の人に体験してもらう！
