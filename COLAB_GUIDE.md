# Google Colabでの実行ガイド

## Colabで実行する方法

### 1. NotebookをColabにアップロード

1. **Google Colabを開く**
   - https://colab.research.google.com/ にアクセス

2. **Notebookをアップロード**
   - ファイル → ノートブックをアップロード
   - `Q_QUEST_量子神託.ipynb` を選択

### 2. 各Cellの実行

#### Cell 0: 基本デモ
- Cell 0を選択
- 実行ボタンをクリック、または **Shift + Enter**

#### Cell 2: 対話型量子神託
- Cell 2を選択して実行
- プロンプトが表示されたら、悩みや気持ちを入力
- **注意**: Cell 2の先頭にある `!apt-get -y install fonts-ipaexfont` はColab専用のコマンドです（ローカルでは不要）

#### Cell 3: 「整い」の視覚化
- Cell 3を選択して実行
- アニメーションが表示されます

#### Cell 4: QUBO × 量子神託 UI
- **方法A**: Cell 4をそのまま実行（プロンプトで入力）
- **方法B**: Cell 5を使用（推奨）
  - Cell 5の `user_input = "世界平和に貢献できる人間になる"` の部分を変更
  - 実行ボタンをクリック

### 3. Cell 5の使用例

```python
# 例1: 平和に関連する願い
user_input = "世界平和に貢献できる人間になる"

# 例2: 成長に関連する願い
user_input = "もっと成長したい、学びたい"

# 例3: 感謝に関連する願い
user_input = "家族や友人に感謝したい"

# 実行
visualize_quantum_oracle(user_input)
```

## Colabでの表示について

- **3D可視化**: Colabでもmatplotlibの3Dプロットは表示されます
- **日本語表示**: フォント警告が出る場合がありますが、実行には影響しません
- **アニメーション**: Cell 3のアニメーションはColabで表示されます

## トラブルシューティング

### パッケージが見つからないエラー

Cell 0またはCell 2の最初に以下を追加して実行：

```python
!pip install numpy matplotlib
```

### 3D表示が表示されない場合

```python
%matplotlib inline
```

をCell 4またはCell 5の最初に追加してください。

### 日本語が正しく表示されない場合

Cell 2の最初のセクションが自動的に日本語フォントをインストールしますが、他のCellでも必要であれば：

```python
!apt-get -y install fonts-ipaexfont > /dev/null
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "IPAexGothic"
```

を追加してください。

## 実行の流れ（推奨）

1. **Cell 0**: 基本デモを確認
2. **Cell 5**: 簡単に願いを設定して実行（Cell 4の代替）
3. **Cell 2**: 対話型でより詳しく体験
4. **Cell 3**: 「整い」のアート的視覚化を体験

---

**Q-Quest 量子神託** - Colabでも体験できる量子の「縁」
