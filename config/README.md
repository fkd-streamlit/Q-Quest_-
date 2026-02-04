# 設定ファイル（config/）

このフォルダーには、Q-Quest 量子神託アプリの設定ファイル（Excel）を配置します。

## 5つのExcel設定ファイル

1. **akiba12_character_list.xlsx**
   - 12神の基本情報（ID、名前、属性、絵文字、説明、格言）

2. **格言.xlsx**（オプション）
   - 格言データ

3. **akiba12_character_to_vow_K.xlsx**
   - k行列（12×12：キャラクター × 誓願）

4. **akiba12_character_to_axis_L.xlsx**
   - l行列（12×4：キャラクター × 世界観軸）

5. **sense_to_vow_initial_filled_from_user.xlsx**
   - sense_to_vow行列（8×12：感覚 × 誓願）

## 📁 配置場所

### ✅ 推奨: `config/` フォルダーに配置

```
Q-Quest_量子神託/
└── config/
    ├── README.md                                    # このファイル
    ├── akiba12_character_list.xlsx                  # ← ここに配置
    ├── 格言.xlsx                                    # ← ここに配置
    ├── akiba12_character_to_vow_K.xlsx             # ← ここに配置
    ├── akiba12_character_to_axis_L.xlsx            # ← ここに配置
    └── sense_to_vow_initial_filled_from_user.xlsx  # ← ここに配置
```

**手順：**
1. プロジェクトルートの `config/` フォルダーに5つのExcelファイルを配置
2. Streamlitアプリを起動
3. サイドバーから「5つのファイル（推奨）」を選択
4. `config/` フォルダーからファイルを選択してアップロード

## ⚠️ 重要な注意事項

### GitHubには含めません

- **Excelファイル（`*.xlsx`, `*.xls`）はGitHubにアップロードされません**
- `.gitignore` で除外されています（個人用設定のため）
- このフォルダーの `README.md` のみがGitHubに含まれます

### Streamlit Cloudでの使用方法

Streamlit Cloudで公開する場合：

1. **アプリのサイドバーからアップロード**
   - 「5つのファイル（推奨）」を選択
   - 各Excelファイルをアップロード
   - ファイルはセッションごとに保持されます

2. **デフォルト設定**
   - ファイルがない場合、アプリ内のデフォルト設定が使用されます

## 📋 ファイルの詳細

各ファイルの詳細は、プロジェクトルートの以下のドキュメントを参照してください：

- `EXCEL_SETUP_GUIDE.md` - Excelファイルの構造と設定方法
- `格言追加ガイド.md` - 格言の追加方法
- `sense_to_vow行列の説明.md` - sense_to_vow行列の説明
- `設計の流れ.md` - 全体の設計フロー
- `config/設定ファイルの配置方法.md` - 配置方法の詳細

## 詳細

各ファイルの詳細は、プロジェクトルートの以下のドキュメントを参照してください：

- `EXCEL_SETUP_GUIDE.md` - Excelファイルの構造と設定方法
- `格言追加ガイド.md` - 格言の追加方法
- `sense_to_vow行列の説明.md` - sense_to_vow行列の説明
- `設計の流れ.md` - 全体の設計フロー