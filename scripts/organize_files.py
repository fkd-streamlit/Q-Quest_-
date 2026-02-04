"""
フォルダー整理スクリプト
GitHub用にファイルを整理します
"""
import os
import shutil
from pathlib import Path

def organize_files():
    """ファイルを整理してGitHub用の構造にする"""
    base_dir = Path(__file__).parent.parent
    
    # ディレクトリを作成
    config_dir = base_dir / "config"
    scripts_dir = base_dir / "scripts"
    docs_dir = base_dir / "docs"
    
    config_dir.mkdir(exist_ok=True)
    scripts_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)
    
    # Excelファイルをconfig/に移動
    excel_files = [
        "akiba12_character_list.xlsx",
        "格言.xlsx",
        "akiba12_character_to_vow_K.xlsx",
        "akiba12_character_to_axis_L.xlsx",
        "sense_to_vow_initial_filled_from_user.xlsx",
    ]
    
    for excel_file in excel_files:
        src = base_dir / excel_file
        if src.exists():
            dst = config_dir / excel_file
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"✓ {excel_file} を config/ にコピーしました")
    
    # バッチファイルをscripts/に移動
    batch_files = [
        "start_app.bat",
        "install_requirements.bat",
        "deploy_to_github.bat",
        "run_streamlit.bat",
    ]
    
    for batch_file in batch_files:
        src = base_dir / batch_file
        if src.exists():
            dst = scripts_dir / batch_file
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"✓ {batch_file} を scripts/ にコピーしました")
    
    # Pythonスクリプトをscripts/に移動
    python_scripts = [
        "create_excel_template.py",
        "setup_notebook.py",
        "test_imports.py",
        "test_mood.py",
    ]
    
    for script in python_scripts:
        src = base_dir / script
        if src.exists():
            dst = scripts_dir / script
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"✓ {script} を scripts/ にコピーしました")
    
    print("\n整理が完了しました！")
    print("config/ フォルダーにExcelファイルが配置されています。")

if __name__ == "__main__":
    organize_files()