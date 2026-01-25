"""
Notebookファイルをワークスペースにコピーするスクリプト
"""
import json
import shutil
from pathlib import Path

def copy_notebook():
    """Notebookファイルをコピー"""
    source = Path(r"C:\Users\FMV\Downloads\Q_QUEST_量子神託.ipynb")
    dest = Path("Q_QUEST_量子神託.ipynb")
    
    if source.exists():
        # JSONとして読み込んで書き込み（エンコーディング問題を回避）
        with open(source, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        with open(dest, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Notebook copied successfully to {dest}")
        return True
    else:
        print(f"✗ Source file not found: {source}")
        print("Please copy the notebook manually or update the source path.")
        return False

if __name__ == "__main__":
    copy_notebook()

