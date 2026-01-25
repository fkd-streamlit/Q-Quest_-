"""依存パッケージのインポートテスト"""
import sys

print("Python version:", sys.version)
print("\n依存パッケージの確認中...\n")

packages = [
    ("streamlit", "st"),
    ("numpy", "np"),
    ("plotly", "px"),
    ("matplotlib", "plt"),
]

missing = []

for package_name, alias in packages:
    try:
        if package_name == "streamlit":
            import streamlit as st
            print(f"✅ {package_name} - OK (version: {st.__version__})")
        elif package_name == "numpy":
            import numpy as np
            print(f"✅ {package_name} - OK (version: {np.__version__})")
        elif package_name == "plotly":
            import plotly.express as px
            print(f"✅ {package_name} - OK")
        elif package_name == "matplotlib":
            import matplotlib
            print(f"✅ {package_name} - OK (version: {matplotlib.__version__})")
    except ImportError as e:
        print(f"❌ {package_name} - インストールされていません")
        missing.append(package_name)

if missing:
    print(f"\n❌ 不足しているパッケージ: {', '.join(missing)}")
    print("\n以下のコマンドでインストールしてください:")
    print(f"pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("\n✅ すべての依存パッケージがインストールされています！")
    print("\napp.pyを実行できます:")
    print("streamlit run app.py")
