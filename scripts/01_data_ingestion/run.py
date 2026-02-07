#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步驟 01：資料提取與前處理。
依序執行：generate_all_indicators.py → extract_indicators_optimized.py
（請將兩腳本置於本目錄，並改為使用 config 路徑。）
"""

import os
import subprocess
import sys
from pathlib import Path

# 專案根目錄
REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402  # 設定 DATA_ROOT 等

SCRIPTS = [
    "generate_all_indicators.py",
    "extract_indicators_optimized.py",
]


def run():
    for name in SCRIPTS:
        path = MODULE_DIR / name
        if not path.exists():
            print(f"[01] 跳過（不存在）: {name}")
            continue
        print(f"[01] 執行: {name}")
        ret = subprocess.run(
            [sys.executable, name],
            cwd=str(MODULE_DIR),
            env=os.environ.copy(),
        )
        if ret.returncode != 0:
            print(f"[01] {name} 回傳碼: {ret.returncode}")
            return False
    return True


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
