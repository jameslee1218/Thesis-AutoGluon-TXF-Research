#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步驟 02：特徵壓縮。
依序執行：split_by_cutoff.py（截點切分）→ autoencoder.py（滾動視窗壓縮）
（請將兩腳本置於本目錄，並改為使用 config 路徑。）
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402

SCRIPTS = [
    "split_by_cutoff.py",  # 對應 01w5 的 01_split_data.py
    "autoencoder.py",      # 對應 01w5 的 02_autoencoder.py
]


def run():
    for name in SCRIPTS:
        path = MODULE_DIR / name
        if not path.exists():
            print(f"[02] 跳過（不存在）: {name}")
            continue
        print(f"[02] 執行: {name}")
        ret = subprocess.run(
            [sys.executable, name],
            cwd=str(MODULE_DIR),
            env=os.environ.copy(),
        )
        if ret.returncode != 0:
            print(f"[02] {name} 回傳碼: {ret.returncode}")
            return False
    return True


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
