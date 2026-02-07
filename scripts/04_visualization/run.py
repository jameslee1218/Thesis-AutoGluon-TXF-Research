#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步驟 04：視覺化。
執行：visualize_results.py（MSE 時間線、壓縮雷達圖、重建散點圖）
（請將腳本置於本目錄，並改為使用 config 路徑。）
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

SCRIPT = "visualize_results.py"  # 對應 01w5 的 03_visualize_results.py


def run():
    path = MODULE_DIR / SCRIPT
    if not path.exists():
        print(f"[04] 跳過（不存在）: {SCRIPT}")
        return True
    print(f"[04] 執行: {SCRIPT}")
    ret = subprocess.run(
        [sys.executable, SCRIPT],
        cwd=str(MODULE_DIR),
        env=os.environ.copy(),
    )
    return ret.returncode == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
