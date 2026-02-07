#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步驟 03：整合與建模。
執行：merge_and_train.py（合併壓縮特徵 + Y 報酬率 → 日表，可選 AutoGluon 訓練）
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

SCRIPT = "merge_and_train.py"  # 對應 01w5 的 04_merge_features.py


def run():
    path = MODULE_DIR / SCRIPT
    if not path.exists():
        print(f"[03] 跳過（不存在）: {SCRIPT}")
        return True
    print(f"[03] 執行: {SCRIPT}")
    ret = subprocess.run(
        [sys.executable, SCRIPT],
        cwd=str(MODULE_DIR),
        env=os.environ.copy(),
    )
    return ret.returncode == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
