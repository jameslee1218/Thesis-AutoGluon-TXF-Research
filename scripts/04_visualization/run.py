#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步驟 04：視覺化。
- visualize_results.py：MSE 時間線、壓縮雷達圖、重建散點圖（需 output_0900）
- txf_ai_analysis.py：台指期 AI 模型全方位績效與特徵分析（需 data/models）
"""

import argparse
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
    "visualize_results.py",
    "txf_ai_analysis.py",
]


def run(script: str) -> bool:
    path = MODULE_DIR / script
    if not path.exists():
        print(f"[04] 跳過（不存在）: {script}")
        return True
    print(f"[04] 執行: {script}")
    ret = subprocess.run(
        [sys.executable, script],
        cwd=str(MODULE_DIR),
        env=os.environ.copy(),
    )
    return ret.returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="04 視覺化")
    parser.add_argument("--script", choices=SCRIPTS + ["all"], default="all", help="執行腳本 (預設 all)")
    args = parser.parse_args()
    if args.script == "all":
        ok = all(run(s) for s in SCRIPTS)
    else:
        ok = run(args.script)
    sys.exit(0 if ok else 1)
