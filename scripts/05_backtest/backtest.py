#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回測占位：僅建立 data/backtest 目錄並結束。
請替換為實際回測邏輯（讀取 data/output_0900/merged_for_autogluon_0900、預測結果，寫入 data/backtest）。
僅讀寫 data/，路徑由 config 提供。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config

def main():
    out = config.get_backtest_dir()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[05_backtest] 已建立目錄: {out}")
    print("請將本腳本替換為實際回測邏輯（讀 merged_for_autogluon_0900、預測，寫 backtest/）。")

if __name__ == "__main__":
    main()
