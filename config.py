#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
專案路徑設定：供五模組腳本統一引用，避免寫死絕對路徑。
所有輸入與輸出均置於 data/ 下，可透過環境變數 DATA_ROOT 覆寫（本機／Colab）。
"""

import os
from pathlib import Path

# 專案根目錄：預設為本檔案所在目錄（repo 根目錄）
_PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = str(Path(__file__).resolve().parent)
PROJECT_ROOT = Path(_PROJECT_ROOT)

# 資料根目錄：所有輸入與產出皆在此目錄下（整個 data/ 不進版控，結構見 data/README.md）
_DATA_ROOT = os.environ.get("DATA_ROOT")
if _DATA_ROOT is None:
    _DATA_ROOT = PROJECT_ROOT / "data"
DATA_ROOT = Path(_DATA_ROOT)

# 為相容舊腳本：OUTPUT_ROOT 指向同一 data（產出與輸入共用 data/）
OUTPUT_ROOT = DATA_ROOT

# 三截點代號（與 data/dataset、output_* 對應）
CUTOFFS = ("0900", "0915", "0930")

# ---------- 輸入（需自備或由 01 產出）----------
def get_raw_kline_dir():
    """原始 1 分鐘 K 線目錄。"""
    return DATA_ROOT / "raw" / "TX2011~20231222-1K"

def get_indicators_complete_dir():
    """完整技術指標目錄（01 產出或自備）。"""
    return DATA_ROOT / "indicators_complete"

def get_extracted_indicators_dir():
    """01 產出：提取後 7 群組技術指標 CSV 目錄。"""
    return DATA_ROOT / "indicators_extracted"

def get_y_file():
    """目標變數檔（收盤－9:00 報酬率）；優先 target/y.xlsx，其次 target/y.csv。"""
    target_dir = DATA_ROOT / "target"
    for name in ("y.xlsx", "y.csv"):
        p = target_dir / name
        if p.exists():
            return str(p)
    return str(target_dir / "y.xlsx")

# ---------- 產出（依研究順序）----------
def get_dataset_dir(cutoff: str = "0900"):
    """02 截點切分產出：dataset/0900、0915、0930。"""
    return DATA_ROOT / "dataset" / cutoff

def get_output_0900_dir():
    """02 壓縮與 03 合併產出：output_0900（W*、merged_for_autogluon）。"""
    return DATA_ROOT / "output_0900"

def get_output_cutoff_dir(cutoff: str):
    """依截點代號取得 output 目錄，如 output_0900、output_0915、output_0930。"""
    return DATA_ROOT / f"output_{cutoff}"

def get_merged_for_autogluon_dir():
    """03 合併表輸出目錄。"""
    return DATA_ROOT / "output_0900" / "merged_for_autogluon"

def get_visualizations_dir():
    """04 視覺化輸出目錄。"""
    return DATA_ROOT / "visualizations"

def get_backtest_dir():
    """05 回測輸出目錄。"""
    return DATA_ROOT / "backtest"
