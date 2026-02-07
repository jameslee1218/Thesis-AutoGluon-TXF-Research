#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從 data/indicators_complete 讀取完整技術指標 CSV，提取 7 群組至 data/indicators_extracted。
僅讀寫 data/，路徑由專案 config 提供。可手動執行或由 main.py --step 1 觸發。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
import numpy as np
import pandas as pd
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 僅讀寫 data/
INPUT_DIR = config.get_indicators_complete_dir()
OUTPUT_DIR = config.get_extracted_indicators_dir()
BATCH_SIZE = 100

INDICATOR_GROUPS = {
    "MACD": ["MACD_12_26", "MACD_signal_12_26", "MACD_hist_12_26"],
    "BBANDS": ["BBANDS_upper_20", "BBANDS_middle_20", "BBANDS_lower_20"],
    "STOCH": ["STOCH_K_14", "STOCH_D_14"],
    "STOCHRSI": ["STOCHRSI_K_14", "STOCHRSI_D_14"],
    "STOCHF": ["STOCHF_K_14", "STOCHF_D_14"],
    "ADX_DMI": ["ADX_14", "ADXR_14", "PDI_14", "MDI_14", "DX_14"],
    "AROON": ["AROON_Down_14", "AROON_Up_14", "AROONOSC_14"],
}


def get_required_columns():
    required_cols = ["datetime"]
    for group_name, cols in INDICATOR_GROUPS.items():
        required_cols.extend(cols)
    basic_cols = ["open", "high", "low", "close", "volume"]
    required_cols.extend(basic_cols)
    return list(set(required_cols))


def extract_single_file(file_path, output_dir):
    try:
        required_cols = get_required_columns()
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        keep = [c for c in required_cols if c in df.columns]
        if keep:
            df = df[keep]
        # 補齊群組欄位（可能部分不存在）
        for group_name, cols in INDICATOR_GROUPS.items():
            group_cols = [c for c in cols if c in df.columns]
            if len(group_cols) > 0:
                df[group_cols] = df[group_cols].ffill().bfill()

        unique_dates = len(np.unique(df.index.date))
        if unique_dates > 1:
            return None, f"多個交易日 ({unique_dates})"

        output_file = output_dir / file_path.name
        df.to_csv(output_file)
        return df, "成功"
    except Exception as e:
        return None, str(e)


def extract_all_data():
    print("開始提取技術指標資料...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"輸出目錄（data/）: {OUTPUT_DIR}")

    all_files = list(INPUT_DIR.glob("*.csv"))
    data_files = [f for f in all_files if not f.name.startswith("combined_")]
    print(f"找到 {len(data_files)} 個檔案需要處理")

    if len(data_files) == 0:
        print("沒有找到任何 CSV 檔案！")
        return None

    required_cols = get_required_columns()
    print(f"需要提取的欄位: {len(required_cols)} 個")

    start_time = time.time()
    successful_extractions = 0
    failed_extractions = 0
    extraction_report = []

    for file_path in tqdm(data_files, desc="提取資料"):
        output_file = OUTPUT_DIR / file_path.name
        if output_file.exists():
            successful_extractions += 1
            extraction_report.append({"file": file_path.name, "status": "已存在", "reason": "跳過重複"})
            continue

        extracted_df, status = extract_single_file(file_path, OUTPUT_DIR)

        if extracted_df is not None:
            successful_extractions += 1
            extraction_report.append({
                "file": file_path.name,
                "status": "成功",
                "reason": status,
                "shape": str(extracted_df.shape),
                "columns": len(extracted_df.columns),
            })
        else:
            failed_extractions += 1
            extraction_report.append({"file": file_path.name, "status": "失敗", "reason": status})

    elapsed = time.time() - start_time
    report_df = pd.DataFrame(extraction_report)
    report_path = OUTPUT_DIR / "extraction_report.csv"
    report_df.to_csv(report_path, index=False)

    print(f"\n資料提取完成！")
    print(f"處理時間: {elapsed:.2f} 秒")
    print(f"成功: {successful_extractions}，失敗: {failed_extractions}")
    print(f"提取報告: {report_path}")
    return report_df


def main():
    print("優化的技術指標資料提取（僅讀寫 data/）")
    print("=" * 60)
    if not INPUT_DIR.exists():
        print(f"輸入目錄不存在: {INPUT_DIR}")
        return
    print(f"輸入目錄（data/）: {INPUT_DIR}")
    total_indicators = sum(len(v) for v in INDICATOR_GROUPS.values())
    print(f"提取技術指標群組: {total_indicators} 個")
    report_df = extract_all_data()
    if report_df is not None:
        print(f"\n提取的資料已儲存至: {OUTPUT_DIR}")
    else:
        print("資料提取失敗")


if __name__ == "__main__":
    main()
