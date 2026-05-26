#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 01（新版）：從 indicators_complete 擷取五大面向欄位到 indicators_extracted。
輸入/輸出皆為 xlsx（每交易日一檔）。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
import pandas as pd
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 僅讀寫 data/
INPUT_DIR = Path(config.get_indicators_complete_dir())
OUTPUT_DIR = Path(config.get_extracted_indicators_dir())
BATCH_SIZE = 100

OUTPUT_COLS = [
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "sp500_prev_return",
    "open_gap_pct",
    "vix_prev_close",
    "vwap_bias_5",
    "volume_roc_5",
    "atr_5",
    "kd_k_9",
    "kd_d_9",
    "linearreg_angle_5",
]


def get_required_columns():
    return OUTPUT_COLS.copy()


def extract_single_file(file_path, output_dir):
    try:
        required_cols = get_required_columns()
        df = pd.read_excel(file_path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        keep = [c for c in required_cols if c in df.columns]
        if keep:
            df = df[keep]

        if "datetime" in df.columns:
            unique_dates = df["datetime"].dt.normalize().nunique(dropna=True)
            if unique_dates > 1:
                return None, f"多個交易日 ({unique_dates})"

        fill_cols = [c for c in df.columns if c != "datetime"]
        if fill_cols:
            df[fill_cols] = df[fill_cols].ffill().bfill()

        output_file = output_dir / file_path.name.replace("_complete.xlsx", "_extracted.xlsx")
        df.to_excel(output_file, index=False, engine="openpyxl")
        return df, "成功"
    except Exception as e:
        return None, str(e)


def extract_all_data():
    print("開始提取技術指標資料...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"輸出目錄（data/）: {OUTPUT_DIR}")

    all_files = list(INPUT_DIR.glob("*.xlsx"))
    data_files = [f for f in all_files if not f.name.startswith("._")]
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
    report_path = OUTPUT_DIR / "extraction_report.xlsx"
    report_df.to_excel(report_path, index=False, engine="openpyxl")

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
    print(f"提取欄位數: {len(OUTPUT_COLS)}")
    report_df = extract_all_data()
    if report_df is not None:
        print(f"\n提取的資料已儲存至: {OUTPUT_DIR}")
    else:
        print("資料提取失敗")


if __name__ == "__main__":
    main()
