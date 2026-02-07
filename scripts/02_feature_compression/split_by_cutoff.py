#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
截點切分：從 data/indicators_extracted 讀取，依三截點寫入 data/dataset/0900, 0915, 0930。
僅讀寫 data/，路徑由 config 提供。可手動執行：python split_by_cutoff.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
import pandas as pd
from datetime import datetime

# 讀取／寫入皆在 data/
SOURCE_DIR = config.get_extracted_indicators_dir()
TIME_CUTOFFS = {
    "0900": "09:01:00",
    "0915": "09:16:00",
    "0930": "09:31:00",
}


def main():
    for folder_name, cutoff_time_str in TIME_CUTOFFS.items():
        output_dir = config.get_dataset_dir(folder_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"已建立目錄: {output_dir}")

    csv_files = [f for f in SOURCE_DIR.glob("*.csv") if not f.name.startswith("._")]
    if not csv_files:
        print(f"找不到 CSV，請確認 {SOURCE_DIR} 存在且內有 *.csv")
        return

    print(f"\n找到 {len(csv_files)} 個 CSV")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["time"] = df["datetime"].dt.time

            for folder_name, cutoff_time_str in TIME_CUTOFFS.items():
                cutoff_time = datetime.strptime(cutoff_time_str, "%H:%M:%S").time()
                filtered_df = df[df["time"] < cutoff_time].copy()
                filtered_df = filtered_df.drop(columns=["time"])
                output_dir = config.get_dataset_dir(folder_name)
                output_file = output_dir / csv_file.name
                filtered_df.to_csv(output_file, index=False)
                print(f"  {csv_file.name} {folder_name}: {len(filtered_df)} 行")
        except Exception as e:
            print(f"  錯誤 {csv_file.name}: {e}")
            continue

    print("\n截點切分完成。")


if __name__ == "__main__":
    main()
