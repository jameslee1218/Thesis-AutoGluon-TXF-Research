#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline 主程式：依序執行 01～05 模組。

使用方式：
  python main.py              # 執行全部
  python main.py --step 1     # 只執行步驟 1
  python main.py --step 2 --step 4
  python main.py --list       # 列出步驟說明

調用關係（main 會呼叫的腳本／函數）：
  main() → load_config() 設定 PROJECT_ROOT, DATA_ROOT
         → run_step(n) 對每個 n 執行 subprocess(scripts/0n_*/run.py)
  run.py（各模組）→ 依序執行同目錄下腳本：
    01: generate_all_indicators.main / extract_indicators_optimized（或直接執行 .py）
    02: split_by_cutoff（對應 01_split_data）, autoencoder.main（對應 02_autoencoder）
    03: merge_and_train.main（對應 04_merge_features）
    04: visualize_results.main（對應 03_visualize_results）
    05: backtest（新建）
  上述腳本需改為從 config 讀取路徑後，置於對應 scripts/0X_*/ 下即可被 main 調用。
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 專案根目錄（main.py 所在目錄）
REPO_ROOT = Path(__file__).resolve().parent


def load_config():
    """載入 config，並將路徑寫入環境變數供子腳本使用。"""
    sys.path.insert(0, str(REPO_ROOT))
    import config
    os.environ["PROJECT_ROOT"] = str(config.PROJECT_ROOT)
    os.environ["DATA_ROOT"] = str(config.DATA_ROOT)
    return config


def run_step(step: int) -> bool:
    """執行單一步驟：呼叫該模組下的 run.py。"""
    step_dirs = {
        1: "01_data_ingestion",
        2: "02_feature_compression",
        3: "03_modeling",
        4: "04_visualization",
        5: "05_backtest",
    }
    name = step_dirs.get(step)
    if not name:
        return False
    run_py = REPO_ROOT / "scripts" / name / "run.py"
    if not run_py.exists():
        print(f"[main] 未找到 scripts/{name}/run.py，跳過步驟 {step}。")
        return False
    cwd = run_py.parent
    ret = subprocess.run(
        [sys.executable, "run.py"],
        cwd=str(cwd),
        env=os.environ.copy(),
    )
    return ret.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Thesis-AutoGluon-TXF-Research 流程 01～05")
    parser.add_argument(
        "--step",
        type=int,
        action="append",
        dest="steps",
        metavar="N",
        help="只執行步驟 N（可重複，如 --step 1 --step 3）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出各步驟說明後結束",
    )
    args = parser.parse_args()

    load_config()

    STEPS_INFO = [
        (1, "01_data_ingestion", "指標計算與篩選 → indicators_complete, indicators_extracted"),
        (2, "02_feature_compression", "截點切分 + Autoencoder 壓縮 → dataset/, output_0900|0915|0930/"),
        (3, "03_modeling", "合併日表 + Y 報酬率 + AutoGluon 訓練 → merged_for_autogluon, 預測/模型"),
        (4, "04_visualization", "MSE / 雷達 / 重建散點 → visualizations/"),
        (5, "05_backtest", "樣本外回測、特徵重要性 → backtest/"),
    ]

    if args.list:
        for n, name, desc in STEPS_INFO:
            print(f"  步驟 {n}: {name} — {desc}")
        return

    if args.steps:
        steps = sorted(set(args.steps))
        if any(s < 1 or s > 5 for s in steps):
            parser.error("--step 須為 1～5")
    else:
        steps = [1, 2, 3, 4, 5]

    print("=" * 60)
    print("Thesis-AutoGluon-TXF-Research Pipeline")
    print("=" * 60)
    print(f"PROJECT_ROOT = {os.environ.get('PROJECT_ROOT')}")
    print(f"DATA_ROOT    = {os.environ.get('DATA_ROOT')}")
    print(f"執行步驟: {steps}")
    print()

    for step in steps:
        print(f"\n--- 步驟 {step}: {STEPS_INFO[step - 1][1]} ---")
        ok = run_step(step)
        if not ok:
            print(f"[main] 步驟 {step} 未成功完成。")
        print()

    print("Pipeline 結束。")


if __name__ == "__main__":
    main()
