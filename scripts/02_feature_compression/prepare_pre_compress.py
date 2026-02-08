#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare cleaned inputs for autoencoder.

Reads data/dataset/{cutoff} and writes cleaned CSVs to
data/pre compress/{cutoff} with the same filenames.
"""

import sys
from pathlib import Path
import argparse
import glob
import os

import numpy as np
import pandas as pd

# Project config (data paths)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import config as _config


def _clean_one_file(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "datetime" not in df.columns:
        raise ValueError(f"Missing 'datetime' column in {os.path.basename(csv_path)}")

    dt = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.loc[~dt.isna()].copy()
    df["datetime"] = dt.loc[~dt.isna()].values

    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    value_cols = [c for c in df.columns if c != "datetime"]
    if value_cols:
        df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
        df[value_cols] = np.nan_to_num(df[value_cols].values, nan=0.0, posinf=1e6, neginf=-1e6)
        df[value_cols] = df[value_cols].astype(float)

    cols = ["datetime"] + [c for c in df.columns if c != "datetime"]
    return df[cols]


def _prepare_cutoff(cutoff: str) -> None:
    input_dir = _config.get_dataset_dir(cutoff)
    output_dir = _config.get_pre_compress_dir(cutoff)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(glob.glob(str(input_dir / "TX*_1K_qlib_indicators_complete.csv")))
    if not csv_files:
        print(f"[WARN] No CSV files found for cutoff {cutoff}: {input_dir}")
        return

    print(f"[INFO] cutoff={cutoff} files={len(csv_files)} input={input_dir} output={output_dir}")
    for csv_path in csv_files:
        cleaned = _clean_one_file(csv_path)
        out_path = output_dir / Path(csv_path).name
        cleaned.to_csv(out_path, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare cleaned pre-compress datasets.")
    parser.add_argument(
        "--cutoffs",
        nargs="*",
        default=list(_config.CUTOFFS),
        help="Cutoffs to process, e.g. 0900 0915 0930",
    )
    args = parser.parse_args()

    for cutoff in args.cutoffs:
        if cutoff not in _config.CUTOFFS:
            print(f"[WARN] Skip unknown cutoff: {cutoff}")
            continue
        _prepare_cutoff(cutoff)

    print("[DONE] Pre-compress data prepared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
