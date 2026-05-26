#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新版主流程（無 Autoencoder）：
將每交易日 indicators_complete xlsx 攤平成 AutoGluon 可用的大表（一天一列）。

輸出：
- data/autogluon/all/merged_for_autogluon_all.xlsx（不含 target_return）
- data/autogluon/0900/merged_for_autogluon_0900.xlsx
- data/autogluon/0915/merged_for_autogluon_0915.xlsx
- data/autogluon/0930/merged_for_autogluon_0930.xlsx

target_return（僅 cutoff 檔）：
    (當日最後一筆 close - cutoff close) / cutoff close
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config

INDICATOR_DIR = Path(config.get_indicators_complete_dir())
RAW_DIR = Path(config.get_raw_kline_dir())
AUTOGLUON_DIR = Path(config.DATA_ROOT) / "autogluon"

STATIC_COLS = ["sp500_prev_return", "open_gap_pct", "vix_prev_close"]
TIME_VARYING_COLS = [
    "close",
    "volume",
    "vwap_bias_5",
    "volume_roc_5",
    "atr_5",
    "kd_k_9",
    "kd_d_9",
    "linearreg_angle_5",
]
COL_ALIAS = {"linearreg_angle_5": "angle_5"}
CUTOFFS = ("0900", "0915", "0930")


def list_indicator_files() -> List[Path]:
    return sorted(
        p for p in INDICATOR_DIR.glob("TX*_qlib_indicators_complete.xlsx") if not p.name.startswith("._")
    )


def load_raw_day_close(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    colmap = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=colmap)
    if "date" in df.columns and "datetime" not in df.columns:
        df = df.rename(columns={"date": "datetime"})
    if "datetime" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{path.name} 缺少 datetime/date 或 close")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["datetime", "close"]).sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "close"]]


def compute_target_map(cutoff_hhmm: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in sorted(RAW_DIR.glob("TX*_1K.csv")):
        if p.name.startswith("._"):
            continue
        day_df = load_raw_day_close(p)
        if day_df.empty:
            continue
        day = day_df["datetime"].iloc[0].strftime("%Y-%m-%d")
        times = day_df["datetime"].dt.strftime("%H%M")
        hit = day_df.loc[times == cutoff_hhmm, "close"]
        if hit.empty:
            out[day] = pd.NA
            continue
        cutoff_close = float(hit.iloc[-1])
        day_close = float(day_df["close"].iloc[-1])
        out[day] = pd.NA if cutoff_close == 0 else (day_close - cutoff_close) / cutoff_close
    return out


def flatten_one_day(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    if dt.isna().all():
        raise ValueError("datetime 欄位無法解析")

    out["date"] = dt.iloc[0].strftime("%Y-%m-%d")
    for c in STATIC_COLS:
        out[c] = df[c].iloc[0] if c in df.columns else pd.NA

    for i in range(len(df)):
        hhmm = dt.iloc[i].strftime("%H%M")
        for c in TIME_VARYING_COLS:
            if c not in df.columns:
                continue
            alias = COL_ALIAS.get(c, c)
            out[f"{alias}_{hhmm}"] = df[c].iloc[i]
    return out


def cut_by_time_columns(df: pd.DataFrame, cutoff_hhmm: str) -> pd.DataFrame:
    base_cols = ["date", "sp500_prev_return", "open_gap_pct", "vix_prev_close"]
    keep = [c for c in base_cols if c in df.columns]

    time_pat = re.compile(r"_(\d{4})$")
    timed_cols = []
    for c in df.columns:
        m = time_pat.search(c)
        if not m:
            continue
        if m.group(1) <= cutoff_hhmm:
            timed_cols.append(c)
    return df[keep + timed_cols].copy()


def main() -> None:
    files = list_indicator_files()
    if not files:
        raise FileNotFoundError(f"找不到 indicators 檔案: {INDICATOR_DIR}")

    target_maps = {cutoff: compute_target_map(cutoff) for cutoff in CUTOFFS}
    valid_dates = set(target_maps["0900"].keys())

    rows: List[Dict[str, float]] = []
    for p in files:
        row = flatten_one_day(pd.read_excel(p))
        if row["date"] not in valid_dates:
            continue
        rows.append(row)

    wide_df = (
        pd.DataFrame(rows)
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    out_all_dir = AUTOGLUON_DIR / "all"
    out_all_dir.mkdir(parents=True, exist_ok=True)
    out_all_path = out_all_dir / "merged_for_autogluon_all.xlsx"
    wide_df.to_excel(out_all_path, index=False, engine="openpyxl")

    for cutoff in CUTOFFS:
        out_dir = AUTOGLUON_DIR / cutoff
        out_dir.mkdir(parents=True, exist_ok=True)
        cut_df = cut_by_time_columns(wide_df, cutoff)
        cut_df["target_return"] = cut_df["date"].map(target_maps[cutoff])
        out_path = out_dir / f"merged_for_autogluon_{cutoff}.xlsx"
        cut_df.to_excel(out_path, index=False, engine="openpyxl")

    print("-" * 60)
    print(f"輸入日檔案數: {len(files)}")
    print(f"all 列數: {len(wide_df)}，欄數: {wide_df.shape[1]}")
    print(f"輸出完成: {AUTOGLUON_DIR}")
    for cutoff in CUTOFFS:
        non_null_count = int(pd.Series(target_maps[cutoff]).notna().sum())
        print(f"{cutoff} target_return 有值筆數: {non_null_count}")


if __name__ == "__main__":
    main()
