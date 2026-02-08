#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge scattered data and organize into AutoGluon-ready format with descriptive statistics.
本腳本僅負責 merge；AutoGluon 訓練請使用同目錄下 train_autogluon_colab.ipynb（可於 Colab 執行）。
- Loads scattered CSV files and merges them
- Builds daily cutoff samples with target (return from cutoff to close)
- Optional: cleaning steps (constant, low variance, binary pattern, specified indicators)
- Outputs descriptive statistics (summary_stats, missing_values) for the merged dataset
- Saves cleaned data (merged_for_autogluon_0900.csv 等)
"""

import re
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from typing import Optional, List, Dict, Tuple
from datetime import datetime as dt_datetime

# ============== Config（僅讀寫 data/，由專案 config 提供）==============
import sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import config as _config
# 截點：0900 時輸出路徑與檔名皆帶 _0900 後綴（merged_for_autogluon_0900/、merged_for_autogluon_0900.csv）
CUTOFF = "0900"
DATA_DIR = str(_config.get_dataset_dir(CUTOFF))
OUTPUT_BASE = str(_config.get_output_0900_dir())
OUTPUT_DIR = str(_config.get_merged_for_autogluon_dir(CUTOFF))
MODEL_DIR = str(_config.get_output_0900_dir() / "models")

# Target
TARGET_COLUMN = 'target_return'
TARGET_CUTOFF_TIME = '09:01:00'   # e.g. '09:00', '09:15', '09:30'

# Daily aggregate stats: turn each day's minutes (before cutoff) into one row of summary statistics
USE_DAILY_AGGREGATE_STATS = True  # if True, daily row = stats (mean/std/min/max/median) over minutes <= cutoff
AGGREGATE_STATS = ("mean", "std", "min", "max", "median")  # statistic names for aggregation

# Compressed features (from 02_autoencoder.py rolling windows)
USE_COMPRESSED_FEATURES = True    # load W*/compressed_data and merge by window/year
DROP_ORIGINAL_INDICATORS = True   # keep only *_compressed_* and drop raw indicator cols before ML

# Indicator groups (must match 02_autoencoder.py) — used to drop original cols when DROP_ORIGINAL_INDICATORS
INDICATOR_GROUPS_ORDER = [
    "STOCH", "STOCHF", "STOCHRSI", "MACD", "BBANDS", "ADX_DMI", "AROON"
]
INDICATOR_GROUPS = {
    "STOCH": ["STOCH_K_14", "STOCH_D_14"],
    "STOCHF": ["STOCHF_K_14", "STOCHF_D_14"],
    "STOCHRSI": ["STOCHRSI_K_14", "STOCHRSI_D_14"],
    "MACD": ["MACD_12_26", "MACD_signal_12_26", "MACD_hist_12_26"],
    "BBANDS": ["BBANDS_upper_20", "BBANDS_middle_20", "BBANDS_lower_20"],
    "ADX_DMI": ["ADX_14", "ADXR_14", "PDI_14", "MDI_14", "DX_14"],
    "AROON": ["AROON_Down_14", "AROON_Up_14", "AROONOSC_14"]
}
ORIGINAL_INDICATOR_COLUMNS = sorted({c for cols in INDICATOR_GROUPS.values() for c in cols})

# Cleaning (set to False to skip)
AUTO_VARIANCE = True
AUTO_VARIANCE_DROP_PCT = 10.0
AUTO_BINARY = True
AUTO_BINARY_DROP_PCT = 10.0
VARIANCE_THRESHOLD: Optional[float] = None
BINARY_THRESHOLD: Optional[float] = None

# Indicators to remove (optional; empty list = keep all) — applied in addition to DROP_ORIGINAL_INDICATORS
INDICATORS_TO_REMOVE: List[str] = []

# External Y: 收盤-9:00 報酬率 (optional). 從 data/target 讀取。
EXTERNAL_Y_FILE: Optional[str] = None
_y_path = _config.get_y_file()
if os.path.isfile(_y_path):
    EXTERNAL_Y_FILE = _y_path
# Known return column names in external file (first match used); 0900 = log return → will convert to simple
EXTERNAL_Y_DATE_COLS = ("day_id", "date", "日期")
EXTERNAL_Y_RETURN_COLS = ("afternoon_return_0900", "target_return", "y", "報酬率", "return")
EXTERNAL_Y_IS_LOG_RETURN = True  # True if external column is log return (e.g. afternoon_return_0900)

def _compute_cv(series: pd.Series) -> float:
    mean = np.abs(series.mean())
    std = series.std()
    if std == 0:
        return 0.0
    if mean == 0:
        return np.inf
    return float(std / mean)


def _parse_cutoff_time(cutoff_str: str):
    if ':' in cutoff_str and cutoff_str.count(':') >= 2:
        return dt_datetime.strptime(cutoff_str.strip(), "%H:%M:%S").time()
    return pd.to_datetime(cutoff_str).time()


def list_window_folders(output_base: str) -> List[Tuple[str, int]]:
    """List all W* folders under output_base; return [(window_name, compress_year)], sorted by window number."""
    if not os.path.isdir(output_base):
        return []
    # W1_2011-2012_compress_2013-2013 → (W1_..., 2013)
    name_re = re.compile(r"^W(\d+)_(.+)$")
    compress_re = re.compile(r"compress_(\d+)-\d+")
    out = []
    for name in os.listdir(output_base):
        if name.startswith("._"):
            continue
        path = os.path.join(output_base, name)
        if not os.path.isdir(path):
            continue
        m = name_re.match(name)
        if not m:
            continue
        wnum = int(m.group(1))
        cm = compress_re.search(name)
        if not cm:
            continue
        compress_year = int(cm.group(1))
        out.append((wnum, name, compress_year))
    out.sort(key=lambda x: x[0])
    return [(name, compress_year) for _, name, compress_year in out]


def load_compressed_for_window(window_dir: str) -> Optional[pd.DataFrame]:
    """Within one W* folder: read all *_compressed.csv in compressed_data, outer join on datetime."""
    compressed_dir = os.path.join(window_dir, "compressed_data")
    if not os.path.isdir(compressed_dir):
        return None
    csv_files = [
        os.path.join(compressed_dir, f)
        for f in os.listdir(compressed_dir)
        if f.endswith(".csv") and not os.path.basename(f).startswith("._")
    ]
    if not csv_files:
        return None
    dfs = []
    for path in sorted(csv_files):
        try:
            df = pd.read_csv(path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            dfs.append(df)
        except Exception as e:
            print(f"    Skip {os.path.basename(path)}: {e}")
    if not dfs:
        return None
    # Outer join on datetime
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(
            df,
            on="datetime",
            how="outer",
            suffixes=("", "_dup")
        )
        dup = [c for c in merged.columns if c.endswith("_dup")]
        if dup:
            merged = merged.drop(columns=dup)
    merged = merged.sort_values("datetime").reset_index(drop=True)
    return merged


def load_and_merge_compressed_features(output_base: str) -> List[Tuple[int, pd.DataFrame]]:
    """
    Traverse all W* folders under output_base; for each window load compressed_data (horizontal join),
    then return list of (compress_year, df) for lookup by year. No vertical concat — each date uses the window for its year.
    """
    print("=" * 60)
    print("Step: Load compressed features from W* / compressed_data")
    print("=" * 60)
    windows = list_window_folders(output_base)
    if not windows:
        print("No W* window folders found; skipped compressed features.")
        return []
    result = []
    for window_name, compress_year in windows:
        window_dir = os.path.join(output_base, window_name)
        df = load_compressed_for_window(window_dir)
        if df is not None and len(df) > 0:
            result.append((compress_year, df))
            print(f"  {window_name} (compress_year={compress_year}): {len(df)} rows, {len(df.columns)} columns")
    return result


def reduce_compressed_to_daily_cutoff(
    df_compressed: pd.DataFrame,
    cutoff_time,
) -> pd.DataFrame:
    """From minute-level compressed df, keep one row per date: last row with datetime <= cutoff that day."""
    if "datetime" not in df_compressed.columns:
        return pd.DataFrame()
    work = df_compressed.copy()
    work["date"] = work["datetime"].dt.date
    work["time"] = work["datetime"].dt.time
    # filter to rows at or before cutoff
    work = work[work["time"] <= cutoff_time]
    if work.empty:
        return pd.DataFrame()
    # last row per date
    daily = work.sort_values("datetime").groupby("date", as_index=False).last()
    daily = daily.drop(columns=["time"], errors="ignore")
    return daily


def reduce_compressed_to_daily_aggregate(
    df_compressed: pd.DataFrame,
    cutoff_time,
    stats: Tuple[str, ...] = None,
) -> pd.DataFrame:
    """From minute-level compressed df, one row per date: aggregate stats over rows with datetime <= cutoff."""
    if stats is None:
        stats = AGGREGATE_STATS
    if "datetime" not in df_compressed.columns:
        return pd.DataFrame()
    work = df_compressed.copy()
    work["date"] = work["datetime"].dt.date
    work = work[work["datetime"].dt.time <= cutoff_time]
    if work.empty:
        return pd.DataFrame()
    numeric_cols = [c for c in work.columns if c not in ("datetime", "date")]
    if not numeric_cols:
        return pd.DataFrame()
    agg_map = {
        "mean": np.nanmean,
        "std": lambda x: x.std() if x.notna().sum() > 1 else np.nan,
        "min": np.nanmin,
        "max": np.nanmax,
        "median": np.nanmedian,
    }
    rows = []
    for date_val, g in work.groupby("date"):
        row = {"date": date_val}
        for col in numeric_cols:
            s = g[col]
            for stat in stats:
                func = agg_map.get(stat)
                if func is not None:
                    try:
                        row[f"{col}_{stat}"] = func(s)
                    except Exception:
                        row[f"{col}_{stat}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def merge_compressed_into_daily(
    df_daily: pd.DataFrame,
    compressed_by_year: List[Tuple[int, pd.DataFrame]],
    cutoff_time,
) -> pd.DataFrame:
    """
    For each row in df_daily, attach compressed features from the window whose compress_year equals row date's year.
    Only compressed columns are added; datetime/date from compressed are not overwritten.
    """
    if not compressed_by_year:
        return df_daily
    year_to_df = {year: reduce_compressed_to_daily_cutoff(df, cutoff_time) for year, df in compressed_by_year}
    compressed_cols = [c for c in compressed_by_year[0][1].columns if c not in ("datetime", "date")]
    out = df_daily.copy()
    for col in compressed_cols:
        out[col] = np.nan
    for idx, row in out.iterrows():
        d = row.get("date")
        if d is None and "datetime" in row:
            d = pd.Timestamp(row["datetime"]).date()
        if d is None:
            continue
        y = d.year if hasattr(d, "year") else pd.Timestamp(d).year
        if y not in year_to_df:
            continue
        daily_comp = year_to_df[y]
        day_rows = daily_comp[daily_comp["date"] == d]
        if day_rows.empty:
            continue
        for col in compressed_cols:
            if col in day_rows.columns:
                out.at[idx, col] = day_rows[col].iloc[0]
    return out


def merge_compressed_into_daily_aggregate(
    df_daily: pd.DataFrame,
    compressed_by_year: List[Tuple[int, pd.DataFrame]],
    cutoff_time,
    stats: Tuple[str, ...] = None,
) -> pd.DataFrame:
    """
    For each (compress_year, df_compressed), reduce to daily aggregate stats; then attach to df_daily by date/year.
    """
    if stats is None:
        stats = AGGREGATE_STATS
    year_to_daily = {}
    for year, df_comp in compressed_by_year:
        daily_agg = reduce_compressed_to_daily_aggregate(df_comp, cutoff_time, stats=stats)
        if not daily_agg.empty:
            year_to_daily[year] = daily_agg
    if not year_to_daily:
        return df_daily
    compressed_cols = [c for c in list(year_to_daily.values())[0].columns if c != "date"]
    out = df_daily.copy()
    for col in compressed_cols:
        out[col] = np.nan
    for idx, row in out.iterrows():
        d = row.get("date")
        if d is None and "datetime" in row:
            d = pd.Timestamp(row["datetime"]).date()
        if d is None:
            continue
        y = d.year if hasattr(d, "year") else pd.Timestamp(d).year
        if y not in year_to_daily:
            continue
        day_df = year_to_daily[y]
        match = day_df[day_df["date"] == d]
        if match.empty:
            continue
        for col in compressed_cols:
            if col in match.columns:
                out.at[idx, col] = match[col].iloc[0]
    return out


def drop_original_indicators(df: pd.DataFrame, aggregate_mode: bool = False) -> pd.DataFrame:
    """Remove raw indicator columns so only compressed (and other) features remain."""
    to_drop = [c for c in ORIGINAL_INDICATOR_COLUMNS if c in df.columns]
    if aggregate_mode:
        for orig in ORIGINAL_INDICATOR_COLUMNS:
            for stat in AGGREGATE_STATS:
                name = f"{orig}_{stat}"
                if name in df.columns:
                    to_drop.append(name)
        to_drop = list(dict.fromkeys(to_drop))
    if not to_drop:
        return df
    out = df.drop(columns=[c for c in to_drop if c in df.columns])
    print(f"  Dropped {len(to_drop)} original indicator columns (keeping compressed only).")
    return out


def load_data(data_dir: str) -> pd.DataFrame:
    """Load all TX*_1K_qlib_indicators_complete.csv and merge."""
    print("=" * 60)
    print("Step 1: Load and merge scattered CSVs")
    print("=" * 60)
    csv_files = glob.glob(os.path.join(data_dir, "TX*_1K_qlib_indicators_complete.csv"))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("._")]
    print(f"Found {len(csv_files)} CSV files")
    if not csv_files:
        raise ValueError(f"No CSV files in {data_dir}")
    dataframes = []
    for path in sorted(csv_files):
        try:
            df = pd.read_csv(path)
            dataframes.append(df)
        except Exception as e:
            print(f"  Skip {os.path.basename(path)}: {e}")
    if not dataframes:
        raise ValueError("No data loaded")
    combined = pd.concat(dataframes, ignore_index=True)
    combined = combined.sort_values('datetime').reset_index(drop=True)
    print(f"Merged: {len(combined)} rows, {len(combined.columns)} columns")
    return combined


def build_cutoff_target_dataset(df: pd.DataFrame, cutoff_time=None) -> pd.DataFrame:
    """One row per day: last row at or before cutoff_time; target = (close_eod - close_cutoff) / close_cutoff."""
    print("\n=== Build daily cutoff samples (AutoGluon-ready format) ===")
    if 'datetime' not in df.columns or 'close' not in df.columns:
        raise ValueError("Need 'datetime' and 'close'")
    if cutoff_time is None:
        cutoff_time = _parse_cutoff_time(TARGET_CUTOFF_TIME)
    work = df.copy()
    work['datetime'] = pd.to_datetime(work['datetime'])
    work['date'] = work['datetime'].dt.date
    samples = []
    for date_val, grp in work.groupby('date'):
        g = grp.sort_values('datetime')
        close_eod = g['close'].iloc[-1]
        before = g[g['datetime'].dt.time <= cutoff_time]
        if before.empty:
            continue
        row = before.tail(1).copy()
        close_cutoff = float(row['close'].values[0])
        if close_cutoff == 0:
            continue
        row[TARGET_COLUMN] = (close_eod - close_cutoff) / close_cutoff
        samples.append(row)
    if not samples:
        raise ValueError("No daily samples; check TARGET_CUTOFF_TIME and data.")
    result = pd.concat(samples, ignore_index=True)
    print(f"Daily samples: {len(result)}")
    return result


def build_daily_aggregate_dataset(
    df: pd.DataFrame,
    cutoff_time,
    stats: Tuple[str, ...] = None,
) -> pd.DataFrame:
    """
    One row per day: aggregate statistics over all rows at or before cutoff_time.
    Columns: date, target_return, and for each numeric col: col_mean, col_std, col_min, col_max, col_median.
    Target = (close_eod - close_cutoff) / close_cutoff (close_cutoff = close at last row <= cutoff).
    """
    if stats is None:
        stats = AGGREGATE_STATS
    print("\n=== Build daily aggregate stats (minute -> one row per day, 2D for AutoGluon) ===")
    if "datetime" not in df.columns or "close" not in df.columns:
        raise ValueError("Need 'datetime' and 'close'")
    work = df.copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    work["date"] = work["datetime"].dt.date
    numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    if "close" not in numeric_cols:
        numeric_cols.append("close")
    agg_map = {
        "mean": np.nanmean,
        "std": lambda x: x.std() if x.notna().sum() > 1 else np.nan,
        "min": np.nanmin,
        "max": np.nanmax,
        "median": np.nanmedian,
    }
    rows = []
    for date_val, grp in work.groupby("date"):
        g = grp.sort_values("datetime")
        close_eod = float(g["close"].iloc[-1])
        before = g[g["datetime"].dt.time <= cutoff_time]
        if before.empty:
            continue
        close_cutoff = float(before["close"].iloc[-1])
        if close_cutoff == 0:
            continue
        target_return = (close_eod - close_cutoff) / close_cutoff
        row = {"date": date_val, TARGET_COLUMN: target_return}
        for col in numeric_cols:
            if col not in before.columns:
                continue
            s = before[col]
            for stat in stats:
                func = agg_map.get(stat)
                if func is None:
                    continue
                try:
                    val = func(s)
                except Exception:
                    val = np.nan
                row[f"{col}_{stat}"] = val
        rows.append(row)
    if not rows:
        raise ValueError("No daily samples; check TARGET_CUTOFF_TIME and data.")
    result = pd.DataFrame(rows)
    print(f"Daily samples: {len(result)}, columns: {len(result.columns)} (aggregate stats per numeric field)")
    return result


def generate_descriptive_statistics(df: pd.DataFrame, output_dir: str) -> None:
    """Export descriptive statistics (describe, missing) for the merged/AutoGluon-ready dataset."""
    print("\n=== Descriptive statistics ===")
    os.makedirs(output_dir, exist_ok=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns for stats.")
        return
    summary = df[numeric_cols].describe()
    summary_path = os.path.join(output_dir, "summary_statistics.csv")
    summary.to_csv(summary_path, encoding='utf-8-sig')
    print(f"Saved: {summary_path}")
    n = len(df)
    missing = pd.DataFrame({
        'Feature': numeric_cols,
        'Missing_Count': [df[c].isna().sum() for c in numeric_cols],
        'Missing_Pct': [df[c].isna().sum() / n * 100 if n else 0 for c in numeric_cols]
    })
    missing_path = os.path.join(output_dir, "missing_values.csv")
    missing.to_csv(missing_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {missing_path}")


def remove_constant_features(df: pd.DataFrame, skip_cols: tuple = ('datetime', TARGET_COLUMN, 'date')) -> pd.DataFrame:
    out = df.copy()
    drop = [c for c in out.columns if c not in skip_cols and out[c].nunique() <= 1]
    if drop:
        out = out.drop(columns=drop)
        print(f"  Dropped {len(drop)} constant features")
    return out


def remove_low_variance_features(df: pd.DataFrame, skip_cols: tuple = ('datetime', TARGET_COLUMN, 'date')) -> pd.DataFrame:
    out = df.copy()
    cv_map = {}
    for c in out.columns:
        if c in skip_cols:
            continue
        if out[c].dtype in ['int64', 'float64']:
            cv_map[c] = _compute_cv(out[c])
    if not cv_map:
        return out
    if AUTO_VARIANCE:
        th = np.nanpercentile(list(cv_map.values()), AUTO_VARIANCE_DROP_PCT)
        drop = [c for c, v in cv_map.items() if np.isfinite(v) and v < th]
    elif VARIANCE_THRESHOLD is not None:
        drop = [c for c, v in cv_map.items() if np.isfinite(v) and v < float(VARIANCE_THRESHOLD)]
    else:
        return out
    if drop:
        out = out.drop(columns=drop)
        print(f"  Dropped {len(drop)} low-variance features")
    return out


def remove_binary_pattern_features(df: pd.DataFrame, skip_cols: tuple = ('datetime', TARGET_COLUMN, 'date')) -> pd.DataFrame:
    out = df.copy()
    minority_ratios = {}
    for c in out.columns:
        if c in skip_cols:
            continue
        uniq = set(out[c].dropna().unique())
        if uniq.issubset({0, 1}):
            vc = out[c].value_counts()
            if len(vc) == 2:
                minority_ratios[c] = vc.min() / len(out[c])
    if not minority_ratios:
        return out
    if AUTO_BINARY:
        th = np.nanpercentile(list(minority_ratios.values()), AUTO_BINARY_DROP_PCT)
        drop = [c for c, r in minority_ratios.items() if r < th]
    elif BINARY_THRESHOLD is not None:
        drop = [c for c, r in minority_ratios.items() if r < float(BINARY_THRESHOLD)]
    else:
        return out
    if drop:
        out = out.drop(columns=drop)
        print(f"  Dropped {len(drop)} binary-pattern features")
    return out


def remove_specified_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if not INDICATORS_TO_REMOVE:
        return df
    existing = [c for c in INDICATORS_TO_REMOVE if c in df.columns]
    if existing:
        return df.drop(columns=existing)
    return df


def prepare_ml_table(df: pd.DataFrame, drop_date_for_training: bool = True) -> Optional[pd.DataFrame]:
    """Return table ready for AutoGluon fit: numeric features + label, no datetime/date (avoid leakage)."""
    out = df.drop(columns=["datetime"], errors="ignore")
    if drop_date_for_training:
        out = out.drop(columns=["date"], errors="ignore")
    if TARGET_COLUMN not in out.columns:
        print(f"Warning: no column '{TARGET_COLUMN}'")
        return None
    out = out.dropna()
    print(f"ML table: {out.shape} (date dropped for training to avoid leakage)")
    return out


def load_external_y(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load external y file (xlsx or csv) with 收盤-9:00 報酬率.
    Returns DataFrame with columns [date, target_return]; date is datetime.date.
    If return column is log return (e.g. afternoon_return_0900), convert to simple return.
    """
    if not file_path or not os.path.isfile(file_path):
        return None
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".xlsx":
            df = pd.read_excel(file_path, engine="openpyxl")
        elif ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        else:
            return None
    except Exception as e:
        print(f"  [WARN] Cannot read external y file {file_path}: {e}")
        return None
    if df.empty:
        return None
    date_col = None
    for c in EXTERNAL_Y_DATE_COLS:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        print(f"  [WARN] External y: no date column among {EXTERNAL_Y_DATE_COLS}")
        return None
    return_col = None
    for c in EXTERNAL_Y_RETURN_COLS:
        if c in df.columns:
            return_col = c
            break
    if return_col is None:
        print(f"  [WARN] External y: no return column among {EXTERNAL_Y_RETURN_COLS}")
        return None
    out = df[[date_col, return_col]].copy()
    out = out.rename(columns={date_col: "date_raw", return_col: "y_value"})
    # Parse date: support YYYYMMDD (int/str) or datetime string
    date_raw = out["date_raw"].astype(str).str.strip()
    date_clean = date_raw.str.replace(r"\D", "", regex=True)
    mask_8 = date_clean.str.len() == 8
    out["date"] = pd.NaT
    out.loc[mask_8, "date"] = pd.to_datetime(date_clean[mask_8], format="%Y%m%d", errors="coerce").dt.date
    out.loc[~mask_8, "date"] = pd.to_datetime(date_raw[~mask_8], errors="coerce").dt.date
    out = out.drop(columns=["date_raw"]).dropna(subset=["date", "y_value"])
    if EXTERNAL_Y_IS_LOG_RETURN:
        out[TARGET_COLUMN] = np.exp(out["y_value"].astype(float)) - 1.0
    else:
        out[TARGET_COLUMN] = out["y_value"].astype(float)
    out = out[["date", TARGET_COLUMN]]
    return out


def merge_external_y_into_daily(df_daily: pd.DataFrame, df_y: pd.DataFrame) -> pd.DataFrame:
    """Left-merge external y on date; use external target_return where matched, else keep computed."""
    if df_y is None or df_y.empty or "date" not in df_daily.columns:
        return df_daily
    computed = df_daily[TARGET_COLUMN].copy() if TARGET_COLUMN in df_daily.columns else None
    df_daily = df_daily.drop(columns=[TARGET_COLUMN], errors="ignore")
    # Normalize date to same type for merge (datetime64[ns] date-only)
    df_daily = df_daily.copy()
    df_y = df_y.copy()
    df_daily["_date_key"] = pd.to_datetime(df_daily["date"], errors="coerce").dt.normalize()
    df_y["_date_key"] = pd.to_datetime(df_y["date"], errors="coerce").dt.normalize()
    df_daily = df_daily.merge(
        df_y[["_date_key", TARGET_COLUMN]], on="_date_key", how="left", suffixes=("", "_y")
    )
    df_daily = df_daily.drop(columns=["_date_key"], errors="ignore")
    if TARGET_COLUMN not in df_daily.columns:
        return df_daily
    if computed is not None:
        df_daily[TARGET_COLUMN] = df_daily[TARGET_COLUMN].fillna(computed)
    return df_daily


def main():
    print("=" * 60)
    print("Merge features for AutoGluon + descriptive statistics")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1) Load and merge raw CSVs
    df_raw = load_data(DATA_DIR)

    cutoff_time = _parse_cutoff_time(TARGET_CUTOFF_TIME)

    # 2) Build daily dataset: either one row per day with aggregate stats (minute -> 2D) or single snapshot at cutoff
    if USE_DAILY_AGGREGATE_STATS:
        df_daily = build_daily_aggregate_dataset(df_raw, cutoff_time, stats=AGGREGATE_STATS)
    else:
        df_daily = build_cutoff_target_dataset(df_raw, cutoff_time=cutoff_time)

    # 2b) Load compressed features from W*/compressed_data and merge into daily (by window/year)
    if USE_COMPRESSED_FEATURES:
        compressed_by_year = load_and_merge_compressed_features(OUTPUT_BASE)
        if compressed_by_year:
            if USE_DAILY_AGGREGATE_STATS:
                df_daily = merge_compressed_into_daily_aggregate(
                    df_daily, compressed_by_year, cutoff_time, stats=AGGREGATE_STATS
                )
            else:
                df_daily = merge_compressed_into_daily(df_daily, compressed_by_year, cutoff_time)
            if DROP_ORIGINAL_INDICATORS:
                df_daily = drop_original_indicators(df_daily, aggregate_mode=USE_DAILY_AGGREGATE_STATS)

    # 2c) Merge external y (收盤-9:00 報酬率 from y.xlsx / y.csv) into daily table
    if EXTERNAL_Y_FILE:
        print("\n=== Merge external Y (收盤-9:00 報酬率) ===")
        df_y = load_external_y(EXTERNAL_Y_FILE)
        if df_y is not None and len(df_y) > 0:
            df_daily = merge_external_y_into_daily(df_daily, df_y)
            print(f"  Loaded {EXTERNAL_Y_FILE}: {len(df_y)} rows; merged into daily table.")
        else:
            print(f"  [WARN] Could not load or empty: {EXTERNAL_Y_FILE}")
    else:
        print("\n(No external y file found: put y.xlsx or y.csv in 01w5 folder to use 收盤-9:00 報酬率)")

    # 3) Descriptive statistics on the AutoGluon-ready dataset (before cleaning)
    generate_descriptive_statistics(df_daily, OUTPUT_DIR)

    # 4) Optional cleaning
    df_clean = remove_constant_features(df_daily)
    df_clean = remove_low_variance_features(df_clean)
    df_clean = remove_binary_pattern_features(df_clean)
    df_clean = remove_specified_indicators(df_clean)

    # 5) Save merged (cleaned) table for AutoGluon
    out_path = os.path.join(OUTPUT_DIR, f"merged_for_autogluon_{CUTOFF}.csv")
    df_clean.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved merged table: {out_path}")

    # 6) Descriptive statistics on cleaned data
    generate_descriptive_statistics(df_clean, os.path.join(OUTPUT_DIR, "stats_after_cleaning"))

    print("\nDone. AutoGluon 訓練請使用同目錄 train_autogluon_colab.ipynb。")


if __name__ == "__main__":
    main()
