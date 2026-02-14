#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 output2 壓縮結果合併為 AutoGluon 訓練格式（敘述統計），取代舊 output_0900 W* 壓縮。
- 讀取 data/output2/year_*/ 的 compressed_data
- 讀取 data/dataset/{0900,0915,0930}/ 的截點前分鐘資料
- 合併 target/y.xlsx（Y）與壓縮特徵（X）
- 依截點 0900、0915、0930 分別產出敘述統計格式
- 輸出至 data/autogluon_ready/{0900,0915,0930}/
"""

import os
import glob
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime as dt_datetime

import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import config as _config

# ============== Config ==============
DATA_ROOT = Path(_config.DATA_ROOT)
OUTPUT2_DIR = DATA_ROOT / "output2"
CUTOFFS = ("0900", "0915", "0930")
CUTOFF_TIMES = {
    "0900": "09:01:00",   # 截點前：<= 09:00
    "0915": "09:16:00",   # 截點前：<= 09:15
    "0930": "09:31:00",   # 截點前：<= 09:30
}
OUTPUT_BASE = DATA_ROOT / "autogluon_ready"

TARGET_COLUMN = "target_return"
AGGREGATE_STATS = ("mean", "std", "min", "max", "median")
INDICATORS = ["ADX_DMI", "AROON", "BBANDS", "MACD", "STOCH", "STOCHF", "STOCHRSI"]
ORIGINAL_INDICATOR_COLUMNS = [
    "STOCH_K_14", "STOCH_D_14", "STOCHF_K_14", "STOCHF_D_14", "STOCHRSI_K_14", "STOCHRSI_D_14",
    "MACD_12_26", "MACD_signal_12_26", "MACD_hist_12_26",
    "BBANDS_upper_20", "BBANDS_middle_20", "BBANDS_lower_20",
    "ADX_14", "ADXR_14", "PDI_14", "MDI_14", "DX_14",
    "AROON_Down_14", "AROON_Up_14", "AROONOSC_14"
]
EXTERNAL_Y_DATE_COLS = ("day_id", "date", "日期")
EXTERNAL_Y_RETURN_COLS = ("afternoon_return_0900", "target_return", "y", "報酬率", "return")
EXTERNAL_Y_IS_LOG_RETURN = True


def _parse_cutoff_time(cutoff_str: str):
    if ':' in cutoff_str and cutoff_str.count(':') >= 2:
        return dt_datetime.strptime(cutoff_str.strip(), "%H:%M:%S").time()
    return pd.to_datetime(cutoff_str).time()


def list_year_folders(output2_dir: Path) -> List[Tuple[str, int]]:
    """List year_* folders; return [(folder_name, year)] sorted by year."""
    if not output2_dir.is_dir():
        return []
    out = []
    for name in os.listdir(output2_dir):
        if name.startswith("year_") and not name.startswith("._"):
            try:
                year = int(name.replace("year_", ""))
                out.append((name, year))
            except ValueError:
                pass
    out.sort(key=lambda x: x[1])
    return out


def load_compressed_for_year(output2_dir: Path, year_folder: str) -> Optional[pd.DataFrame]:
    """Load all 7 indicators from year_YYYY/{INDICATOR}/compressed_data/ and merge on datetime."""
    dfs = []
    for ind in INDICATORS:
        path = output2_dir / year_folder / ind / "compressed_data" / f"{ind}_compressed.csv"
        if path.is_file():
            try:
                df = pd.read_csv(path)
                df["datetime"] = pd.to_datetime(df["datetime"])
                dfs.append(df)
            except Exception as e:
                print(f"    Skip {path.name}: {e}")
    if not dfs:
        return None
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="datetime", how="outer", suffixes=("", "_dup"))
        dup = [c for c in merged.columns if c.endswith("_dup")]
        if dup:
            merged = merged.drop(columns=dup)
    merged = merged.sort_values("datetime").reset_index(drop=True)
    return merged


def load_and_merge_compressed_from_output2(output2_dir: Path) -> List[Tuple[int, pd.DataFrame]]:
    """Load compressed features from output2/year_*; return [(year, df)]."""
    print("=" * 60)
    print("Step: Load compressed features from output2/year_*")
    print("=" * 60)
    years = list_year_folders(output2_dir)
    if not years:
        print("No year_* folders found in output2.")
        return []
    result = []
    for folder, year in years:
        df = load_compressed_for_year(output2_dir, folder)
        if df is not None and len(df) > 0:
            result.append((year, df))
            print(f"  {folder} (year={year}): {len(df)} rows, {len(df.columns)} columns")
    return result


def reduce_compressed_to_daily_aggregate(
    df_compressed: pd.DataFrame,
    cutoff_time,
    stats: Tuple[str, ...] = None,
) -> pd.DataFrame:
    """From minute-level compressed df, one row per date: aggregate stats over rows <= cutoff."""
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


def merge_compressed_into_daily_aggregate(
    df_daily: pd.DataFrame,
    compressed_by_year: List[Tuple[int, pd.DataFrame]],
    cutoff_time,
    stats: Tuple[str, ...] = None,
) -> pd.DataFrame:
    """Attach compressed features (daily aggregate) to df_daily by date/year."""
    if stats is None:
        stats = AGGREGATE_STATS
    if not compressed_by_year:
        return df_daily
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


def drop_original_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Remove raw indicator columns; keep only compressed and other features."""
    to_drop = []
    for c in df.columns:
        for orig in ORIGINAL_INDICATOR_COLUMNS:
            if c == orig or c.startswith(orig + "_"):
                to_drop.append(c)
                break
    to_drop = list(dict.fromkeys([c for c in to_drop if c in df.columns]))
    if not to_drop:
        return df
    out = df.drop(columns=to_drop)
    print(f"  Dropped {len(to_drop)} original indicator columns (keeping compressed only).")
    return out


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load all TX*_1K_qlib_indicators_complete.csv from data_dir and merge."""
    csv_files = list(data_dir.glob("TX*_1K_qlib_indicators_complete.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith("._")]
    if not csv_files:
        raise ValueError(f"No CSV files in {data_dir}")
    dfs = []
    for path in sorted(csv_files):
        try:
            dfs.append(pd.read_csv(path))
        except Exception as e:
            print(f"  Skip {path.name}: {e}")
    if not dfs:
        raise ValueError("No data loaded")
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("datetime").reset_index(drop=True)
    return combined


def build_daily_aggregate_dataset(
    df: pd.DataFrame,
    cutoff_time,
    stats: Tuple[str, ...] = None,
) -> pd.DataFrame:
    """One row per day: aggregate stats over rows <= cutoff; target = (close_eod - close_cutoff)/close_cutoff."""
    if stats is None:
        stats = AGGREGATE_STATS
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
        raise ValueError("No daily samples; check cutoff_time and data.")
    return pd.DataFrame(rows)


def load_external_y(file_path: str) -> Optional[pd.DataFrame]:
    """Load y.xlsx/y.csv with 收盤-9:00 報酬率; return [date, target_return]."""
    if not file_path or not Path(file_path).is_file():
        return None
    ext = Path(file_path).suffix.lower()
    try:
        if ext == ".xlsx":
            df = pd.read_excel(file_path, engine="openpyxl")
        elif ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        else:
            return None
    except Exception as e:
        print(f"  [WARN] Cannot read y file {file_path}: {e}")
        return None
    if df.empty:
        return None
    date_col = next((c for c in EXTERNAL_Y_DATE_COLS if c in df.columns), None)
    return_col = next((c for c in EXTERNAL_Y_RETURN_COLS if c in df.columns), None)
    if date_col is None or return_col is None:
        return None
    out = df[[date_col, return_col]].copy()
    out = out.rename(columns={date_col: "date_raw", return_col: "y_value"})
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
    return out[["date", TARGET_COLUMN]]


def merge_external_y_into_daily(df_daily: pd.DataFrame, df_y: pd.DataFrame) -> pd.DataFrame:
    """Left-merge external y on date."""
    if df_y is None or df_y.empty or "date" not in df_daily.columns:
        return df_daily
    computed = df_daily[TARGET_COLUMN].copy() if TARGET_COLUMN in df_daily.columns else None
    df_daily = df_daily.drop(columns=[TARGET_COLUMN], errors="ignore")
    df_daily = df_daily.copy()
    df_y = df_y.copy()
    df_daily["_date_key"] = pd.to_datetime(df_daily["date"], errors="coerce").dt.normalize()
    df_y["_date_key"] = pd.to_datetime(df_y["date"], errors="coerce").dt.normalize()
    df_daily = df_daily.merge(
        df_y[["_date_key", TARGET_COLUMN]], on="_date_key", how="left"
    )
    df_daily = df_daily.drop(columns=["_date_key"], errors="ignore")
    if TARGET_COLUMN not in df_daily.columns:
        return df_daily
    if computed is not None:
        df_daily[TARGET_COLUMN] = df_daily[TARGET_COLUMN].fillna(computed)
    return df_daily


def remove_constant_features(df: pd.DataFrame, skip_cols: tuple = ("datetime", TARGET_COLUMN, "date")) -> pd.DataFrame:
    drop = [c for c in df.columns if c not in skip_cols and df[c].nunique() <= 1]
    if drop:
        df = df.drop(columns=drop)
        print(f"  Dropped {len(drop)} constant features")
    return df


def generate_descriptive_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """Export summary_statistics.csv and missing_values.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return
    summary = df[numeric_cols].describe()
    summary.to_csv(output_dir / "summary_statistics.csv", encoding="utf-8-sig")
    n = len(df)
    missing = pd.DataFrame({
        "Feature": numeric_cols,
        "Missing_Count": [df[c].isna().sum() for c in numeric_cols],
        "Missing_Pct": [df[c].isna().sum() / n * 100 if n else 0 for c in numeric_cols]
    })
    missing.to_csv(output_dir / "missing_values.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved summary_statistics.csv, missing_values.csv")


def process_cutoff(cutoff: str, compressed_by_year: List[Tuple[int, pd.DataFrame]], y_file: Optional[str]) -> None:
    """Process one cutoff: load data, merge compressed, merge y, output."""
    print("\n" + "=" * 60)
    print(f"Processing cutoff: {cutoff}")
    print("=" * 60)
    data_dir = _config.get_dataset_dir(cutoff)
    if not data_dir.exists():
        print(f"  [SKIP] {data_dir} does not exist.")
        return
    cutoff_time_str = CUTOFF_TIMES.get(cutoff, "09:01:00")
    cutoff_time = _parse_cutoff_time(cutoff_time_str)

    # 1) Load raw data
    df_raw = load_data(data_dir)
    print(f"  Loaded {len(df_raw)} rows from {data_dir}")

    # 2) Build daily aggregate
    df_daily = build_daily_aggregate_dataset(df_raw, cutoff_time, stats=AGGREGATE_STATS)
    print(f"  Daily samples: {len(df_daily)}")

    # 3) Merge compressed from output2
    if compressed_by_year:
        df_daily = merge_compressed_into_daily_aggregate(
            df_daily, compressed_by_year, cutoff_time, stats=AGGREGATE_STATS
        )
        df_daily = drop_original_indicators(df_daily)

    # 4) Merge external y (0900 only uses y.xlsx; 0915/0930 use computed target from step 2)
    if y_file and cutoff == "0900":
        df_y = load_external_y(y_file)
        if df_y is not None and len(df_y) > 0:
            df_daily = merge_external_y_into_daily(df_daily, df_y)
            print(f"  Merged external y from {y_file}")

    # 5) Cleaning
    df_clean = remove_constant_features(df_daily)

    # 6) Output
    out_dir = OUTPUT_BASE / cutoff
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"merged_for_autogluon_{cutoff}.csv"
    df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {out_path} ({len(df_clean)} rows, {len(df_clean.columns)} columns)")

    # 7) Descriptive statistics
    generate_descriptive_statistics(df_clean, out_dir)
    generate_descriptive_statistics(df_clean, out_dir / "stats_after_cleaning")


def main():
    print("=" * 60)
    print("Merge output2 → AutoGluon-ready (0900, 0915, 0930)")
    print("=" * 60)
    print(f"OUTPUT2_DIR: {OUTPUT2_DIR}")
    print(f"OUTPUT_BASE: {OUTPUT_BASE}")

    if not OUTPUT2_DIR.exists():
        raise FileNotFoundError(f"output2 not found: {OUTPUT2_DIR}")

    # Load compressed from output2 (shared across cutoffs)
    compressed_by_year = load_and_merge_compressed_from_output2(OUTPUT2_DIR)
    if not compressed_by_year:
        print("[WARN] No compressed data from output2; proceeding without compressed features.")

    # External y (for 0900)
    y_file = _config.get_y_file()
    if not Path(y_file).exists():
        y_file = None

    # Process each cutoff
    for cutoff in CUTOFFS:
        process_cutoff(cutoff, compressed_by_year, y_file)

    print("\n" + "=" * 60)
    print("Done. Output: data/autogluon_ready/{0900,0915,0930}/")
    print("  - merged_for_autogluon_0900.csv (0915, 0930)")
    print("  - summary_statistics.csv, missing_values.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
