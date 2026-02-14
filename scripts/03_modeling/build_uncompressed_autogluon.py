#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建立「未壓縮」版 AutoGluon 訓練資料，並與壓縮版進行敘述統計與 A/B 比較。

- 讀取 data/dataset/{0900,0915,0930}/ 的截點前分鐘資料（原始技術指標）
- 建立日頻彙總（mean/std/min/max/median），不進行 autoencoder 壓縮
- 合併 target/y.xlsx
- 輸出至 data/autogluon_ready_uncompress/{0900,0915,0930}/
- 產出敘述統計與 A/B 比較至 data/analysis/compressed_vs_uncompressed/
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import config as _config

# ============== Config ==============
DATA_ROOT = Path(_config.DATA_ROOT)
CUTOFFS = ("0900", "0915", "0930")
CUTOFF_TIMES = {
    "0900": "09:01:00",
    "0915": "09:16:00",
    "0930": "09:31:00",
}
AUTOGLUON_READY_DIR = DATA_ROOT / "autogluon_ready"
OUTPUT_UNCOMPRESS_DIR = DATA_ROOT / "autogluon_ready_uncompress"
ANALYSIS_DIR = DATA_ROOT / "analysis" / "compressed_vs_uncompressed"

TARGET_COLUMN = "target_return"
MIN_YEAR = 2013
AGGREGATE_STATS = ("mean", "std", "min", "max", "median")
EXTERNAL_Y_DATE_COLS = ("day_id", "date", "日期")
EXTERNAL_Y_RETURN_COL_BY_CUTOFF = {
    "0900": "afternoon_return_0900",
    "0915": "afternoon_return_0915",
    "0930": "afternoon_return_0930",
}
EXTERNAL_Y_RETURN_COLS = ("afternoon_return_0900", "target_return", "y", "報酬率", "return")
EXTERNAL_Y_IS_LOG_RETURN = True


def _parse_cutoff_time(cutoff_str: str):
    from datetime import datetime as dt_datetime
    if ":" in cutoff_str and cutoff_str.count(":") >= 2:
        return dt_datetime.strptime(cutoff_str.strip(), "%H:%M:%S").time()
    return pd.to_datetime(cutoff_str).time()


def _find_data_dir(cutoff: str) -> Optional[Path]:
    """Try dataset first, then indicators_complete as fallback."""
    dataset_dir = _config.get_dataset_dir(cutoff)
    if dataset_dir.exists():
        csvs = list(dataset_dir.glob("TX*_1K_qlib_indicators_complete.csv"))
        csvs = [f for f in csvs if not f.name.startswith("._")]
        if csvs:
            return dataset_dir
    indicators_dir = _config.get_indicators_complete_dir()
    if indicators_dir.exists():
        csvs = list(indicators_dir.glob("TX*_1K_qlib_indicators_complete.csv"))
        csvs = [f for f in csvs if not f.name.startswith("._")]
        if csvs:
            return Path(indicators_dir)
    return None


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load all TX*_1K_qlib_indicators_complete.csv from data_dir."""
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
    combined["datetime"] = pd.to_datetime(combined["datetime"])
    combined = combined.sort_values("datetime").reset_index(drop=True)
    return combined


def build_daily_aggregate_dataset(
    df: pd.DataFrame,
    cutoff_time,
    stats: Tuple[str, ...] = None,
) -> pd.DataFrame:
    """One row per day: aggregate stats over rows <= cutoff."""
    if stats is None:
        stats = AGGREGATE_STATS
    if "datetime" not in df.columns or "close" not in df.columns:
        raise ValueError("Need 'datetime' and 'close'")
    work = df.copy()
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
        raise ValueError("No daily samples")
    return pd.DataFrame(rows)


def load_external_y(file_path: str, return_col: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load y.xlsx/y.csv."""
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
    except Exception:
        return None
    if df.empty:
        return None
    date_col = next((c for c in EXTERNAL_Y_DATE_COLS if c in df.columns), None)
    if return_col is None:
        return_col = next((c for c in EXTERNAL_Y_RETURN_COLS if c in df.columns), None)
    elif return_col not in df.columns:
        return None
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
    df_daily = df_daily.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
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
    return df


def process_cutoff_uncompressed(cutoff: str, y_file: Optional[str]) -> Optional[pd.DataFrame]:
    """Build uncompressed autogluon-ready for one cutoff."""
    data_dir = _find_data_dir(cutoff)
    if data_dir is None:
        print(f"  [SKIP] No data for cutoff {cutoff}")
        return None
    cutoff_time = _parse_cutoff_time(CUTOFF_TIMES.get(cutoff, "09:01:00"))
    df_raw = load_data(data_dir)
    df_daily = build_daily_aggregate_dataset(df_raw, cutoff_time, stats=AGGREGATE_STATS)
    if y_file:
        return_col = EXTERNAL_Y_RETURN_COL_BY_CUTOFF.get(cutoff)
        df_y = load_external_y(y_file, return_col=return_col)
        if df_y is not None and len(df_y) > 0:
            df_daily = merge_external_y_into_daily(df_daily, df_y)
    df_daily["date"] = pd.to_datetime(df_daily["date"], errors="coerce")
    df_daily = df_daily[df_daily["date"].dt.year >= MIN_YEAR].copy()
    df_daily["date"] = df_daily["date"].dt.date
    df_clean = remove_constant_features(df_daily)
    return df_clean


def descriptive_stats(df: pd.DataFrame) -> dict:
    """Compute descriptive statistics for a dataset."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return {}
    return {
        "n_rows": len(df),
        "n_features": len([c for c in numeric.columns if c != TARGET_COLUMN]),
        "n_target_nonnull": df[TARGET_COLUMN].notna().sum() if TARGET_COLUMN in df.columns else 0,
        "target_mean": float(df[TARGET_COLUMN].mean()) if TARGET_COLUMN in df.columns else None,
        "target_std": float(df[TARGET_COLUMN].std()) if TARGET_COLUMN in df.columns else None,
        "missing_pct": float(numeric.isna().sum().sum() / (numeric.size + 1e-10) * 100),
        "feature_mean_of_means": float(numeric.mean().mean()),
        "feature_std_of_stds": float(numeric.std().mean()) if numeric.std().notna().any() else None,
    }


def ab_test_correlation_with_target(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation of each feature with target. Returns top abs correlations."""
    if TARGET_COLUMN not in df.columns or df[TARGET_COLUMN].isna().all():
        return pd.DataFrame()
    numeric = df.select_dtypes(include=[np.number])
    feat_cols = [c for c in numeric.columns if c != TARGET_COLUMN]
    if not feat_cols:
        return pd.DataFrame()
    corrs = []
    for c in feat_cols:
        r = df[c].corr(df[TARGET_COLUMN])
        if not np.isnan(r):
            corrs.append({"feature": c, "corr_with_target": r})
    if not corrs:
        return pd.DataFrame()
    out = pd.DataFrame(corrs).sort_values("corr_with_target", key=abs, ascending=False)
    return out.head(30)


def run_ab_test(compressed: pd.DataFrame, uncompressed: pd.DataFrame, cutoff: str) -> dict:
    """A/B comparison: compressed vs uncompressed."""
    from scipy import stats
    result = {
        "cutoff": cutoff,
        "compressed_n_rows": len(compressed),
        "uncompressed_n_rows": len(uncompressed),
        "compressed_n_features": len([c for c in compressed.columns if c not in ("date", TARGET_COLUMN)]),
        "uncompressed_n_features": len([c for c in uncompressed.columns if c not in ("date", TARGET_COLUMN)]),
    }
    if TARGET_COLUMN in compressed.columns and TARGET_COLUMN in uncompressed.columns:
        # Align by date for fair comparison
        common_dates = set(compressed["date"].astype(str)) & set(uncompressed["date"].astype(str))
        if len(common_dates) >= 10:
            c_sub = compressed[compressed["date"].astype(str).isin(common_dates)].sort_values("date")
            u_sub = uncompressed[uncompressed["date"].astype(str).isin(common_dates)].sort_values("date")
            c_y = c_sub[TARGET_COLUMN].values
            u_y = u_sub[TARGET_COLUMN].values
            if len(c_y) == len(u_y):
                # Paired t-test: same target, so should be identical; if different merge, we compare
                t_stat, p_val = stats.ttest_rel(c_y, u_y)
                result["target_paired_t_stat"] = round(float(t_stat), 4)
                result["target_paired_p_value"] = round(float(p_val), 4)
                result["target_identical"] = p_val > 0.05
    return result


def main():
    print("=" * 60)
    print("Build Uncompressed AutoGluon Data + A/B Comparison")
    print("=" * 60)
    print(f"Output (uncompressed): {OUTPUT_UNCOMPRESS_DIR}")
    print(f"Output (analysis):     {ANALYSIS_DIR}")
    print("=" * 60)

    y_file = _config.get_y_file()
    if not Path(y_file).exists():
        y_file = None
        print("[WARN] No y.xlsx/y.csv found; target will use EOD-close from data.")

    # 1) Build uncompressed for each cutoff
    uncompressed_by_cutoff = {}
    for cutoff in CUTOFFS:
        print(f"\n[Cutoff {cutoff}] Building uncompressed...")
        df = process_cutoff_uncompressed(cutoff, y_file)
        if df is not None:
            out_dir = OUTPUT_UNCOMPRESS_DIR / cutoff
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"merged_for_autogluon_{cutoff}.csv"
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  Saved: {out_path} ({len(df)} rows, {len(df.columns)} cols)")
            uncompressed_by_cutoff[cutoff] = df
            # Descriptive stats
            summary = df.select_dtypes(include=[np.number]).describe()
            summary.to_csv(out_dir / "summary_statistics.csv", encoding="utf-8-sig")
            n = len(df)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            missing = pd.DataFrame({
                "Feature": numeric_cols,
                "Missing_Count": [df[c].isna().sum() for c in numeric_cols],
                "Missing_Pct": [df[c].isna().sum() / n * 100 if n else 0 for c in numeric_cols],
            })
            missing.to_csv(out_dir / "missing_values.csv", index=False, encoding="utf-8-sig")

    if not uncompressed_by_cutoff:
        print("\n[FAIL] No uncompressed data produced. Check dataset/ or indicators_complete/.")
        return

    # 2) Load compressed, run descriptive stats + A/B
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_rows = []

    for cutoff in CUTOFFS:
        if cutoff not in uncompressed_by_cutoff:
            continue
        df_uncomp = uncompressed_by_cutoff[cutoff]
        stats_uncomp = descriptive_stats(df_uncomp)
        stats_uncomp["cutoff"] = cutoff
        stats_uncomp["version"] = "uncompressed"

        # Load compressed
        comp_path = AUTOGLUON_READY_DIR / cutoff / f"merged_for_autogluon_{cutoff}.csv"
        if comp_path.exists():
            df_comp = pd.read_csv(comp_path)
            df_comp["date"] = pd.to_datetime(df_comp["date"], errors="coerce")
            stats_comp = descriptive_stats(df_comp)
            stats_comp["cutoff"] = cutoff
            stats_comp["version"] = "compressed"
            comparison_rows.append(stats_comp)
            comparison_rows.append(stats_uncomp)

            # A/B test
            ab = run_ab_test(df_comp, df_uncomp, cutoff)
            comparison_rows.append({**ab, "version": "ab_test"})

            # Correlation with target (top features)
            corr_uncomp = ab_test_correlation_with_target(df_uncomp)
            corr_comp = ab_test_correlation_with_target(df_comp)
            if not corr_uncomp.empty:
                corr_uncomp.to_csv(ANALYSIS_DIR / f"{cutoff}_uncompressed_top_corr_target.csv", index=False)
            if not corr_comp.empty:
                corr_comp.to_csv(ANALYSIS_DIR / f"{cutoff}_compressed_top_corr_target.csv", index=False)
        else:
            comparison_rows.append(stats_uncomp)
            print(f"  [INFO] No compressed data at {comp_path}; skip A/B for {cutoff}")

    # 3) Write comparison summary to xlsx
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        xlsx_path = ANALYSIS_DIR / "compressed_vs_uncompressed_comparison.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            comp_df.to_excel(writer, sheet_name="Summary", index=False)
            # Per-cutoff summary
            for cutoff in CUTOFFS:
                sub = comp_df[comp_df["cutoff"] == cutoff]
                if not sub.empty:
                    sub.to_excel(writer, sheet_name=f"Cutoff_{cutoff}", index=False)
        print(f"\n[OK] Comparison: {xlsx_path}")

    print("\n" + "=" * 60)
    print("Done.")
    print(f"  Uncompressed: {OUTPUT_UNCOMPRESS_DIR}")
    print(f"  Analysis:     {ANALYSIS_DIR}")
    print("  Use merged_for_autogluon_*.csv in autogluon_ready_uncompress/ for training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
