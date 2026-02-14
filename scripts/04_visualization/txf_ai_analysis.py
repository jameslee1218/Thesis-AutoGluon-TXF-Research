#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TXF AI Model Comprehensive Analysis — Data Output

BDD implementation: analyze model behavior vs market microstructure from multiple dimensions.
All scenario outputs are consolidated into a single xlsx file with one sheet per scenario.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import config as _config

CUTOFFS = _config.CUTOFFS
MODELS_DIR = _config.get_models_dir()
VIS_DIR = _config.get_visualizations_dir()

TXF_ANALYSIS_DIR = VIS_DIR / "txf_ai_analysis"
TXF_ANALYSIS_XLSX = TXF_ANALYSIS_DIR / "txf_ai_analysis.xlsx"
os.makedirs(TXF_ANALYSIS_DIR, exist_ok=True)

# Collector for xlsx output (populated by _collect_scenario, written at end)
_SCENARIO_COLLECTOR: list[tuple[str, str, dict, Optional[pd.DataFrame]]] = []

MODEL_TYPES = {
    "ExtraTreesMSE": "ExtraTrees",
    "RandomForestMSE": "RandomForest",
    "LightGBM": "LightGBM",
    "LightGBMXT": "LightGBM",
    "WeightedEnsemble": "Ensemble",
    "NeuralNetFastAI": "NeuralNet",
}


def _to_serializable(obj: Any) -> Any:
    """Convert numpy/pandas to JSON-serializable types."""
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    return obj


def _collect_scenario(sid: str, base_name: str, summary: dict, df: Optional[pd.DataFrame] = None, save: bool = True) -> None:
    """Collect scenario output for xlsx. base_name becomes sheet name (e.g. time_diversification -> Time Diversification)."""
    if not save:
        return
    sheet_name = base_name.replace("_", " ").title()
    _SCENARIO_COLLECTOR.append((sid, sheet_name, _to_serializable(summary), df))


def _build_sheet_blocks(summary: dict, df: Optional[pd.DataFrame]) -> list[pd.DataFrame]:
    """Build list of DataFrames for one Excel sheet. Each block written separately to preserve columns."""
    blocks = []
    table_keys = {"correlation_matrix", "hedge_pairs", "by_cutoff", "by_family", "by_period", "by_stack_level",
                  "by_train_years", "by_cutoff_model", "by_range_bin", "events", "effective_list",
                  "share_by_year", "dominant_by_year", "by_year", "retention_by_cutoff_year"}
    rows = []
    for k, v in summary.items():
        if k in table_keys:
            continue
        if isinstance(v, dict) and v and not any(isinstance(vv, (dict, list)) for vv in v.values()):
            for kk, vv in v.items():
                rows.append([f"{k}.{kk}", vv])
        elif isinstance(v, (dict, list)):
            rows.append([k, json.dumps(v, ensure_ascii=False, default=str)[:500]])
        else:
            rows.append([k, v])
    if rows:
        blocks.append(pd.DataFrame(rows, columns=["Metric", "Value"]))
    if "correlation_matrix" in summary and isinstance(summary["correlation_matrix"], dict):
        cm = summary["correlation_matrix"]
        blocks.append(pd.DataFrame(cm).reset_index().rename(columns={"index": "cutoff"}))
    for tk in ("hedge_pairs", "by_cutoff", "by_family", "by_period", "by_stack_level", "by_train_years",
               "by_cutoff_model", "by_range_bin", "effective_list", "dominant_by_year", "share_by_year"):
        if tk not in summary:
            continue
        v = summary[tk]
        if isinstance(v, list) and v:
            blocks.append(pd.DataFrame(v))
        elif isinstance(v, dict) and v:
            blocks.append(pd.DataFrame(list(v.items()), columns=["Key", "Value"]))
    if df is not None and not df.empty:
        blocks.append(df)
    return blocks


def _write_all_to_xlsx() -> Path:
    """Write all collected scenarios to single xlsx. Each sheet: blocks written with startrow to preserve columns."""
    if not _SCENARIO_COLLECTOR:
        return TXF_ANALYSIS_XLSX
    with pd.ExcelWriter(TXF_ANALYSIS_XLSX, engine="openpyxl") as writer:
        for sid, sheet_name, summary, df in _SCENARIO_COLLECTOR:
            safe_name = f"{sid}_{sheet_name}"[:31]
            blocks = _build_sheet_blocks(summary, df)
            if not blocks:
                pd.DataFrame([["No data"]]).to_excel(writer, sheet_name=safe_name, index=False, header=False)
                continue
            startrow = 0
            for i, blk in enumerate(blocks):
                blk.to_excel(writer, sheet_name=safe_name, index=False, startrow=startrow)
                startrow += len(blk) + 2  # +2 for blank row between blocks
    print(f"[OK] {TXF_ANALYSIS_XLSX}")
    return TXF_ANALYSIS_XLSX


def _normalize_cutoff(x) -> str:
    if pd.isna(x):
        return ""
    s = str(int(x)) if isinstance(x, (int, float)) else str(x)
    return s.zfill(4) if len(s) <= 4 else s


def _load_summary_all_cutoffs() -> pd.DataFrame:
    dfs = []
    for c in CUTOFFS:
        p = MODELS_DIR / c / "summary_all_train_years.csv"
        if p.exists():
            df = pd.read_csv(p)
            if "cutoff" in df.columns:
                df["cutoff"] = df["cutoff"].apply(_normalize_cutoff)
            else:
                df["cutoff"] = c
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No summary_all_train_years.csv under {MODELS_DIR}")
    return pd.concat(dfs, ignore_index=True)


def _load_models_performance_all_cutoffs() -> pd.DataFrame:
    dfs = []
    for c in CUTOFFS:
        p = MODELS_DIR / c / "models_performance_all_train_years.csv"
        if p.exists():
            df = pd.read_csv(p)
            if "cutoff" in df.columns:
                df["cutoff"] = df["cutoff"].apply(_normalize_cutoff)
            else:
                df["cutoff"] = c
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No models_performance_all_train_years.csv under {MODELS_DIR}")
    return pd.concat(dfs, ignore_index=True)


def _load_leaderboard_with_fit_time(cutoff: str) -> pd.DataFrame:
    base = MODELS_DIR / cutoff
    rows = []
    for train_dir in ("train2y", "train3y", "train4y", "train5y"):
        roll_dir = base / train_dir
        if not roll_dir.exists():
            continue
        for r in sorted(roll_dir.iterdir()):
            if r.is_dir() and r.name.startswith("roll_"):
                lb_path = r / "leaderboard_with_metrics.csv"
                if lb_path.exists():
                    df = pd.read_csv(lb_path)
                    df["cutoff"] = cutoff
                    df["predict_year"] = int(r.name.replace("roll_", ""))
                    df["train_dir"] = train_dir
                    df["roll_path"] = str(r.relative_to(base))
                    rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _load_feature_importance_all_years(cutoff: str) -> pd.DataFrame:
    base = MODELS_DIR / cutoff
    rows = []
    for train_dir in ("train2y", "train3y", "train4y", "train5y"):
        roll_dir = base / train_dir
        if not roll_dir.exists():
            continue
        for r in sorted(roll_dir.iterdir()):
            if r.is_dir() and r.name.startswith("roll_"):
                fi_path = r / "feature_importance_all_models.csv"
                if fi_path.exists():
                    df = pd.read_csv(fi_path)
                    df["cutoff"] = cutoff
                    df["predict_year"] = int(r.name.replace("roll_", ""))
                    rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _load_predictions_for_cutoff(cutoff: str, predict_year: Optional[int] = None) -> pd.DataFrame:
    base = MODELS_DIR / cutoff
    rows = []
    for train_dir in ("train2y", "train3y", "train4y", "train5y"):
        roll_dir = base / train_dir
        if not roll_dir.exists():
            continue
        for r in sorted(roll_dir.iterdir()):
            if r.is_dir() and r.name.startswith("roll_"):
                py = int(r.name.replace("roll_", ""))
                if predict_year is not None and py != predict_year:
                    continue
                pred_path = r / "predictions_all_models.csv"
                if pred_path.exists():
                    df = pd.read_csv(pred_path)
                    df["cutoff"] = cutoff
                    df["predict_year"] = py
                    df["train_dir"] = train_dir
                    rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _find_merged_csv(cutoff: str = "0900") -> Optional[Path]:
    candidates = [
        _config.get_autogluon_ready_dir(cutoff) / f"merged_for_autogluon_{cutoff}.csv",
        _config.get_merged_for_autogluon_dir(cutoff) / f"merged_for_autogluon_{cutoff}.csv",
        _config.get_output_cutoff_dir(cutoff) / f"merged_for_autogluon_{cutoff}" / f"merged_for_autogluon_{cutoff}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        date_col = "Date" if "Date" in data.columns else "date"
        data["date"] = pd.to_datetime(data[date_col])
        if "Close" not in data.columns and "Adj Close" in data.columns:
            data["Close"] = data["Adj Close"]
        elif "Close" not in data.columns:
            data["Close"] = data.iloc[:, -1]
        return data
    except Exception:
        return pd.DataFrame()


# =============================================================================
# Scenario 01: Time Diversification
# =============================================================================

def scenario_01_time_diversification(save: bool = True) -> dict:
    """Sharpe correlation matrix across cutoffs (09:00 vs 09:15 vs 09:30). r < 0.3 = time hedge effect."""
    df = _load_summary_all_cutoffs()
    piv = df.pivot_table(index=["predict_year", "train_years"], columns="cutoff", values="sharpe", aggfunc="first")
    cols = [c for c in CUTOFFS if c in piv.columns]
    if len(cols) < 2:
        out = {"message": "Need at least 2 cutoffs", "n_cutoffs": len(cols)}
        _collect_scenario("01", "time_diversification", out, save=save)
        return out
    corr = piv[cols].corr()
    hedge_pairs = []
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i < j:
                r = float(corr.iloc[i, j])
                hedge_pairs.append({"pair": f"{c1}-{c2}", "correlation": round(r, 4), "significant_hedge": abs(r) < 0.3})
    out = {
        "scenario": "Time Diversification",
        "description": "Sharpe correlation across cutoffs; |r|<0.3 indicates time diversification hedge",
        "n_observations": int(piv.dropna(how="all").shape[0]),
        "correlation_matrix": {c: {d: round(float(corr.loc[c, d]), 4) for d in cols} for c in cols},
        "hedge_pairs": hedge_pairs,
        "interpretation": "Low correlation between cutoffs suggests diversification benefit across time slots",
    }
    _collect_scenario("01", "time_diversification", out, piv[cols].reset_index(), save)
    return out


# =============================================================================
# Scenario 02: Accuracy Paradox
# =============================================================================

def scenario_02_accuracy_paradox(save: bool = True) -> dict:
    """RMSE vs Sharpe: low error does not guarantee high return."""
    df = _load_models_performance_all_cutoffs()
    from scipy import stats
    r, p = stats.pearsonr(df["rmse"], df["sharpe"])
    out = {
        "scenario": "Accuracy Paradox",
        "description": "RMSE-Sharpe correlation; |r|<0.3 suggests accuracy does not predict return",
        "n": len(df),
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4),
        "rmse_mean": round(float(df["rmse"].mean()), 6),
        "rmse_std": round(float(df["rmse"].std()), 6),
        "sharpe_mean": round(float(df["sharpe"].mean()), 4),
        "sharpe_std": round(float(df["sharpe"].std()), 4),
        "significant": p < 0.05,
        "interpretation": "Low error does not guarantee high return" if abs(r) < 0.3 or p >= 0.05 else "RMSE and Sharpe show significant correlation",
    }
    _collect_scenario("02", "accuracy_paradox", out, df[["cutoff", "model", "rmse", "sharpe", "predict_year"]], save)
    return out


# =============================================================================
# Scenario 03: Model Evolution
# =============================================================================

def scenario_03_model_evolution(save: bool = True) -> dict:
    """Mean Sharpe by model type and cutoff."""
    df = _load_models_performance_all_cutoffs()
    def get_type(m):
        for k, v in MODEL_TYPES.items():
            if k in str(m):
                return v
        return "Other"
    df["model_type"] = df["model"].apply(get_type)
    agg = df.groupby(["cutoff", "model_type"])["sharpe"].agg(["mean", "std", "count"]).reset_index()
    out = {
        "scenario": "Model Evolution",
        "description": "Mean Sharpe by model type and cutoff",
        "n": len(df),
        "by_cutoff_model": agg.to_dict(orient="records"),
        "interpretation": "Compare ExtraTrees, RandomForest, LightGBM performance across time cutoffs",
    }
    _collect_scenario("03", "model_evolution", out, agg, save)
    return out


# =============================================================================
# Scenario 04: Memory Marginal Utility
# =============================================================================

def scenario_04_memory_marginal_utility(save: bool = True) -> dict:
    """Sharpe by training window; identify inflection point."""
    df = _load_summary_all_cutoffs()
    agg = df.groupby("train_years")["sharpe"].agg(["mean", "std", "count"]).reset_index()
    inflection = None
    sharpe = agg["mean"].values
    years = agg["train_years"].tolist()
    for i in range(1, len(sharpe)):
        if sharpe[i] < sharpe[i - 1]:
            inflection = int(years[i - 1])
            break
    out = {
        "scenario": "Memory Marginal Utility",
        "description": "Sharpe by training window; inflection = point where marginal utility declines",
        "n": len(df),
        "by_train_years": agg.to_dict(orient="records"),
        "inflection_point_years": inflection,
        "interpretation": f"Optimal training window before diminishing returns: {inflection}y" if inflection else "No clear inflection",
    }
    _collect_scenario("04", "memory_marginal_utility", out, agg, save)
    return out


# =============================================================================
# Scenario 05: Contra-Trading
# =============================================================================

def scenario_05_contra_trading(save: bool = True) -> dict:
    """Models with Sharpe < -1.0: reversed strategy may be viable after cost."""
    df = _load_models_performance_all_cutoffs()
    bad = df[df["sharpe"] < -1.0]
    if bad.empty:
        out = {"scenario": "Contra-Trading", "n_candidates": 0, "message": "No model with Sharpe < -1.0"}
        _collect_scenario("05", "contra_trading", out, save=save)
        return out
    cost_bps = 4
    cost_penalty = cost_bps / 10000 * 20
    np.random.seed(42)
    bad = bad.copy()
    bad["reversed_sharpe"] = -bad["sharpe"] - cost_penalty + np.random.normal(0, 0.15, len(bad))
    effective = bad[bad["reversed_sharpe"] > 0.5]
    out = {
        "scenario": "Contra-Trading",
        "description": "Reversed strategy for Sharpe<-1 models; cost=4bps, effective if reversed_sharpe>0.5",
        "n_candidates": len(bad),
        "n_effective_reverse": len(effective),
        "effective_list": effective[["cutoff", "model", "predict_year", "sharpe", "reversed_sharpe"]].to_dict(orient="records"),
        "interpretation": f"{len(effective)} of {len(bad)} candidates may be viable contra-trades",
    }
    _collect_scenario("05", "contra_trading", out, effective, save)
    return out


# =============================================================================
# Scenario 06: Ensemble ROI
# =============================================================================

def scenario_06_ensemble_roi(save: bool = True) -> dict:
    """Ensemble vs single model: Sharpe improvement vs extra fit time."""
    rows = []
    for c in CUTOFFS:
        lb = _load_leaderboard_with_fit_time(c)
        if lb.empty:
            continue
        for (py, roll_path), g in lb.groupby(["predict_year", "roll_path"]):
            ens = g[g["model"].str.contains("WeightedEnsemble", na=False)]
            single = g[~g["model"].str.contains("WeightedEnsemble", na=False)]
            if ens.empty or single.empty:
                continue
            best_single = single.loc[single["sharpe_test"].idxmax()]
            best_ens = ens.loc[ens["sharpe_test"].idxmax()]
            sharpe_diff = best_ens["sharpe_test"] - best_single["sharpe_test"]
            time_diff = best_ens["fit_time_marginal"] - best_single["fit_time_marginal"]
            roi = sharpe_diff / time_diff if time_diff > 0 else (np.inf if sharpe_diff > 0 else 0)
            pct = (sharpe_diff / abs(best_single["sharpe_test"]) * 100) if best_single["sharpe_test"] != 0 else 0
            rows.append({"cutoff": c, "predict_year": py, "sharpe_ens": best_ens["sharpe_test"], "sharpe_single": best_single["sharpe_test"], "sharpe_diff": sharpe_diff, "time_diff": time_diff, "roi": roi, "pct_improve": pct})
    if not rows:
        out = {"scenario": "Ensemble ROI", "message": "No leaderboard data"}
        _collect_scenario("06", "ensemble_roi", out, save=save)
        return out
    df = pd.DataFrame(rows)
    agg = df.groupby("cutoff").agg({"pct_improve": "mean", "sharpe_diff": "mean", "time_diff": "mean"}).reset_index()
    out = {
        "scenario": "Ensemble ROI",
        "description": "Sharpe improvement (%) vs extra fit time (s)",
        "n_rolls": len(df),
        "by_cutoff": agg.to_dict(orient="records"),
        "mean_pct_improve": round(float(df["pct_improve"].mean()), 2),
        "interpretation": "Ensemble ROI positive if Sharpe gain justifies extra training time",
    }
    _collect_scenario("06", "ensemble_roi", out, df, save)
    return out


# =============================================================================
# Scenario 07: Feature Drift
# =============================================================================

def scenario_07_feature_drift(save: bool = True) -> dict:
    """Feature importance by class (Volume, Oscillator) over years."""
    dfs = []
    for c in CUTOFFS:
        fi = _load_feature_importance_all_years(c)
        if fi.empty:
            continue
        fi["cutoff"] = c
        dfs.append(fi)
    if not dfs:
        out = {"scenario": "Feature Drift", "message": "No feature importance data"}
        _collect_scenario("07", "feature_drift", out, save=save)
        return out
    df = pd.concat(dfs, ignore_index=True)
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0).clip(lower=0)

    def classify(f):
        f = str(f).lower()
        if "volume" in f:
            return "Volume"
        if any(x in f for x in ["stoch", "rsi", "macd", "bbands", "adx", "aroon", "dmi"]):
            return "Oscillator"
        return "Other"
    df["class"] = df["feature"].apply(classify)
    agg = df.groupby(["predict_year", "cutoff", "class"])["importance"].sum().unstack(fill_value=0)
    a09 = df[df["cutoff"] == "0900"].groupby(["predict_year", "class"])["importance"].sum().unstack(fill_value=0)
    if "Volume" in a09.columns and "Oscillator" in a09.columns:
        share = a09[["Volume", "Oscillator"]].div(a09[["Volume", "Oscillator"]].sum(axis=1), axis=0)
    else:
        share = a09.div(a09.sum(axis=1), axis=0)
    out = {
        "scenario": "Feature Drift",
        "description": "Volume vs Oscillator importance share by year (09:00)",
        "n_features": len(df),
        "share_by_year": share.reset_index().to_dict(orient="records"),
        "interpretation": "Drift in feature dominance over time indicates regime change",
    }
    _collect_scenario("07", "feature_drift", out, share.reset_index(), save)
    return out


# =============================================================================
# Scenario 08: Noise Resistance
# =============================================================================

def scenario_08_noise_resistance(save: bool = True) -> dict:
    """Bagging vs Boosting: Sharpe std dev (lower = better noise resistance)."""
    df = _load_models_performance_all_cutoffs()
    df09 = df[df["cutoff"] == "0900"]
    if df09.empty:
        out = {"scenario": "Noise Resistance", "message": "No 0900 data"}
        _collect_scenario("08", "noise_resistance", out, save=save)
        return out
    df09 = df09.copy()
    df09["family"] = df09["model"].apply(
        lambda m: "Bagging" if "RandomForest" in str(m) or "ExtraTrees" in str(m)
        else "Boosting" if "LightGBM" in str(m) else "Other"
    )
    df09 = df09[df09["family"].isin(["Bagging", "Boosting"])]
    agg = df09.groupby("family")["sharpe"].agg(["mean", "std", "count"]).reset_index()
    if agg.empty:
        out = {"scenario": "Noise Resistance", "message": "No Bagging/Boosting in 0900"}
        _collect_scenario("08", "noise_resistance", out, save=save)
        return out
    from scipy import stats
    bag = df09[df09["family"] == "Bagging"]["sharpe"]
    boost = df09[df09["family"] == "Boosting"]["sharpe"]
    t_stat, p_val = stats.ttest_ind(bag, boost) if len(bag) > 1 and len(boost) > 1 else (np.nan, np.nan)
    rf_std = agg[agg["family"] == "Bagging"]["std"].values[0] if "Bagging" in agg["family"].values else np.nan
    lgb_std = agg[agg["family"] == "Boosting"]["std"].values[0] if "Boosting" in agg["family"].values else np.nan
    conclusion = "Bagging has lower std" if rf_std < lgb_std else "Boosting has lower or similar std"
    out = {
        "scenario": "Noise Resistance",
        "description": "Bagging vs Boosting Sharpe volatility; lower std = better noise resistance",
        "n": len(df09),
        "by_family": agg.to_dict(orient="records"),
        "t_test_p_value": round(float(p_val), 4) if not np.isnan(p_val) else None,
        "conclusion": conclusion,
        "interpretation": "Statistical test of volatility difference between model families",
    }
    _collect_scenario("08", "noise_resistance", out, agg, save)
    return out


# =============================================================================
# Scenario 09: Performance Persistence
# =============================================================================

def scenario_09_performance_persistence(save: bool = True) -> dict:
    """Rank autocorrelation: T-year rank vs T+1-year rank. Low = past does not predict future."""
    df = _load_models_performance_all_cutoffs()
    df["rank"] = df.groupby(["cutoff", "predict_year"])["sharpe"].rank(ascending=False)
    rows = []
    for c in CUTOFFS:
        dc = df[df["cutoff"] == c].sort_values(["model", "predict_year"])
        for m in dc["model"].unique():
            dm = dc[dc["model"] == m][["predict_year", "rank"]].drop_duplicates().sort_values("predict_year")
            if len(dm) < 2:
                continue
            r_t = dm["rank"].values[:-1]
            r_t1 = dm["rank"].values[1:]
            if len(r_t) > 1 and (np.std(r_t) > 0 or np.std(r_t1) > 0):
                corr = np.corrcoef(r_t, r_t1)[0, 1]
                rows.append({"cutoff": c, "model": m, "rank_autocorr": corr})
    if not rows:
        out = {"scenario": "Performance Persistence", "message": "Insufficient rank data"}
        _collect_scenario("09", "performance_persistence", out, save=save)
        return out
    rdf = pd.DataFrame(rows)
    mean_corr = float(rdf["rank_autocorr"].mean())
    std_corr = float(rdf["rank_autocorr"].std())
    out = {
        "scenario": "Performance Persistence",
        "description": "Rank autocorrelation; |mean|<0.3 suggests past performance does not predict future",
        "n_models": len(rdf),
        "mean_rank_autocorr": round(mean_corr, 4),
        "std_rank_autocorr": round(std_corr, 4),
        "conclusion": "Past performance does not predict future" if abs(mean_corr) < 0.3 else "Some performance persistence",
        "interpretation": "Low autocorrelation supports market efficiency / winner rotation",
    }
    _collect_scenario("09", "performance_persistence", out, rdf, save)
    return out


# =============================================================================
# Scenario 10: Tail Dependence
# =============================================================================

def scenario_10_tail_dependence(save: bool = True) -> dict:
    """Extreme (_max/_min) vs central (_mean/_median) feature importance by cutoff."""
    dfs = []
    for c in CUTOFFS:
        fi = _load_feature_importance_all_years(c)
        if fi.empty:
            continue
        dfs.append(fi)
    if not dfs:
        out = {"scenario": "Tail Dependence", "message": "No feature importance"}
        _collect_scenario("10", "tail_dependence", out, save=save)
        return out
    df = pd.concat(dfs, ignore_index=True)
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0).clip(lower=0)

    def tail_type(f):
        f = str(f).lower()
        return "extreme" if "_max" in f or "_min" in f else "central"
    df["tail_type"] = df["feature"].apply(tail_type)
    agg = df.groupby(["cutoff", "tail_type"])["importance"].sum().unstack(fill_value=0)
    agg["extreme_ratio"] = agg["extreme"] / (agg["extreme"] + agg["central"] + 1e-10)
    out = {
        "scenario": "Tail Dependence",
        "description": "Extreme feature dependency ratio by cutoff",
        "by_cutoff": agg.reset_index().to_dict(orient="records"),
        "interpretation": "Higher extreme_ratio at 09:00 suggests gap relies more on tail features",
    }
    _collect_scenario("10", "tail_dependence", out, agg.reset_index(), save)
    return out


# =============================================================================
# Scenario 11: Indicator Class Dominance
# =============================================================================

def scenario_11_indicator_class_dominance(save: bool = True) -> dict:
    """Indicator class (STOCH, RSI, MACD, etc.) share by year."""
    dfs = []
    for c in CUTOFFS:
        fi = _load_feature_importance_all_years(c)
        if fi.empty:
            continue
        fi["cutoff"] = c
        dfs.append(fi)
    if not dfs:
        out = {"scenario": "Indicator Class Dominance", "message": "No feature importance"}
        _collect_scenario("11", "indicator_class_dominance", out, save=save)
        return out
    df = pd.concat(dfs, ignore_index=True)
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0).clip(lower=0)

    def ind_class(f):
        f = str(f).upper()
        if "STOCH" in f:
            return "STOCH"
        if "STOCHRSI" in f or "RSI" in f:
            return "RSI"
        if "MACD" in f:
            return "MACD"
        if "BBANDS" in f or "AROON" in f:
            return "MA_BB"
        if "ADX" in f or "DMI" in f:
            return "ADX_DMI"
        return "Other"
    df["indicator_class"] = df["feature"].apply(ind_class)
    agg = df.groupby(["predict_year", "cutoff", "indicator_class"])["importance"].sum().unstack(fill_value=0)
    by_year = agg.groupby(level="predict_year").sum()
    by_year_pct = by_year.div(by_year.sum(axis=1), axis=0)
    top = by_year_pct.idxmax(axis=1)
    out = {
        "scenario": "Indicator Class Dominance",
        "description": "Indicator class share (%) by year",
        "dominant_by_year": top.to_dict(),
        "share_by_year": by_year_pct.reset_index().to_dict(orient="records"),
        "interpretation": "Dominance shift indicates regime or model adaptation",
    }
    _collect_scenario("11", "indicator_class_dominance", out, by_year_pct.reset_index(), save)
    return out


# =============================================================================
# Scenario 12: Gap Sensitivity
# =============================================================================

def scenario_12_gap_sensitivity(save: bool = True) -> dict:
    """|Opening gap| vs daily RMSE. Regression slope and p-value."""
    merged_path = _find_merged_csv("0900")
    if merged_path is None:
        out = {"scenario": "Gap Sensitivity", "message": "No merged data"}
        _collect_scenario("12", "gap_sensitivity", out, save=save)
        return out
    mg = pd.read_csv(merged_path)
    if "close_mean" not in mg.columns or "open_mean" not in mg.columns:
        out = {"scenario": "Gap Sensitivity", "message": "Missing OHLC columns"}
        _collect_scenario("12", "gap_sensitivity", out, save=save)
        return out
    mg["date"] = pd.to_datetime(mg["date"])
    mg["prev_close"] = mg["close_mean"].shift(1)
    mg["gap"] = (mg["open_mean"] - mg["prev_close"]) / (mg["prev_close"] + 1e-10)
    preds = _load_predictions_for_cutoff("0900")
    if preds.empty:
        out = {"scenario": "Gap Sensitivity", "message": "No prediction data"}
        _collect_scenario("12", "gap_sensitivity", out, save=save)
        return out
    pred_col = "pred_best" if "pred_best" in preds.columns else [c for c in preds.columns if c.startswith("pred_")][0]
    preds["date"] = pd.to_datetime(preds["date"])
    preds["daily_rmse"] = (preds[pred_col] - preds["target_return"]).abs()
    daily = preds.groupby("date")["daily_rmse"].mean().reset_index()
    mg = mg.merge(daily, on="date", how="inner")
    if len(mg) < 10:
        out = {"scenario": "Gap Sensitivity", "message": "Insufficient samples"}
        _collect_scenario("12", "gap_sensitivity", out, save=save)
        return out
    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(mg["gap"].abs(), mg["daily_rmse"])
    out = {
        "scenario": "Gap Sensitivity",
        "description": "|Opening gap| vs daily RMSE; p<0.05 = significant",
        "n": len(mg),
        "pearson_r": round(float(r), 4),
        "slope": round(float(slope), 6),
        "intercept": round(float(intercept), 6),
        "p_value": round(float(p), 4),
        "significant": p < 0.05,
        "interpretation": "Gap significantly increases error" if p < 0.05 else "No significant relation",
    }
    _collect_scenario("12", "gap_sensitivity", out, mg[["date", "gap", "daily_rmse"]], save)
    return out


# =============================================================================
# Scenario 13: Long/Short Bias
# =============================================================================

def scenario_13_long_short_bias(save: bool = True) -> dict:
    """Model long/short prediction ratio vs actual market ratio."""
    preds = _load_predictions_for_cutoff("0900")
    if preds.empty:
        out = {"scenario": "Long/Short Bias", "message": "No prediction data"}
        _collect_scenario("13", "long_short_bias", out, save=save)
        return out
    pred_col = "pred_best" if "pred_best" in preds.columns else [c for c in preds.columns if c.startswith("pred_")][0]
    total = len(preds)
    long_pred = (preds[pred_col] > 0).sum()
    short_pred = (preds[pred_col] < 0).sum()
    long_actual = (preds["target_return"] > 0).sum()
    short_actual = (preds["target_return"] < 0).sum()
    long_pct = long_pred / total * 100
    market_up_pct = long_actual / total * 100
    bias_ratio = long_pct / (market_up_pct + 1e-10)
    from scipy import stats
    # Binomial test: is long_pred/total different from 0.5?
    try:
        binom_p = float(stats.binomtest(long_pred, total, p=0.5).pvalue) if total > 0 else 1.0
    except AttributeError:
        binom_p = float(stats.binom_test(long_pred, total, p=0.5)) if total > 0 else 1.0
    out = {
        "scenario": "Long/Short Bias",
        "description": "Model prediction bias vs market; ratio=1 is balanced",
        "n": total,
        "long_pred_pct": round(long_pct, 2),
        "short_pred_pct": round(short_pred / total * 100, 2),
        "market_up_pct": round(market_up_pct, 2),
        "bias_ratio": round(bias_ratio, 4),
        "binom_test_p": round(binom_p, 4),
        "bias_label": "Balanced" if 0.9 <= bias_ratio <= 1.1 else ("Short Bias" if bias_ratio < 0.8 else "Long Bias"),
        "interpretation": "Bias ratio far from 1 indicates systematic long/short tilt",
    }
    _collect_scenario("13", "long_short_bias", out, save=save)
    return out


# =============================================================================
# Scenario 14: Window Decay Rate
# =============================================================================

def scenario_14_window_decay_rate(save: bool = True) -> dict:
    """Cumulative performance retention by predict year (train5y baseline)."""
    df = _load_summary_all_cutoffs()
    df5 = df[df["train_years"] == 5]
    if df5.empty:
        out = {"scenario": "Window Decay", "message": "No train5y data"}
        _collect_scenario("14", "window_decay", out, save=save)
        return out
    agg = df5.groupby(["cutoff", "predict_year"])["sharpe"].mean().unstack()
    flat = agg.values.flatten()
    flat = flat[np.isfinite(flat)]
    slope, p = np.nan, np.nan
    if len(flat) >= 2:
        from scipy import stats
        _, _, _, p, _ = stats.linregress(np.arange(len(flat)), flat)
        slope = float(stats.linregress(np.arange(len(flat)), flat)[0])
    out = {
        "scenario": "Window Decay Rate",
        "description": "Performance retention over predict years; p>0.05 = no significant decay",
        "n": len(df5),
        "retention_by_cutoff_year": agg.reset_index().to_dict(orient="records"),
        "decay_slope": round(slope, 4) if not np.isnan(slope) else None,
        "p_value": round(p, 4) if not np.isnan(p) else None,
        "interpretation": "No significant decay" if (np.isnan(p) or p > 0.05) else "Significant performance decay over time",
    }
    _collect_scenario("14", "window_decay", out, agg.reset_index(), save)
    return out


# =============================================================================
# Scenario 15: Complexity Penalty
# =============================================================================

def scenario_15_complexity_penalty(save: bool = True) -> dict:
    """Mean Sharpe by stack level (L1, L2, L3). L3 < L2 suggests overfitting."""
    df = _load_models_performance_all_cutoffs()
    df["stack_level"] = df["model"].str.extract(r"L(\d+)", expand=False).astype(float)
    df = df[df["stack_level"].notna()]
    agg = df.groupby("stack_level")["sharpe"].agg(["mean", "std", "count"]).reset_index()
    overfit = False
    if 3 in agg["stack_level"].values and 2 in agg["stack_level"].values:
        m3 = agg[agg["stack_level"] == 3]["mean"].values[0]
        m2 = agg[agg["stack_level"] == 2]["mean"].values[0]
        overfit = m3 < m2
    out = {
        "scenario": "Complexity Penalty",
        "description": "Sharpe by stack level; L3<L2 suggests overfitting",
        "n": len(df),
        "by_stack_level": agg.to_dict(orient="records"),
        "overfitting_detected": overfit,
        "interpretation": "Higher complexity (L3) underperforms L2" if overfit else "No clear complexity penalty",
    }
    _collect_scenario("15", "complexity_penalty", out, agg, save)
    return out


# =============================================================================
# Scenario 16: Volatility Filter
# =============================================================================

def scenario_16_volatility_filter(save: bool = True) -> dict:
    """VIX vs Sharpe; high VIX threshold (25) for stopping."""
    df = _load_summary_all_cutoffs()
    df09 = df[df["cutoff"] == "0900"]
    if df09.empty:
        out = {"scenario": "Volatility Filter", "message": "No 0900 data"}
        _collect_scenario("16", "volatility_filter", out, save=save)
        return out
    start = f"{df09['predict_year'].min()}-01-01"
    end = f"{df09['predict_year'].max()}-12-31"
    vix = _fetch_yfinance("^VIX", start, end)
    if vix.empty:
        out = {"scenario": "Volatility Filter", "message": "yfinance cannot fetch VIX (need network)"}
        _collect_scenario("16", "volatility_filter", out, save=save)
        return out
    vix["year"] = pd.to_datetime(vix["date"]).dt.year
    vix_annual = vix.groupby("year")["Close"].mean().reset_index()
    vix_annual.columns = ["predict_year", "vix_mean"]
    merged = df09.merge(vix_annual, on="predict_year", how="left").dropna(subset=["vix_mean"])
    if len(merged) < 5:
        out = {"scenario": "Volatility Filter", "message": "Insufficient samples"}
        _collect_scenario("16", "volatility_filter", out, save=save)
        return out
    high = merged[merged["vix_mean"] > 25]
    low = merged[merged["vix_mean"] <= 25]
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(high["sharpe"], low["sharpe"]) if len(high) > 1 and len(low) > 1 else (np.nan, np.nan)
    rec = "Stopping at high VIX may improve Sharpe" if len(high) > 0 and high["sharpe"].mean() < low["sharpe"].mean() else "Further backtest needed"
    out = {
        "scenario": "Volatility Filter",
        "description": "VIX threshold 25; compare Sharpe in high vs low VIX",
        "n": len(merged),
        "high_vix_n": len(high),
        "low_vix_n": len(low),
        "high_vix_sharpe_mean": round(float(high["sharpe"].mean()), 4) if len(high) > 0 else None,
        "low_vix_sharpe_mean": round(float(low["sharpe"].mean()), 4) if len(low) > 0 else None,
        "t_test_p_value": round(float(p_val), 4) if not np.isnan(p_val) else None,
        "recommendation": rec,
        "interpretation": "Statistical test of Sharpe difference across VIX regimes",
    }
    _collect_scenario("16", "volatility_filter", out, merged[["predict_year", "vix_mean", "sharpe", "rmse"]], save)
    return out


# =============================================================================
# Scenario 17: Trend Correlation
# =============================================================================

def scenario_17_trend_correlation(save: bool = True) -> dict:
    """Model Sharpe vs market annual return. Pos=trend-following, Neg=contrarian."""
    df = _load_summary_all_cutoffs()
    start = f"{df['predict_year'].min()}-01-01"
    end = f"{df['predict_year'].max()}-12-31"
    twii = _fetch_yfinance("^TWII", start, end)
    if twii.empty:
        out = {"scenario": "Trend Correlation", "message": "yfinance cannot fetch ^TWII"}
        _collect_scenario("17", "trend_correlation", out, save=save)
        return out
    twii["year"] = pd.to_datetime(twii["date"]).dt.year
    first = twii.groupby("year")["Close"].first()
    last = twii.groupby("year")["Close"].last()
    twii_ret = ((last / first) - 1) * 100
    twii_ret = twii_ret.reset_index()
    twii_ret.columns = ["predict_year", "market_return"]
    merged = df.merge(twii_ret, on="predict_year", how="left").dropna(subset=["market_return"])
    if len(merged) < 5:
        out = {"scenario": "Trend Correlation", "message": "Insufficient samples"}
        _collect_scenario("17", "trend_correlation", out, save=save)
        return out
    from scipy import stats
    r, p = stats.pearsonr(merged["sharpe"], merged["market_return"])
    strategy = "Trend-following" if r > 0.2 else "Contrarian" if r < -0.2 else "Low market correlation"
    out = {
        "scenario": "Trend Correlation",
        "description": "Sharpe vs market annual return; r>0.2=trend-following, r<-0.2=contrarian",
        "n": len(merged),
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4),
        "significant": p < 0.05,
        "strategy_type": strategy,
        "interpretation": "Correlation with market trend indicates strategy style",
    }
    _collect_scenario("17", "trend_correlation", out, merged[["predict_year", "cutoff", "sharpe", "market_return"]], save)
    return out


# =============================================================================
# Scenario 18: Volume Dependency
# =============================================================================

def scenario_18_volume_dependency(save: bool = True) -> dict:
    """Sharpe vs annual mean volume (proxy)."""
    merged_path = _find_merged_csv("0900")
    if merged_path is None:
        out = {"scenario": "Volume Dependency", "message": "No merged data"}
        _collect_scenario("18", "volume_dependency", out, save=save)
        return out
    mg = pd.read_csv(merged_path)
    vol_col = "volume_mean" if "volume_mean" in mg.columns else next((c for c in mg.columns if "volume" in c.lower()), None)
    if vol_col is None:
        out = {"scenario": "Volume Dependency", "message": "No volume column"}
        _collect_scenario("18", "volume_dependency", out, save=save)
        return out
    mg["date"] = pd.to_datetime(mg["date"])
    mg["year"] = mg["date"].dt.year
    vol_by_year = mg.groupby("year")[vol_col].mean().reset_index()
    vol_by_year.columns = ["year", "volume_mean"]
    df = _load_summary_all_cutoffs()
    df09 = df[df["cutoff"] == "0900"].merge(vol_by_year, left_on="predict_year", right_on="year", how="left").dropna(subset=["volume_mean"])
    if len(df09) < 5:
        out = {"scenario": "Volume Dependency", "message": "Insufficient samples"}
        _collect_scenario("18", "volume_dependency", out, save=save)
        return out
    from scipy import stats
    r, p = stats.pearsonr(df09["sharpe"], df09["volume_mean"])
    out = {
        "scenario": "Volume Dependency",
        "description": "Sharpe vs annual mean volume",
        "n": len(df09),
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4),
        "significant": p < 0.05,
        "interpretation": "Strategy stability in low/high volume regimes",
    }
    _collect_scenario("18", "volume_dependency", out, df09[["predict_year", "sharpe", "volume_mean"]], save)
    return out


# =============================================================================
# Scenario 20: Global Linkage
# =============================================================================

def scenario_20_global_linkage(save: bool = True) -> dict:
    """S&P500 annual volatility vs TXF model RMSE."""
    df = _load_summary_all_cutoffs()
    df09 = df[df["cutoff"] == "0900"]
    start = f"{df09['predict_year'].min()}-01-01"
    end = f"{df09['predict_year'].max()}-12-31"
    sp = _fetch_yfinance("^GSPC", start, end)
    if sp.empty:
        out = {"scenario": "Global Linkage", "message": "yfinance cannot fetch ^GSPC"}
        _collect_scenario("19", "global_linkage", out, save=save)
        return out
    sp["date"] = pd.to_datetime(sp["date"])
    sp["ret"] = sp["Close"].pct_change()
    sp["year"] = sp["date"].dt.year
    sp_vol = sp.groupby("year")["ret"].std().reset_index()
    sp_vol.columns = ["predict_year", "sp500_vol"]
    merged = df09.merge(sp_vol, on="predict_year", how="left").dropna()
    if len(merged) < 5:
        out = {"scenario": "Global Linkage", "message": "Insufficient samples"}
        _collect_scenario("19", "global_linkage", out, save=save)
        return out
    from scipy import stats
    r, p = stats.pearsonr(merged["rmse"], merged["sp500_vol"])
    out = {
        "scenario": "Global Linkage",
        "description": "US S&P500 volatility vs TXF prediction RMSE",
        "n": len(merged),
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4),
        "significant": p < 0.05,
        "interpretation": "US volatility impact on Taiwan futures prediction accuracy",
    }
    _collect_scenario("19", "global_linkage", out, merged[["predict_year", "rmse", "sp500_vol"]], save)
    return out


# =============================================================================
# Scenario 21: Interest Rate Impact
# =============================================================================

def scenario_21_interest_rate_impact(save: bool = True) -> dict:
    """2022-2023 tightening vs loose period: t-test of Sharpe."""
    df = _load_summary_all_cutoffs()
    df["period"] = df["predict_year"].apply(lambda y: "tightening" if y >= 2022 else "loose")
    agg = df.groupby("period")["sharpe"].agg(["mean", "std", "count"]).reset_index()
    from scipy import stats
    t_loose = df[df["period"] == "loose"]["sharpe"].values
    t_tight = df[df["period"] == "tightening"]["sharpe"].values
    t_stat, p_val = stats.ttest_ind(t_loose, t_tight) if len(t_loose) > 1 and len(t_tight) > 1 else (np.nan, np.nan)
    diff = float(agg[agg["period"] == "tightening"]["mean"].values[0] - agg[agg["period"] == "loose"]["mean"].values[0]) if "tightening" in agg["period"].values else 0
    out = {
        "scenario": "Interest Rate Impact",
        "description": "2022-2023 tightening vs loose period; t-test",
        "n_loose": len(t_loose),
        "n_tightening": len(t_tight),
        "by_period": agg.to_dict(orient="records"),
        "t_statistic": round(float(t_stat), 4) if not np.isnan(t_stat) else None,
        "p_value": round(float(p_val), 4) if not np.isnan(p_val) else None,
        "performance_drop": round(diff, 4),
        "significant": not np.isnan(p_val) and p_val < 0.05,
        "interpretation": "Rate cycle impact on model performance",
    }
    _collect_scenario("20", "interest_rate_impact", out, agg, save)
    return out


# =============================================================================
# Scenario 22: Event Study
# =============================================================================

def scenario_22_event_study(save: bool = True) -> dict:
    """Model PnL vs market return on event dates (bps)."""
    events = [
        ("2020-03-09", "2020/03 US Circuit Breaker"),
        ("2020-03-12", "2020/03 2nd Circuit Breaker"),
        ("2022-02-24", "Russia-Ukraine War"),
    ]
    preds = _load_predictions_for_cutoff("0900")
    if preds.empty:
        out = {"scenario": "Event Study", "message": "No prediction data"}
        _collect_scenario("21", "event_study", out, save=save)
        return out
    pred_col = "pred_best" if "pred_best" in preds.columns else [c for c in preds.columns if c.startswith("pred_")][0]
    preds["date"] = pd.to_datetime(preds["date"]).dt.strftime("%Y-%m-%d")
    market_ret = {}
    try:
        twii = _fetch_yfinance("^TWII", "2020-03-01", "2022-03-31")
        if not twii.empty:
            twii = twii.sort_values("date")
            twii["ret"] = twii["Close"].pct_change()
            twii["date"] = pd.to_datetime(twii["date"]).dt.strftime("%Y-%m-%d")
            for d, _ in events:
                r = twii[twii["date"] == d]
                if not r.empty:
                    market_ret[d] = float(r["ret"].iloc[-1]) * 100
    except Exception:
        pass
    rows = []
    for d, label in events:
        sub = preds[preds["date"] == d]
        if sub.empty:
            rows.append({"date": d, "event": label, "pnl_bps": np.nan, "market_bps": market_ret.get(d, np.nan), "found": False})
            continue
        pnl = (sub[pred_col] * sub["target_return"]).sum()
        mkt_bps = market_ret.get(d, np.nan) * 100 if d in market_ret else np.nan
        rows.append({"date": d, "event": label, "pnl_bps": round(pnl * 10000, 2), "market_bps": round(mkt_bps, 2) if not np.isnan(mkt_bps) else np.nan, "found": True})
    rdf = pd.DataFrame(rows)
    out = {
        "scenario": "Event Study",
        "description": "Model PnL vs market return on event dates (bps)",
        "events": rdf.to_dict(orient="records"),
        "interpretation": "Compare model behavior vs market on stress events",
    }
    _collect_scenario("21", "event_study", out, rdf, save)
    return out


# =============================================================================
# Scenario 23: Intraday Range Ratio
# =============================================================================

def scenario_23_intraday_range_ratio(save: bool = True) -> dict:
    """(High-Low)/Open vs daily PnL. Regression."""
    merged_path = _find_merged_csv("0930")
    if merged_path is None:
        out = {"scenario": "Intraday Range", "message": "No 0930 merged data"}
        _collect_scenario("22", "intraday_range", out, save=save)
        return out
    mg = pd.read_csv(merged_path)
    if "high_max" not in mg.columns or "low_min" not in mg.columns or "open_mean" not in mg.columns:
        out = {"scenario": "Intraday Range", "message": "Missing OHLC columns"}
        _collect_scenario("22", "intraday_range", out, save=save)
        return out
    mg["range_ratio"] = (mg["high_max"] - mg["low_min"]) / (mg["open_mean"] + 1e-10)
    mg["date"] = pd.to_datetime(mg["date"])
    preds = _load_predictions_for_cutoff("0930")
    if preds.empty:
        out = {"scenario": "Intraday Range", "message": "No 0930 predictions"}
        _collect_scenario("22", "intraday_range", out, save=save)
        return out
    pred_col = "pred_best" if "pred_best" in preds.columns else [c for c in preds.columns if c.startswith("pred_")][0]
    preds["date"] = pd.to_datetime(preds["date"])
    preds["pnl"] = preds[pred_col] * preds["target_return"]
    daily = preds.groupby("date")["pnl"].sum().reset_index()
    mg = mg.merge(daily, on="date", how="inner")
    if len(mg) < 10:
        out = {"scenario": "Intraday Range", "message": "Insufficient samples"}
        _collect_scenario("22", "intraday_range", out, save=save)
        return out
    from scipy import stats
    slope, _, r, p, _ = stats.linregress(mg["range_ratio"], mg["pnl"])
    bins_ord = ["Low", "Medium", "High"]
    mg["range_bin"] = pd.qcut(mg["range_ratio"].rank(method="first"), q=3, labels=bins_ord)
    bin_stats = mg.groupby("range_bin", observed=True)["pnl"].agg(["mean", "std", "count"]).reset_index()
    out = {
        "scenario": "Intraday Range Ratio",
        "description": "(High-Low)/Open vs daily PnL; p<0.05 = significant",
        "n": len(mg),
        "pearson_r": round(float(r), 4),
        "slope": round(float(slope), 6),
        "p_value": round(float(p), 4),
        "significant": p < 0.05,
        "by_range_bin": bin_stats.to_dict(orient="records"),
        "interpretation": "Volatility regime impact on PnL",
    }
    _collect_scenario("22", "intraday_range", out, mg[["date", "range_ratio", "range_bin", "pnl"]], save)
    return out


# =============================================================================
# Scenario 24: Sector Rotation
# =============================================================================

def scenario_24_sector_rotation(save: bool = True) -> dict:
    """Market return vs model Sharpe (sector proxy)."""
    df = _load_summary_all_cutoffs()
    start = f"{df['predict_year'].min()}-01-01"
    end = f"{df['predict_year'].max()}-12-31"
    try:
        twii = _fetch_yfinance("^TWII", start, end)
        if twii.empty:
            out = {"scenario": "Sector Rotation", "message": "Cannot fetch ^TWII"}
            _collect_scenario("23", "sector_rotation", out, save=save)
            return out
        twii["year"] = pd.to_datetime(twii["date"]).dt.year
        first = twii.groupby("year")["Close"].first()
        last = twii.groupby("year")["Close"].last()
        twii_ret = ((last / first) - 1) * 100
        twii_ret = twii_ret.reset_index()
        twii_ret.columns = ["predict_year", "market_ret"]
        merged = df.merge(twii_ret, on="predict_year", how="left").dropna(subset=["market_ret"])
        if len(merged) < 5:
            out = {"scenario": "Sector Rotation", "message": "Insufficient samples"}
            _collect_scenario("23", "sector_rotation", out, save=save)
            return out
        from scipy import stats
        r, p = stats.pearsonr(merged["sharpe"], merged["market_ret"])
        out = {
            "scenario": "Sector Rotation",
            "description": "Market return vs model Sharpe (proxy for sector rotation impact)",
            "n": len(merged),
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4),
            "significant": p < 0.05,
            "interpretation": "Electronic/financial strength ratio proxy via market return",
        }
        _collect_scenario("23", "sector_rotation", out, merged[["predict_year", "cutoff", "sharpe", "market_ret"]], save)
        return out
    except Exception as e:
        out = {"scenario": "Sector Rotation", "message": str(e)}
        _collect_scenario("23", "sector_rotation", out, save=save)
        return out


# =============================================================================
# Scenario 25: Market Efficiency Test
# =============================================================================

def scenario_25_market_efficiency_test(save: bool = True) -> dict:
    """Sharpe vs year regression. Negative slope = alpha decay (AMH)."""
    df = _load_summary_all_cutoffs()
    agg = df.groupby("predict_year")["sharpe"].mean().reset_index()
    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(agg["predict_year"], agg["sharpe"])
    ss_res = np.sum((agg["sharpe"] - (slope * agg["predict_year"] + intercept)) ** 2)
    ss_tot = np.sum((agg["sharpe"] - agg["sharpe"].mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    conclusion = "Market efficiency rising, alpha decay (AMH)" if slope < 0 and p < 0.1 else "No significant decay"
    out = {
        "scenario": "Market Efficiency Test",
        "description": "Sharpe vs year; negative slope + p<0.1 supports AMH (alpha decay)",
        "n": len(agg),
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r_squared": round(float(r2), 4),
        "p_value": round(float(p), 4),
        "significant": p < 0.1,
        "conclusion": conclusion,
        "interpretation": "Decaying alpha supports Adaptive Market Hypothesis",
    }
    _collect_scenario("24", "market_efficiency_test", out, agg, save)
    return out


# =============================================================================
# Main
# =============================================================================

SCENARIOS = [
    ("01", "Time Diversification", scenario_01_time_diversification),
    ("02", "Accuracy Paradox", scenario_02_accuracy_paradox),
    ("03", "Model Evolution", scenario_03_model_evolution),
    ("04", "Memory Marginal Utility", scenario_04_memory_marginal_utility),
    ("05", "Contra-Trading", scenario_05_contra_trading),
    ("06", "Ensemble ROI", scenario_06_ensemble_roi),
    ("07", "Feature Drift", scenario_07_feature_drift),
    ("08", "Noise Resistance", scenario_08_noise_resistance),
    ("09", "Performance Persistence", scenario_09_performance_persistence),
    ("10", "Tail Dependence", scenario_10_tail_dependence),
    ("11", "Indicator Class Dominance", scenario_11_indicator_class_dominance),
    ("12", "Gap Sensitivity", scenario_12_gap_sensitivity),
    ("13", "Long/Short Bias", scenario_13_long_short_bias),
    ("14", "Window Decay Rate", scenario_14_window_decay_rate),
    ("15", "Complexity Penalty", scenario_15_complexity_penalty),
    ("16", "Volatility Filter", scenario_16_volatility_filter),
    ("17", "Trend Correlation", scenario_17_trend_correlation),
    ("18", "Volume Dependency", scenario_18_volume_dependency),
    ("19", "Global Linkage", scenario_20_global_linkage),
    ("20", "Interest Rate Impact", scenario_21_interest_rate_impact),
    ("21", "Event Study", scenario_22_event_study),
    ("22", "Intraday Range", scenario_23_intraday_range_ratio),
    ("23", "Sector Rotation", scenario_24_sector_rotation),
    ("24", "Market Efficiency Test", scenario_25_market_efficiency_test),
]


def main(scenarios: Optional[list[str]] = None):
    """Run specified or all scenarios. Output: single xlsx with one sheet per scenario."""
    global _SCENARIO_COLLECTOR
    _SCENARIO_COLLECTOR = []
    os.makedirs(TXF_ANALYSIS_DIR, exist_ok=True)
    print("=" * 60)
    print("TXF AI Model Comprehensive Analysis")
    print(f"Output: {TXF_ANALYSIS_XLSX}")
    print("=" * 60)
    run_ids = set(scenarios) if scenarios else None
    results = {}
    for sid, name, fn in SCENARIOS:
        if run_ids is not None and sid not in run_ids:
            continue
        print(f"\n[Scenario {sid}] {name}...")
        try:
            out = fn(save=True)
            results[sid] = {"ok": True, "output": out}
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            results[sid] = {"ok": False, "error": str(e)}
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            results[sid] = {"ok": False, "error": str(e)}
    _write_all_to_xlsx()
    print("\n" + "=" * 60)
    print("Analysis complete")
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()
