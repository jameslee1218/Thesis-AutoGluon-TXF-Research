#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab 一鍵訓練矩陣核心模組（Tabular + TimeSeries）。

用途：
1) 讀取 data/autogluon/{cutoff}/merged_for_autogluon_{cutoff}.xlsx
2) 依 train_years=[2,3,5] 做滾動訓練與測試
3) 跑多組 metric/config（RMSE、Sharpe）
4) 輸出可續跑結果（done.marker + metrics.json）與完整 xlsx 報告
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from autogluon.core.metrics import make_scorer
    from autogluon.tabular import TabularPredictor
except Exception:  # pragma: no cover
    TabularPredictor = None
    make_scorer = None

try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
except Exception:  # pragma: no cover
    TimeSeriesDataFrame = None
    TimeSeriesPredictor = None


STATIC_PREMARKET_FEATURES = ["sp500_prev_return", "open_gap_pct", "vix_prev_close"]


@dataclass
class MatrixConfig:
    project_root: str
    cutoff_list: List[str]
    feature_set: str  # full | premarket3
    train_years_list: List[int]
    time_limit: int
    presets: str
    resume: bool
    label_col: str
    random_seed: int


def compute_sharpe_backtest(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    signal = np.sign(y_pred)
    realized = signal * y_true
    std = realized.std(ddof=1) if realized.size > 1 else 0.0
    if std == 0 or np.isnan(std):
        return 0.0
    return float((realized.mean() / std) * math.sqrt(252))


def _sharpe_score_func(y_true, y_pred, **kwargs):
    return compute_sharpe_backtest(np.asarray(y_true), np.asarray(y_pred))


def _build_tabular_sharpe_scorer():
    if make_scorer is None:
        return "rmse"
    return make_scorer(
        name="annualized_sharpe_reg",
        score_func=_sharpe_score_func,
        optimum=1,
        greater_is_better=True,
    )


def experiment_grid(feature_set: str) -> List[Dict[str, Any]]:
    # default = 完全使用 AutoGluon 預設訓練 metric（不手動指定 eval_metric）
    # Full features: 保留 4 組核心比較
    # Premarket3: 僅保留 2 組資訊含量基準比較
    if feature_set == "premarket3":
        return [
            {"id": "tab_default_sel_rmse", "family": "tabular", "train_metric": "default", "select_metric": "rmse"},
            {"id": "tab_sharpe_sel_sharpe", "family": "tabular", "train_metric": "sharpe", "select_metric": "sharpe"},
        ]
    return [
        {"id": "tab_default_sel_rmse", "family": "tabular", "train_metric": "default", "select_metric": "rmse"},
        {"id": "tab_sharpe_sel_sharpe", "family": "tabular", "train_metric": "sharpe", "select_metric": "sharpe"},
        {"id": "ts_default_sel_rmse", "family": "timeseries", "train_metric": "default", "select_metric": "rmse"},
        {"id": "ts_default_sel_sharpe", "family": "timeseries", "train_metric": "default", "select_metric": "sharpe"},
    ]


def _data_root(project_root: Path) -> Path:
    return project_root / "data"


def _autogluon_input_path(project_root: Path, cutoff: str) -> Path:
    return _data_root(project_root) / "autogluon" / cutoff / f"merged_for_autogluon_{cutoff}.xlsx"


def _results_root(project_root: Path) -> Path:
    return _data_root(project_root) / "models" / "matrix_runs"


def load_cutoff_df(project_root: Path, cutoff: str, feature_set: str, label_col: str) -> pd.DataFrame:
    p = _autogluon_input_path(project_root, cutoff)
    if not p.exists():
        raise FileNotFoundError(f"找不到資料檔：{p}")
    df = pd.read_excel(p)
    if "date" not in df.columns:
        raise ValueError(f"{p.name} 缺少 date 欄位")
    if label_col not in df.columns:
        raise ValueError(f"{p.name} 缺少 {label_col} 欄位")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", label_col]).copy()
    if feature_set == "premarket3":
        keep = ["date", label_col] + [c for c in STATIC_PREMARKET_FEATURES if c in df.columns]
        df = df[keep].copy()
    return df.sort_values("date").reset_index(drop=True)


def preflight(project_root: Path, cutoff_list: Iterable[str], feature_set: str, label_col: str) -> pd.DataFrame:
    rows = []
    for cutoff in cutoff_list:
        p = _autogluon_input_path(project_root, cutoff)
        ok = p.exists()
        msg = ""
        n_rows = 0
        n_cols = 0
        if ok:
            try:
                df = load_cutoff_df(project_root, cutoff, feature_set, label_col)
                n_rows = len(df)
                n_cols = len(df.columns)
            except Exception as e:  # pragma: no cover
                ok = False
                msg = str(e)
        rows.append({"cutoff": cutoff, "path": str(p), "ok": ok, "rows": n_rows, "cols": n_cols, "note": msg})
    return pd.DataFrame(rows)


def _predict_years(df: pd.DataFrame, train_years: int) -> List[int]:
    years = sorted(df["date"].dt.year.unique())
    return [y for y in years if all((y - i) in years for i in range(1, train_years + 1))]


def _unit_dir(base: Path, cutoff: str, feature_set: str, exp_id: str, train_years: int, predict_year: int) -> Path:
    return base / cutoff / feature_set / exp_id / f"train{train_years}y" / f"roll_{predict_year}"


def _save_metrics_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _detect_num_gpus() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    return 0


def _train_tabular(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    train_metric: str,
    time_limit: int,
    presets: str,
    seed: int,
    model_path: Path,
    num_gpus: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if TabularPredictor is None:
        raise RuntimeError("autogluon.tabular 未安裝")

    train_ag = train_df.drop(columns=["date"], errors="ignore").dropna()
    test_ag = test_df.drop(columns=["date"], errors="ignore").dropna()
    y_true = test_ag[label_col].to_numpy(dtype=float)

    if train_metric == "sharpe":
        eval_metric = _build_tabular_sharpe_scorer()
        eval_metric_name = "custom_sharpe_scorer"
        predictor = TabularPredictor(
            label=label_col,
            problem_type="regression",
            eval_metric=eval_metric,
            path=str(model_path),
        ).fit(
            train_data=train_ag,
            time_limit=time_limit,
            presets=presets,
            dynamic_stacking=True,
            ag_args_fit={"num_gpus": num_gpus, "random_state": seed},
        )
    else:
        eval_metric_name = "autogluon_default"
        predictor = TabularPredictor(
            label=label_col,
            problem_type="regression",
            path=str(model_path),
        ).fit(
            train_data=train_ag,
            time_limit=time_limit,
            presets=presets,
            # Keep training path stable across environments.
            dynamic_stacking=False,
            ag_args_fit={"num_gpus": num_gpus, "random_state": seed},
        )

    lb = predictor.leaderboard(test_ag, silent=True)
    perf_rows = []
    fi_rows = []
    pred_pack = {"date": test_df.loc[test_ag.index, "date"].astype(str).values, label_col: y_true}

    for model_name in lb["model"].tolist():
        preds = np.asarray(predictor.predict(test_ag, model=model_name), dtype=float)
        rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))
        sharpe = compute_sharpe_backtest(y_true, preds)
        perf_rows.append({"model": model_name, "rmse": rmse, "sharpe": sharpe})
        pred_pack[f"pred_{model_name.replace(' ', '_').replace('/', '_')}"] = preds
        try:
            fi = predictor.feature_importance(data=test_ag, model=model_name).reset_index().rename(columns={"index": "feature"})
            fi["model"] = model_name
            fi_rows.append(fi)
        except Exception:
            pass

    model_perf = pd.DataFrame(perf_rows).sort_values("rmse", ascending=True).reset_index(drop=True)
    pred_df = pd.DataFrame(pred_pack)
    fi_df = pd.concat(fi_rows, ignore_index=True) if fi_rows else pd.DataFrame()
    lb.to_csv(model_path / "leaderboard.csv", index=False)
    model_perf.to_csv(model_path / "models_performance.csv", index=False)
    pred_df.to_csv(model_path / "predictions_all_models.csv", index=False)
    fi_df.to_csv(model_path / "feature_importance.csv", index=False)
    return model_perf, pred_df, fi_df, eval_metric_name


def _to_ts_frame(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    out = df[["date", label_col]].copy()
    out = out.rename(columns={"date": "timestamp", label_col: "target"})
    out["item_id"] = "txf"
    return out[["item_id", "timestamp", "target"]].sort_values("timestamp")


def _rolling_ts_predict(
    predictor: Any,
    full_df: pd.DataFrame,
    test_dates: List[pd.Timestamp],
    model_name: str | None = None,
) -> pd.DataFrame:
    rows = []
    for d in test_dates:
        hist = full_df[full_df["timestamp"] < d]
        if len(hist) < 10:
            continue
        ts_hist = TimeSeriesDataFrame.from_data_frame(hist, id_column="item_id", timestamp_column="timestamp")
        pred = predictor.predict(ts_hist, model=model_name) if model_name else predictor.predict(ts_hist)
        pred_mean = float(pred["mean"].iloc[0])
        rows.append({"date": d, "pred": pred_mean})
    return pd.DataFrame(rows)


def _train_timeseries(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_metric: str,
    time_limit: int,
    presets: str,
    model_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if TimeSeriesPredictor is None or TimeSeriesDataFrame is None:
        raise RuntimeError("autogluon.timeseries 未安裝")

    train_ts_df = _to_ts_frame(train_df, "target_return")
    full_ts_df = pd.concat([train_ts_df, _to_ts_frame(test_df, "target_return")], ignore_index=True).sort_values("timestamp")
    test_dates = sorted(pd.to_datetime(test_df["date"]).tolist())
    ts_train = TimeSeriesDataFrame.from_data_frame(train_ts_df, id_column="item_id", timestamp_column="timestamp")

    # default / sharpe_proxy 都不硬指定 eval_metric，走框架預設
    eval_metric_name = "autogluon_default"
    predictor = TimeSeriesPredictor(
        prediction_length=1,
        target="target",
        freq="B",
        path=str(model_path),
    ).fit(train_data=ts_train, time_limit=time_limit, presets=presets)

    model_names = predictor.model_names()
    perf_rows = []
    pred_frames = []
    for m in model_names:
        pred_df = _rolling_ts_predict(predictor, full_ts_df, test_dates, model_name=m)
        if pred_df.empty:
            continue
        joined = pred_df.merge(test_df[["date", "target_return"]], on="date", how="inner")
        y_true = joined["target_return"].to_numpy(dtype=float)
        y_pred = joined["pred"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2))) if len(joined) else np.nan
        sharpe = compute_sharpe_backtest(y_true, y_pred) if len(joined) else np.nan
        perf_rows.append({"model": m, "rmse": rmse, "sharpe": sharpe})
        j = joined.rename(columns={"pred": f"pred_{m.replace(' ', '_').replace('/', '_')}"})
        pred_frames.append(j[["date", f"pred_{m.replace(' ', '_').replace('/', '_')}"]])

    model_perf = pd.DataFrame(perf_rows).sort_values("rmse", ascending=True).reset_index(drop=True)
    base = test_df[["date", "target_return"]].copy()
    for pf in pred_frames:
        base = base.merge(pf, on="date", how="left")
    base.to_csv(model_path / "predictions_all_models.csv", index=False)
    model_perf.to_csv(model_path / "ts_model_scores.csv", index=False)

    # TimeSeries 無通用 permutation FI：輸出 proxy（特徵與目標相關性）
    fi_proxy = pd.DataFrame(columns=["feature", "proxy_corr_abs"])
    model_perf.to_csv(model_path / "leaderboard.csv", index=False)
    fi_proxy.to_csv(model_path / "feature_importance.csv", index=False)
    if train_metric == "sharpe_proxy":
        eval_metric_name = "autogluon_default_with_sharpe_proxy_experimental"
    return model_perf, base, fi_proxy, eval_metric_name


def run_matrix(cfg: MatrixConfig) -> Dict[str, str]:
    project_root = Path(cfg.project_root)
    results_root = _results_root(project_root)
    results_root.mkdir(parents=True, exist_ok=True)
    detected_num_gpus = _detect_num_gpus()
    compute_device = "gpu" if detected_num_gpus > 0 else "cpu"

    all_rows = []
    all_fi = []
    all_model_perf = []

    for cutoff in cfg.cutoff_list:
        df = load_cutoff_df(project_root, cutoff, cfg.feature_set, cfg.label_col)
        for train_years in cfg.train_years_list:
            for predict_year in _predict_years(df, train_years):
                train_year_range = range(predict_year - train_years, predict_year)
                train_df = df[df["date"].dt.year.isin(train_year_range)].copy()
                test_df = df[df["date"].dt.year == predict_year].copy()
                if len(train_df) < 50 or len(test_df) < 10:
                    continue

                for exp in experiment_grid(cfg.feature_set):
                    udir = _unit_dir(results_root, cutoff, cfg.feature_set, exp["id"], train_years, predict_year)
                    udir.mkdir(parents=True, exist_ok=True)
                    done_marker = udir / "done.marker"
                    metrics_path = udir / "metrics.json"
                    if cfg.resume and (done_marker.exists() or metrics_path.exists()):
                        try:
                            with open(metrics_path, "r", encoding="utf-8") as f:
                                m = json.load(f)
                            all_rows.append(m)
                        except Exception:
                            pass
                        continue

                    try:
                        if exp["family"] == "tabular":
                            model_perf, pred_df, fi_df, actual_train_eval_metric = _train_tabular(
                                train_df=train_df,
                                test_df=test_df,
                                label_col=cfg.label_col,
                                train_metric=exp["train_metric"],
                                time_limit=cfg.time_limit,
                                presets=cfg.presets,
                                seed=cfg.random_seed,
                                model_path=udir,
                                num_gpus=detected_num_gpus,
                            )
                        else:
                            ts_train = train_df.rename(columns={cfg.label_col: "target_return"})
                            ts_test = test_df.rename(columns={cfg.label_col: "target_return"})
                            model_perf, pred_df, fi_df, actual_train_eval_metric = _train_timeseries(
                                train_df=ts_train,
                                test_df=ts_test,
                                train_metric=exp["train_metric"],
                                time_limit=cfg.time_limit,
                                presets=cfg.presets,
                                model_path=udir,
                            )

                        if model_perf.empty:
                            continue

                        if exp["select_metric"] == "sharpe":
                            best_row = model_perf.sort_values("sharpe", ascending=False).iloc[0]
                        else:
                            best_row = model_perf.sort_values("rmse", ascending=True).iloc[0]

                        best_model = best_row["model"]
                        pred_col = f"pred_{str(best_model).replace(' ', '_').replace('/', '_')}"
                        if pred_col in pred_df.columns:
                            y_true = pred_df[cfg.label_col].to_numpy(dtype=float)
                            y_pred = pred_df[pred_col].to_numpy(dtype=float)
                        else:
                            y_true = np.array([])
                            y_pred = np.array([])

                        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2))) if y_true.size else float("nan")
                        sharpe = compute_sharpe_backtest(y_true, y_pred) if y_true.size else float("nan")
                        rec = {
                            "cutoff": cutoff,
                            "feature_set": cfg.feature_set,
                            "family": exp["family"],
                            "exp_id": exp["id"],
                            "train_metric": exp["train_metric"],
                            "select_metric": exp["select_metric"],
                            "actual_train_eval_metric": actual_train_eval_metric,
                            "actual_select_rule": f"best_by_{exp['select_metric']}",
                            "train_years": int(train_years),
                            "predict_year": int(predict_year),
                            "train_period": f"{predict_year - train_years}-{predict_year - 1}",
                            "best_model": str(best_model),
                            "rmse": rmse,
                            "sharpe": sharpe,
                            "num_gpus_used": int(detected_num_gpus),
                            "compute_device": compute_device,
                            "unit_dir": str(udir),
                        }
                        _save_metrics_json(metrics_path, rec)
                        done_marker.write_text("done\n", encoding="utf-8")
                        all_rows.append(rec)

                        mp = model_perf.copy()
                        mp["cutoff"] = cutoff
                        mp["feature_set"] = cfg.feature_set
                        mp["exp_id"] = exp["id"]
                        mp["train_metric"] = exp["train_metric"]
                        mp["select_metric"] = exp["select_metric"]
                        mp["actual_train_eval_metric"] = actual_train_eval_metric
                        mp["actual_select_rule"] = f"best_by_{exp['select_metric']}"
                        mp["train_years"] = int(train_years)
                        mp["predict_year"] = int(predict_year)
                        all_model_perf.append(mp)

                        if not fi_df.empty:
                            fi = fi_df.copy()
                            fi["cutoff"] = cutoff
                            fi["feature_set"] = cfg.feature_set
                            fi["exp_id"] = exp["id"]
                            fi["train_years"] = int(train_years)
                            fi["predict_year"] = int(predict_year)
                            all_fi.append(fi)
                    except Exception as e:
                        err_rec = {
                            "cutoff": cutoff,
                            "feature_set": cfg.feature_set,
                            "family": exp["family"],
                            "exp_id": exp["id"],
                            "train_metric": exp["train_metric"],
                            "select_metric": exp["select_metric"],
                            "train_years": int(train_years),
                            "predict_year": int(predict_year),
                            "error": str(e),
                            "num_gpus_used": int(detected_num_gpus),
                            "compute_device": compute_device,
                            "unit_dir": str(udir),
                        }
                        _save_metrics_json(udir / "error.json", err_rec)

    report_dir = results_root / "reports" / cfg.feature_set
    report_dir.mkdir(parents=True, exist_ok=True)
    run_summary = pd.DataFrame(all_rows)
    model_perf_all = pd.concat(all_model_perf, ignore_index=True) if all_model_perf else pd.DataFrame()
    fi_all = pd.concat(all_fi, ignore_index=True) if all_fi else pd.DataFrame()

    comparison_by_metric = (
        run_summary.groupby(["family", "train_metric", "select_metric"], dropna=False)[["rmse", "sharpe"]]
        .mean()
        .reset_index()
        if not run_summary.empty
        else pd.DataFrame()
    )
    comparison_tab_vs_ts = (
        run_summary.groupby(["family"], dropna=False)[["rmse", "sharpe"]].mean().reset_index()
        if not run_summary.empty
        else pd.DataFrame()
    )
    rolling_year_detail = run_summary.sort_values(["cutoff", "train_years", "predict_year"]) if not run_summary.empty else pd.DataFrame()
    signal_perf = model_perf_all if not model_perf_all.empty else pd.DataFrame()

    out_run = report_dir / "run_summary.xlsx"
    out_cmp_metric = report_dir / "comparison_by_metric.xlsx"
    out_cmp_family = report_dir / "comparison_tabular_vs_timeseries.xlsx"
    out_roll = report_dir / "rolling_year_detail.xlsx"
    out_signal = report_dir / "signal_performance.xlsx"
    out_fi = report_dir / "feature_importance_pack.xlsx"
    out_all = report_dir / "summary_all_configs.xlsx"

    run_summary.to_excel(out_run, index=False, engine="openpyxl")
    comparison_by_metric.to_excel(out_cmp_metric, index=False, engine="openpyxl")
    comparison_tab_vs_ts.to_excel(out_cmp_family, index=False, engine="openpyxl")
    rolling_year_detail.to_excel(out_roll, index=False, engine="openpyxl")
    signal_perf.to_excel(out_signal, index=False, engine="openpyxl")
    fi_all.to_excel(out_fi, index=False, engine="openpyxl")

    with pd.ExcelWriter(out_all, engine="openpyxl") as w:
        run_summary.to_excel(w, index=False, sheet_name="run_summary")
        comparison_by_metric.to_excel(w, index=False, sheet_name="comparison_by_metric")
        comparison_tab_vs_ts.to_excel(w, index=False, sheet_name="tabular_vs_timeseries")
        rolling_year_detail.to_excel(w, index=False, sheet_name="rolling_year_detail")
        signal_perf.to_excel(w, index=False, sheet_name="signal_performance")
        fi_all.to_excel(w, index=False, sheet_name="feature_importance")

    return {
        "run_summary": str(out_run),
        "comparison_by_metric": str(out_cmp_metric),
        "comparison_tabular_vs_timeseries": str(out_cmp_family),
        "rolling_year_detail": str(out_roll),
        "signal_performance": str(out_signal),
        "feature_importance_pack": str(out_fi),
        "summary_all_configs": str(out_all),
    }
