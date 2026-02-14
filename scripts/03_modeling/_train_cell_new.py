# This file contains the new training cell source - will be injected into notebook
TRAIN_CELL_SOURCE = r'''import torch
import gc
import shutil
import json
import numpy as np
import pandas as pd
from datetime import datetime
from autogluon.tabular import TabularPredictor
from pathlib import Path

DATA_ROOT = Path(DRIVE_PROJECT_ROOT) / "data" if IN_COLAB else Path(LOCAL_PROJECT_ROOT) / "data"

HAS_GPU = torch.cuda.is_available()
print(f"系統檢查: GPU {'可用 ✅' if HAS_GPU else '未偵測到 ⚠️ (將使用 CPU)'}")
print(f"訓練設定: 依序 {CUTOFFS}, 滾動視窗={TRAIN_YEARS}年→預測第四年, 限時={TIME_LIMIT}秒")
print(f"斷線續跑: 若 roll_YYYY 內已有 predictions.csv 則跳過\n")

all_summary = []

for cutoff in CUTOFFS:
    merged_path = DATA_ROOT / "autogluon_ready" / cutoff / f"merged_for_autogluon_{cutoff}.csv"
    if not merged_path.exists():
        print(f"⏭️ 跳過 {cutoff}: 檔案不存在 {merged_path}")
        continue

    print(f"\n{'#'*50}")
    print(f"# 截點 {cutoff}")
    print(f"{'#'*50}")

    df = pd.read_csv(merged_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop(columns=["datetime"], errors="ignore")
    df["year"] = df["date"].dt.year
    df = df.dropna()
    if LABEL not in df.columns:
        print(f"❌ {cutoff}: 無 {LABEL} 欄位")
        continue

    years = sorted(df["year"].unique())
    predict_years = [y for y in years if all((y - i) in years for i in range(1, TRAIN_YEARS + 1))]
    predict_years = sorted(set(predict_years))
    print(f"  資料: {len(df)} 列, 預測年: {predict_years}")

    ROLL_OUTPUT = DATA_ROOT / "models" / cutoff
    ROLL_OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_list = []
    per_year_reports = {}

    for predict_year in predict_years:
        required_train_years = range(predict_year - TRAIN_YEARS, predict_year)
        path_roll = ROLL_OUTPUT / f"roll_{predict_year}"

        if (path_roll / "predictions.csv").exists():
            print(f"  ⏭️ {predict_year}: 已存在，跳過訓練")
            try:
                with open(path_roll / "metrics.json") as f:
                    m = json.load(f)
                summary_list.append({
                    "cutoff": cutoff, "predict_year": int(predict_year),
                    "train_period": m.get("train_period", ""),
                    "rmse": m.get("rmse"), "sharpe": m.get("sharpe"),
                    "best_model": m.get("best_model", ""), "model_path": f"roll_{predict_year}", "skipped": True
                })
                lb = pd.read_csv(path_roll / "leaderboard_with_metrics.csv") if (path_roll / "leaderboard_with_metrics.csv").exists() else pd.DataFrame()
                perf = pd.read_csv(path_roll / "models_performance.csv") if (path_roll / "models_performance.csv").exists() else pd.DataFrame()
                fi = pd.read_csv(path_roll / "feature_importance_all_models.csv") if (path_roll / "feature_importance_all_models.csv").exists() else pd.DataFrame()
                per_year_reports[int(predict_year)] = {"leaderboard": lb, "model_performance": perf, "feature_importance": fi}
            except Exception as e:
                print(f"    ⚠️ 載入既有結果失敗: {e}")
            continue

        missing_years = [y for y in required_train_years if y not in df["year"].unique()]
        if missing_years:
            print(f"  ❌ 跳過 {predict_year}: 缺少 {missing_years}")
            continue

        train_df = df[df["year"].isin(required_train_years)].copy()
        test_df = df[df["year"] == predict_year].copy()
        train_ag = train_df.drop(columns=["date", "datetime", "year"], errors="ignore").dropna()
        test_ag = test_df.drop(columns=["date", "datetime", "year"], errors="ignore").dropna()

        if len(train_ag) < 50 or len(test_ag) < 10:
            print(f"  ⚠️ 跳過 {predict_year}: 資料過少")
            continue

        print(f"  🚀 {predict_year}: 訓練 {min(required_train_years)}-{max(required_train_years)} → 預測 {predict_year} ({len(train_ag)}/{len(test_ag)} 筆)")

        if 'predictor' in locals():
            del predictor
        gc.collect()
        torch.cuda.empty_cache() if HAS_GPU else None

        path_roll.mkdir(parents=True, exist_ok=True)

        try:
            predictor = TabularPredictor(
                label=LABEL, problem_type="regression", eval_metric="rmse", path=str(path_roll),
            ).fit(
                train_data=train_ag, time_limit=TIME_LIMIT, presets="best_quality",
                dynamic_stacking=True, ag_args_fit={'num_gpus': 1} if HAS_GPU else {'num_gpus': 0}
            )
        except Exception as e:
            print(f"  ❌ 訓練錯誤 {predict_year}: {e}")
            continue

        preds = predictor.predict(test_ag)
        y_true = test_ag[LABEL].values
        y_pred = np.asarray(preds)
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        sharpe = compute_sharpe_backtest(y_true, y_pred)

        lb = predictor.leaderboard(test_ag, silent=True)
        model_names = lb["model"].tolist()
        model_perf_rows = []
        fi_rows = []
        for model_name in model_names:
            try:
                preds_m = np.asarray(predictor.predict(test_ag, model=model_name))
                model_perf_rows.append({
                    "predict_year": int(predict_year), "train_period": f"{min(required_train_years)}-{max(required_train_years)}",
                    "model": model_name, "rmse": float(np.sqrt(np.mean((preds_m - y_true) ** 2))),
                    "sharpe": compute_sharpe_backtest(y_true, preds_m),
                })
            except Exception:
                pass
            try:
                fi_m = predictor.feature_importance(data=test_ag, model=model_name).reset_index().rename(columns={"index": "feature"})
                fi_m["model"] = model_name
                fi_m["predict_year"] = int(predict_year)
                fi_rows.append(fi_m)
            except Exception:
                pass

        df_model_perf = pd.DataFrame(model_perf_rows)
        lb_with_metrics = lb.merge(df_model_perf, on="model", how="left")
        df_fi_all = pd.concat(fi_rows, ignore_index=True) if fi_rows else pd.DataFrame()

        lb.to_csv(path_roll / "leaderboard.csv", index=False)
        lb_with_metrics.to_csv(path_roll / "leaderboard_with_metrics.csv", index=False)
        df_model_perf.to_csv(path_roll / "models_performance.csv", index=False)
        df_fi_all.to_csv(path_roll / "feature_importance_all_models.csv", index=False)

        out_pred = test_df.loc[test_ag.index].copy()
        out_pred["pred"] = y_pred
        out_pred[["date", LABEL, "pred"]].to_csv(path_roll / "predictions.csv", index=False)

        metrics = {
            "cutoff": cutoff, "predict_year": int(predict_year),
            "train_period": f"{min(required_train_years)}-{max(required_train_years)}",
            "rmse": rmse, "sharpe": sharpe, "best_model": predictor.model_best,
            "num_models": len(model_names), "model_path": f"roll_{predict_year}",
        }
        with open(path_roll / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        summary_list.append({
            "cutoff": cutoff, "predict_year": int(predict_year),
            "train_period": metrics["train_period"], "rmse": rmse, "sharpe": sharpe,
            "best_model": predictor.model_best, "num_models": len(model_names),
            "model_path": f"roll_{predict_year}", "skipped": False
        })
        per_year_reports[int(predict_year)] = {"leaderboard": lb_with_metrics, "model_performance": df_model_perf, "feature_importance": df_fi_all}
        print(f"  🎉 完成 {predict_year}: Sharpe={sharpe:.4f}, RMSE={rmse:.5f}, Best={predictor.model_best}")

    if summary_list:
        df_summary = pd.DataFrame(summary_list)
        summary_path = ROLL_OUTPUT / "rolling_summary_final.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"\n✅ {cutoff} 完成，總表: {summary_path}")
        all_summary.append(df_summary)

        excel_path = ROLL_OUTPUT / "rolling_models_by_year.xlsx"
        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                for year in sorted(per_year_reports.keys()):
                    report = per_year_reports[year]
                    sheet_name = str(year)[:31]
                    startrow = 0
                    if not report["leaderboard"].empty:
                        pd.DataFrame({"section": ["leaderboard_with_metrics"]}).to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=startrow)
                        startrow += 1
                        report["leaderboard"].to_excel(writer, sheet_name=sheet_name, index=False, startrow=startrow)
                        startrow += len(report["leaderboard"]) + 2
                    if not report["model_performance"].empty:
                        pd.DataFrame({"section": ["model_performance"]}).to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=startrow)
                        startrow += 1
                        report["model_performance"].to_excel(writer, sheet_name=sheet_name, index=False, startrow=startrow)
                        startrow += len(report["model_performance"]) + 2
                    if not report["feature_importance"].empty:
                        pd.DataFrame({"section": ["feature_importance_all_models"]}).to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=startrow)
                        startrow += 1
                        report["feature_importance"].to_excel(writer, sheet_name=sheet_name, index=False, startrow=startrow)
            print(f"  Excel: {excel_path}")
        except Exception as e:
            print(f"  ⚠️ Excel 輸出失敗: {e}")

if all_summary:
    combined = pd.concat(all_summary, ignore_index=True)
    combined.to_csv(DATA_ROOT / "models" / "rolling_summary_all_cutoffs.csv", index=False)
    print(f"\n✅ 全部完成！彙總: data/models/rolling_summary_all_cutoffs.csv")
    display(combined)
else:
    print("\n❌ 沒有產生任何結果，請檢查資料或路徑。")
'''
