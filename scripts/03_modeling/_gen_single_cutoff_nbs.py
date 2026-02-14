#!/usr/bin/env python3
"""Generate train_autogluon_colab_0900/0915/0930.ipynb and update train_autogluon_colab.ipynb with TRAIN_YEARS loop 2,3,4,5."""
import json
from pathlib import Path

with open("scripts/03_modeling/train_autogluon_colab.ipynb") as f:
    nb = json.load(f)

PATH_TEMPLATE = '''from pathlib import Path

# ========== 路徑設定（Colab 請先跑上方「掛載 Drive」） ==========
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/Thesis-AutoGluon-TXF-Research"
LOCAL_PROJECT_ROOT = "/Volumes/Transcend/thesis/github_clone/Thesis-AutoGluon-TXF-Research"

PROJECT_ROOT = Path(DRIVE_PROJECT_ROOT) if IN_COLAB else Path(LOCAL_PROJECT_ROOT)
DATA_ROOT = PROJECT_ROOT / "data"

# 本 notebook 僅處理單一截點（可同時開三個 Colab 分別跑 0900/0915/0930，互不干擾）
CUTOFF = "%s"
# 訓練年數迴圈：2,3,4,5 年訓練 → 預測下一年，比較哪個較佳
TRAIN_YEARS_LIST = [2, 3, 4, 5]
LABEL = "target_return"
TIME_LIMIT = 30  # 秒；每段訓練時間

print("PROJECT_ROOT:", PROJECT_ROOT)
print("CUTOFF:", CUTOFF, "| TRAIN_YEARS_LIST:", TRAIN_YEARS_LIST)
p = DATA_ROOT / "autogluon_ready" / CUTOFF / ("merged_for_autogluon_" + CUTOFF + ".csv")
print("  資料:", "✅" if p.exists() else "❌", p)
'''

def to_nb_lines(s):
    lines = s.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1 or s.endswith("\n"):
            result.append(line + "\n")
        else:
            result.append(line + "\n" if line else "\n")
    return result if result else ["\n"]

TRAIN_SINGLE = r'''import torch
import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularPredictor

DATA_ROOT = Path(DRIVE_PROJECT_ROOT) / "data" if IN_COLAB else Path(LOCAL_PROJECT_ROOT) / "data"

HAS_GPU = torch.cuda.is_available()
print(f"系統檢查: GPU {'可用 ✅' if HAS_GPU else '未偵測到 ⚠️ (將使用 CPU)'}")
print(f"訓練設定: 截點 {CUTOFF}, 訓練年數={TRAIN_YEARS_LIST}, 限時={TIME_LIMIT}秒")
print(f"斷線續跑: 若 train{{N}}y/roll_YYYY 內已有 predictions.csv 則跳過\n")

cutoff = CUTOFF
merged_path = DATA_ROOT / "autogluon_ready" / cutoff / f"merged_for_autogluon_{cutoff}.csv"
if not merged_path.exists():
    print(f"❌ 檔案不存在: {merged_path}")
else:
    df = pd.read_csv(merged_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop(columns=["datetime"], errors="ignore")
    df["year"] = df["date"].dt.year
    df = df.dropna()
    if LABEL not in df.columns:
        print(f"❌ 無 {LABEL} 欄位")
    else:
        years = sorted(df["year"].unique())
        ROLL_OUTPUT = DATA_ROOT / "models" / cutoff
        ROLL_OUTPUT.mkdir(parents=True, exist_ok=True)
        all_summary = []
        all_models_perf = []

        for TRAIN_YEARS in TRAIN_YEARS_LIST:
            predict_years = [y for y in years if all((y - i) in years for i in range(1, TRAIN_YEARS + 1))]
            predict_years = sorted(set(predict_years))
            print(f"\n{'='*50}")
            print(f"# train{TRAIN_YEARS}y → 預測年: {predict_years}")
            print(f"{'='*50}")

            train_out = ROLL_OUTPUT / f"train{TRAIN_YEARS}y"
            train_out.mkdir(parents=True, exist_ok=True)
            summary_list = []
            per_year_reports = {}

            for predict_year in predict_years:
                required_train_years = range(predict_year - TRAIN_YEARS, predict_year)
                path_roll = train_out / f"roll_{predict_year}"

                if (path_roll / "predictions.csv").exists():
                    print(f"  ⏭️ {predict_year}: 已存在，跳過訓練")
                    try:
                        with open(path_roll / "metrics.json") as f:
                            m = json.load(f)
                        summary_list.append({
                            "cutoff": cutoff, "train_years": TRAIN_YEARS, "predict_year": int(predict_year),
                            "train_period": m.get("train_period", ""),
                            "rmse": m.get("rmse"), "sharpe": m.get("sharpe"),
                            "best_model": m.get("best_model", ""), "model_path": f"train{TRAIN_YEARS}y/roll_{predict_year}", "skipped": True
                        })
                        lb = pd.read_csv(path_roll / "leaderboard_with_metrics.csv") if (path_roll / "leaderboard_with_metrics.csv").exists() else pd.DataFrame()
                        perf = pd.read_csv(path_roll / "models_performance.csv") if (path_roll / "models_performance.csv").exists() else pd.DataFrame()
                        fi = pd.read_csv(path_roll / "feature_importance_all_models.csv") if (path_roll / "feature_importance_all_models.csv").exists() else pd.DataFrame()
                        per_year_reports[int(predict_year)] = {"leaderboard": lb, "model_performance": perf, "feature_importance": fi}
                        if not perf.empty:
                            perf["train_years"] = TRAIN_YEARS
                            perf["cutoff"] = cutoff
                            all_models_perf.append(perf)
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
                pred_cols = {"date": test_df.loc[test_ag.index]["date"].values, LABEL: y_true, "pred_best": y_pred}

                for model_name in model_names:
                    try:
                        preds_m = np.asarray(predictor.predict(test_ag, model=model_name))
                        safe_name = model_name.replace(" ", "_").replace("/", "_")
                        pred_cols[f"pred_{safe_name}"] = preds_m
                        model_perf_rows.append({
                            "cutoff": cutoff, "train_years": TRAIN_YEARS, "predict_year": int(predict_year),
                            "train_period": f"{min(required_train_years)}-{max(required_train_years)}",
                            "model": model_name, "rmse": float(np.sqrt(np.mean((preds_m - y_true) ** 2))),
                            "sharpe": compute_sharpe_backtest(y_true, preds_m),
                        })
                    except Exception:
                        pass

                df_model_perf = pd.DataFrame(model_perf_rows)
                # 特徵重要性：僅計算 RMSE 前3 與 Sharpe 前3，去重後節省時間
                top_rmse = df_model_perf.nsmallest(3, "rmse")["model"].tolist()
                top_sharpe = df_model_perf.nlargest(3, "sharpe")["model"].tolist()
                fi_models = list(dict.fromkeys(top_rmse + top_sharpe))
                fi_rows = []
                for model_name in fi_models:
                    try:
                        fi_m = predictor.feature_importance(data=test_ag, model=model_name).reset_index().rename(columns={"index": "feature"})
                        fi_m["model"] = model_name
                        fi_m["predict_year"] = int(predict_year)
                        fi_m["train_years"] = TRAIN_YEARS
                        fi_rows.append(fi_m)
                    except Exception:
                        pass
                all_models_perf.append(df_model_perf)
                lb_with_metrics = lb.merge(
                    df_model_perf[["model", "rmse", "sharpe"]].rename(columns={"rmse": "rmse_test", "sharpe": "sharpe_test"}),
                    on="model", how="left"
                )
                df_fi_all = pd.concat(fi_rows, ignore_index=True) if fi_rows else pd.DataFrame()

                lb.to_csv(path_roll / "leaderboard.csv", index=False)
                lb_with_metrics.to_csv(path_roll / "leaderboard_with_metrics.csv", index=False)
                df_model_perf.to_csv(path_roll / "models_performance.csv", index=False)
                df_fi_all.to_csv(path_roll / "feature_importance_all_models.csv", index=False)

                out_pred = test_df.loc[test_ag.index].copy()
                out_pred["pred"] = y_pred
                out_pred[["date", LABEL, "pred"]].to_csv(path_roll / "predictions.csv", index=False)

                pred_all_df = pd.DataFrame(pred_cols)
                pred_all_df.to_csv(path_roll / "predictions_all_models.csv", index=False)

                metrics = {
                    "cutoff": cutoff, "train_years": TRAIN_YEARS, "predict_year": int(predict_year),
                    "train_period": f"{min(required_train_years)}-{max(required_train_years)}",
                    "rmse": rmse, "sharpe": sharpe, "best_model": predictor.model_best,
                    "num_models": len(model_names), "model_path": f"train{TRAIN_YEARS}y/roll_{predict_year}",
                }
                with open(path_roll / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)

                summary_list.append({
                    "cutoff": cutoff, "train_years": TRAIN_YEARS, "predict_year": int(predict_year),
                    "train_period": metrics["train_period"], "rmse": rmse, "sharpe": sharpe,
                    "best_model": predictor.model_best, "num_models": len(model_names),
                    "model_path": metrics["model_path"], "skipped": False
                })
                per_year_reports[int(predict_year)] = {"leaderboard": lb_with_metrics, "model_performance": df_model_perf, "feature_importance": df_fi_all}
                print(f"  🎉 完成 {predict_year}: Sharpe={sharpe:.4f}, RMSE={rmse:.5f}, Best={predictor.model_best}")

            if summary_list:
                df_summary = pd.DataFrame(summary_list)
                df_summary.to_csv(train_out / f"summary_train{TRAIN_YEARS}y.csv", index=False)
                all_summary.append(df_summary)
                excel_path = ROLL_OUTPUT / f"rolling_models_by_year_train{TRAIN_YEARS}y.xlsx"
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
            combined.to_csv(ROLL_OUTPUT / "summary_all_train_years.csv", index=False)
            print(f"\n✅ 彙總: {ROLL_OUTPUT / 'summary_all_train_years.csv'}")
            if all_models_perf:
                perf_all = pd.concat(all_models_perf, ignore_index=True)
                perf_all.to_csv(ROLL_OUTPUT / "models_performance_all_train_years.csv", index=False)
                print(f"✅ 所有模型紀錄: {ROLL_OUTPUT / 'models_performance_all_train_years.csv'}")
            display(combined)
        else:
            print("\n❌ 沒有產生任何結果，請檢查資料或路徑。")
'''

for cutoff in ("0900", "0915", "0930"):
    nb_copy = json.loads(json.dumps(nb))
    nb_copy["cells"][0]["source"] = [
        f"# AutoGluon 滾動訓練 — 截點 {cutoff}（訓練年 2,3,4,5 比較）\n",
        "\n",
        f"- **輸入**：`data/autogluon_ready/{cutoff}/merged_for_autogluon_{cutoff}.csv`\n",
        "- **流程**：迴圈 train_years=2,3,4,5，各「N年訓練→預測第N+1年」\n",
        f"- **輸出**：`data/models/{cutoff}/train{{N}}y/roll_YYYY/`，含所有模型紀錄\n",
        "- **斷線續跑**：若 `predictions.csv` 已存在則跳過\n",
        "- **並行**：可同時開三個 Colab 分別跑 0900、0915、0930\n",
        "- **Colab**：掛載 Drive 後設定 `DRIVE_PROJECT_ROOT`\n"
    ]
    nb_copy["cells"][4]["source"] = to_nb_lines(PATH_TEMPLATE % cutoff)
    nb_copy["cells"][11]["source"] = [f"## 5. 滾動訓練：截點 {cutoff}，train_years=2,3,4,5 迴圈，已存在則跳過\n"]
    nb_copy["cells"][12]["source"] = to_nb_lines(TRAIN_SINGLE)
    nb_copy["cells"][14]["source"] = [
        f"# 輸出架構見 data/models/README.md\n",
        f"# 彙總表：data/models/{cutoff}/summary_all_train_years.csv\n",
        f"# 所有模型：data/models/{cutoff}/models_performance_all_train_years.csv\n",
        "from pathlib import Path\n",
        f'summary_path = Path(DRIVE_PROJECT_ROOT if IN_COLAB else LOCAL_PROJECT_ROOT) / "data" / "models" / "{cutoff}" / "summary_all_train_years.csv"\n',
        "if summary_path.exists():\n",
        "    display(pd.read_csv(summary_path))\n",
        "else:\n",
        "    print(\"尚未產生彙總表，請先執行上方滾動訓練 cell。\")\n"
    ]
    for c in nb_copy["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
    out_path = f"scripts/03_modeling/train_autogluon_colab_{cutoff}.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb_copy, f, ensure_ascii=False, indent=2)
    print(f"Created {out_path}")

# 註：依序版 train_autogluon_colab.ipynb 維持原結構；train_years 2,3,4,5 迴圈請使用上述三個單一截點 notebook
