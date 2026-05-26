# Colab Matrix Runbook

本文件對應以下 4 本 notebook：

- `train_matrix_0900.ipynb`
- `train_matrix_0915.ipynb`
- `train_matrix_0930.ipynb`
- `train_matrix_premarket3.ipynb`

## 最小上線步驟

1. 把專案放到 Google Drive（含 `data/autogluon/...xlsx`）。
2. 開啟任一本 notebook。
3. Cell 1 執行 Drive 掛載。
4. Cell 2 只改 `DRIVE_PROJECT_ROOT`（必要時調整 `time_limit`、`presets`）。
5. Cell 3 跑 preflight，確認資料可讀與欄位齊全。
6. Cell 4 一鍵執行全組合。

## 續跑機制

- 每個實驗單位路徑：
  - `data/models/matrix_runs/{cutoff}/{feature_set}/{exp_id}/train{N}y/roll_{year}/`
- 若已有 `done.marker` 或 `metrics.json`，重跑時會自動跳過。

## 輸出報告（每次執行都會更新）

報告根目錄：
- `data/models/matrix_runs/reports/{feature_set}/`

固定輸出：
- `run_summary.xlsx`
- `comparison_by_metric.xlsx`
- `comparison_tabular_vs_timeseries.xlsx`
- `rolling_year_detail.xlsx`
- `signal_performance.xlsx`
- `feature_importance_pack.xlsx`
- `summary_all_configs.xlsx`

## 建議的執行順序

1. `train_matrix_0900.ipynb`
2. `train_matrix_0915.ipynb`
3. `train_matrix_0930.ipynb`
4. `train_matrix_premarket3.ipynb`

## 常見問題

- **找不到資料檔**：先確認 `data/autogluon/{cutoff}/merged_for_autogluon_{cutoff}.xlsx` 是否存在。
- **TimeSeries 套件錯誤**：在 Colab 先安裝 AutoGluon time series 依賴。
- **執行很久**：先把 Cell 2 的 `train_years_list` 改為 `[2]` 做 smoke test，再恢復完整設定。
