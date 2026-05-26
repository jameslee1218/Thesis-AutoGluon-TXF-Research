# data/models 說明（v2.0）

v2.0 主要模型輸出為 matrix 架構，路徑如下：

- `data/models/matrix_runs/{cutoff}/{feature_set}/{exp_id}/train{N}y/roll_{year}/`
- `data/models/matrix_runs/reports/{feature_set}/`

## 單次實驗單位常見檔案

- `metrics.json`
- `predictions.csv`
- `predictions_all_models.csv`
- `leaderboard.csv`
- `leaderboard_with_metrics.csv`
- `feature_importance_all_models.csv`
- `done.marker`（續跑判斷）

## 報告彙總

`data/models/matrix_runs/reports/{feature_set}/` 常見輸出：

- `run_summary.xlsx`
- `comparison_by_metric.xlsx`
- `comparison_tabular_vs_timeseries.xlsx`
- `rolling_year_detail.xlsx`
- `signal_performance.xlsx`
- `feature_importance_pack.xlsx`
- `summary_all_configs.xlsx`

## 注意事項

- 請避免將 Legacy 流程舊輸出與 `matrix_runs` 混放。
- 若重跑同一組實驗，請先確認是否要保留既有 `done.marker`。
