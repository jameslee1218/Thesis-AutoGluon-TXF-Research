# AutoGluon 滾動訓練輸出架構

本目錄存放 AutoGluon 滾動訓練（三年訓練、預測第四年）之所有產出，供後續分析與回測使用。

---

## 一、目錄樹

```
data/models/
├── README.md                    ← 本說明
│
├── 0900/                        截點 09:00 前
│   ├── train2y/                 2年訓練 → 預測第3年
│   │   ├── roll_2014/           2012-2013 訓練 → 2014 預測
│   │   │   ├── predictions.csv
│   │   │   ├── predictions_all_models.csv
│   │   │   ├── metrics.json
│   │   │   ├── leaderboard.csv
│   │   │   ├── leaderboard_with_metrics.csv
│   │   │   ├── models_performance.csv
│   │   │   ├── feature_importance_all_models.csv
│   │   │   └── models/           AutoGluon 產出之模型檔
│   │   ├── roll_2015/
│   │   └── ...
│   ├── train3y/                 3年訓練 → 預測第4年
│   ├── train4y/                 4年訓練 → 預測第5年
│   ├── train5y/                 5年訓練 → 預測第6年
│   ├── summary_train2y.csv
│   ├── summary_train3y.csv
│   ├── summary_train4y.csv
│   ├── summary_train5y.csv
│   ├── summary_all_train_years.csv
│   ├── models_performance_all_train_years.csv
│   ├── rolling_models_by_year_train2y.xlsx
│   ├── rolling_models_by_year_train3y.xlsx
│   └── ...
│
├── 0915/                        截點 09:15 前（結構同上）
├── 0930/                        截點 09:30 前（結構同上）
│
└── rolling_summary_all_cutoffs.csv   ← 依序版 notebook 產出
```

---

## 二、單一 roll 目錄內檔案說明

| 檔案 | 說明 |
|------|------|
| `predictions.csv` | 最佳模型預測：`date`, `target_return`, `pred` |
| `predictions_all_models.csv` | 所有模型預測：`date`, `target_return`, `pred`(best), `pred_Model1`, `pred_Model2`, ... |
| `metrics.json` | 該 roll 的 rmse、sharpe、best_model、train_period 等 |
| `leaderboard.csv` | AutoGluon 原始 leaderboard |
| `leaderboard_with_metrics.csv` | 各模型 RMSE、Sharpe 回測 |
| `models_performance.csv` | 每模型：predict_year, train_period, model, rmse, sharpe |
| `feature_importance_all_models.csv` | 每模型：feature, importance, model, predict_year |
| `models/` | AutoGluon 產出之模型檔（可載入 predictor） |

---

## 三、彙總表說明

| 檔案 | 說明 |
|------|------|
| `summary_train{N}y.csv` | 該 train_years 下各 roll 的 rmse、sharpe、best_model |
| `summary_all_train_years.csv` | 所有 train_years 合併，含 `train_years` 欄 |
| `models_performance_all_train_years.csv` | 所有模型、所有 train_years 的 rmse/sharpe |
| `rolling_models_by_year_train{N}y.xlsx` | 各年 leaderboard、model_performance、feature_importance |

---

## 四、訓練年數與預測年對應

| train_years | 範例 |
|-------------|------|
| 2 | 2012-2013 訓練 → 2014 |
| 3 | 2012-2013-2014 訓練 → 2015 |
| 4 | 2012-2013-2014-2015 訓練 → 2016 |
| 5 | 2012-2013-2014-2015-2016 訓練 → 2017 |

---

## 五、後續分析建議

1. **比較 train_years**：讀 `summary_all_train_years.csv`，依 `train_years` 分組比較 rmse/sharpe。
2. **單模型分析**：讀 `models_performance_all_train_years.csv`，篩選特定 model。
3. **特徵重要性**：讀 `feature_importance_all_models.csv`，依 model、predict_year 分析。
4. **回測**：讀 `predictions_all_models.csv`，依不同模型預測做策略回測。
