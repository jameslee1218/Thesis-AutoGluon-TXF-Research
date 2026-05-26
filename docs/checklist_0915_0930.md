# 0915 / 0930 / 0900（v2.0）執行檢查清單

本清單已對齊 v2.0 主流程：`01 -> 03 -> matrix -> 04/05`。  
`02_feature_compression` 僅保留 Legacy，不是必經步驟。

---

## 流程總覽（v2.0）

```
[1] 資料準備與指標產生 -> [2] 03 合併寬表 -> [3] matrix 訓練 -> [4] 視覺化與回測
```

---

## 1) 資料與前置檢查（必做）

- `data/raw/TX2011~20231222-1K/` 存在且含 `TX*_1K.csv`
- `scripts/01_data_ingestion` 可產生：
  - `data/indicators_complete/TX*_qlib_indicators_complete.xlsx`
  - `data/indicators_extracted/TX*_qlib_indicators_extracted.xlsx`
- 研究口徑確認：
  - cutoff：`0900`、`0915`、`0930`
  - `target_return = (close_EOD - close_cutoff) / close_cutoff`

---

## 2) Step 3 合併寬表（主線）

執行：

```bash
cd scripts/03_modeling
python merge_for_autogluon.py
```

應產出：

- `data/autogluon/all/merged_for_autogluon_all.xlsx`
- `data/autogluon/0900/merged_for_autogluon_0900.xlsx`
- `data/autogluon/0915/merged_for_autogluon_0915.xlsx`
- `data/autogluon/0930/merged_for_autogluon_0930.xlsx`

核對：

- 三個 cutoff 檔案都有 `target_return`
- `all` 檔不含 `target_return`
- `date` 可正確解析為交易日

---

## 3) matrix 訓練（主線）

依序執行 notebook：

1. `train_matrix_0900.ipynb`
2. `train_matrix_0915.ipynb`
3. `train_matrix_0930.ipynb`
4. （選擇）`train_matrix_premarket3.ipynb`

對應說明：`scripts/03_modeling/COLAB_MATRIX_RUNBOOK.md`

核對輸出：

- `data/models/matrix_runs/{cutoff}/{feature_set}/{exp_id}/train{N}y/roll_{year}/`
- `data/models/matrix_runs/reports/{feature_set}/run_summary.xlsx`
- `comparison_by_metric.xlsx`
- `comparison_tabular_vs_timeseries.xlsx`

---

## 4) Step 4/5 視覺化與回測

- Step 4：`python main.py --step 4`
- Step 5：`python main.py --step 5`

核對重點：

- 分析腳本應讀取 `autogluon/*` 與 `models/matrix_runs/*`，不再硬綁舊版 `output_0900`
- 回測腳本需支援多 cutoff 路徑（至少 0900/0915/0930）

---

## 5) Legacy 邊界（避免混用）

以下不屬於 v2.0 主流程：

- `scripts/02_feature_compression/*`
- `scripts/03_modeling/merge_and_train.py`
- `scripts/03_modeling/merge_output2_for_autogluon.py`
- `scripts/03_modeling/build_uncompressed_autogluon.py`

---

## 建議最終核對順序（摘要）

1. Step 1 輸出齊全（indicators_complete/extracted）
2. Step 3 四份寬表齊全（all/0900/0915/0930）
3. matrix 四本 notebook 都可跑通（至少 full 三截點）
4. reports 指標檔齊全
5. Step 4/5 輸入路徑已對齊 v2.0
