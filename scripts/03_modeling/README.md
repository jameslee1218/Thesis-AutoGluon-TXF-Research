# 03_modeling

- **讀取（data/）**：`indicators_complete/TX*_qlib_indicators_complete.xlsx`
- **寫入（data/）**：
  - `autogluon/all/merged_for_autogluon_all.xlsx`（不含 `target_return`）
  - `autogluon/0900|0915|0930/merged_for_autogluon_*.xlsx`（含 cutoff `target_return`）
- **目標值**：`target_return = (close_EOD-close_cutoff)/close_cutoff`

## 手動執行

```bash
cd scripts/03_modeling
python merge_for_autogluon.py
```

或由 main 觸發：`python main.py --step 3`

## Legacy 腳本

- `merge_and_train.py`
- `merge_output2_for_autogluon.py`
- `build_uncompressed_autogluon.py`

以上保留作歷史流程參考，不屬於目前主線。
