# 03_modeling

- **讀取（data/）**：`output_0900/`（壓縮結果）、`target/y.xlsx` 或 `y.csv`
- **寫入（data/）**：`output_0900/merged_for_autogluon/`、可選 AutoGluon 模型

## 手動執行

```bash
cd scripts/03_modeling
python merge_and_train.py
```

或由 main 觸發：`python main.py --step 3`
