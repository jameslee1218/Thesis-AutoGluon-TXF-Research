# 03_modeling

- **讀取（data/）**：`output_0900/`（壓縮結果）、`target/y.xlsx` 或 `y.csv`
- **寫入（data/）**：`output_0900/merged_for_autogluon_0900/`（0900 截點加後綴）
- **訓練**：AutoGluon 訓練請用同目錄 **`train_autogluon_colab.ipynb`**（可於 Colab 或本機執行，讀取上述合併表後 fit 並存模型至 `output_0900/models/`）

## 手動執行

```bash
cd scripts/03_modeling
python merge_and_train.py
```

或由 main 觸發：`python main.py --step 3`
