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

## 未壓縮版資料（build_uncompressed_autogluon.py）

產出**未經 autoencoder 壓縮**的 AutoGluon 訓練資料，並與壓縮版進行敘述統計與 A/B 比較。

```bash
python build_uncompressed_autogluon.py
```

- **輸出**：`data/autogluon_ready_uncompress/{0900,0915,0930}/merged_for_autogluon_*.csv`
- **分析**：`data/analysis/compressed_vs_uncompressed/compressed_vs_uncompressed_comparison.xlsx`
- **訓練**：將 Colab notebook 的 `MERGED_CSV_PATH` 指向 `autogluon_ready_uncompress/0900/merged_for_autogluon_0900.csv` 即可用未壓縮資料重新訓練
