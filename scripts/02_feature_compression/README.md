# 02_feature_compression

- **讀取（data/）**：`indicators_extracted/`
- **寫入（data/）**：`dataset/0900`, `0915`, `0930/`；`output_0900/W*/`、`all_windows_results_*.json`

## 手動執行

依序執行（請在專案根目錄或本目錄，確保可 import config）：

```bash
cd scripts/02_feature_compression
python split_by_cutoff.py    # 截點切分
python autoencoder.py        # 滾動視窗壓縮
```

或由 main 觸發：`python main.py --step 2`
