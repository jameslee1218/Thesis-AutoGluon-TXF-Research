# 02_feature_compression

> Legacy 模組：目前主流程已停用 Autoencoder，不再進入此步驟。

- 舊版讀取（data/）：`indicators_extracted/`
- 舊版寫入（data/）：`dataset/0900`, `0915`, `0930/`；`output_0900/W*/`、`all_windows_results_*.json`

## 手動執行

若要重現歷史 AE 實驗，可依序執行（非主流程）：

```bash
cd scripts/02_feature_compression
python split_by_cutoff.py    # 截點切分
python autoencoder.py        # 滾動視窗壓縮
```

目前 `main.py --step 2` 會顯示 Legacy 並跳過。
