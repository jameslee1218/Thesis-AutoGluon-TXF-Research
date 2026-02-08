# 05_backtest

- **讀取（data/）**：`output_0900/merged_for_autogluon_0900/`、預測結果
- **寫入（data/）**：`backtest/`（權益曲線、特徵重要性、回測報告）

## 手動執行

請將回測腳本置於本目錄並命名為 `backtest.py`，改為使用專案 `config`（僅讀寫 data/）後執行：

```bash
cd scripts/05_backtest
python backtest.py
```

或由 main 觸發：`python main.py --step 5`

若尚未實作，執行 `run.py` 會跳過並回傳 0。
