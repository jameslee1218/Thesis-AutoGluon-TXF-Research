# 05_backtest

- **讀取（data/）**：`autogluon/{0900,0915,0930}/`、模型預測結果（建議使用 matrix_runs）
- **寫入（data/）**：`backtest/`（權益曲線、特徵重要性、回測報告）

## 手動執行

請將回測腳本置於本目錄並命名為 `backtest.py`，改為使用專案 `config`（僅讀寫 data/）後執行：

```bash
cd scripts/05_backtest
python backtest.py
```

或由 main 觸發：`python main.py --step 5`

若尚未實作，執行 `run.py` 會跳過並回傳 0。

## v2.0 建議前置條件

1. 先執行 `python main.py --step 3` 產生 `autogluon/{cutoff}` 輸入資料。
2. 完成 matrix notebooks，確保 `data/models/matrix_runs/` 有預測與評估輸出。
3. 回測腳本統一讀取新主線資料，不再硬綁單一 `0900` 路徑。
