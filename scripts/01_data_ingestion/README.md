# 01_data_ingestion

- **讀取（data/）**：`raw/TX2011~20231222-1K/`（1 分鐘 K 線 CSV）→ `indicators_complete/`
- **寫入（data/）**：`indicators_complete/`、`indicators_extracted/`（7 群組）

## 手動執行

依序執行（路徑由 config 提供，僅讀寫 data/）：

1. `generate_all_indicators.py` — 從 raw 產出 indicators_complete（完整技術指標）
2. `extract_indicators_optimized.py` — 從 indicators_complete 產出 indicators_extracted

```bash
cd scripts/01_data_ingestion
python generate_all_indicators.py
python extract_indicators_optimized.py
```

或由 main 觸發：`python main.py --step 1`
