# 01_data_ingestion

- **讀取（data/）**：`raw/TX2011~20231222-1K/`（1 分鐘 K 線 CSV）
- **寫入（data/）**：`indicators_complete/`、`indicators_extracted/`（五大面向）

## 手動執行

依序執行（路徑由 config 提供，僅讀寫 data/，輸出均為 xlsx）：

1. `generate_all_indicators.py` — 從 raw 產出五大面向技術指標  
   （`sp500_prev_return`, `open_gap_pct`, `vix_prev_close`, `vwap_bias_5`, `volume_roc_5`, `atr_5`, `kd_k_9`, `kd_d_9`, `linearreg_angle_5`）
2. `extract_indicators_optimized.py` — 從 indicators_complete 再提取精簡欄位到 indicators_extracted

```bash
cd scripts/01_data_ingestion
python generate_all_indicators.py
python extract_indicators_optimized.py
```

或由 main 觸發：`python main.py --step 1`
