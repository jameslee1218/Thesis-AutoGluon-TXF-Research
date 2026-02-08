# 0915 與 0930 截點準備檢查清單

要跑 0915、0930 與 0900 並行，**建議從「資料與 target 準備」開始**，再依序檢查 02 → 03 → 04/05。

---

## 流程總覽（依序）

```
[1] 資料／target 準備  →  [2] 02 截點切分＋壓縮  →  [3] 03 合併＋訓練  →  [4] 04/05 視覺化與回測
```

---

## 1. 從「資料清理／target 準備」開始（必做）

### 1.1 target 報酬率欄位（Y）

- **現狀**：`merge_and_train.py` 從 `config.get_y_file()` 讀 `target/y.xlsx`（或 y.csv），並用 **第一個匹配** 的欄位：`EXTERNAL_Y_RETURN_COLS = ("afternoon_return_0900", "target_return", "y", "報酬率", "return")`，目前實務上會用到 `afternoon_return_0900`。
- **0915/0930 需求**：同一份 target 檔需具備：
  - `day_id`（或 `date`、`日期`）— 已有
  - `afternoon_return_0915`：收盤－9:15 報酬率（與 0900 同口徑，建議 log 再於 03 轉 simple）
  - `afternoon_return_0930`：收盤－9:30 報酬率
- **你要做的**：
  1. 若尚未有 0915/0930 報酬率：從原始價量（例如 1 分鐘 K 線）計算「收盤價－該截點價格」的報酬率，寫入 `data/target/y.xlsx`（或 y.csv）的新欄位 `afternoon_return_0915`、`afternoon_return_0930`。
  2. 確認格式與 0900 一致（例如皆為 log return，由 03 統一轉 simple）。

### 1.2 上游資料（與截點無關）

- **indicators_extracted/**：一份即可，`split_by_cutoff.py` 會依時間切出 0900/0915/0930。確認目錄存在且 CSV 完整。
- **raw / indicators_complete**：若 01 或手動已產出 `indicators_extracted`，不需為 0915/0930 重做「資料清理」；僅需補 target 欄位即可。

**結論：是的，要先從「資料／target 準備」開始** — 至少要有 `afternoon_return_0915`、`afternoon_return_0930` 再跑 02/03。

---

## 2. 02_feature_compression

| 項目 | 現狀 | 0915/0930 要做的 |
|------|------|-------------------|
| **split_by_cutoff.py** | 已支援 0900/0915/0930，寫入 `dataset/0900`, `0915`, `0930` | 執行一次即可，無需改程式。 |
| **autoencoder.py** | 路徑寫死：`dataset/0900`、`output_0900` | 需改為可指定截點（例如指令列 `--cutoff 0915`），或對 0915、0930 各跑一次並寫入 `output_0915`、`output_0930`。 |

- 確認 `data/dataset/0915`、`data/dataset/0930` 在執行 `split_by_cutoff.py` 後存在且有資料。
- 修改或執行 `autoencoder.py`，使 0915、0930 分別產出 `data/output_0915/`、`data/output_0930/`（含 W*/compressed_data 等）。

---

## 3. 03_modeling（merge_and_train + notebook）

| 項目 | 現狀 | 0915/0930 要做的 |
|------|------|-------------------|
| **merge_and_train.py** | `CUTOFF = "0900"`，`OUTPUT_BASE = output_0900`，Y 欄位取 `EXTERNAL_Y_RETURN_COLS` 第一個匹配 | 改為依 CUTOFF 選擇：路徑用 `get_output_cutoff_dir(cutoff)`、合併表用 `merged_for_autogluon_{cutoff}`；Y 欄位改為依 CUTOFF 選用 `afternoon_return_{cutoff}`（或將 0915/0930 加入 EXTERNAL_Y_RETURN_COLS 並依 CUTOFF 選欄位）。可支援指令列參數 `--cutoff 0915` / `0930` 或跑三次。 |
| **train_autogluon_colab.ipynb** | 讀取 `merged_for_autogluon_0900.csv`，輸出到 `data/models/` | 路徑改為可選截點（例如 0915/0930 的 merged 表與對應 output 目錄），或複製一份 notebook 改路徑。 |

- 確認 `config.py` 已有 `get_output_cutoff_dir(cutoff)`、`get_merged_for_autogluon_dir(cutoff)`（已有），merge 與訓練都改為使用該 cutoff。

---

## 4. 04_visualization、05_backtest

- 目前多數腳本讀取 `output_0900`。若要對 0915/0930 做視覺化與回測，再將讀取路徑改為 `output_0915`、`output_0930`（或改為指令列/參數指定 cutoff）。

---

## 建議執行順序（摘要）

1. **資料／target**：補齊 `afternoon_return_0915`、`afternoon_return_0930`，確認 `indicators_extracted/` 存在。
2. **02**：執行 `split_by_cutoff.py` → 檢查 `dataset/0915`、`dataset/0930`；修改/執行 `autoencoder.py` 產出 `output_0915`、`output_0930`。
3. **03**：修改 `merge_and_train.py` 支援 CUTOFF 與對應 Y 欄位；產出 `merged_for_autogluon_0915`、`merged_for_autogluon_0930`；調整訓練 notebook 路徑以支援 0915/0930。
4. **04/05**：依需要改讀取 0915/0930 的 output 與合併表。

**結論：要從「資料清理／target 準備」開始**，再依序檢查並修改 02 → 03 → 04/05 的檔案與程式碼。
