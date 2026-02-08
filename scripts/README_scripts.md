# scripts 總覽：手動執行與 data/ 對照

所有腳本**僅讀寫 data/**，路徑由專案根目錄 `config.py` 提供。可手動執行單一腳本，不一定要透過 `main.py`。

---

## 執行前

- 在**專案根目錄**設定好 `config`（或已設 `DATA_ROOT` 環境變數）。
- 手動執行時請在**專案根目錄**下跑，或先 `import sys; sys.path.insert(0, "專案根路徑")` 再執行，以便載入 `config`。

---

## 01_data_ingestion

| 腳本 | 手動執行 | 讀取（data/） | 寫入（data/） |
|------|----------|----------------|----------------|
| generate_all_indicators.py | `cd scripts/01_data_ingestion && python generate_all_indicators.py` | raw/ | indicators_complete/ |
| extract_indicators_optimized.py | `python extract_indicators_optimized.py` | indicators_complete/ | indicators_extracted/ |

（兩檔已置於本目錄，使用 config 僅讀寫 data/；目錄內有 `README.md`。）

---

## 02_feature_compression

（目錄內有 `README.md`。）

| 腳本 | 手動執行 | 讀取（data/） | 寫入（data/） |
|------|----------|----------------|----------------|
| split_by_cutoff.py | `cd scripts/02_feature_compression && python split_by_cutoff.py` | indicators_extracted/ | dataset/0900, 0915, 0930/ |
| autoencoder.py | `python autoencoder.py` | dataset/0900/ | output_0900/W*/, all_windows_results_*.json |

---

## 03_modeling

（目錄內有 `README.md`。）

| 腳本 | 手動執行 | 讀取（data/） | 寫入（data/） |
|------|----------|----------------|----------------|
| merge_and_train.py | `cd scripts/03_modeling && python merge_and_train.py` | output_0900/, target/y.xlsx | output_0900/merged_for_autogluon_0900/ |

---

## 04_visualization

（目錄內有 `README.md`。）

| 腳本 | 手動執行 | 讀取（data/） | 寫入（data/） |
|------|----------|----------------|----------------|
| visualize_results.py | `cd scripts/04_visualization && python visualize_results.py` | output_0900/（JSON、W*） | visualizations/ |

---

## 05_backtest

（目錄內有 `README.md`；`backtest.py` 為占位腳本，會建立 data/backtest 並結束，可替換為實際回測邏輯。）

| 腳本 | 手動執行 | 讀取（data/） | 寫入（data/） |
|------|----------|----------------|----------------|
| backtest.py | `cd scripts/05_backtest && python backtest.py` | output_0900/merged_for_autogluon_0900/, 預測結果 | backtest/ |

---

## 依賴順序

1. 01 → 02（需 indicators_extracted）
2. 02 → 03（需 output_0900 壓縮結果）
3. 02 → 04（需 output_0900）
4. 03 → 05（需 merged_for_autogluon_0900 與預測）
