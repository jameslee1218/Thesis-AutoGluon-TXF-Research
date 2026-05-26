# data 目錄說明（v2.0）

本專案所有輸入、過程產出、模型輸出皆放在 `data/` 下。  
為避免「少放資料」或路徑錯置，請依本檔維持固定結構。

## 主要目錄

- `raw/`：原始 1 分鐘 K 線（例如 `TX2011~20231222-1K/TX*_1K.csv`）
- `indicators_complete/`：Step 1 產生的完整日指標檔
- `indicators_extracted/`：Step 1 提取後欄位（供比對/歷史流程）
- `autogluon/all/`：Step 3 全特徵寬表（不含 `target_return`）
- `autogluon/0900/`：Step 3 cutoff=0900 寬表（含 `target_return`）
- `autogluon/0915/`：Step 3 cutoff=0915 寬表（含 `target_return`）
- `autogluon/0930/`：Step 3 cutoff=0930 寬表（含 `target_return`）
- `models/matrix_runs/`：matrix 訓練單位輸出與報告
- `visualizations/`：Step 4 圖表與分析輸出
- `backtest/`：Step 5 回測輸出

## v2.0 最小必備資料

1. `raw/TX2011~20231222-1K/` 存在且有 `TX*_1K.csv`
2. 執行 Step 1 後有 `indicators_complete/*.xlsx`
3. 執行 Step 3 後有：
   - `autogluon/all/merged_for_autogluon_all.xlsx`
   - `autogluon/0900/merged_for_autogluon_0900.xlsx`
   - `autogluon/0915/merged_for_autogluon_0915.xlsx`
   - `autogluon/0930/merged_for_autogluon_0930.xlsx`

## 注意事項

- 不同流程的輸出不要混用（Legacy 與 v2.0 請分開）。
- 新機器重建環境時，請先核對本檔再執行 pipeline。
- 若要改資料根目錄，請用 `DATA_ROOT` 環境變數，不要硬改腳本內路徑。
