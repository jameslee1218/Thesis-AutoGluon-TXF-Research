# scripts 總覽（v2.0）

所有腳本僅讀寫 `data/`；路徑由 `config.py` 統一管理。v2.0 主流程不依賴 Autoencoder。

## 執行前

- 在專案根目錄執行（或確保 `PROJECT_ROOT`、`DATA_ROOT` 已正確設定）。
- 先安裝依賴：`pip install -r requirements.txt`。

## 01_data_ingestion（主流程）

| 腳本 | 手動執行 | 讀取（data/） | 寫入（data/） |
|------|----------|----------------|----------------|
| `generate_all_indicators.py` | `cd scripts/01_data_ingestion && python generate_all_indicators.py` | `raw/` | `indicators_complete/` |
| `extract_indicators_optimized.py` | `cd scripts/01_data_ingestion && python extract_indicators_optimized.py` | `indicators_complete/` | `indicators_extracted/` |

## 02_feature_compression（Legacy）

此模組保留歷史重現用途，不是 v2.0 主流程必經步驟。

| 腳本 | 手動執行 | 讀取（data/） | 寫入（data/） |
|------|----------|----------------|----------------|
| `split_by_cutoff.py` | `cd scripts/02_feature_compression && python split_by_cutoff.py` | `indicators_extracted/` | `dataset/0900|0915|0930/` |
| `autoencoder.py` | `cd scripts/02_feature_compression && python autoencoder.py` | `dataset/0900/` | `output_0900/W*/` |

## 03_modeling（主流程核心）

| 腳本 | 手動執行 | 讀取（data/） | 寫入（data/） |
|------|----------|----------------|----------------|
| `merge_for_autogluon.py` | `cd scripts/03_modeling && python merge_for_autogluon.py` | `indicators_complete/` + `raw/` | `autogluon/all|0900|0915|0930/` |
| `colab_train_matrix_core.py` | 由 `train_matrix_*.ipynb` 呼叫 | `autogluon/{cutoff}/` | `models/matrix_runs/` |

## 04_visualization

| 腳本 | 手動執行 | 讀取（data/） | 寫入（data/） |
|------|----------|----------------|----------------|
| `visualize_results.py` | `cd scripts/04_visualization && python visualize_results.py` | `models/matrix_runs/`、`autogluon/` | `visualizations/` |
| `txf_ai_analysis.py` | `cd scripts/04_visualization && python txf_ai_analysis.py` | `models/`、`autogluon/` | `visualizations/txf_ai_analysis/` |

## 05_backtest

| 腳本 | 手動執行 | 讀取（data/） | 寫入（data/） |
|------|----------|----------------|----------------|
| `backtest.py` | `cd scripts/05_backtest && python backtest.py` | `autogluon/{cutoff}/`、模型預測結果 | `backtest/` |

## v2.0 建議執行順序

1. `python main.py --step 1`
2. `python main.py --step 3`
3. 依序執行 `scripts/03_modeling/train_matrix_0900.ipynb`、`train_matrix_0915.ipynb`、`train_matrix_0930.ipynb`
4. 視需求執行 `train_matrix_premarket3.ipynb`
5. `python main.py --step 4`
6. `python main.py --step 5`

補充：`python main.py --step 2` 會顯示 Legacy 並跳過。
