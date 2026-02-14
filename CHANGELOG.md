# 更新日誌

格式依 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/)，版本號採 [語意化版本](https://semver.org/lang/zh-TW/)。

---

## [Unreleased]

### 新增

- **03_modeling**：新增 `train_autogluon_colab.ipynb`，供 Colab 或本機執行 AutoGluon 訓練；輸入為 `merged_for_autogluon_0900.csv`，可掛載 Google Drive、設定路徑與時間切分後直接訓練並存模型至 `data/output_0900/models/`。
- **03_modeling**：新增 `build_uncompressed_autogluon.py`，產出未經 autoencoder 壓縮的訓練資料至 `data/autogluon_ready_uncompress/`，並與壓縮版進行敘述統計與 A/B 比較（輸出至 `data/analysis/compressed_vs_uncompressed/`）。
- **04_visualization**：`txf_ai_analysis.py` 改為產出結構化數據（單一 xlsx，24 個 Scenario 各一分頁），含敘述統計與統計檢定；Scenario 編號改為連續 01–24。

### 變更

- **train_autogluon_colab**（2025-02 重構）：
  - **依序三組**：依序執行 0900 → 0915 → 0930 三組截點訓練。
  - **滾動視窗**：改為「三年訓練、預測第四年」（`TRAIN_YEARS=3`），取代原本兩年訓練預測第三年。
  - **斷線續跑**：若 `data/models/{cutoff}/roll_{predict_year}/predictions.csv` 已存在則跳過該次訓練，便於 Colab 斷線後續跑。
  - **輸出**：各 cutoff 產出 `rolling_summary_final.csv`、`rolling_models_by_year.xlsx`；全部彙總於 `data/models/rolling_summary_all_cutoffs.csv`。
  - **並行版**：新增 `train_autogluon_colab_0900.ipynb`、`train_autogluon_colab_0915.ipynb`、`train_autogluon_colab_0930.ipynb`，各處理單一截點，可同時開三個 Colab 並行跑，互不干擾。
  - **訓練年比較**：迴圈 `TRAIN_YEARS_LIST = [2, 3, 4, 5]`，比較不同訓練年數效果；輸出至 `data/models/{cutoff}/train{N}y/roll_YYYY/`。
  - **完整輸出**：`predictions_all_models.csv`、`models_performance_all_train_years.csv`、`data/models/README.md` 說明輸出架構。
  - **特徵重要性優化**：僅計算 RMSE 前 3 與 Sharpe 前 3 模型的特徵重要性，去重後節省時間。
- **03_modeling**：`merge_and_train.py` 僅負責合併（merge）與敘述統計，**不再內含 AutoGluon 訓練**；訓練改由同目錄 `train_autogluon_colab.ipynb` 執行。
- **合併表路徑**：merged 目錄與檔名加入截點後綴，目前為 `merged_for_autogluon_0900/`、`merged_for_autogluon_0900.csv`（由 `config.py` 之 `get_merged_for_autogluon_dir(cutoff="0900")` 決定）。
- **資料目錄**：舊版合併表已移至 `data/legacy/merged_for_autogluon/`，並於 `data/legacy/README.md` 說明。
- **config.py**：新增 `get_autogluon_ready_uncompress_dir(cutoff)`、`get_models_dir()`。

### 說明

- 01～02 產出之壓縮結果與 03 之 merge 皆就緒後，於 Colab 或本機開啟 `scripts/03_modeling/train_autogluon_colab.ipynb`，設定 `DATA_ROOT` 或 `MERGED_CSV_PATH` 即可訓練；模型可存於 Drive 或本機 `data/output_0900/models/autogluon_merged/`。

---

## 過往版本

（此前變更未以 CHANGELOG 記錄；上述為近期重構與訓練流程分離之重點。）
