# 更新日誌

格式依 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/)，版本號採 [語意化版本](https://semver.org/lang/zh-TW/)。

---

## [Unreleased]

### 新增

- **03_modeling**：新增 `train_autogluon_colab.ipynb`，供 Colab 或本機執行 AutoGluon 訓練；輸入為 `merged_for_autogluon_0900.csv`，可掛載 Google Drive、設定路徑與時間切分後直接訓練並存模型至 `data/output_0900/models/`。

### 變更

- **train_autogluon_colab.ipynb**（2025-02 重構）：
  - **依序三組**：依序執行 0900 → 0915 → 0930 三組截點訓練。
  - **滾動視窗**：改為「三年訓練、預測第四年」（`TRAIN_YEARS=3`），取代原本兩年訓練預測第三年。
  - **斷線續跑**：若 `data/models/{cutoff}/roll_{predict_year}/predictions.csv` 已存在則跳過該次訓練，便於 Colab 斷線後續跑。
  - **輸出**：各 cutoff 產出 `rolling_summary_final.csv`、`rolling_models_by_year.xlsx`；全部彙總於 `data/models/rolling_summary_all_cutoffs.csv`。
- **03_modeling**：`merge_and_train.py` 僅負責合併（merge）與敘述統計，**不再內含 AutoGluon 訓練**；訓練改由同目錄 `train_autogluon_colab.ipynb` 執行。
- **合併表路徑**：merged 目錄與檔名加入截點後綴，目前為 `merged_for_autogluon_0900/`、`merged_for_autogluon_0900.csv`（由 `config.py` 之 `get_merged_for_autogluon_dir(cutoff="0900")` 決定）。
- **資料目錄**：舊版合併表已移至 `data/legacy/merged_for_autogluon/`，並於 `data/legacy/README.md` 說明。

### 說明

- 01～02 產出之壓縮結果與 03 之 merge 皆就緒後，於 Colab 或本機開啟 `scripts/03_modeling/train_autogluon_colab.ipynb`，設定 `DATA_ROOT` 或 `MERGED_CSV_PATH` 即可訓練；模型可存於 Drive 或本機 `data/output_0900/models/autogluon_merged/`。

---

## 過往版本

（此前變更未以 CHANGELOG 記錄；上述為近期重構與訓練流程分離之重點。）
