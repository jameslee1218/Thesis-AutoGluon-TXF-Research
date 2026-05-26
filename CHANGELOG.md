# 更新日誌

格式依 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/)，版本號採 [語意化版本](https://semver.org/lang/zh-TW/)。

---

## [Unreleased]

### 新增

- **03_modeling**：新增 `merge_for_autogluon.py`，直接由 `indicators_complete` 產生 `data/autogluon/all|0900|0915|0930` 寬表。
- **03_modeling**：新增 matrix 訓練核心 `colab_train_matrix_core.py` 與 `train_matrix_*.ipynb`、`COLAB_MATRIX_RUNBOOK.md`。
- **README**：重寫為 v2.0 版，主流程、matrix 設定、核對清單與已知問題集中於單一入口。
- **docs**：更新 `docs/checklist_0915_0930.md` 為 v2.0 執行順序（`01 -> 03 -> matrix -> 04/05`）。

### 變更

- **main.py**：Step 2（`02_feature_compression`）明確標記為 Legacy 並跳過，不再作為主流程依賴。
- **scripts 文件**：`scripts/README_scripts.md`、`scripts/04_visualization/README.md`、`scripts/05_backtest/README.md` 改為以 v2.0 路徑與 matrix 輸出為主。
- **流程邊界**：將 `merge_and_train.py`、`merge_output2_for_autogluon.py`、`build_uncompressed_autogluon.py` 標記為歷史參考，不再是主線建議。

### 說明

- v2.0 主流程建議：`01_data_ingestion -> 03_modeling -> matrix notebooks -> 04_visualization -> 05_backtest`。
- 舊版 Autoencoder 流程仍可保留重現，但不建議與 v2.0 輸出混用。

---

## 過往版本

（此前變更未以 CHANGELOG 記錄；上述為近期重構與訓練流程分離之重點。）
