# data/ 目錄說明

本目錄為**所有輸入與產出的唯一存放位置**，整個 `data/` 已於 `.gitignore` 中忽略（僅本 README 被版控追蹤）。請依下列結構自建目錄或由腳本自動建立。

---

## 一、目錄樹與邏輯對應

```
data/
├── README.md                    ← 本說明（唯一追蹤檔）
│
├── raw/                         【輸入】原始 K 線
│   └── TX2011~20231222-1K/     ← 約 3171 個 CSV，檔名 TX*_1K.csv
│
├── indicators_complete/         【輸入/中繼】完整技術指標（01 可產出）
│   └── *.csv                    ← TX*_1K_qlib_indicators_complete.csv，約 2724 檔
│
├── indicators_extracted/       【01 產出 → 02 輸入】提取後 7 群組
│   └── *.csv                    ← 僅 STOCH, STOCHF, STOCHRSI, MACD, BBANDS, ADX_DMI, AROON
│
├── target/                      【輸入】目標變數
│   ├── y.xlsx  或  y.csv        ← 欄位：day_id, afternoon_return_0900（收盤－9:00 報酬率）
│
├── dataset/                     【02 產出】截點前分鐘資料
│   ├── 0900/                    ← 09:01 前，依日分檔 CSV
│   ├── 0915/                    ← 09:16 前
│   └── 0930/                    ← 09:31 前
│
├── output_0900/                 【02／03 產出】壓縮與合併
│   ├── W1_2011-2012_compress_2013-2013/
│   │   ├── compressed_data/    ← *_compressed.csv（7 群組）
│   │   ├── models/              ← .h5, .pkl
│   │   └── plots/               ← .png
│   ├── W2_... , ... , W11_...
│   ├── all_windows_results_*.json
│   └── merged_for_autogluon_0900/   ← 0900 截點加後綴（0915→merged_for_autogluon_0915）
│       ├── merged_for_autogluon_0900.csv
│       ├── summary_statistics.csv
│       ├── missing_values.csv
│       └── stats_after_cleaning/
│
├── visualizations/              【04 產出】圖檔
│   ├── 01_mse_error_timeline.png
│   ├── 02_compression_radar.png
│   └── 03_reconstruction_scatter.png
│
├── backtest/                    【05 產出】回測報告與圖
│   ├── equity_curve.png
│   ├── feature_importance.csv
│   └── backtest_report.csv
│
├── autogluon_ready/              【03 產出】AutoGluon 訓練用合併表（output2 壓縮版，取代舊 output_0900 W*）
│   ├── 0900/                     ← 截點 09:00 前：merged_for_autogluon_0900.csv, summary_statistics.csv, missing_values.csv
│   ├── 0915/                     ← 截點 09:15 前
│   └── 0930/                     ← 截點 09:30 前
│
├── output2/                     【歸檔】按「年度」產出的 autoencoder 壓縮實驗（與 output_0900 視窗制並列）
│   ├── window_2011-2013/        ← 視窗 2011–2013：compressed_data/, models/, plots/
│   ├── year_2013/ … year_2023/  ← 各年度：results_*_*.json, autoencoder_results_*.xlsx, 以及 7 群組子目錄
│   │   └── {ADX_DMI, AROON, BBANDS, MACD, STOCH, STOCHF, STOCHRSI}/
│   │       ├── progress.json, result.json
│   │       ├── compressed_data/*_compressed.csv
│   │       ├── models/*_scaler.pkl, *_search_best.h5
│   │       └── plots/*_training_history.png
│   └── （來源：temp/output2，歸檔日期 2026-02-13）
│
└── legacy/                      【僅存檔】已棄用／可能錯誤 encoder 的舊資料，勿用於訓練
    ├── README.md                ← 說明
    └── merged_for_autogluon/    ← 舊合併表（新表由 03 產出至 output_0900/merged_for_autogluon_0900/）
```

---

## 二、依模組的輸入／產出對照

| 模組 | 讀取（輸入） | 寫入（產出） |
|------|--------------|--------------|
| **01_data_ingestion** | `raw/`, 或既有 `indicators_complete/` | `indicators_complete/`, `indicators_extracted/` |
| **02_feature_compression** | `indicators_extracted/` | `dataset/0900|0915|0930/`, `output_0900/W*/`, `all_windows_results_*.json` |
| **03_modeling** | `output_0900/` 壓縮結果、`target/y.xlsx` | `output_0900/merged_for_autogluon_0900/`（依截點後綴）, AutoGluon 模型 |
| **04_visualization** | `output_0900/`（JSON、W*） | `visualizations/` |
| **05_backtest** | `output_0900/merged_for_autogluon_0900/`, 預測結果 | `backtest/` |

---

## 三、欄位與格式摘要

- **raw/**：台灣期貨 1 分鐘 K 線，開高低收量等，時間約 2011-01-03～2023-12-22。
- **indicators_extracted/**：每檔 CSV 含 `datetime` 與 7 群組共 20 欄（如 STOCH_K_14, STOCH_D_14, ...）。
- **target/y.xlsx**：`day_id`（YYYYMMDD 整數）、`afternoon_return_0900`（log 報酬）；03 會自動轉 simple。
- **dataset/0900/**：與 `indicators_extracted` 同檔名，僅保留截點前分鐘列。
- **output_0900/W*/compressed_data/**：檔名如 `STOCH_W1_2011-2012_compress_2013-2013_compressed.csv`，內含 `datetime` 與壓縮欄（如 `*_compressed_0`）。
- **merged_for_autogluon_0900.csv**：日頻表（0900 截點），每列一日，欄含壓縮特徵彙總與 `target_return`；0915/0930 則為 `_0915`、`_0930` 後綴。
- **output2/**：依「年度」劃分的 autoencoder 壓縮實驗歸檔。每年度含多版 `results_*_YYYYMMDD_HHMMSS.json`（各技術指標 best_config、final_scores、history）、對應 `autoencoder_results_*.xlsx`，以及 7 個技術指標子目錄（STOCH, STOCHF, STOCHRSI, MACD, BBANDS, ADX_DMI, AROON），各含壓縮資料、模型與訓練曲線圖。

---

## 四、使用方式

1. **首次使用**：依上列樹狀建立 `raw/`、`target/` 等，並放入原始 K 線與 `y.xlsx`（或 `y.csv`）。
2. **執行順序**：01 → 02 → 03 → 04 → 05；各腳本依 `config.py` 讀寫上述路徑。
3. **路徑覆寫**：若 `data/` 不在 repo 下，可設環境變數 `DATA_ROOT` 指向實際目錄。
