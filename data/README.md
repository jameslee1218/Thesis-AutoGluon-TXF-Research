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

---

## 四、使用方式

1. **首次使用**：依上列樹狀建立 `raw/`、`target/` 等，並放入原始 K 線與 `y.xlsx`（或 `y.csv`）。
2. **執行順序**：01 → 02 → 03 → 04 → 05；各腳本依 `config.py` 讀寫上述路徑。
3. **路徑覆寫**：若 `data/` 不在 repo 下，可設環境變數 `DATA_ROOT` 指向實際目錄。
