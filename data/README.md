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
│   └── merged_for_autogluon/
│       ├── merged_for_autogluon.csv
│       ├── summary_statistics.csv
│       ├── missing_values.csv
│       └── stats_after_cleaning/
│
├── visualizations/              【04 產出】圖檔
│   ├── 01_mse_error_timeline.png
│   ├── 02_compression_radar.png
│   └── 03_reconstruction_scatter.png
│
└── backtest/                    【05 產出】回測報告與圖
    ├── equity_curve.png
    ├── feature_importance.csv
    └── backtest_report.csv
```

---

## 二、依模組的輸入／產出對照

| 模組 | 讀取（輸入） | 寫入（產出） |
|------|--------------|--------------|
| **01_data_ingestion** | `raw/`, 或既有 `indicators_complete/` | `indicators_complete/`, `indicators_extracted/` |
| **02_feature_compression** | `indicators_extracted/` | `dataset/0900|0915|0930/`, `output_0900/W*/`, `all_windows_results_*.json` |
| **03_modeling** | `output_0900/` 壓縮結果、`target/y.xlsx` | `output_0900/merged_for_autogluon/`, AutoGluon 模型 |
| **04_visualization** | `output_0900/`（JSON、W*） | `visualizations/` |
| **05_backtest** | `output_0900/merged_for_autogluon/`, 預測結果 | `backtest/` |

---

## 三、欄位與格式摘要

- **raw/**：台灣期貨 1 分鐘 K 線，開高低收量等，時間約 2011-01-03～2023-12-22。
- **indicators_extracted/**：每檔 CSV 含 `datetime` 與 7 群組共 20 欄（如 STOCH_K_14, STOCH_D_14, ...）。
- **target/y.xlsx**：`day_id`（YYYYMMDD 整數）、`afternoon_return_0900`（log 報酬）；03 會自動轉 simple。
- **dataset/0900/**：與 `indicators_extracted` 同檔名，僅保留截點前分鐘列。
- **output_0900/W*/compressed_data/**：檔名如 `STOCH_W1_2011-2012_compress_2013-2013_compressed.csv`，內含 `datetime` 與壓縮欄（如 `*_compressed_0`）。
- **merged_for_autogluon.csv**：日頻表，每列一日，欄含壓縮特徵彙總與 `target_return`。

---

## 四、研究紀錄

以下為開發過程中遇到的問題、心得與改進方式，供日後重現與避免重蹈覆轍。

### 4.1 Autoencoder／壓縮特徵「偷看未來」（資料洩漏）

- **問題**：若用「全時段」資料一次訓練 autoencoder，再對所有年份做壓縮，等於用**未來資訊**來壓縮過去日期的特徵；合併後交給 AutoGluon 預測時，模型會間接用到未來資訊，結果不公正（尤其 MACD 等技術指標壓縮值會帶有洩漏）。
- **改進**：
  - **02_feature_compression** 改為**滾動視窗**：每個窗口「前 2 年訓練、第 3 年壓縮」（如 W1：2011–2012 訓練 → 2013 壓縮；W2：2013–2014 訓練 → 2015 壓縮），訓練時絕不使用該壓縮年之後的資料。
  - **03_modeling** 合併時**依年對齊**：每日只取「該日所屬年份」對應窗口的壓縮特徵（例如 2013 年某日只用 W1 的壓縮，2014 年只用 W2），不混用窗口，避免任何未來資訊進入特徵。

### 4.2 目標變數與訓練用表

- **Y**：使用「收盤－截點」報酬率（如 afternoon_return_0900）；若來源為 log 報酬，03 會轉成 simple return 再寫入 `target_return`。
- **訓練時**：合併表產出後，訓練 AutoGluon 前會 **drop 日期欄**（以及僅用截點前分鐘彙總），避免時間本身被當成特徵造成洩漏。

### 4.3 特徵清理與壓縮專一

- 合併階段會做常數欄、低變異、二元型態等清理，並可設定**只保留壓縮特徵、剔除原始技術指標欄**（`DROP_ORIGINAL_INDICATORS`），讓模型只吃壓縮後的彙總，不重複使用原始指標。

### 4.4 路徑與重現性

- 所有輸入與產出**僅在 data/**，由 `config.py`（及可選的 `DATA_ROOT`）統一管理，方便本機／Colab 或不同機器重現；整個 `data/` 不進版控，僅本 README 被追蹤。

---

### 4.5 為什麼刪除部分技術指標（精簡特徵）

- **CDL 蠟燭圖形態（61 個）**  
  以 `CDL` 開頭的 TA-Lib 圖形識別指標（如 CDL2CROWS、CDLENGULFING 等）為**二元或三元離散**（常為 0 / 100 / -100），在機器學習中效果不佳且欄位過多；刪除後欄位由約 369 降至 308。見根目錄 `要刪除的圖形類指標列表.md`。

- **常數／高缺失／索引類**  
  - 常數或幾乎常數（如 DEMA_200、TEMA_200 在單日分鐘資料下 100% 缺失）對模型無用。  
  - 長週期（如 200 日）在「單日僅數百筆分鐘」的資料下缺漏嚴重，故刪除高缺失欄位。  
  - 索引類（MAXINDEX_*、MININDEX_*）表示極值**位置**而非價格／趨勢，預測幫助有限。  
  見 `建議刪除的指標分析.md`。

- **週期過長欄位**  
  有「週期大於 5」的欄位列表與刪除腳本：單日僅約 301 筆分鐘，過長週期指標易缺值或無意義，故只保留要壓縮的 7 群組（且多為 14／20 週期）。

- **策略取捨**  
  採「保守刪除」：先刪明顯無用（CDL、常數、高缺失、索引類），其餘讓模型或後續特徵選擇決定；避免一次刪太多導致資訊損失。

---

### 4.6 壓縮結果未寫回與合併閉環

- **問題**：早期單一視窗版（如 working/1105）只產出 autoencoder 模型與 plots，**compressed_data 未系統性寫回**；整體／資料進度總結均註記「步驟 3：合併壓縮特徵回完整資料 ⏳ 尚未完成」，僅有零散測試檔（如 1119/STOCH_compressed.csv）。
- **解法**：改為**滾動視窗**（02 每窗產出 W*/compressed_data/），並在 03（merge）中實作「依 compress_year 載入對應 W*、按日／年對齊後併入日表」，形成完整閉環；不再依賴 1105 的單一 compressed_data 目錄。

---

### 4.7 時間序列切分與評估（會議／文獻要點）

- **交叉驗證**：一般 K 折假設樣本 iid，時間序列會違反；應採用**時間序列切分**或 **rolling / expanding window** 的驗證方式（如 sklearn 3.1.2.6 Time Series Split、block 切分），避免未來資訊進入訓練。
- **比例**：訓練／驗證／測試可採 80/10/10 或固定筆數（如各 10000）；小資料常用 60/20/20，資料量足夠時可加大訓練比例。
- **評估指標**：除 accuracy 外，建議用 **Sharpe ratio**、**F1** 或下方風險等更貼近實務的指標作為優化目標；AutoGluon 可自訂 eval_metric。
- **模型比較**：不同模型需做**統計檢定**或明確比較方式，並在論文中說明為何選用某些模型（文獻支持、實證結果）。

---

### 4.8 特徵重要性與共線性

- **問題**：特徵重要性若採「移除單一特徵看表現」時，若存在**高度相關**的特徵，移除其一可由另一補上，導致重要性被低估或失真。
- **做法**：先做相關性／Spearman 分析，必要時用**主成分或因素分析**萃取、或依相關群組保留代表特徵，再跑特徵重要性；IC test／Spearman 與機器學習特徵重要性可並存，不必一致（線性 vs 非線性），但需在論文中說明取捨理由。

---

### 4.9 明確排除的洩漏欄位

- 在日級或分鐘級特徵中，下列欄位**不得**進入模型（會洩漏未來）：  
  **P_close**（收盤價）、**P_0900 / P_0915 / P_0930**（該時點價格）、**close_return**（全日報酬）。  
  僅能使用「截點前」的價格與報酬、以及「截點至收盤」的目標 Y。

---

### 4.10 研究心得（早盤與午盤、條件式有效）

- **早盤與午盤相關性**：曾因樣本自相關導致初版結果不可信；重做後發現早盤與午盤（09:00 後至收盤）的**線性相關並不高**，需結合多特徵與非線性模型。
- **顯著因子**：IC test／Spearman 顯示**昨日美股報酬、開盤缺口**對「截點至收盤」報酬較顯著；在**前一交易日台股大跌**等特定 regime 下，昨日美股與午盤呈更強負相關（情緒過度反應後反彈）。
- **技術分析條件式有效**：技術訊號往往僅在**特定市場狀態**下有效，關係具非線性與交互；傳統統計難以全面驗證，故以 AutoML（AutoGluon）系統性檢驗特徵資訊含量與策略可行性。
- **特徵重要性實證**：美股前日、缺口最重要；早盤波動（std、range）次之；單純成交量／VWAP／早盤漲跌幅在部分設定下幾乎無效，開盤價多已 price in。

---

### 4.11 資料散落、格式與 y 對齊

- **散落與路徑**：歷史資料曾分散於根目錄、`之前使用過的內容/`、working 多個日期資料夾；且存在 **Windows 絕對路徑**（如 F:\thesis\...），搬遷或跨機器需改為 config 或環境變數。
- **格式**：部分為 xlsx、部分 csv；欄位名稱大小寫不一致（如 open vs Open）。現行流程統一為 CSV 與固定檔名 pattern（如 TX*_1K_qlib_indicators_complete.csv）。
- **y 與日表合併**：目標變數檔（y.xlsx）的 **day_id**（YYYYMMDD 整數）與日表的 **date**（datetime.date）型別不一致會導致 merge 失敗。解法：在讀取 y 時將 day_id 轉成標準日期，與日表用同一種日期鍵（如 normalize 後）再合併；並將 log 報酬轉成 simple（與專案其餘報酬一致）。

---

### 4.12 Keras 模型載入與 Colab 路徑

- **載入已存 Keras 模型**時若出現「無法解析 mse / 找不到 metric」：在 `load_model` 時傳入 **custom_objects**（如 `mse`、`mean_squared_error`），若仍失敗可改為 **compile=False** 載入。
- **Colab**：部分流程曾在 Google Colab 執行，路徑為 Google Drive；若在本機重跑需改為本機 data/ 或 config 路徑。Colab 版 notebook 曾發生 ENCODER_DIMS 等參數未定義，需與本地 02_autoencoder 保持一致。
