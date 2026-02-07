# Thesis-AutoGluon-TXF-Research

本專案為**台灣期貨市場（TXF）量化研究流程**：以盤前／早盤技術指標經 Autoencoder 滾動視窗壓縮後，合併為日頻特徵表，再以 AutoGluon 建模預測「截點至收盤」報酬率，並支援視覺化與回測。

---

## 總體流程圖

本流程**依三組截點（09:00 / 09:15 / 09:30）**分別產出 X 與 Y：特徵為「截點前」分鐘資料與壓縮結果，目標變數 Y 為**報酬率**（收盤－該截點），並在建模前做報酬率加工（如 log → simple）。

```mermaid
flowchart TB
    subgraph INPUT["📥 輸入（data/）"]
        raw["raw/<br/>TX*_1K.csv<br/>原始 K 線"]
        target["target/<br/>y.xlsx / y.csv<br/>依截點之報酬率欄位"]
    end

    subgraph Y_PROCESS["Y 加工：報酬率"]
        y_def["目標變數 = 收盤－截點 報酬率<br/>e.g. afternoon_return_0900, 0915, 0930"]
        y_convert["log 報酬 → simple 報酬<br/>與特徵一致"]
    end

    subgraph M01["01_data_ingestion"]
        gen["generate_all_indicators"]
        ext["extract_indicators_optimized<br/>+ 篩選（圖形類、週期>5）"]
    end

    subgraph DATA01["data/ 中繼"]
        comp["indicators_complete/"]
        extr["indicators_extracted/<br/>7 群組（共用）"]
    end

    subgraph CUTOFFS["三截點：09:01 / 09:16 / 09:31"]
        t9["09:00 組"]
        t15["09:15 組"]
        t30["09:30 組"]
    end

    subgraph M02["02_feature_compression"]
        split["split_by_cutoff<br/>依截點切出「截點前」分鐘"]
        ae["autoencoder<br/>滾動視窗壓縮<br/>（每組各做）"]
    end

    subgraph DATA02["data/ 產出 — 三組並列"]
        ds9["dataset/0900/"]
        ds15["dataset/0915/"]
        ds30["dataset/0930/"]
        w9["output_0900/<br/>W*, compressed_data"]
        w15["output_0915/"]
        w30["output_0930/"]
    end

    subgraph M03["03_modeling"]
        merge["merge_and_train<br/>合併壓縮特徵 + Y 報酬率 → 日表"]
        ag["AutoGluon 訓練<br/>（每截點一組）"]
    end

    subgraph DATA03["data/ 產出 — 三組"]
        mfg9["merged_for_autogluon/<br/>0900"]
        mfg15["0915"]
        mfg30["0930"]
    end

    subgraph M04["04_visualization"]
        viz["visualize_results<br/>MSE / 雷達 / 重建散點<br/>（可依截點產出）"]
    end

    subgraph M05["05_backtest"]
        bt["backtest<br/>權益曲線、特徵重要性<br/>（可依截點評估）"]
    end

    subgraph OUT["📤 產出（data/）"]
        vis["visualizations/"]
        bto["backtest/"]
    end

    raw --> gen
    gen --> comp
    comp --> ext
    ext --> extr
    target --> y_def
    y_def --> y_convert
    extr --> split
    split --> t9
    split --> t15
    split --> t30
    t9 --> ds9
    t15 --> ds15
    t30 --> ds30
    ds9 --> ae
    ds15 --> ae
    ds30 --> ae
    ae --> w9
    ae --> w15
    ae --> w30
    w9 --> merge
    w15 --> merge
    w30 --> merge
    y_convert --> merge
    merge --> mfg9
    merge --> mfg15
    merge --> mfg30
    merge --> ag
    w9 --> viz
    w15 --> viz
    w30 --> viz
    viz --> vis
    mfg9 --> bt
    mfg15 --> bt
    mfg30 --> bt
    bt --> bto
```

---

## 執行順序與依賴

**三組截點**：所有 X（特徵）與 Y（目標）皆依 **09:00、09:15、09:30** 三種截點分別產出；Y 為**報酬率**（收盤－該截點），會經 log → simple 加工後再與特徵合併。

| 步驟 | 模組 | 輸入（data/） | 產出（data/） |
|------|------|----------------|----------------|
| 1 | **01_data_ingestion** | `raw/`, 或既有 `indicators_complete/` | `indicators_complete/`, `indicators_extracted/`（共用） |
| 2 | **02_feature_compression** | `indicators_extracted/` | 三組：`dataset/0900`, `0915`, `0930/`；`output_0900`, `output_0915`, `output_0930/`（各含 W*） |
| 3 | **03_modeling** | 各截點之 `output_*/` 壓縮結果、`target/` 內**依截點之報酬率欄位**（如 afternoon_return_0900） | 三組：`merged_for_autogluon/`（0900, 0915, 0930），AutoGluon 模型 |
| 4 | **04_visualization** | 各截點 `output_*/`（JSON、W*） | `visualizations/`（可依截點分檔） |
| 5 | **05_backtest** | 各截點 `merged_for_autogluon/`、預測結果 | `backtest/`（可依截點評估） |

**執行順序**：`01 → 02 → 03 → 04 → 05`（各腳本路徑由 `config.py` 統一指向 `data/`）。

---

## 目錄結構

```
Thesis-AutoGluon-TXF-Research/
├── config.py              # 路徑設定（DATA_ROOT = data/）
├── data/                   # 所有輸入與產出（見 data/README.md）
├── scripts/
│   ├── 01_data_ingestion/
│   ├── 02_feature_compression/
│   ├── 03_modeling/
│   ├── 04_visualization/
│   ├── 05_backtest/
│   └── utils/              # config 引用、plotting_engine
└── docs/
```

---

## 使用方式

1. **資料準備**：將原始 K 線放入 `data/raw/TX2011~20231222-1K/`，目標變數放入 `data/target/y.xlsx`（或 `y.csv`）。詳見 [data/README.md](data/README.md)。
2. **路徑覆寫**：若 `data/` 不在 repo 下，可設環境變數 `DATA_ROOT` 指向實際目錄（本機或 Colab 皆可）。
3. **依序執行**：進入各模組目錄執行對應腳本，或依 `scripts/README_scripts.md` 總覽執行。
