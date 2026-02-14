# 04_visualization

## 腳本說明

| 腳本 | 讀取 | 寫入 | 說明 |
|------|------|------|------|
| `visualize_results.py` | `output_0900/` (JSON, W*) | `visualizations/` | MSE 時間線、壓縮雷達圖、重建散點 |
| `txf_ai_analysis.py` | `data/models/`, `autogluon_ready/` | `visualizations/txf_ai_analysis/txf_ai_analysis.xlsx` | 台指期 AI 模型全方位分析（單一 xlsx，每 Scenario 一分頁） |

## txf_ai_analysis 規格 (BDD)

**第一類：內部數據交叉分析**
- 01 時間避險效果、02 準確度悖論、03 模型適應性演變、04 訓練記憶邊際效益
- 05 反向策略、06 集成 ROI、07 特徵權重漂移、08 抗噪能力、09 績效持續性
- 10 極端值依賴、11 技術指標類別優勢、12 跳空敏感度、13 多空偏差
- 14 窗口衰退率、15 複雜度懲罰

**第二類：外部數據對照分析**
- 16 波動率濾網、17 趨勢相關性、18 成交量相依性、19 國際連動
- 20 升息衝擊、21 黑天鵝壓力測試、22 日內波幅、23 類股輪動、24 市場效率檢定

外部數據需 `yfinance`（VIX、^TWII、^GSPC 等）。

## 手動執行

```bash
cd scripts/04_visualization

# 執行全部
python run.py

# 僅執行 AI 分析
python run.py --script txf_ai_analysis.py

# 直接執行
python txf_ai_analysis.py
python visualize_results.py
```

或由 main 觸發：`python main.py --step 4`
