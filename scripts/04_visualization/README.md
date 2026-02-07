# 04_visualization

- **讀取（data/）**：`output_0900/`（all_windows_results_*.json、W*）
- **寫入（data/）**：`visualizations/`（MSE 時間線、雷達圖、重建散點）

## 手動執行

```bash
cd scripts/04_visualization
python visualize_results.py
```

或由 main 觸發：`python main.py --step 4`
