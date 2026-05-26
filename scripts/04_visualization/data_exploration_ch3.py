#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三章研究方法：資料來源與資料探索

產出：
1. 日成交量分佈圖（以日為單位，三千多個交易日的成交量分配）
2. 每日報酬率統計（平均值、標準差、偏態、峰度）— 展現厚尾特性
3. 台股期貨收盤價走勢圖（標註重大事件）
4. 日內波動度與成交量集中度（各時段佔比）

每日報酬率以開盤至收盤價格計算：return = (close - open) / open
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# 中文字體：依系統選擇
import matplotlib.font_manager as fm
_cjk_fonts = [f.name for f in fm.fontManager.ttflist if "PingFang" in f.name or "Heiti" in f.name or "Noto Sans CJK" in f.name or "Microsoft" in f.name]
if _cjk_fonts:
    plt.rcParams["font.sans-serif"] = _cjk_fonts[:3] + ["DejaVu Sans"]
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from scipy import stats

# 專案路徑
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import config as _config

# 資料路徑：優先使用 Thesis-AutoGluon 的 raw，其次搜尋 thesis 根目錄
RAW_KLINE_DIR = _config.get_raw_kline_dir()
VIS_DIR = _config.get_visualizations_dir()
OUTPUT_DIR = VIS_DIR / "ch3_data_exploration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 若 raw 不存在，嘗試從 thesis 根目錄搜尋
THESIS_ROOT = Path("/Volumes/Transcend 1/thesis")
if not Path(RAW_KLINE_DIR).exists():
    _candidates = list(THESIS_ROOT.glob("**/TX2011*1K/*.csv"))
    if _candidates:
        RAW_KLINE_DIR = str(_candidates[0].parent)
    else:
        RAW_KLINE_DIR = str(THESIS_ROOT / "github_clone/Thesis-AutoGluon-TXF-Research/data/raw/TX2011~20231222-1K")

# 重大事件（日期, 標籤）— 使用 MMDD 日期格式
MAJOR_EVENTS = [
    ("2011-08-05", "0805 歐債危機"),
    ("2015-06-01", "0601 漲跌幅放寬"),
    ("2020-03-13", "0313 疫情大跌"),
    ("2022-02-24", "0224 俄烏戰爭"),
]


def load_daily_aggregated(raw_dir: str) -> pd.DataFrame:
    """從 1K 原始檔彙總為日資料：開盤、收盤、成交量。"""
    pattern = os.path.join(raw_dir, "TX*_1K.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"找不到資料檔：{pattern}")

    rows = []
    for fp in files:
        try:
            df = pd.read_csv(fp, encoding="utf-8")
        except Exception:
            df = pd.read_csv(fp, encoding="big5")
        df.columns = [c.strip() for c in df.columns]
        # 欄位可能為 Date/Open/Close/Volume 或 date/open/close/volume
        col_map = {c.lower(): c for c in df.columns}
        date_col = col_map.get("date", "Date")
        open_col = col_map.get("open", "Open")
        close_col = col_map.get("close", "Close")
        vol_col = col_map.get("volume", "Volume")

        if date_col not in df.columns or open_col not in df.columns or close_col not in df.columns:
            continue

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        if df.empty:
            continue

        first = df.iloc[0]
        last = df.iloc[-1]
        day_open = float(first[open_col])
        day_close = float(last[close_col])
        day_vol = df[vol_col].sum() if vol_col in df.columns else 0
        n_bars = len(df)
        day_avg_vol = day_vol / n_bars if n_bars > 0 else 0  # 日內平均成交量（每分鐘平均）

        day_str = first[date_col].strftime("%Y-%m-%d") if hasattr(first[date_col], "strftime") else str(first[date_col])[:10]
        rows.append({
            "date": pd.Timestamp(day_str),
            "open": day_open,
            "close": day_close,
            "volume": day_vol,
            "avg_volume": day_avg_vol,
        })

    daily = pd.DataFrame(rows)
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def load_intraday_all(raw_dir: str) -> pd.DataFrame:
    """載入所有 1K 分鐘資料，用於日內集中度分析。"""
    pattern = os.path.join(raw_dir, "TX*_1K.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"找不到資料檔：{pattern}")

    rows = []
    for fp in files:
        try:
            df = pd.read_csv(fp, encoding="utf-8")
        except Exception:
            df = pd.read_csv(fp, encoding="big5")
        df.columns = [c.strip() for c in df.columns]
        col_map = {c.lower(): c for c in df.columns}
        date_col = col_map.get("date", "Date")
        open_col = col_map.get("open", "Open")
        high_col = col_map.get("high", "High")
        low_col = col_map.get("low", "Low")
        close_col = col_map.get("close", "Close")
        vol_col = col_map.get("volume", "Volume")
        if date_col not in df.columns:
            continue
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        for _, r in df.iterrows():
            dt = r[date_col]
            rows.append({
                "datetime": dt,
                "time_str": dt.strftime("%H:%M") if hasattr(dt, "strftime") else str(dt)[11:16],
                "open": float(r[open_col]),
                "high": float(r[high_col]),
                "low": float(r[low_col]),
                "close": float(r[close_col]),
                "volume": int(r[vol_col]) if vol_col in df.columns else 0,
            })
    return pd.DataFrame(rows)


def compute_daily_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """計算開盤至收盤報酬率：return = (close - open) / open"""
    daily = daily.copy()
    daily["return"] = (daily["close"] - daily["open"]) / daily["open"]
    daily["return_pct"] = daily["return"] * 100
    return daily


def plot_volume_distribution(daily: pd.DataFrame, out_dir: Path) -> None:
    """日成交量分佈圖：以日為單位，三千多個交易日的日成交量分配。"""
    vol = daily["volume"].dropna()
    vol = vol[vol > 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vol, bins=60, color="#2E86AB", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(vol.mean(), color="#E94F37", linestyle="--", linewidth=2, label=f"Mean: {vol.mean():,.0f}")
    ax.axvline(vol.median(), color="#44AF69", linestyle="-.", linewidth=1.5, label=f"Median: {vol.median():,.0f}")
    ax.set_xlabel("Daily Volume (contracts)", fontsize=12)
    ax.set_ylabel("Number of Trading Days", fontsize=12)
    ax.set_title(f"TX Daily Volume Distribution\n({len(vol):,} trading days)", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(True)
    plt.tight_layout()
    plt.savefig(out_dir / "01_volume_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"已儲存: {out_dir / '01_volume_distribution.png'}")


def plot_intraday_concentration(raw_dir: str, out_dir: Path) -> None:
    """日內波動度與成交量集中度：各時段佔總量比例（以 15 分鐘為單位）。"""
    print("  載入 1K 分鐘資料（日內集中度）...")
    intra = load_intraday_all(raw_dir)
    intra["dt"] = pd.to_datetime(intra["datetime"])
    intra["hour"] = intra["dt"].dt.hour
    intra["minute"] = intra["dt"].dt.minute
    intra["slot"] = intra["hour"] * 60 + (intra["minute"] // 15) * 15
    intra["slot_label"] = intra.apply(lambda r: f"{r['hour']:02d}:{r['minute']//15*15:02d}", axis=1)
    intra["range_pct"] = (intra["high"] - intra["low"]) / intra["open"] * 100

    by_slot = intra.groupby("slot").agg(
        volume=("volume", "sum"),
        vol_mean=("volume", "mean"),
        range_pct_mean=("range_pct", "mean"),
    ).reset_index()
    by_slot["slot_label"] = by_slot["slot"].apply(
        lambda x: f"{x//60:02d}:{x%60:02d}"
    )
    total_vol = by_slot["volume"].sum()
    by_slot["vol_pct"] = by_slot["volume"] / total_vol * 100

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    slots = by_slot["slot_label"].tolist()
    x = np.arange(len(slots))

    ax1 = axes[0]
    bars = ax1.bar(x - 0.2, by_slot["vol_pct"], width=0.4, color="#2E86AB", alpha=0.8, label="Volume %")
    ax1.set_ylabel("Volume Share (%)", fontsize=11)
    ax1.set_title("Intraday Concentration: Volume & Volatility by 15-min Slot\n(All sample days aggregated)", fontsize=12)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    ax2.bar(x + 0.2, by_slot["range_pct_mean"], width=0.4, color="#E94F37", alpha=0.8, label="Avg Range % (High-Low)/Open")
    ax2.set_ylabel("Avg Range (%)", fontsize=11)
    ax2.set_xlabel("Time (Session)", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.xticks(x, slots, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "04_intraday_concentration.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"已儲存: {out_dir / '04_intraday_concentration.png'}")


def plot_return_statistics(daily: pd.DataFrame, out_dir: Path) -> None:
    """報酬率統計：平均值、標準差、偏態、峰度，並展示厚尾特性。"""
    ret = daily["return"].dropna()
    ret = ret[np.isfinite(ret)]

    mean_ret = ret.mean() * 100
    std_ret = ret.std() * 100
    skew = stats.skew(ret)
    kurt = stats.kurtosis(ret)  # excess kurtosis

    # 圖1：報酬率直方圖 + 常態分佈對照（展示厚尾）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    n, bins, _ = ax1.hist(ret * 100, bins=80, density=True, color="#2E86AB", alpha=0.7, edgecolor="white", label="實際報酬率")
    x = np.linspace(ret.min() * 100, ret.max() * 100, 200)
    norm_pdf = stats.norm.pdf(x, mean_ret, std_ret)
    ax1.plot(x, norm_pdf, "r-", linewidth=2, label="常態分佈")
    ax1.set_xlabel("每日報酬率（%）", fontsize=11)
    ax1.set_ylabel("密度", fontsize=11)
    ax1.set_title("每日報酬率分佈 vs 常態分佈", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    stats_text = (
        f"Sample N: {len(ret):,}\n"
        f"Mean: {mean_ret:.4f}%\n"
        f"Std: {std_ret:.4f}%\n"
        f"Skewness: {skew:.4f}\n"
        f"Kurtosis: {kurt:.4f}\n\n"
        "Skew>0: right skew\n"
        "Kurt>0: fat tails"
    )
    ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment="center", fontfamily="sans-serif",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax2.axis("off")
    ax2.set_title("Descriptive Statistics", fontsize=12)

    plt.suptitle("台指期貨每日報酬率（開盤→收盤）— 樣本期間描述統計", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "02_return_statistics.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"已儲存: {out_dir / '02_return_statistics.png'}")

    # 輸出統計表（CSV，英文）
    stats_df = pd.DataFrame({
        "Metric": ["Sample N", "Mean (%)", "Std (%)", "Skewness", "Kurtosis"],
        "Value": [len(ret), f"{mean_ret:.4f}", f"{std_ret:.4f}", f"{skew:.4f}", f"{kurt:.4f}"],
    })
    stats_df.to_csv(out_dir / "02_return_statistics.csv", index=False, encoding="utf-8-sig")
    print(f"已儲存: {out_dir / '02_return_statistics.csv'}")


def load_sp500_daily(thesis_root: Path) -> pd.DataFrame | None:
    """載入 S&P 500 日收盤價。優先本地檔，其次從 merged_tw_us 的 US_Return 重建，最後用 yfinance。"""
    # 1. 搜尋本地 S&P 500 收盤價（含 close 欄位）
    for pattern in ["**/sp500*.csv", "**/sp500*.xlsx", "**/GSPC*.csv"]:
        for fp in thesis_root.glob(pattern):
            try:
                if fp.suffix == ".csv":
                    df = pd.read_csv(fp)
                else:
                    df = pd.read_excel(fp)
                df.columns = [c.strip().lower() for c in df.columns]
                if "date" in df.columns and "close" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    return df[["date", "close"]].dropna().sort_values("date")
            except Exception:
                pass
    # 2. 從 merged_tw_us_sp500 的 US_Return 重建指數（100 * cumprod(1 + return)）
    for fp in thesis_root.glob("**/merged_tw_us_sp500.csv"):
        try:
            df = pd.read_csv(fp)
            df.columns = [c.strip() for c in df.columns]
            date_col = "Date" if "Date" in df.columns else "date"
            ret_col = "US_Return" if "US_Return" in df.columns else "us_return"
            if date_col not in df.columns or ret_col not in df.columns:
                continue
            df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
            df["close"] = 100.0 * (1 + df[ret_col].astype(float)).cumprod()
            return df[["date", "close"]].dropna().sort_values("date")
        except Exception:
            pass
    # 3. 嘗試 yfinance
    try:
        import yfinance as yf
        df = yf.download("^GSPC", start="2010-12-31", end="2024-01-01", interval="1d", progress=False, auto_adjust=False)
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        if "date" not in df.columns and "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df[["date", "close"]].dropna().sort_values("date")
    except ImportError:
        return None


def plot_closing_price_with_events(daily: pd.DataFrame, out_dir: Path) -> None:
    """台股期貨與 S&P 500 收盤價走勢圖（標準化起點=100），標註重大事件。"""
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()

    # 載入 S&P 500
    sp500 = load_sp500_daily(THESIS_ROOT)
    if sp500 is not None:
        merged = daily.merge(sp500, on="date", how="inner", suffixes=("_tx", "_sp"))
        tx_norm = merged["close_tx"] / merged["close_tx"].iloc[0] * 100
        sp_norm = merged["close_sp"] / merged["close_sp"].iloc[0] * 100
        plot_dates = merged["date"]
        has_sp500 = True
    else:
        tx_norm = daily["close"] / daily["close"].iloc[0] * 100
        plot_dates = daily["date"]
        has_sp500 = False

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(plot_dates, tx_norm, color="#1a1a2e", linewidth=1.2, alpha=0.9, label="TX (Taiwan Index Futures)")
    if has_sp500:
        ax.plot(plot_dates, sp_norm, color="#2E86AB", linewidth=1.2, alpha=0.8, linestyle="-", label="S&P 500")

    # Y 軸：僅顯示指數（無實際點數）
    ax.set_ylabel("Index (Base = 100)", fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # 標註重大事件：奇偶交錯 y 位置，避免重疊
    ax.set_xlabel("Date", fontsize=12)
    title = "Taiwan Index Futures & S&P 500 (Normalized, Base=100)" if has_sp500 else "Taiwan Index Futures (Normalized, Base=100)"
    ax.set_title(f"{title}\nMajor Market Events (2011–2023)", fontsize=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=30)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.draw()
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    y_label_base = y_min - y_range * 0.06
    for i, (event_date_str, label) in enumerate(MAJOR_EVENTS):
        try:
            ed = pd.Timestamp(event_date_str)
        except Exception:
            continue
        idx = (plot_dates - ed).abs().argmin()
        ex = plot_dates.iloc[idx]
        ey = float(tx_norm.iloc[idx]) if hasattr(tx_norm, "iloc") else float(tx_norm.values[idx])
        y_label = y_label_base - (i % 2) * y_range * 0.04  # 奇偶交錯
        ax.axvline(ex, color="#E94F37", linestyle="--", alpha=0.5, linewidth=1)
        ax.annotate(
            label,
            xy=(ex, ey),
            xytext=(ex, y_label),
            fontsize=8,
            ha="center",
            va="top",
            rotation=45,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF8E7", edgecolor="#E94F37", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#E94F37", lw=0.8),
        )
    ax.set_ylim(y_min - y_range * 0.18, y_max)  # 留空間給標籤
    plt.tight_layout()
    plt.savefig(out_dir / "03_closing_price_with_events.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"已儲存: {out_dir / '03_closing_price_with_events.png'}")


def main() -> None:
    print("載入原始 1K 資料...")
    daily = load_daily_aggregated(RAW_KLINE_DIR)
    print(f"共 {len(daily)} 個交易日，期間 {daily['date'].min()} ~ {daily['date'].max()}")

    daily = compute_daily_returns(daily)
    print(f"每日報酬率範圍: {daily['return_pct'].min():.2f}% ~ {daily['return_pct'].max():.2f}%")

    print("\n產出圖表...")
    plot_volume_distribution(daily, OUTPUT_DIR)
    plot_return_statistics(daily, OUTPUT_DIR)
    plot_closing_price_with_events(daily, OUTPUT_DIR)
    plot_intraday_concentration(RAW_KLINE_DIR, OUTPUT_DIR)

    print("\n完成。輸出目錄:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
