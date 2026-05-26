#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 01（新版）：由 raw 1 分鐘 K 線產生五大面向特徵（每交易日一個 xlsx）。

五大面向：
1) 跨日：sp500_prev_return, open_gap_pct, vix_prev_close
2) 量價：vwap_bias_5, volume_roc_5
3) 波動：atr_5（ATR/close）
4) 動能：kd_k_9, kd_d_9（KD 9,3,3，含前日暖機）
5) 趨勢：linearreg_angle_5
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import talib

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config

try:
    import yfinance as yf
except Exception:
    yf = None

INPUT_DIR = Path(config.get_raw_kline_dir())
OUTPUT_DIR = Path(config.get_indicators_complete_dir())
VERBOSE = True
MIN_BARS = 5


@dataclass
class DailyMeta:
    trade_date: pd.Timestamp
    first_open: float
    last_close: float


def _log(msg: str) -> None:
    if VERBOSE:
        print(msg)


def list_input_csvs(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.glob("TX*_1K.csv") if not p.name.startswith("._"))


def load_day_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    colmap = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=colmap)
    if "date" in df.columns and "datetime" not in df.columns:
        df = df.rename(columns={"date": "datetime"})

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        first = df.columns[0]
        maybe_dt = pd.to_datetime(df[first], errors="coerce")
        if maybe_dt.notna().sum() == 0:
            raise ValueError(f"{path.name} 找不到 datetime/date 欄位")
        df["datetime"] = maybe_dt

    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{path.name} 缺少必要欄位: {required}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["datetime"] + required)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError(f"{path.name} 清洗後無有效資料")
    return df


def build_daily_meta(daily_data: List[Tuple[Path, pd.DataFrame]]) -> Dict[pd.Timestamp, DailyMeta]:
    meta: Dict[pd.Timestamp, DailyMeta] = {}
    for _, df in daily_data:
        d = pd.Timestamp(df["datetime"].iloc[0]).normalize()
        meta[d] = DailyMeta(
            trade_date=d,
            first_open=float(df["open"].iloc[0]),
            last_close=float(df["close"].iloc[-1]),
        )
    return meta


def fetch_market_series(start_date: pd.Timestamp, end_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if yf is None:
        raise RuntimeError("缺少 yfinance 套件，請先安裝: pip install yfinance")

    start = (start_date - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    end = (end_date + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    sp = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=False)
    vx = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
    if sp.empty or vx.empty:
        raise RuntimeError("無法抓到 S&P500 或 VIX 資料")

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df[["Close"]].copy().rename(columns={"Close": "close"})
        out.index = pd.to_datetime(out.index).normalize()
        out = out.reset_index().rename(columns={"Date": "date", "index": "date"})
        out["prev_return"] = out["close"].pct_change()
        return out

    return _prep(sp), _prep(vx)


def get_prev_market_row(market_df: pd.DataFrame, trade_date: pd.Timestamp) -> Optional[pd.Series]:
    rows = market_df[market_df["date"] < trade_date]
    if rows.empty:
        return None
    return rows.iloc[-1]


def add_requested_indicators_with_history(df: pd.DataFrame, prev_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    cur = df.copy().reset_index(drop=True)
    if prev_df is not None and not prev_df.empty:
        hist = prev_df.copy().reset_index(drop=True)
        merged = pd.concat([hist, cur], ignore_index=True)
        cur_start = len(hist)
    else:
        merged = cur.copy()
        cur_start = 0

    high = merged["high"].to_numpy(dtype=float)
    low = merged["low"].to_numpy(dtype=float)
    close = merged["close"].to_numpy(dtype=float)
    volume = merged["volume"].to_numpy(dtype=float)

    pv = merged["close"] * merged["volume"]
    merged["vwap_5"] = (
        pv.rolling(5, min_periods=1).sum()
        / merged["volume"].rolling(5, min_periods=1).sum().replace(0, np.nan)
    )
    merged["vwap_bias_5"] = (merged["close"] - merged["vwap_5"]) / merged["vwap_5"]
    merged["volume_roc_5"] = talib.ROC(volume, timeperiod=5)
    merged["atr_5"] = talib.ATR(high, low, close, timeperiod=5) / merged["close"].replace(0, np.nan)

    k, d = talib.STOCH(
        high,
        low,
        close,
        fastk_period=9,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    merged["kd_k_9"] = k
    merged["kd_d_9"] = d
    merged["linearreg_angle_5"] = talib.LINEARREG_ANGLE(close, timeperiod=5)

    out = merged.iloc[cur_start:].copy().reset_index(drop=True)
    out = out.drop(columns=["vwap_5"], errors="ignore")
    return out


def process_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_files = list_input_csvs(INPUT_DIR)
    if not input_files:
        raise FileNotFoundError(f"在 {INPUT_DIR} 找不到可處理的 CSV")

    _log(f"找到 {len(input_files)} 個日內 CSV，開始讀取...")
    daily_data: List[Tuple[Path, pd.DataFrame]] = []
    for p in input_files:
        try:
            df = load_day_csv(p)
            if len(df) < MIN_BARS:
                _log(f"[SKIP] {p.name}: 有效資料小於 {MIN_BARS}")
                continue
            daily_data.append((p, df))
        except Exception as e:
            _log(f"[SKIP] {p.name}: {e}")

    if not daily_data:
        raise RuntimeError("沒有可處理的有效日內資料")

    daily_data.sort(key=lambda x: pd.Timestamp(x[1]["datetime"].iloc[0]))
    meta = build_daily_meta(daily_data)
    sorted_dates = sorted(meta.keys())
    start_date = sorted_dates[0]
    end_date = sorted_dates[-1]

    _log(f"交易日範圍: {start_date.date()} ~ {end_date.date()}")
    _log("抓取跨日市場資料（S&P500 / VIX）...")
    sp_df, vx_df = fetch_market_series(start_date, end_date)

    success = 0
    for idx, (path, day_df) in enumerate(daily_data):
        trade_date = pd.Timestamp(day_df["datetime"].iloc[0]).normalize()
        prev_day_df = daily_data[idx - 1][1] if idx > 0 else None

        prev_local_close = np.nan
        if idx > 0:
            prev_date = pd.Timestamp(daily_data[idx - 1][1]["datetime"].iloc[0]).normalize()
            prev_local_close = meta[prev_date].last_close

        out = add_requested_indicators_with_history(day_df, prev_day_df)

        first_open = float(out["open"].iloc[0])
        if np.isfinite(prev_local_close) and prev_local_close != 0:
            out["open_gap_pct"] = (first_open - prev_local_close) / prev_local_close
        else:
            out["open_gap_pct"] = np.nan

        sp_row = get_prev_market_row(sp_df, trade_date)
        vx_row = get_prev_market_row(vx_df, trade_date)
        out["sp500_prev_return"] = (
            float(sp_row["prev_return"].iloc[0]) if sp_row is not None else np.nan
        )
        out["vix_prev_close"] = float(vx_row["close"].iloc[0]) if vx_row is not None else np.nan

        out_name = f"{path.stem}_qlib_indicators_complete.xlsx"
        out_path = OUTPUT_DIR / out_name
        out.to_excel(out_path, index=False, engine="openpyxl")
        success += 1
        _log(f"[OK] {path.name} -> {out_name} ({len(out)} rows)")

    _log("-" * 60)
    _log(f"完成！成功輸出 {success} 個 xlsx 至: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_all()
