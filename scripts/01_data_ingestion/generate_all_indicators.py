#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從 data/raw 讀取 1 分鐘 K 線 CSV，產出完整技術指標至 data/indicators_complete。
僅讀寫 data/，路徑由專案 config 提供。可手動執行或由 main.py --step 1 觸發。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
import numpy as np
import pandas as pd
import talib
import warnings

warnings.filterwarnings("ignore")

# 僅讀寫 data/：輸入 raw K 線目錄，輸出 indicators_complete
INPUT_DIR = config.get_raw_kline_dir()
OUTPUT_DIR = config.get_indicators_complete_dir()
VERBOSE = True


def create_output_directory():
    """建立輸出目錄"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"輸出目錄已建立: {OUTPUT_DIR}")


def generate_all_indicators_for_single_day(df):
    """
    為單一交易日的資料生成所有可用的技術指標
    """
    if len(df) < 50:
        print(f"警告：資料點不足 ({len(df)} < 50)，跳過技術指標計算")
        return df

    open_prices = df["open"].values.astype(float)
    high_prices = df["high"].values.astype(float)
    low_prices = df["low"].values.astype(float)
    close_prices = df["close"].values.astype(float)
    volumes = df["volume"].values.astype(float)

    result_df = df.copy()

    try:
        for period in [3, 5, 7, 10, 14, 20, 30, 50, 100, 200]:
            if len(df) >= period:
                result_df[f"SMA_{period}"] = talib.SMA(close_prices, timeperiod=period)
                result_df[f"EMA_{period}"] = talib.EMA(close_prices, timeperiod=period)
                result_df[f"WMA_{period}"] = talib.WMA(close_prices, timeperiod=period)
                result_df[f"DEMA_{period}"] = talib.DEMA(close_prices, timeperiod=period)
                result_df[f"TEMA_{period}"] = talib.TEMA(close_prices, timeperiod=period)
                result_df[f"TRIMA_{period}"] = talib.TRIMA(close_prices, timeperiod=period)
                result_df[f"KAMA_{period}"] = talib.KAMA(close_prices, timeperiod=period)

        if len(df) >= 14:
            mama, fama = talib.MAMA(close_prices, fastlimit=0.5, slowlimit=0.05)
            result_df["MAMA"] = mama
            result_df["FAMA"] = fama

        for period in [6, 9, 14, 21, 28]:
            if len(df) >= period:
                result_df[f"RSI_{period}"] = talib.RSI(close_prices, timeperiod=period)
                result_df[f"STOCH_K_{period}"], result_df[f"STOCH_D_{period}"] = talib.STOCH(
                    high_prices, low_prices, close_prices, fastk_period=period, slowk_period=3, slowd_period=3
                )
                result_df[f"STOCHF_K_{period}"], result_df[f"STOCHF_D_{period}"] = talib.STOCHF(
                    high_prices, low_prices, close_prices, fastk_period=period, fastd_period=3
                )
                result_df[f"STOCHRSI_K_{period}"], result_df[f"STOCHRSI_D_{period}"] = talib.STOCHRSI(
                    close_prices, timeperiod=period, fastk_period=5, fastd_period=3
                )
                result_df[f"WILLR_{period}"] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=period)
                result_df[f"ADX_{period}"] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=period)
                result_df[f"ADXR_{period}"] = talib.ADXR(high_prices, low_prices, close_prices, timeperiod=period)
                result_df[f"AROON_Down_{period}"], result_df[f"AROON_Up_{period}"] = talib.AROON(
                    high_prices, low_prices, timeperiod=period
                )
                result_df[f"AROONOSC_{period}"] = talib.AROONOSC(high_prices, low_prices, timeperiod=period)
                result_df[f"CCI_{period}"] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=period)
                result_df[f"MFI_{period}"] = talib.MFI(high_prices, low_prices, close_prices, volumes, timeperiod=period)
                result_df[f"MOM_{period}"] = talib.MOM(close_prices, timeperiod=period)
                result_df[f"ROC_{period}"] = talib.ROC(close_prices, timeperiod=period)
                result_df[f"ROCP_{period}"] = talib.ROCP(close_prices, timeperiod=period)
                result_df[f"ROCR_{period}"] = talib.ROCR(close_prices, timeperiod=period)
                result_df[f"ROCR100_{period}"] = talib.ROCR100(close_prices, timeperiod=period)
                result_df[f"DX_{period}"] = talib.DX(high_prices, low_prices, close_prices, timeperiod=period)
                result_df[f"PDI_{period}"] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=period)
                result_df[f"MDI_{period}"] = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=period)

        for fast, slow in [(5, 10), (8, 17), (12, 26), (19, 39)]:
            if len(df) >= slow:
                macd, macd_signal, macd_hist = talib.MACD(
                    close_prices, fastperiod=fast, slowperiod=slow, signalperiod=9
                )
                result_df[f"MACD_{fast}_{slow}"] = macd
                result_df[f"MACD_signal_{fast}_{slow}"] = macd_signal
                result_df[f"MACD_hist_{fast}_{slow}"] = macd_hist

        for period in [5, 10, 14, 20, 30]:
            if len(df) >= period:
                result_df[f"ATR_{period}"] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
                result_df[f"NATR_{period}"] = talib.NATR(high_prices, low_prices, close_prices, timeperiod=period)
                result_df[f"TRANGE_{period}"] = talib.TRANGE(high_prices, low_prices, close_prices)
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    close_prices, timeperiod=period, nbdevup=2, nbdevdn=2
                )
                result_df[f"BBANDS_upper_{period}"] = bb_upper
                result_df[f"BBANDS_middle_{period}"] = bb_middle
                result_df[f"BBANDS_lower_{period}"] = bb_lower
                result_df[f"BBANDS_width_{period}"] = (bb_upper - bb_lower) / bb_middle
                result_df[f"BBANDS_percent_{period}"] = (close_prices - bb_lower) / (bb_upper - bb_lower)

        result_df["AD"] = talib.AD(high_prices, low_prices, close_prices, volumes)
        result_df["ADOSC"] = talib.ADOSC(
            high_prices, low_prices, close_prices, volumes, fastperiod=3, slowperiod=10
        )
        result_df["OBV"] = talib.OBV(close_prices, volumes)
        for period in [5, 10, 20]:
            if len(df) >= period:
                result_df[f"VOLUME_SMA_{period}"] = talib.SMA(volumes, timeperiod=period)
                result_df[f"VOLUME_EMA_{period}"] = talib.EMA(volumes, timeperiod=period)

        result_df["AVGPRICE"] = talib.AVGPRICE(open_prices, high_prices, low_prices, close_prices)
        result_df["MEDPRICE"] = talib.MEDPRICE(high_prices, low_prices)
        result_df["TYPPRICE"] = talib.TYPPRICE(high_prices, low_prices, close_prices)
        result_df["WCLPRICE"] = talib.WCLPRICE(high_prices, low_prices, close_prices)

        for period in [5, 10, 14, 20, 30]:
            if len(df) >= period:
                result_df[f"STDDEV_{period}"] = talib.STDDEV(close_prices, timeperiod=period)
                result_df[f"VAR_{period}"] = talib.VAR(close_prices, timeperiod=period)
                result_df[f"LINEARREG_{period}"] = talib.LINEARREG(close_prices, timeperiod=period)
                result_df[f"LINEARREG_SLOPE_{period}"] = talib.LINEARREG_SLOPE(close_prices, timeperiod=period)
                result_df[f"LINEARREG_ANGLE_{period}"] = talib.LINEARREG_ANGLE(close_prices, timeperiod=period)
                result_df[f"LINEARREG_INTERCEPT_{period}"] = talib.LINEARREG_INTERCEPT(
                    close_prices, timeperiod=period
                )
                result_df[f"TSF_{period}"] = talib.TSF(close_prices, timeperiod=period)

        for period in [5, 10, 14, 20, 30]:
            if len(df) >= period:
                result_df[f"MAX_{period}"] = talib.MAX(close_prices, timeperiod=period)
                result_df[f"MIN_{period}"] = talib.MIN(close_prices, timeperiod=period)
                result_df[f"MAXINDEX_{period}"] = talib.MAXINDEX(close_prices, timeperiod=period)
                result_df[f"MININDEX_{period}"] = talib.MININDEX(close_prices, timeperiod=period)
                result_df[f"SUM_{period}"] = talib.SUM(close_prices, timeperiod=period)
                result_df[f"BETA_{period}"] = talib.BETA(high_prices, low_prices, timeperiod=period)
                result_df[f"CORREL_{period}"] = talib.CORREL(high_prices, low_prices, timeperiod=period)

        cdl_names = [
            "CDL2CROWS", "CDL3BLACKCROWS", "CDL3INSIDE", "CDL3LINESTRIKE", "CDL3OUTSIDE",
            "CDL3STARSINSOUTH", "CDL3WHITESOLDIERS", "CDLABANDONEDBABY", "CDLADVANCEBLOCK",
            "CDLBELTHOLD", "CDLBREAKAWAY", "CDLCLOSINGMARUBOZU", "CDLCONCEALBABYSWALL",
            "CDLCOUNTERATTACK", "CDLDARKCLOUDCOVER", "CDLDOJI", "CDLDOJISTAR", "CDLDRAGONFLYDOJI",
            "CDLENGULFING", "CDLEVENINGDOJISTAR", "CDLEVENINGSTAR", "CDLGAPSIDESIDEWHITE",
            "CDLGRAVESTONEDOJI", "CDLHAMMER", "CDLHANGINGMAN", "CDLHARAMI", "CDLHARAMICROSS",
            "CDLHIGHWAVE", "CDLHIKKAKE", "CDLHIKKAKEMOD", "CDLHOMINGPIGEON", "CDLIDENTICAL3CROWS",
            "CDLINNECK", "CDLINVERTEDHAMMER", "CDLKICKING", "CDLKICKINGBYLENGTH", "CDLLADDERBOTTOM",
            "CDLLONGLEGGEDDOJI", "CDLLONGLINE", "CDLMARUBOZU", "CDLMATCHINGLOW", "CDLMATHOLD",
            "CDLMORNINGDOJISTAR", "CDLMORNINGSTAR", "CDLONNECK", "CDLPIERCING", "CDLRICKSHAWMAN",
            "CDLRISEFALL3METHODS", "CDLSEPARATINGLINES", "CDLSHOOTINGSTAR", "CDLSHORTLINE",
            "CDLSPINNINGTOP", "CDLSTALLEDPATTERN", "CDLSTICKSANDWICH", "CDLTAKURI", "CDLTASUKIGAP",
            "CDLTHRUSTING", "CDLTRISTAR", "CDLUNIQUE3RIVER", "CDLUPSIDEGAP2CROWS", "CDLXSIDEGAP3METHODS",
        ]
        for name in cdl_names:
            func = getattr(talib, name, None)
            if func is not None:
                result_df[name] = func(open_prices, high_prices, low_prices, close_prices)

    except Exception as e:
        print(f"生成技術指標時發生錯誤: {str(e)}")
        return df

    return result_df


def process_single_file(file_path):
    """處理單一檔案，按交易日分別計算技術指標"""
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        if VERBOSE:
            print(f"處理檔案: {file_path.name}")
            print(f"原始資料形狀: {df.shape}")
            print(f"時間範圍: {df.index.min()} 到 {df.index.max()}")

        df_with_indicators = generate_all_indicators_for_single_day(df)

        if VERBOSE:
            print(f"加入技術指標後形狀: {df_with_indicators.shape}")
            print(f"新增欄位數: {df_with_indicators.shape[1] - df.shape[1]}")
            nan_count = df_with_indicators.isnull().sum().sum()
            print(f"缺失值數量: {nan_count}")
            print("-" * 50)

        return df_with_indicators

    except Exception as e:
        print(f"處理檔案 {file_path} 時發生錯誤: {str(e)}")
        return None


def process_batch_files():
    """批次處理所有 CSV，排除 combined_*。"""
    create_output_directory()

    csv_files = [f for f in INPUT_DIR.glob("*.csv") if not f.name.startswith("combined_")]
    total_files = len(csv_files)

    print(f"找到 {total_files} 個 CSV 檔案（已排除 combined_*）")
    print(f"開始按交易日分別處理...")

    processed_count = 0
    failed_count = 0
    all_dataframes = []

    for i, file_path in enumerate(csv_files):
        try:
            df_with_indicators = process_single_file(file_path)

            if df_with_indicators is not None and not df_with_indicators.empty:
                output_filename = file_path.stem + "_indicators_complete.csv"
                output_path = OUTPUT_DIR / output_filename
                df_with_indicators.to_csv(output_path)
                all_dataframes.append(df_with_indicators)
                processed_count += 1
                if VERBOSE and processed_count % 50 == 0:
                    print(f"已處理 {processed_count}/{total_files} 個檔案")
            else:
                failed_count += 1
                print(f"檔案 {file_path.name} 處理失敗")
        except Exception as e:
            failed_count += 1
            print(f"處理檔案 {file_path.name} 時發生錯誤: {str(e)}")

    if all_dataframes:
        print("\n合併所有資料...")
        combined_df = pd.concat(all_dataframes, axis=0)
        combined_df = combined_df.sort_index()
        combined_output_path = OUTPUT_DIR / "combined_with_indicators_complete.csv"
        combined_df.to_csv(combined_output_path)
        print(f"\n處理完成!")
        print(f"成功處理: {processed_count} 個檔案")
        print(f"處理失敗: {failed_count} 個檔案")
        print(f"合併後資料形狀: {combined_df.shape}")
        print(f"合併資料已儲存至: {combined_output_path}")
        return combined_df
    else:
        print("沒有成功處理任何檔案")
        return None


if __name__ == "__main__":
    print("開始完整技術指標生成程序...")
    print(f"輸入目錄（data/）: {INPUT_DIR}")
    print(f"輸出目錄（data/）: {OUTPUT_DIR}")
    print("=" * 60)
    result_df = process_batch_files()
    if result_df is not None:
        print(f"\n完整技術指標生成完成，已儲存至: {OUTPUT_DIR}")
    else:
        print("完整技術指標生成失敗！")
