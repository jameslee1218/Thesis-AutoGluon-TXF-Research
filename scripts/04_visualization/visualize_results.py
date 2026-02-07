#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoencoder Results Visualization
Reads JSON and models from output dir and produces:
(1) MSE Error Timeline
(2) Compression Radar Chart
(3) Reconstruction Scatter (original vs reconstructed)
"""

import os
import sys
import json
import glob
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths（僅讀寫 data/，由專案 config 提供）
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import config as _config
OUTPUT_BASE = str(_config.get_output_0900_dir())
DATASET_DIR = str(_config.get_dataset_dir("0900"))
VIS_OUTPUT_DIR = str(_config.get_visualizations_dir())
os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

# Indicator groups (same as 02_autoencoder.py)
INDICATOR_GROUPS = {
    "STOCH": ["STOCH_K_14", "STOCH_D_14"],
    "STOCHF": ["STOCHF_K_14", "STOCHF_D_14"],
    "STOCHRSI": ["STOCHRSI_K_14", "STOCHRSI_D_14"],
    "MACD": ["MACD_12_26", "MACD_signal_12_26", "MACD_hist_12_26"],
    "BBANDS": ["BBANDS_upper_20", "BBANDS_middle_20", "BBANDS_lower_20"],
    "ADX_DMI": ["ADX_14", "ADXR_14", "PDI_14", "MDI_14", "DX_14"],
    "AROON": ["AROON_Down_14", "AROON_Up_14", "AROONOSC_14"]
}

GROUP_ORDER = list(INDICATOR_GROUPS.keys())


def find_results_json(output_base):
    """Find all_windows_results_*.json under output (e.g. 0900/0915/0930)."""
    candidates = []
    for root, _, files in os.walk(output_base):
        for f in files:
            if f.startswith("all_windows_results_") and f.endswith(".json"):
                path = os.path.join(root, f)
                if not os.path.basename(path).startswith("._"):
                    candidates.append(path)
    if not candidates:
        candidates = glob.glob(os.path.join(output_base, "all_windows_results_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No all_windows_results_*.json under {output_base}")
    candidates.sort(key=lambda p: os.path.basename(p), reverse=True)
    return candidates[0], os.path.dirname(candidates[0])


def load_mse_by_window_and_group(results_dir):
    """Load Test MSE per window and group from JSON. Returns (windows, mse_matrix, groups, run_subdir)."""
    json_path, run_subdir = find_results_json(OUTPUT_BASE)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def window_sort_key(name):
        m = re.match(r"W(\d+)_", name)
        return int(m.group(1)) if m else 0

    windows = sorted(data.keys(), key=window_sort_key)
    groups = GROUP_ORDER
    mse_matrix = np.full((len(windows), len(groups)), np.nan)
    group_to_idx = {g: i for i, g in enumerate(groups)}

    for wi, win in enumerate(windows):
        for group_name, group_data in data[win].items():
            if group_name not in group_to_idx:
                continue
            test_mse = group_data.get("final_scores", {}).get("test_mse")
            if test_mse is not None:
                mse_matrix[wi, group_to_idx[group_name]] = test_mse

    return windows, mse_matrix, groups, run_subdir


def plot_mse_timeline(windows, mse_matrix, groups, save_dir):
    """(1) MSE Error Timeline: x=Window, y=Test MSE, one line per indicator group."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(windows))
    x_labels = [w.split("_")[0] if "_" in w else w for w in windows]

    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    for gi, group in enumerate(groups):
        ax.plot(x, mse_matrix[:, gi], marker='o', markersize=4, label=group, color=colors[gi], linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel("Window", fontsize=12)
    ax.set_ylabel("Test MSE", fontsize=12)
    ax.set_title("The Error Timeline — Test MSE by Window (peaks = harder to compress)", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "01_mse_error_timeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def plot_radar_chart(windows, mse_matrix, groups, save_dir):
    """(2) Compression Radar Chart: mean Test MSE per indicator group (inner = easier to compress)."""
    mean_mse = np.nanmean(mse_matrix, axis=0)
    n = len(groups)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    values = mean_mse.tolist()
    values += values[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    ax.plot(angles, values, "o-", linewidth=2, label="Mean Test MSE")
    ax.fill(angles, values, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("Mean Test MSE", fontsize=10)
    ax.set_title("Compression Radar — Mean Test MSE by Group (inner = easier to compress)", fontsize=14, fontweight="bold")
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(save_dir, "02_compression_radar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def load_compress_year_data(dataset_dir, compress_year):
    """Load and merge all CSV files for the given year from dataset dir (for reconstruction scatter)."""
    pattern = os.path.join(dataset_dir, f"TX*{compress_year}*_1K_qlib_indicators_complete.csv")
    files = glob.glob(pattern)
    files = [f for f in files if not os.path.basename(f).startswith("._")]
    if not files:
        raise FileNotFoundError(f"No data for year {compress_year}: {pattern}")
    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f)
        df["datetime"] = pd.to_datetime(df["datetime"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).sort_values("datetime").reset_index(drop=True)


def _get_keras_load_model_with_custom_objects():
    """Return load_model and custom_objects so that saved 'mse' loss/metric can be resolved."""
    custom = {}
    try:
        import keras
        load_model = keras.models.load_model
        if hasattr(keras.losses, "mean_squared_error"):
            custom["mse"] = keras.losses.mean_squared_error
            custom["mean_squared_error"] = keras.losses.mean_squared_error
        if hasattr(keras.metrics, "mean_squared_error"):
            custom["mean_squared_error"] = keras.metrics.mean_squared_error
        if hasattr(keras.metrics, "MeanSquaredError"):
            custom["MeanSquaredError"] = keras.metrics.MeanSquaredError
    except Exception:
        import tensorflow as tf
        load_model = tf.keras.models.load_model
        custom["mse"] = tf.keras.losses.mean_squared_error
        custom["mean_squared_error"] = tf.keras.losses.mean_squared_error
        if hasattr(tf.keras.metrics, "MeanSquaredError"):
            custom["MeanSquaredError"] = tf.keras.metrics.MeanSquaredError
    return load_model, custom


def get_reconstruction(window_dir, group_name, indicator_cols, df_raw, scaler_path, model_path, max_points=3000):
    """
    Get (original_scaled, reconstructed) using the window's model and scaler.
    Returns 1d arrays, sampled to at most max_points; only finite values kept for plotting.
    """
    load_model, custom_objects = _get_keras_load_model_with_custom_objects()
    data = df_raw[indicator_cols].values
    data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X = np.asarray(scaler.transform(data), dtype=np.float64)
    try:
        model = load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        if "mse" in str(e).lower() or "metric" in str(e).lower() or "loss" in str(e).lower() or "locate" in str(e).lower():
            model = load_model(model_path, compile=False)
        else:
            raise
    recon = model.predict(X, verbose=0)
    # Ensure numpy 1d arrays (model may return tensor)
    orig_flat = np.asarray(X, dtype=np.float64).flatten()
    recon_flat = np.asarray(recon, dtype=np.float64).flatten()
    assert len(orig_flat) == len(recon_flat), "Original and reconstruction length mismatch"
    # Keep only finite for plotting
    mask = np.isfinite(orig_flat) & np.isfinite(recon_flat)
    orig_flat = orig_flat[mask]
    recon_flat = recon_flat[mask]
    if len(orig_flat) == 0:
        raise ValueError("No finite values after filtering")
    if len(orig_flat) > max_points:
        idx = np.random.RandomState(42).choice(len(orig_flat), max_points, replace=False)
        orig_flat = orig_flat[idx]
        recon_flat = recon_flat[idx]
    return orig_flat, recon_flat


def plot_reconstruction_scatter(run_subdir, windows, mse_matrix, groups, dataset_dir, save_dir,
                                good_group="BBANDS", bad_group="MACD", sample_window_idx=-1):
    """
    (3) Reconstruction scatter: one good and one poor MSE group; x=original (scaled), y=reconstructed; 45deg = perfect.
    """
    win_name = windows[sample_window_idx]
    m = re.search(r"compress_(\d+)-\d+", win_name)
    if not m:
        raise ValueError(f"Cannot parse compress year from: {win_name}")
    compress_year = int(m.group(1))
    window_dir = os.path.join(run_subdir, win_name)

    df_raw = load_compress_year_data(dataset_dir, compress_year)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (group_name, title_suffix) in zip(axes, [
        (good_group, "(easier to compress)"),
        (bad_group, "(harder to compress)")
    ]):
        indicator_cols = INDICATOR_GROUPS.get(group_name)
        if not indicator_cols:
            ax.set_title(f"{group_name} — not defined")
            continue
        missing = [c for c in indicator_cols if c not in df_raw.columns]
        if missing:
            ax.set_title(f"{group_name} — missing cols: {missing[:3]}")
            continue
        scaler_path = os.path.join(window_dir, "models", f"{group_name}_scaler.pkl")
        model_path = os.path.join(window_dir, "models", f"{group_name}_search_best.h5")
        if not os.path.isfile(scaler_path) or not os.path.isfile(model_path):
            ax.set_title(f"{group_name} — model/scaler not found")
            continue
        try:
            orig, recon = get_reconstruction(
                window_dir, group_name, indicator_cols, df_raw,
                scaler_path, model_path, max_points=5000
            )
        except Exception as e:
            ax.set_title(f"{group_name} — error: {e}")
            continue
        n_pts = len(orig)
        if n_pts == 0:
            ax.set_title(f"{group_name} — no data")
            continue
        # Draw scatter with visible point size and opacity
        ax.scatter(orig, recon, alpha=0.4, s=12, c="steelblue", edgecolors="none")
        lim_lo = min(orig.min(), recon.min())
        lim_hi = max(orig.max(), recon.max())
        if lim_hi <= lim_lo:
            lim_lo -= 0.5
            lim_hi += 0.5
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=2, label="Perfect (45 deg)")
        ax.set_xlabel("Original (scaled)", fontsize=11)
        ax.set_ylabel("Reconstructed", fontsize=11)
        ax.set_title(f"{group_name} {title_suffix}\nWindow: {win_name} | n={n_pts}", fontsize=12, fontweight="bold")
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    plt.suptitle("Reconstruction Scatter — Original vs Reconstructed (deviation from 45 deg = information loss)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "03_reconstruction_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def main():
    print("=" * 60)
    print("[START] Autoencoder results visualization")
    print("=" * 60)
    print(f"OUTPUT_BASE: {OUTPUT_BASE}")
    print(f"VIS_OUTPUT_DIR: {VIS_OUTPUT_DIR}")

    windows, mse_matrix, groups, run_subdir = load_mse_by_window_and_group(OUTPUT_BASE)
    print(f"\n[INFO] {len(windows)} windows, {len(groups)} groups")
    print(f"[INFO] Results dir: {run_subdir}")

    print("\n(1) Plot MSE error timeline...")
    plot_mse_timeline(windows, mse_matrix, groups, VIS_OUTPUT_DIR)

    print("\n(2) Plot compression radar...")
    plot_radar_chart(windows, mse_matrix, groups, VIS_OUTPUT_DIR)

    print("\n(3) Plot reconstruction scatter...")
    try:
        plot_reconstruction_scatter(
            run_subdir, windows, mse_matrix, groups,
            DATASET_DIR, VIS_OUTPUT_DIR,
            good_group="BBANDS", bad_group="MACD",
            sample_window_idx=-1
        )
    except FileNotFoundError as e:
        print(f"[WARN] Skip reconstruction scatter (missing data): {e}")
    except Exception as e:
        print(f"[WARN] Reconstruction scatter failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("[DONE] Visualization finished")
    print(f"Output dir: {VIS_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
