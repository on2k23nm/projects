"""
================================================================================
CSIRO BIOMASS SHARED UTILITIES
================================================================================
A comprehensive library for biomass data analysis, visualization, and modeling.
Includes data loading, statistical analysis, distribution visualization, and
evaluation metrics for multi-target biomass regression tasks.

KEY COMPONENTS:
1. Data Loading & Preparation (prepare_data, load_frozen_split, save_frozen_split)
2. Distribution Analysis (plot_distributions, strong_points_from_distributions)
3. Feature Engineering Utilities (ndvi_weighted_sanity_report)
4. Visualization Tools (print_biomass_2xN, show_images_grid, plot_species_*)
5. Statistical Metrics (compute_weighted_sst, compute_weighted_sse)
6. Species Analysis (species_biomass_summary, species_dist_compare)
7. Image & Batch Processing (show_dl_batch_original_vs_transformed)
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image
import torch
import cv2
import math

from pathlib import Path
import json
import pandas as pd
from pathlib import Path
import json
import pandas as pd
import difflib

from sklearn.model_selection import train_test_split

# ================================================================================
# GLOBAL CONFIGURATION & CONSTANTS
# ================================================================================

DEFAULT_WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g":  0.1,
    "Dry_Clover_g":0.1,
    "GDM_g":       0.2,
    "Dry_Total_g": 0.5,
}

cols = ["Pre_GSHH_NDVI", "Height_Ave_cm",
        "Dry_Clover_g", "Dry_Dead_g",
        "Dry_Green_g", "Dry_Total_g", "GDM_g"]


# ================================================================================
# SECTION 1: FEATURE ENGINEERING & CORRELATION ANALYSIS
# ================================================================================

def ndvi_weighted_sanity_report(
    df: pd.DataFrame,
    x_col: str,
    y_weights: dict,
    *,
    corr_method: str = "pearson",   # "pearson" or "spearman"
    dropna: bool = True,
    sort_by: str = "expected_contrib",  # or "w_sst_share", "abs_corr"
):
    """
    PURPOSE: Analyze correlation between a feature (e.g., NDVI) and multiple targets 
             with weighted importance accounting.
    
    USAGE: Called in csiro-biomass-data-analysis.ipynb to assess NDVI's predictive power
           across all biomass targets, weighted by their importance scores.
    
    INPUT:
      - df: DataFrame with x_col and all target columns
      - x_col: Feature column name (e.g., "Pre_GSHH_NDVI")
      - y_weights: Dict {target_col: weight} indicating target importance
      - corr_method: "pearson" or "spearman" correlation
      - sort_by: Sort output by "expected_contrib", "w_sst_share", or "abs_corr"
    
    OUTPUT:
      - rep: DataFrame with correlation stats per target
      - r2w_expected: Expected weighted R² if using x_col alone
    
    EXAMPLE:
      rep, r2w = ndvi_weighted_sanity_report(
          full_df, 
          x_col="Pre_GSHH_NDVI", 
          y_weights=DEFAULT_WEIGHTS
      )
    """
    assert corr_method in ("pearson", "spearman"), "corr_method must be 'pearson' or 'spearman'"

    cols_needed = [x_col] + list(y_weights.keys())
    d = df[cols_needed].copy()

    if dropna:
        d = d.dropna()

    x = d[x_col].astype(float).values
    if len(x) < 3:
        raise ValueError("Not enough rows after dropna to compute correlations (need >= 3).")

    rows = []
    # compute w*SST denominator
    w_sst_list = []
    tmp_stats = {}

    for y_col, w in y_weights.items():
        y = d[y_col].astype(float).values
        y_mean = np.mean(y)
        sst = float(np.sum((y - y_mean) ** 2))

        tmp_stats[y_col] = (w, y_mean, float(np.std(y, ddof=1)) if len(y) > 1 else 0.0, sst)
        w_sst_list.append(w * sst)

    denom_w_sst = float(np.sum(w_sst_list)) if np.sum(w_sst_list) > 0 else np.nan

    for y_col, (w, y_mean, y_std, sst) in tmp_stats.items():
        y = d[y_col].astype(float)

        # correlation (handles constant columns -> NaN)
        corr = float(pd.Series(x).corr(y, method=corr_method))
        r2_ndvi = float(corr ** 2) if np.isfinite(corr) else np.nan

        w_sst = w * sst
        w_sst_share = (w_sst / denom_w_sst) if (np.isfinite(denom_w_sst) and denom_w_sst > 0) else np.nan

        expected_contrib = (w_sst_share * r2_ndvi) if (np.isfinite(w_sst_share) and np.isfinite(r2_ndvi)) else np.nan

        rows.append({
            "target": y_col,
            "weight": w,
            "n": int(len(d)),
            "mean": y_mean,
            "std": y_std,
            "SST": sst,
            f"{corr_method}_corr": corr,
            "r2_ndvi_only_linear": r2_ndvi,
            "w*SST": w_sst,
            "w*SST_share": w_sst_share,
            "expected_contrib": expected_contrib,
            "abs_corr": abs(corr) if np.isfinite(corr) else np.nan,
        })

    rep = pd.DataFrame(rows)

    # Expected weighted-R2 using NDVI-only linear model
    r2w_expected = float(np.nansum(rep["expected_contrib"].values))

    if sort_by in rep.columns:
        rep = rep.sort_values(sort_by, ascending=False).reset_index(drop=True)

    print(f"\nNDVI sanity report (x='{x_col}', corr='{corr_method}')")
    print(f"Expected weighted-R2 (NDVI-only linear) ≈ {r2w_expected:.4f}")
    print("(This is a proxy: it answers 'how much can NDVI alone explain, aligned with your weights?')\n")

    return rep, r2w_expected

# ================================================================================
# SECTION 2: DATA LOADING & PREPARATION
# ================================================================================

def prepare_data(data_root, in_csv_fl):
    """
    PURPOSE: Load raw Kaggle CSV data, clean, pivot by targets, and return 
             one-row-per-sample format for modeling.
    
    USAGE: Called at data loading stage (csiro-biomass-data-loading.ipynb) to 
           convert multi-row pivot format into clean sample-level DataFrame.
    
    INPUT:
      - data_root: Path to directory containing the CSV
      - in_csv_fl: Filename (e.g., "train.csv")
    
    OUTPUT:
      - DataFrame with columns: [sample_id, Species, Pre_GSHH_NDVI, Height_Ave_cm, 
                                  Dry_Clover_g, Dry_Dead_g, Dry_Green_g, Dry_Total_g, GDM_g]
    
    DATA INTEGRITY CHECK:
      - Validates no duplicate group keys after sample_id cleaning (prevents silent averaging)
    
    EXAMPLE:
      full_df = prepare_data(data_root="/kaggle/input/csiro-biomass", 
                            in_csv_fl="train.csv")
    """
    # ---------------------------------
    # Read the data from CSV
    train_df = pd.read_csv(f"{data_root}/{in_csv_fl}")
    # print(f"INPUT:\n{train_df}")

    cols_to_drop = ["State", "Sampling_Date", "image_path"]
    biomass_cols_order = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]
        
    train_df = train_df.drop(columns=cols_to_drop)
    # print(train_df)

    # Change the sample_id to have just ID and nothing else
    train_df["sample_id"] = train_df["sample_id"].str.split("__").str[0]
    
    # DATA INTEGRITY: Check that sample_id is still unique within each (Species, NDVI, Height) group
    # to detect silent averaging during pivot_table
    group_cols = ["sample_id", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"]
    duplicates = train_df[group_cols].duplicated(keep=False).sum()
    if duplicates > 0:
        raise ValueError(
            f"After splitting sample_id, found {duplicates} rows with duplicate group keys. "
            "This suggests pivot_table will average silently. Check raw data for collisions."
        )

    train_pivot = (
        train_df.pivot_table(
            index=["sample_id", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"],
            columns="target_name",
            values="target"
        ).reset_index()
    )

    train_pivot = train_pivot.rename_axis(columns=None)

    # ensure biomass columns are in a fixed order
    train_pivot = train_pivot [
        ["sample_id", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"] + biomass_cols_order
    ]

    return train_pivot.reset_index(drop=True)


# ================================================================================
# SECTION 3: INTERNAL CORRELATION HELPERS (NO SCIPY DEPENDENCY)
# ================================================================================

def _rankdata_avg(a: np.ndarray) -> np.ndarray:
    """
    PURPOSE: Compute average ranks for correlation calculations (Spearman).
    
    USAGE: Internal helper for _spearmanr() - converts values to ranks with 
           tie-handling (average rank for tied values).
    
    EXAMPLE:
      ranks = _rankdata_avg(np.array([5, 2, 5, 8]))  
      # Returns: [2.5, 1, 2.5, 4]  (two 5's share ranks 2-3)
    """
    a = np.asarray(a)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks start at 1
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks

def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """
    PURPOSE: Compute Pearson correlation coefficient without scipy.
    
    USAGE: Internal helper for print_biomass_2xN() and other correlation calculations.
           Handles edge cases (constant arrays, insufficient data).
    
    RETURNS: Float in [-1, 1] or NaN if inputs are invalid/constant.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm * xm).sum()) * np.sqrt((ym * ym).sum())
    if denom <= 1e-12:
        return np.nan
    return float((xm * ym).sum() / denom)

def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    """
    PURPOSE: Compute Spearman rank correlation coefficient (rank-based, robust to outliers).
    
    USAGE: Internal helper used in print_biomass_2xN() to show both Pearson (P) 
           and Spearman (S) correlations for feature-target pairs.
    
    RETURNS: Float in [-1, 1] or NaN if insufficient data.
    """
    rx = _rankdata_avg(x)
    ry = _rankdata_avg(y)
    return _pearsonr(rx, ry)

def _corr_ps(x: np.ndarray, y: np.ndarray):
    """
    PURPOSE: Compute Pearson AND Spearman correlation together for comparison.
    
    USAGE: Internal helper for print_biomass_2xN() to display both correlation types.
           Filters finite values automatically.
    
    RETURNS: Tuple (pearsonr, spearmanr) - both floats or NaN.
    """
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan
    xx = x[m]
    yy = y[m]
    return _pearsonr(xx, yy), _spearmanr(xx, yy)

# ================================================================================
# SECTION 4: DATA VISUALIZATION & EXPLORATION
# ================================================================================

def print_biomass_2xN(
    df,
    *,
    target_col="Pre_GSHH_NDVI",
    y_cols=("Dry_Clover_g", "Dry_Green_g", "Dry_Dead_g", "Dry_Total_g", "GDM_g"),
    hue_col=None,
    max_points=5000,
    log1p_bottom=True,
    point_size=10,
    alpha=0.6,
    title=None,
    legend_below=True,
    legend_ncol=6,
    legend_fontsize=9,
):
    """
    PURPOSE: Create 2×N visualization grid (raw scale top, log1p bottom) for feature-target
             relationships, with optional species/category coloring.
    
    USAGE: Called in csiro-biomass-data-analysis.ipynb to visualize correlations between
           a feature (e.g., NDVI) and all biomass targets, showing both raw and log scales.
    
    INPUT:
      - df: DataFrame with target_col and y_cols
      - target_col: Feature column (X-axis)
      - y_cols: List of target columns to plot (each becomes one column)
      - hue_col: Optional categorical column for color-coding points (e.g., "Species")
      - log1p_bottom: If True, show log1p(y) in bottom row
      - max_points: Downsample to this many points (for large datasets)
    
    OUTPUT: 2×N subplot grid with scatter plots + Pearson/Spearman correlations
    
    EXAMPLE:
      print_biomass_2xN(full_df, target_col="Pre_GSHH_NDVI", 
                       y_cols=["Dry_Total_g", "GDM_g"], hue_col="Species")
    """
    if max_points is not None and len(df) > max_points:
        d = df.sample(n=max_points, random_state=42).copy()
    else:
        d = df.copy()

    x = d[target_col].astype(float).to_numpy()

    hue_vals = None
    legend_labels = None
    if hue_col is not None:
        hue_vals = d[hue_col].astype(str).to_numpy()
        legend_labels = np.unique(hue_vals)

    ncols = len(y_cols)
    nrows = 2 if log1p_bottom else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.8 * nrows), squeeze=False)

    if title is None:
        title = f"{target_col} vs targets"
    fig.suptitle(title, fontsize=14)

    def scatter_with_hue(ax, x, y, hue_vals):
        if hue_vals is None:
            ax.scatter(x, y, s=point_size, alpha=alpha)
            return
        for lab in legend_labels:
            m = (hue_vals == lab)
            ax.scatter(x[m], y[m], s=point_size, alpha=alpha, label=lab)

    for j, ycol in enumerate(y_cols):
        y = d[ycol].astype(float).to_numpy()

        # --- TOP (raw y) ---
        P, S = _corr_ps(x, y)
        ax0 = axes[0, j]
        scatter_with_hue(ax0, x, y, hue_vals)
        ax0.set_xlabel(target_col)
        ax0.set_ylabel(ycol)
        ax0.set_title(f"{ycol}\nP={P:.2f}, S={S:.2f}", fontsize=11)
        ax0.grid(True, alpha=0.2)

        # --- BOTTOM (log1p y) ---
        if log1p_bottom:
            y_log = np.log1p(np.maximum(y, -1 + 1e-12))
            P2, S2 = _corr_ps(x, y_log)

            ax1 = axes[1, j]
            scatter_with_hue(ax1, x, y_log, hue_vals)
            ax1.set_xlabel(target_col)
            ax1.set_ylabel(f"log1p({ycol})")
            ax1.set_title(f"log1p({ycol})\nP={P2:.2f}, S={S2:.2f}", fontsize=11)
            ax1.grid(True, alpha=0.2)

    # --- Legend placement ---
    if hue_col is not None:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            for ax in axes.ravel():
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()

            if legend_below:
                fig.legend(
                    handles, labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.02),
                    ncol=legend_ncol,
                    fontsize=legend_fontsize,
                    title=hue_col,
                    frameon=True,
                )
                plt.tight_layout(rect=[0, 0.08, 1, 0.95])
            else:
                fig.legend(
                    handles, labels,
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.5),
                    title=hue_col,
                )
                plt.tight_layout(rect=[0, 0, 0.86, 0.95])
        else:
            plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

def scatter_raw_and_log1p_one_row(
    df,
    cols,
    *,
    max_points=5000,
    log1p_eps=0.0,     # keep 0 unless you have small negatives (not typical for NDVI)
    point_size=10,
    alpha=0.6,
    show_titles=True,
):
    """
    PURPOSE: Visualize 1D distributions of columns in raw and log1p scales 
             (index vs value scatter plots, 2 rows × N columns).
    
    USAGE: Called in exploratory analysis to detect skew/tail behavior and validate 
           log transformation effectiveness.
    
    INPUT:
      - df: DataFrame
      - cols: List of column names to visualize
      - max_points: Downsample if len(df) > max_points
    
    OUTPUT: 2×N grid (top = raw, bottom = log1p)
    
    EXAMPLE:
      scatter_raw_and_log1p_one_row(full_df, ["Dry_Total_g", "GDM_g"])
    """
    if isinstance(cols, str):
        cols = [cols]

    ncols = len(cols)
    if ncols == 0:
        print("scatter_raw_and_log1p_one_row: cols is empty.")
        return

    # sample rows (same rows used for all columns)
    n = len(df)
    if n == 0:
        print("scatter_raw_and_log1p_one_row: df is empty.")
        return

    if max_points is None or max_points <= 0 or max_points >= n:
        d = df
    else:
        d = df.sample(n=max_points, random_state=42)

    x_idx = np.arange(len(d))

    fig, axes = plt.subplots(
        2, ncols,
        figsize=(5 * ncols, 8),
        squeeze=False,
        constrained_layout=False
    )

    for j, col in enumerate(cols):
        y = d[col].astype(float).to_numpy()

        # --- TOP: RAW ---
        ax0 = axes[0, j]
        ax0.scatter(x_idx, y, s=point_size, alpha=alpha)
        ax0.set_xlabel("index")
        ax0.set_ylabel(col)
        if show_titles:
            ax0.set_title(f"RAW {col}")

        # --- BOTTOM: LOG1P ---
        ax1 = axes[1, j]
        # guard: log1p requires >= -1; clamp if tiny negatives exist
        y_log = np.log1p(np.maximum(y, -1 + 1e-12) + log1p_eps)
        ax1.scatter(x_idx, y_log, s=point_size, alpha=alpha)
        ax1.set_xlabel("index")
        ax1.set_ylabel(f"log1p({col})")
        if show_titles:
            ax1.set_title(f"LOG1P {col}")

    plt.tight_layout()
    plt.show()

# ================================================================================
# SECTION 5: FROZEN SPLIT MANAGEMENT (REPRODUCIBILITY)
# ================================================================================

def save_frozen_split(df3: pd.DataFrame, out_dir="dataset/frozen_split",
                      id_col="sample_id", split_col="split", overwrite=False):
    """
    PURPOSE: Save a train/val/test split (by ID) to disk for reproducible, 
             deterministic experiments across runs.
    
    USAGE: Called after creating stratified splits to lock them permanently.
           Ensures all notebooks use the exact same Train/Val/Test sets.
    
    INPUT:
      - df3: DataFrame with split_col indicating "train"/"val"/"test"
      - out_dir: Directory to save artifacts (creates if missing)
      - id_col: Column name with unique sample IDs
      - split_col: Column name with split labels ("train"/"val"/"test")
    
    OUTPUT:
      - df3.parquet: Full DataFrame (all columns)
      - splits.json: Manifest with ID lists for each split
    
    EXAMPLE:
      save_frozen_split(df3, out_dir="dataset/frozen_split", 
                       id_col="sample_id", split_col="split")
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_path = out_dir / "df3.parquet"
    split_path = out_dir / "splits.json"

    if (df_path.exists() or split_path.exists()) and not overwrite:
        raise FileExistsError(f"Artifacts exist in {out_dir}. Set overwrite=True to replace.")

    # save full df3 (ALL columns)
    df3.to_parquet(df_path, index=False)

    # save split IDs
    manifest = {
        "id_col": id_col,
        "split_col": split_col,
        "splits": {
            "train": df3.loc[df3[split_col] == "train", id_col].tolist(),
            "val":   df3.loc[df3[split_col] == "val",   id_col].tolist(),
            "test":  df3.loc[df3[split_col] == "test",  id_col].tolist(),
        }
    }
    split_path.write_text(json.dumps(manifest, indent=2))

    print("Saved:", df_path)
    print("Saved:", split_path)
    print({k: len(v) for k, v in manifest["splits"].items()})
    return manifest



# ================================================================================
# SECTION 6: INTERNAL UTILITIES (COLUMN VALIDATION, STANDARDIZATION)
# ================================================================================

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    PURPOSE: Remove BOM markers and strip whitespace from DataFrame column names.
    
    USAGE: Internal helper to normalize CSV imports with encoding issues.
    
    EXAMPLE:
      df = _standardize_columns(df)  # Cleans up malformed column names
    """
    df = df.copy()
    # remove BOM, strip spaces, keep exact names stable
    df.columns = [c.replace("\ufeff", "").strip() if isinstance(c, str) else c for c in df.columns]
    return df

def _assert_has_columns(df: pd.DataFrame, cols, name="df"):
    """
    PURPOSE: Validate that DataFrame has all required columns (with helpful error messages).
    
    USAGE: Internal validation in prepare_data(), load_frozen_split(), etc. 
           Shows close matches for typos (using difflib).
    
    EXAMPLE:
      _assert_has_columns(df, ["sample_id", "Species", "Dry_Total_g"], name="train_df")
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = [f"{name} is missing columns: {missing}"]
        for m in missing:
            close = difflib.get_close_matches(m, list(df.columns), n=5, cutoff=0.6)
            msg.append(f"  close matches for '{m}': {close}")
        msg.append(f"Available columns ({len(df.columns)}): {list(df.columns)}")
        raise KeyError("\n".join(msg))

def load_frozen_split(out_dir="dataset/frozen_split"):
    """
    PURPOSE: Load a previously saved frozen train/val/test split from disk.
    
    USAGE: Called at start of training/validation notebooks to ensure all runs 
           use the exact same splits saved by save_frozen_split().
    
    INPUT:
      - out_dir: Directory containing df3.parquet and splits.json
    
    OUTPUT:
      - df3: Full DataFrame with all rows and columns
      - train_df, val_df, test_df: DataFrames for each split (by ID lists)
    
    EXAMPLE:
      full_df, train_df, val_df, test_df = load_frozen_split("dataset/frozen_split")
    """
    out_dir = Path(out_dir)
    df3 = pd.read_parquet(out_dir / "df3.parquet")
    manifest = json.loads((out_dir / "splits.json").read_text())

    df3 = _standardize_columns(df3)
    id_col = manifest["id_col"]

    # build dfs by ID lists (most stable)
    id_to_row = df3.set_index(id_col, drop=False)

    train_df = id_to_row.loc[manifest["splits"]["train"]].reset_index(drop=True)
    val_df   = id_to_row.loc[manifest["splits"]["val"]].reset_index(drop=True)
    test_df  = id_to_row.loc[manifest["splits"]["test"]].reset_index(drop=True)

    return df3, train_df, val_df, test_df

def plot_distributions(df, cols, title_prefix="RAW", bins=30, figsize_per_col=(4, 3)):
    """
    PURPOSE: Create histograms + boxplots (2 rows) for each column to visualize 
             distributions, skew, and outliers.
    
    USAGE: Called in csiro-biomass-data-analysis.ipynb to get first look at 
           feature/target distributions before modeling.
    
    INPUT:
      - df: DataFrame
      - cols: List of column names to plot
      - title_prefix: String prefix for subplot titles (e.g., "RAW", "STANDARDIZED")
      - bins: Number of histogram bins
    
    OUTPUT: 2×N grid (top = histogram, bottom = boxplot)
    
    EXAMPLE:
      plot_distributions(full_df, ["Dry_Total_g", "GDM_g", "Height_Ave_cm"], 
                        title_prefix="DATA OVERVIEW")
    """
    import numpy as np
    import matplotlib.pyplot as plt

    df = _standardize_columns(df)
    cols = [c.replace("\ufeff", "").strip() for c in cols]

    _assert_has_columns(df, cols, name=f"{title_prefix} df")

    cols = list(cols)
    n = len(cols)

    fig_w = max(6, figsize_per_col[0] * n)
    fig_h = figsize_per_col[1] * 2
    fig, axes = plt.subplots(2, n, figsize=(fig_w, fig_h))
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for i, col in enumerate(cols):
        x = df[col].dropna()

        ax_hist = axes[0, i]
        ax_hist.hist(x, bins=bins)
        ax_hist.set_title(f"{title_prefix} {col}")
        ax_hist.set_xlabel(col)

        ax_box = axes[1, i]
        ax_box.boxplot(x, vert=False)
        ax_box.set_title(f"{title_prefix} {col}")
        ax_box.set_xlabel(col)
        ax_box.tick_params(axis="y", left=False, labelleft=False)

    plt.tight_layout()
    plt.show()

# # call it
# plot_distributions(full_df, cols)

# ================================================================================
# SECTION 7: DISTRIBUTION ANALYSIS & TAIL DETECTION
# ================================================================================

def _strong_points_with_print(
    df,
    cols,                    # REQUIRED: list/tuple of columns to analyze
    bands=(8, 20),
    tail_frac=0.10,
    weights=None,            # optional dict {col: weight} (else 1.0)
    default_weight=1.0,
    dropna=True,
    top_by="value",          # "value" or "sst"
    print_output=True
):
    """
    Internal variant: prints high-signal distribution facts + tail SST concentration.
    Returns a DataFrame summary (one row per col).
    DEPRECATED: Use strong_points_from_distributions() instead (no printing by default).
    """

    if cols is None or len(cols) == 0:
        raise ValueError("Please pass cols=['col1','col2',...]")

    weights = {} if weights is None else dict(weights)
    rows = []

    for col in cols:
        if col not in df.columns:
            if print_output:
                print(f"\n**{col}**: column not found in df")
            continue

        x = df[col]
        x = x.dropna() if dropna else x
        x = pd.to_numeric(x, errors="coerce").dropna()

        n = len(x)
        if n == 0:
            if print_output:
                print(f"\n**{col}**: no valid numeric values")
            continue

        w = float(weights.get(col, default_weight))

        # core stats
        pct_zero = 100.0 * x.eq(0).mean()
        mean = float(x.mean())
        var = float(x.var(ddof=1))     # sample variance
        std = float(x.std(ddof=1))     # sample std
        cv = float(std / mean) if mean != 0 else np.inf

        med = float(x.median())
        q75 = float(x.quantile(0.75))
        q90 = float(x.quantile(0.90))
        q95 = float(x.quantile(0.95))
        q99 = float(x.quantile(0.99))
        xmax = float(x.max())

        band_pct = {b: float(100.0 * x.le(b).mean()) for b in bands}

        # weighted SST for THIS target (variance vs global mean, same idea as metric denominator)
        y = x.to_numpy(dtype=float)
        ybar = y.mean()
        sst_i = w * (y - ybar) ** 2
        sst_total = float(sst_i.sum() + 1e-12)

        k = max(1, int(np.ceil(tail_frac * n)))
        if top_by == "sst":
            idx = np.argsort(sst_i)[-k:]
        else:  # "value"
            idx = np.argsort(y)[-k:]

        top_sst_share = float(100.0 * sst_i[idx].sum() / sst_total)

        if print_output:
            print(f"\n**{col} — strong points (n={n}, w={w})**")
            print(f"- Range: 0 → {xmax:.2f}")
            print(f"- Mean/Std/Var: mean={mean:.3f}, std={std:.3f}, var={var:.3f} (CV={cv:.3f})")
            print(f"- Zero inflation: {pct_zero:.1f}% are exactly 0")
            print(f"- Typical values: median={med:.2f}, 75th%={q75:.2f}  (~75% ≤ {q75:.2f})")
            print(f"- Tail/outliers: 95th%={q95:.2f}, 99th%={q99:.2f}")
            for b in bands:
                print(f"- Mass near low end: {band_pct[b]:.1f}% are in 0–{b}")
            print(f"- Tail drives variance: top {int(tail_frac*100)}% contribute {top_sst_share:.1f}% of this target’s weighted SST")

        row = {
            "col": col,
            "n": n,
            "weight": w,

            "mean": mean,
            "std": std,
            "var": var,
            "cv": cv,

            "pct_zero": pct_zero,
            "median": med,
            "q75": q75,
            "q90": q90,
            "q95": q95,
            "q99": q99,
            "max": xmax,
            "top_tail_weighted_sst_share_pct": top_sst_share,
        }
        for b in bands:
            row[f"pct_in_0_{b}"] = band_pct[b]

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def show_images_grid(
    sample_ids, *,
    base_dir: str, split: str = "train",
    cols: int = 3, max_images: int = 24, cell_size: float = 8,
    df=None, title_col: str | None = None
):
    ids = [str(x) for x in list(sample_ids)[:max_images]]
    n = len(ids)
    if n == 0:
        return

    # optional: map sample_id -> extra title value
    extra_map = {}
    if df is not None and title_col is not None:
        tmp = df[["sample_id", title_col]].copy()
        tmp["sample_id"] = tmp["sample_id"].astype(str)
        extra_map = dict(zip(tmp["sample_id"].tolist(), tmp[title_col].tolist()))

    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)

    base = Path(base_dir) / split

    # compute aspect ratio (w/h) from the first readable image
    aspect = None
    for sid in ids:
        p = base / f"{sid}.jpg"
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is not None:
            h0, w0 = im.shape[:2]
            if h0 > 0 and w0 > 0:
                aspect = w0 / h0
                break
    if aspect is None:
        return

    # subplot width = cell_size, height = width / aspect (no hardcoding)
    cell_h = cell_size / aspect

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * cell_size, rows * cell_h),
        squeeze=False
    )
    plt.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.06, wspace=0.10, hspace=0.12)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        if i >= n:
            ax.axis("off")
            continue

        sid = ids[i]
        img_path = base / f"{sid}.jpg"
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if img_bgr is None:
            ax.axis("off")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        ax.imshow(img_rgb, origin="upper")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

        ax.tick_params(
            axis="both", which="both",
            direction="out", labelsize=6, pad=2, length=3,
            top=False, right=False
        )
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

        if extra_map and (sid in extra_map):
            ax.set_title(f"{sid} | {title_col}={extra_map[sid]}", fontsize=8, pad=4)
        else:
            ax.set_title(sid, fontsize=8, pad=4)

        ax.axis("on")

    plt.show()

def ndvi_vs_species_hscatter(
    df,
    ndvi_col="Pre_GSHH_NDVI",
    species_col="Species",
    max_points_per_species=500,
    jitter=0.15,
    figsize=(16, 7),
    legend_ncol=4,          # columns in legend row(s)
    legend_fontsize=8
):
    """
    x = NDVI, y = Species. Different color per species.
    Legend is placed at the bottom (consistent).
    """
    d = df[[species_col, ndvi_col]].dropna().copy()
    d[species_col] = d[species_col].astype(str)

    # order species by median NDVI (nice visual)
    order = (
        d.groupby(species_col)[ndvi_col]
         .median()
         .sort_values()
         .index
         .tolist()
    )
    sp2y = {sp: i for i, sp in enumerate(order)}

    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=figsize)

    for sp in order:
        sub = d[d[species_col] == sp]
        if len(sub) > max_points_per_species:
            sub = sub.sample(max_points_per_species, random_state=0)

        y0 = sp2y[sp]
        yj = y0 + rng.uniform(-jitter, jitter, size=len(sub))

        # one call per species -> different color automatically
        ax.scatter(sub[ndvi_col].to_numpy(), yj, s=16, alpha=0.75, label=sp)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel(ndvi_col)
    ax.set_ylabel(species_col)
    ax.set_title(f"{ndvi_col} vs {species_col}")
    ax.grid(True, axis="x", alpha=0.2)

    # Legend at bottom
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(legend_ncol, max(1, len(labels))),
        fontsize=legend_fontsize,
        frameon=False
    )

    # leave space for legend
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    plt.show()

# ================================================================================
# SECTION 9: NUMERIC SERIES SUMMARIZATION
# ================================================================================

def summarize_numeric_series(
    x: pd.Series,
    *,
    weight: float = 1.0,
    bands=(8, 20),
    tail_frac: float = 0.10,
    top_by: str = "value",   # "value" or "sst"
    dropna: bool = True
) -> dict:
    """
    PURPOSE: Compute distribution statistics for a single numeric column 
             (mean, std, quantiles, tail SST concentration).
    
    USAGE: Internal helper for strong_points_from_distributions() - 
           processes one column at a time without printing.
    
    INPUT:
      - x: pandas Series
      - weight: Weight for this variable (used in weighted SST calculation)
      - tail_frac: Fraction of top values (0.10 = top 10%) to highlight
      - top_by: Sort tail by "value" (highest) or "sst" (highest contribution to variance)
    
    OUTPUT: Dict with keys {n, weight, mean, std, var, cv, pct_zero, min, q10, ..., q99, max, 
                            top_tail_weighted_sst_share_pct, pct_in_0_X, ...}
    
    EXAMPLE:
      stats = summarize_numeric_series(df["Dry_Total_g"], weight=0.5, tail_frac=0.10)
    """
    if dropna:
        x = x.dropna()

    x = pd.to_numeric(x, errors="coerce").dropna()
    n = int(len(x))
    if n == 0:
        return {"n": 0}

    w = float(weight)

    # core stats
    y = x.to_numpy(dtype=float)
    mean = float(np.mean(y))
    # ddof=1 only valid if n>1
    var = float(np.var(y, ddof=1)) if n > 1 else 0.0
    std = float(np.std(y, ddof=1)) if n > 1 else 0.0
    cv = float(std / mean) if mean != 0 else np.inf

    pct_zero = float(100.0 * np.mean(y == 0))

    # quantiles
    q = x.quantile([0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]).to_dict()

    # band mass near low end
    band_pct = {b: float(100.0 * np.mean(y <= b)) for b in bands}

    # weighted SST and tail SST share
    ybar = float(np.mean(y))
    sst_i = w * (y - ybar) ** 2
    sst_total = float(sst_i.sum()) + 1e-12

    k = max(1, int(np.ceil(tail_frac * n)))
    if top_by == "sst":
        idx = np.argsort(sst_i)[-k:]
    else:  # "value"
        idx = np.argsort(y)[-k:]

    top_sst_share = float(100.0 * sst_i[idx].sum() / sst_total)

    out = {
        "n": n,
        "weight": w,
        "mean": mean,
        "std": std,
        "var": var,
        "cv": cv,
        "pct_zero": pct_zero,

        "min": float(q[0.00]),
        "q10": float(q[0.10]),
        "q25": float(q[0.25]),
        "median": float(q[0.50]),
        "q75": float(q[0.75]),
        "q90": float(q[0.90]),
        "q95": float(q[0.95]),
        "q99": float(q[0.99]),
        "max": float(q[1.00]),

        "top_tail_weighted_sst_share_pct": top_sst_share,
    }

    for b in bands:
        out[f"pct_in_0_{b}"] = band_pct[b]

    return out


def strong_points_from_distributions(
    df: pd.DataFrame,
    cols,                    # REQUIRED: list/tuple of columns to analyze
    bands=(8, 20),
    tail_frac=0.10,
    weights=None,            # optional dict {col: weight}
    default_weight=1.0,
    dropna=True,
    top_by="value",
) -> pd.DataFrame:
    """
    Single responsibility: compute a summary DataFrame (one row per col).
    NO printing.
    """
    if cols is None or len(cols) == 0:
        raise ValueError("Please pass cols=['col1','col2',...]")

    weights = {} if weights is None else dict(weights)
    rows = []

    for col in cols:
        if col not in df.columns:
            rows.append({"col": col, "n": 0})
            continue

        w = float(weights.get(col, default_weight))
        stats = summarize_numeric_series(
            df[col],
            weight=w,
            bands=bands,
            tail_frac=tail_frac,
            top_by=top_by,
            dropna=dropna,
        )
        stats["col"] = col
        rows.append(stats)

    return pd.DataFrame(rows)
## ===============================================================================
## SECTION 10: WEIGHTED VARIANCE & ERROR METRICS (FOR VALIDATION)
## ===============================================================================

def compute_weighted_sst(
    df,
    species_col="Species",
    target_cols=("Dry_Green_g","Dry_Dead_g","Dry_Clover_g","GDM_g","Dry_Total_g"),
    weights=None,
    top_k=None,
):
    """
    PURPOSE: Compute weighted Sum of Squares Total (SST) per species to understand 
             which species contribute most to overall variance in weighted metrics.
    
    USAGE: Called in validation/evaluation to diagnose if model performance is 
           balanced across species (important for fair scoring).
    
    INPUT:
      - df: DataFrame with targets
      - target_cols: List of target columns to include
      - weights: Dict {col: weight} (defaults to DEFAULT_WEIGHTS)
      - top_k: Keep top K species by count, collapse rest to "Other"
    
    OUTPUT:
      - d: Used DataFrame (after optional top_k filtering)
      - counts: Series of species counts
      - sst_sum: Weighted SST per species
      - sst_share_pct: Percentage contribution of each species to total SST
    
    MATH: SST_i = sum_t w_t * (y_it - ybar_t)^2  for each sample i and target t
    
    EXAMPLE:
      d, counts, sst_sum, sst_pct = compute_weighted_sst(val_df, top_k=12)
      print(sst_pct)  # See which species drive variance
    """
    weights = DEFAULT_WEIGHTS if weights is None else weights
    target_cols = tuple(target_cols)
    w = np.array([weights[c] for c in target_cols], dtype=float)

    d = df[[species_col, *target_cols]].copy()
    d[species_col] = d[species_col].astype(str)

    # Optional: group rare species into "Other"
    if top_k is not None and top_k > 0:
        top = d[species_col].value_counts().head(top_k).index
        d[species_col] = d[species_col].where(d[species_col].isin(top), other="Other")

    counts = d[species_col].value_counts()
    order = counts.index.tolist()

    Y = d[list(target_cols)].astype(float).to_numpy()
    ybar = Y.mean(axis=0, keepdims=True)
    sst_i = ((Y - ybar) ** 2 * w).sum(axis=1)

    sst_df = pd.DataFrame({species_col: d[species_col].to_numpy(), "sst": sst_i})
    sst_sum = sst_df.groupby(species_col)["sst"].sum().reindex(order).fillna(0.0)
    sst_share_pct = 100.0 * sst_sum / (sst_sum.sum() + 1e-12)

    return d, counts.reindex(order), sst_sum, sst_share_pct


def compute_weighted_sse(
    df,
    yhat_df,
    species_col="Species",
    target_cols=("Dry_Green_g","Dry_Dead_g","Dry_Clover_g","GDM_g","Dry_Total_g"),
    weights=None,
    top_k=None,
):
    """
    PURPOSE: Compute weighted Sum of Squares Error (SSE) per species after model predictions 
             to diagnose which species have highest prediction errors.
    
    USAGE: Called during model validation to identify if certain species are harder to 
           predict (e.g., rare species with smaller sample size or noisier data).
    
    INPUT:
      - df: True labels DataFrame
      - yhat_df: Predictions DataFrame (same structure as df)
      - target_cols: List of target columns
      - weights: Dict {col: weight}
      - top_k: Keep top K species, collapse rest to "Other"
    
    OUTPUT:
      - d: Used DataFrame (after top_k filtering)
      - counts: Species counts
      - sse_sum: Weighted SSE per species
      - sse_share_pct: Percentage contribution of each species to total error
    
    MATH: SSE_i = sum_t w_t * (y_it - yhat_it)^2  for each sample i and target t
    
    EXAMPLE:
      d, counts, sse_sum, sse_pct = compute_weighted_sse(val_df, val_pred_df, top_k=12)
      print("Species with highest error:", sse_pct.idxmax())
    """
    if not isinstance(yhat_df, pd.DataFrame):
        raise ValueError("yhat_df must be a pandas DataFrame with the same target_cols.")

    weights = DEFAULT_WEIGHTS if weights is None else weights
    target_cols = tuple(target_cols)
    w = np.array([weights[c] for c in target_cols], dtype=float)

    d = df[[species_col, *target_cols]].copy()
    d[species_col] = d[species_col].astype(str)

    # Optional: group rare species into "Other" (must be identical logic to SST)
    if top_k is not None and top_k > 0:
        top = d[species_col].value_counts().head(top_k).index
        d[species_col] = d[species_col].where(d[species_col].isin(top), other="Other")

    counts = d[species_col].value_counts()
    order = counts.index.tolist()

    Y = d[list(target_cols)].astype(float).to_numpy()

    # align preds by index if possible; else assume same row order
    if yhat_df.index.equals(df.index):
        yhat = yhat_df.loc[df.index, list(target_cols)].to_numpy()
    else:
        yhat = yhat_df[list(target_cols)].to_numpy()

    sse_i = ((Y - yhat) ** 2 * w).sum(axis=1)

    sse_df = pd.DataFrame({species_col: d[species_col].to_numpy(), "sse": sse_i})
    sse_sum = sse_df.groupby(species_col)["sse"].sum().reindex(order).fillna(0.0)
    sse_share_pct = 100.0 * sse_sum / (sse_sum.sum() + 1e-12)

    return d, counts.reindex(order), sse_sum, sse_share_pct


def species_distribution_count(df, species_col="Species", top_k=None, other_label="Other",
                               figsize=(18, 5),            # <- bigger plot
                               legend_cols=6,
                               legend_fontsize=8,          # <- smaller legend
                               legend_title_fontsize=9,
                               label_fontsize=8,           # <- smaller % text on bars
                               pct_decimals=1,
                               bar_width=0.85):
    """
    PURPOSE: Create bar chart showing count and percentage of each species in dataset.
    
    USAGE: Called in EDA to understand species balance (imbalance = modeling challenge).
    
    INPUT:
      - df: DataFrame
      - species_col: Column name with species labels
      - top_k: Keep top K species by count, collapse rest to "Other"
    
    OUTPUT: Bar chart + returns (counts Series, pct Series)
    
    EXAMPLE:
      counts, pct = species_distribution_count(train_df, species_col="Species", top_k=15)
    """
    s = df[species_col].astype(str)

    if top_k is not None:
        top = s.value_counts().head(top_k).index
        s = s.where(s.isin(top), other=other_label)

    counts = s.value_counts(dropna=False).sort_values(ascending=False)
    total = int(counts.sum())
    pct = (counts / total) * 100

    labels = counts.index.tolist()
    values = counts.values

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(values))]

    x = range(len(values))
    bars = ax.bar(x, values, width=bar_width, color=colors)

    ax.set_xticks([])  # hide long labels
    ax.set_ylabel("Count")
    ax.set_title(f"{species_col} distribution (count)")

    ymax = max(values) if len(values) else 1
    ax.set_ylim(0, ymax * 1.25)

    for b, c, p in zip(bars, values, pct.values):
        if c <= 0:
            continue
        inside = c >= 0.18 * ymax
        y = c * 0.55 if inside else c + ymax * 0.03
        va = "center" if inside else "bottom"
        ax.text(
            b.get_x() + b.get_width() / 2,
            y,
            f"{c} ({p:.{pct_decimals}f}%)",
            ha="center",
            va=va,
            fontsize=label_fontsize
        )

    ax.legend(
        bars, labels,
        title=species_col,
        ncol=legend_cols,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize
    )

    plt.tight_layout()
    plt.show()
    return counts, pct
    return counts, pct

# # usage
# counts, pct = species_distribution_count(full_df)

# ================================================================================

def species_dist_compare(full_df, train_df, val_df, test_df, species_col="Species",
                         top_k=None, show="heatmap", figsize=(14, 10)):
    """
    PURPOSE: Compare species distribution across train/val/test splits to verify 
             stratification was successful and fair coverage.
    
    USAGE: Called during split validation to check if all species are well-represented 
           in each fold (no fold has only rare species, etc.).
    
    INPUT:
      - full_df, train_df, val_df, test_df: DataFrames for each set
      - species_col: Column with species labels
      - top_k: Keep top K species (optional)
      - show: "heatmap" to visualize as color matrix
    
    OUTPUT: 
      - mat: DataFrame with counts per split
      - pct: DataFrame with percentages per split
      - Heatmap visualization
    
    EXAMPLE:
      mat, pct = species_dist_compare(full_df, train_df, val_df, test_df, 
                                     species_col="Species", top_k=20)
    """
    dfs = {"full": full_df, "train": train_df, "val": val_df, "test": test_df}

    # counts per split
    counts = {}
    for name, df in dfs.items():
        counts[name] = df[species_col].value_counts(dropna=False)

    # union index, ordered by full frequency
    all_species = counts["full"].index
    mat = pd.concat([counts[k].reindex(all_species, fill_value=0) for k in ["full","train","val","test"]], axis=1)
    mat.columns = ["full","train","val","test"]

    if top_k is not None:
        mat = mat.iloc[:top_k]

    if show == "heatmap":
        plt.figure(figsize=figsize)
        plt.imshow(mat.values, aspect="auto")
        plt.xticks(range(mat.shape[1]), mat.columns)
        plt.yticks(range(mat.shape[0]), mat.index)
        plt.title("Species distribution (count) — aligned across splits")
        plt.colorbar(label="count")
        plt.tight_layout()
        plt.show()

    # also return percent table (useful for sanity check)
    pct = (mat / mat.sum(axis=0).replace(0, np.nan)) * 100
    return mat.astype(int), pct.round(2)

# # usage
# counts_mat, pct_mat = species_dist_compare(full_df, train_df, val_df, test_df,
#                                            species_col="Species",
#                                            top_k=20, show="heatmap")
# display(counts_mat)
# display(pct_mat)

# ================================================================================
# SECTION 13: IMAGE NORMALIZATION & PREPROCESSING
# ================================================================================
# keep these as module-level defaults
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def unnormalize(img_t: torch.Tensor) -> torch.Tensor:
    """
    PURPOSE: Reverse ImageNet normalization for visualization (convert from 
             normalized [0,1] to denormalized RGB [0,1]).
    
    USAGE: Internal helper for show_dl_batch_original_vs_transformed() to make 
           transformed images visually interpretable.
    
    INPUT: Normalized image tensor [3, H, W]
    OUTPUT: Denormalized image tensor [3, H, W] clamped to [0, 1]
    
    EXAMPLE:
      original = unnormalize(normalized_img)
      plt.imshow(original.permute(1,2,0).numpy())
    """
    x = img_t.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return x.clamp(0, 1)

@torch.no_grad()
def show_dl_batch_original_vs_transformed(dl):
    """
    PURPOSE: Visualize original and transformed (augmented/normalized) images from 
             a PyTorch DataLoader for quality control and augmentation inspection.
    
    USAGE: Called in training notebooks to verify image preprocessing/augmentation 
           are working correctly and not corrupting data.
    
    INPUT: PyTorch DataLoader with batch structure: {image, orig, sample_id}
           - orig: Original image (before transforms)
           - image: Transformed/augmented image (normalized)
    
    OUTPUT: 2-row grid showing: [top row = original, bottom row = transformed]
           for first 3 images per batch
    
    EXAMPLE:
      show_dl_batch_original_vs_transformed(train_dataloader)
    """
    MAX_COLS = 3
    FONT_SIZE = 9

    batch = next(iter(dl))

    if "image" not in batch or "sample_id" not in batch:
        raise KeyError("Batch must contain keys: 'image' and 'sample_id'.")

    img = batch["image"]                 # [B,3,H,W] normalized
    ids = batch["sample_id"]             # list length B (or tensor/np)
    orig = batch.get("orig", None)       # list length B OR [B,3,h,w]

    if orig is None:
        raise ValueError("Batch has no 'orig'. Create dataset with return_original=True and collate it into 'orig'.")

    # make ids a python list of strings
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    ids = [str(x) for x in ids]

    B = img.shape[0]

    # helper: get original tensor [3,h,w] for index i
    def get_orig(i):
        if torch.is_tensor(orig):
            return orig[i]
        return orig[i]

    plt.rcParams.update({"font.size": FONT_SIZE})

    for start in range(0, B, MAX_COLS):
        end = min(start + MAX_COLS, B)
        m = end - start

        fig, axes = plt.subplots(2, m, figsize=(6*m, 7))
        if m == 1:
            axes = np.array(axes).reshape(2, 1)

        for j, i in enumerate(range(start, end)):
            # top: original
            o = get_orig(i).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            axes[0, j].imshow(o)
            axes[0, j].set_title(ids[i], fontsize=FONT_SIZE)
            axes[0, j].axis("on")

            # bottom: transformed (unnormalized for display)
            t = unnormalize(img[i]).permute(1, 2, 0).numpy()
            axes[1, j].imshow(t)
            axes[1, j].set_title("transformed", fontsize=FONT_SIZE)
            axes[1, j].axis("on")

        plt.tight_layout(pad=0.4)
        plt.show()


# ================================================================================
# SECTION 14: SPECIES EFFECTS & BIOMASS MODELING INSIGHTS
# ================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def species_biomass_summary(
    df: pd.DataFrame,
    *,
    species_col: str = "Species",
    y_cols=("Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"),
    min_n_per_species: int = 5,
    eps: float = 1e-12,
    plot_effect: bool = False,          # optional: bar plot of eta^2 per target
    effect_figsize=(7, 3.5),
):
    """
    PURPOSE: Analyze how strongly Species (categorical variable) explains variance 
             in each biomass target using one-way ANOVA effect size (eta²).
    
    USAGE: Called in EDA to identify if certain species have systematically different 
           biomass levels (e.g., some species are harder to grow/accumulate mass).
    
    INPUT:
      - df: DataFrame with species_col and y_cols
      - y_cols: List of biomass target columns
      - min_n_per_species: Exclude species with < N samples (reduces noise)
      - plot_effect: If True, create bar chart of eta² values
    
    OUTPUT:
      - summary_df: One row per target with eta², overall_mean/std, top/bottom species
      - by_species: Dict mapping target → per-species stats (count, mean, std, median)
    
    METRICS:
      - eta² = SS_between / SS_total (0 to 1, higher = species explains more variance)
        • 0.00-0.05: negligible effect
        • 0.05-0.15: small-moderate effect
        • 0.15-0.30: moderate-strong effect
        • >0.30: strong effect
    
    EXAMPLE:
      summary, by_sp = species_biomass_summary(full_df, y_cols=["Dry_Total_g", "GDM_g"],
                                              plot_effect=True)
      print(summary)  # Which targets are most influenced by Species?
    """
    if isinstance(y_cols, str):
        y_cols = (y_cols,)

    d0 = df.copy()
    d0[species_col] = d0[species_col].astype(str)

    rows = []
    by_species = {}

    for ycol in y_cols:
        d = d0[[species_col, ycol]].dropna().copy()
        if len(d) == 0:
            rows.append({
                "target": ycol,
                "n_total": 0,
                "n_species": 0,
                "eta2": np.nan,
                "overall_mean": np.nan,
                "overall_std": np.nan,
                "top_species": None,
                "top_mean": np.nan,
                "bottom_species": None,
                "bottom_mean": np.nan,
                "delta_top_bottom_mean": np.nan,
            })
            by_species[ycol] = pd.DataFrame()
            continue

        # filter rare species
        counts = d[species_col].value_counts()
        keep = counts[counts >= min_n_per_species].index
        d = d[d[species_col].isin(keep)].copy()

        if len(d) == 0:
            rows.append({
                "target": ycol,
                "n_total": 0,
                "n_species": 0,
                "eta2": np.nan,
                "overall_mean": np.nan,
                "overall_std": np.nan,
                "top_species": None,
                "top_mean": np.nan,
                "bottom_species": None,
                "bottom_mean": np.nan,
                "delta_top_bottom_mean": np.nan,
            })
            by_species[ycol] = pd.DataFrame()
            continue

        # per-species stats
        g = d.groupby(species_col)[ycol]
        stats = g.agg(count="count", mean="mean", std="std", median="median").sort_values("count", ascending=False)
        by_species[ycol] = stats

        y = d[ycol].astype(float).to_numpy()
        ybar = float(np.mean(y))
        ss_tot = float(np.sum((y - ybar) ** 2))

        # SS_between = sum n_g * (mean_g - overall_mean)^2
        ss_between = float(np.sum(stats["count"].to_numpy() * (stats["mean"].to_numpy() - ybar) ** 2))
        eta2 = ss_between / (ss_tot + eps)

        # top/bottom by mean
        stats_by_mean = stats.sort_values("mean", ascending=False)
        top_species = stats_by_mean.index[0]
        top_mean = float(stats_by_mean.iloc[0]["mean"])
        bottom_species = stats_by_mean.index[-1]
        bottom_mean = float(stats_by_mean.iloc[-1]["mean"])

        rows.append({
            "target": ycol,
            "n_total": int(len(d)),
            "n_species": int(stats.shape[0]),
            "eta2": float(eta2),
            "overall_mean": float(np.mean(y)),
            "overall_std": float(np.std(y)),
            "top_species": str(top_species),
            "top_mean": top_mean,
            "bottom_species": str(bottom_species),
            "bottom_mean": bottom_mean,
            "delta_top_bottom_mean": float(top_mean - bottom_mean),
        })

    summary_df = pd.DataFrame(rows).set_index("target")

    # optional: plot eta^2 bar chart (like a quick "species effect" view)
    if plot_effect:
        eff = summary_df["eta2"].replace([np.inf, -np.inf], np.nan)
        fig, ax = plt.subplots(figsize=effect_figsize)
        ax.bar(eff.index.astype(str), eff.to_numpy())
        ax.set_title("Species effect per target (eta^2)")
        ax.set_ylabel("eta^2")
        ax.set_ylim(0, min(1.0, max(0.05, np.nanmax(eff.to_numpy()) + 0.05)) if np.isfinite(np.nanmax(eff.to_numpy())) else 1.0)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.2)
        plt.tight_layout()
        plt.show()

    return summary_df, by_species

def scatter_species_vs_height(
    df,
    *,
    height_col="Height_Ave_cm",
    species_col="Species",
    max_points=5000,
    jitter=0.18,
    point_size=10,
    alpha=0.6,
    legend_ncol=6,
    legend_fontsize=9,
    title="Species vs Height",
):
    """
    PURPOSE: Create scatter plot with jitter showing Height distribution across Species 
             (to visualize which species grow taller/shorter).
    
    USAGE: Called in EDA to spot-check for known biological patterns or data collection biases.
    
    INPUT:
      - df: DataFrame
      - height_col: Column with height measurements
      - species_col: Column with species labels
      - max_points: Downsample if > max_points
      - jitter: Add small random noise on Y (species axis) for visibility
    
    OUTPUT: Scatter plot
    
    EXAMPLE:
      scatter_species_vs_height(full_df, height_col="Height_Ave_cm", 
                               species_col="Species", max_points=2000)
    """
    d = df[[height_col, species_col]].dropna().copy()

    # optional subsample
    if max_points is not None and len(d) > max_points:
        d = d.sample(n=max_points, random_state=42)

    # stable ordering by frequency (or alphabetic if you prefer)
    species = d[species_col].astype(str)
    order = species.value_counts().index.tolist()
    idx = {sp: i for i, sp in enumerate(order)}

    x = d[height_col].astype(float).to_numpy()
    sp = species.to_numpy()

    # numeric y + jitter
    y0 = np.array([idx[s] for s in sp], dtype=float)
    y = y0 + (np.random.rand(len(y0)) - 0.5) * 2 * jitter

    fig, ax = plt.subplots(figsize=(14, max(4, 0.35 * len(order))))

    # plot one series per species so we can have a legend
    for s in order:
        m = (sp == s)
        ax.scatter(x[m], y[m], s=point_size, alpha=alpha, label=s)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel(height_col)
    ax.set_ylabel(species_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    # legend below
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        frameon=True,
        title=species_col,
    )

    plt.tight_layout()
    plt.show()


def plot_target_vs_height_ndvi_species(
    df: pd.DataFrame,
    target_col: str = "Dry_Total_g",
    height_col: str = "Height_Ave_cm",
    ndvi_col: str = "Pre_GSHH_NDVI",
    species_col: str = "Species",
    sample: int | None = None,          # e.g., 2000 to speed up plotting
    alpha: float = 0.7,
    s: float = 18,
    jitter: float = 0.15,               # y-jitter for species plot
    title_prefix: str | None = None,
    show_corr: bool = True,             # <-- adds Pearson + Spearman on panels 1&2
):
    # ---------- utils ----------
    def _coerce_numeric(x: pd.Series) -> pd.Series:
        return pd.to_numeric(x, errors="coerce")

    def _pearson_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        # Pearson
        r = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else np.nan

        # Spearman: rank both and compute Pearson on ranks
        # (ties handled by average ranks via pandas)
        rx = pd.Series(x).rank(method="average").to_numpy()
        ry = pd.Series(y).rank(method="average").to_numpy()
        rho = float(np.corrcoef(rx, ry)[0, 1]) if len(x) >= 2 else np.nan
        return r, rho

    def _annotate_corr(ax, xvals, yvals):
        if not show_corr:
            return
        r, rho = _pearson_spearman(xvals, yvals)
        ax.text(
            0.02, 0.98,
            f"P (Pearson) = {r:.2f}\nS (Spearman) = {rho:.2f}",
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9),
        )

    # ---------- optional sampling ----------
    dff = df
    if sample is not None and len(df) > sample:
        dff = df.sample(sample, random_state=0)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)

    # ---------- 1) Height vs target ----------
    a = dff[[height_col, target_col]].copy()
    a[height_col] = _coerce_numeric(a[height_col])
    a[target_col] = _coerce_numeric(a[target_col])
    a = a.dropna()

    axes[0].scatter(a[height_col], a[target_col], s=s, alpha=alpha)
    axes[0].set_xlabel(height_col)
    axes[0].set_ylabel(target_col)
    axes[0].set_title(f"{height_col} vs {target_col}")
    _annotate_corr(axes[0], a[height_col].to_numpy(), a[target_col].to_numpy())

    # ---------- 2) NDVI vs target ----------
    b = dff[[ndvi_col, target_col]].copy()
    b[ndvi_col] = _coerce_numeric(b[ndvi_col])
    b[target_col] = _coerce_numeric(b[target_col])
    b = b.dropna()

    axes[1].scatter(b[ndvi_col], b[target_col], s=s, alpha=alpha)
    axes[1].set_xlabel(ndvi_col)
    axes[1].set_ylabel(target_col)
    axes[1].set_title(f"{ndvi_col} vs {target_col}")
    _annotate_corr(axes[1], b[ndvi_col].to_numpy(), b[target_col].to_numpy())

    # ---------- 3) target vs species (colored, more separated colors) ----------
    c = dff[[species_col, target_col]].copy()
    c[target_col] = _coerce_numeric(c[target_col])
    c[species_col] = c[species_col].astype(str)
    c = c.dropna()

    species_order = c[species_col].value_counts().index.tolist()
    sp2y = {sp: i for i, sp in enumerate(species_order)}

    y = c[species_col].map(sp2y).to_numpy(dtype=float)

    if jitter and jitter > 0:
        rng = np.random.default_rng(0)
        y = y + rng.uniform(-jitter, jitter, size=len(y))

    # Build species -> color mapping using tab20 but spread indices to maximize separation
    K = len(species_order)
    cmap = plt.get_cmap("tab20")
    # Spread indices across the colormap range for better separation:
    # e.g., for K=15, picks ~evenly spaced colors in [0, 1]
    color_positions = np.linspace(0, 1, max(K, 2), endpoint=False)[:K]
    sp2color = {sp: cmap(color_positions[i]) for i, sp in enumerate(species_order)}

    colors = c[species_col].map(sp2color).to_list()

    axes[2].scatter(
        c[target_col].to_numpy(),
        y,
        c=colors,
        s=s,
        alpha=alpha,
    )

    axes[2].set_xlabel(target_col)
    axes[2].set_yticks(range(len(species_order)))
    axes[2].set_yticklabels(species_order)
    axes[2].set_title(f"{target_col} vs {species_col} (colored)")

    if title_prefix is not None:
        fig.suptitle(f"{title_prefix} — {target_col}", fontsize=14)
    
    plt.show()