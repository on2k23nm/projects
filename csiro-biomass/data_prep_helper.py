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

from matplotlib.lines import Line2D
import matplotlib as mpl

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

all_cols = ["Pre_GSHH_NDVI", "Height_Ave_cm",
        "Dry_Clover_g", "Dry_Dead_g",
        "Dry_Green_g", "Dry_Total_g", "GDM_g"]


def check_biomass_nonnegative(
    df: pd.DataFrame,
    cols=("Dry_Dead_g", "Dry_Green_g", "Dry_Clover_g", "Dry_Total_g", "GDM_g"),
):
    """
    Row-wise check: all specified biomass targets should be >= 0.
    Returns:
      summary: dict
      bad_rows: DataFrame (only rows where any target < 0)
    """
    bad_mask = (df.loc[:, cols] < 0).any(axis=1)
    bad_rows = df.loc[bad_mask, list(cols)].copy()

    summary = {
        "n_rows": int(len(df)),
        "n_bad": int(bad_mask.sum()),
        "pct_bad": float(bad_mask.mean() * 100.0),
    }
    return summary, bad_rows


def check_dry_total_sumlike(
    df: pd.DataFrame,
    total_col: str = "Dry_Total_g",
    parts: tuple = ("Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"),
):
    """
    Checks:
      Dry_Total_g >= Dry_Green_g + Dry_Dead_g + Dry_Clover_g

    Returns:
      summary: dict
      bad_rows: rows where condition is violated
      out: df copy with helper columns
    """
    d = df.copy()
    d["parts_sum"] = d[list(parts)].sum(axis=1)
    d["diff"] = d[total_col] - d["parts_sum"]  # should be >= 0

    d["passes_check"] = d["diff"] >= 0

    bad_rows = d.loc[~d["passes_check"], [total_col, *parts, "parts_sum", "diff"]].sort_values("diff")

    summary = {
        "n_rows": int(len(d)),
        "n_pass": int(d["passes_check"].sum()),
        "n_fail": int((~d["passes_check"]).sum()),
        "pct_fail": float((~d["passes_check"]).mean() * 100.0),
        "min_diff": float(d["diff"].min()),
        "median_diff": float(d["diff"].median()),
    }

    return summary, bad_rows, d

def plot_targets_and_log1p(
    df,
    target_cols=("Dry_Dead_g", "Dry_Green_g", "Dry_Clover_g", "Dry_Total_g", "GDM_g"),
    bins=40,
):
    """
    Row 1: histograms of raw targets
    Row 2: histograms of log1p(targets)
    """
    cols = list(target_cols)
    n = len(cols)

    fig, axes = plt.subplots(2, n, figsize=(4*n, 7), constrained_layout=True)

    for j, c in enumerate(cols):
        x = df[c].dropna().astype(float).to_numpy()

        # ---- row 1: raw ----
        ax = axes[0, j]
        ax.hist(x, bins=bins)
        ax.set_title(f"RAW {c}")
        ax.set_xlabel(c)
        ax.set_ylabel("count")

        # ---- row 2: log1p ----
        ax = axes[1, j]
        ax.hist(np.log1p(np.clip(x, 0, None)), bins=bins)
        ax.set_title(f"log1p({c})")
        ax.set_xlabel(f"log1p({c})")
        ax.set_ylabel("count")

    return fig, axes

def species_distribution_table(
    df: pd.DataFrame,
    splits: dict,                 # {"train": idx, "valid": idx, "test": idx} (np arrays / lists)
    species_col: str = "Species",
    split_order=("train", "valid", "test"),
    include_full: bool = False,
    sort_by: str = "full_count",  # "full_count" | "train_count" | "species"
) -> pd.DataFrame:
    """
    Returns ONE combined table:
      index = Species
      columns = <split>_count, <split>_pct

    Example columns:
      train_count, train_pct, valid_count, valid_pct, test_count, test_pct
    """
    def _one(name: str, idx: np.ndarray) -> pd.DataFrame:
        s = df.iloc[idx][species_col].astype(str).fillna("NA")
        vc = s.value_counts()
        out = pd.DataFrame({f"{name}_count": vc})
        out[f"{name}_pct"] = (vc / vc.sum() * 100.0)
        return out

    frames = []

    if include_full:
        frames.append(_one("full", np.arange(len(df), dtype=np.int64)))

    for k in split_order:
        frames.append(_one(k, np.asarray(splits[k])))

    tab = pd.concat(frames, axis=1).fillna(0)

    # nice formatting
    pct_cols = [c for c in tab.columns if c.endswith("_pct")]
    tab[pct_cols] = tab[pct_cols].round(2)

    # ints for counts
    count_cols = [c for c in tab.columns if c.endswith("_count")]
    tab[count_cols] = tab[count_cols].astype(int)

    # ordering
    if sort_by == "species":
        tab = tab.sort_index()
    elif sort_by in tab.columns:
        tab = tab.sort_values(sort_by, ascending=False)
    else:
        # fallback: sort by first available count col
        tab = tab.sort_values(count_cols[0], ascending=False)

    tab.index.name = species_col
    return tab


