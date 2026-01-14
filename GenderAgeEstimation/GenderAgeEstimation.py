from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from dataclasses import dataclass
from sklearn.model_selection import train_test_split


def plot_race_composition_with_labels(
    dfs: dict,
    race_col: str = "race",
    race_order: list | None = None,
    rename_map: dict | None = None,
    title: str = "Racial compositions",
    figsize=(10, 4),
    min_pct_to_label: float = 3.0,   # skip tiny segments
):
    """
    dfs: {"Train": train_df, "Valid": valid_df, ...}
    Plots 100% stacked bars and labels each segment as:  'x% (n)'
    """

    # Counts table: rows = dataset name, cols = race
    rows = []
    for name, df in dfs.items():
        s = df[race_col].astype(str)
        if rename_map is not None:
            s = s.replace(rename_map)

        counts = s.value_counts(dropna=False)
        counts.name = name
        rows.append(counts)

    count_df = pd.DataFrame(rows).fillna(0)

    # Column order
    if race_order is not None:
        for r in race_order:
            if r not in count_df.columns:
                count_df[r] = 0
        count_df = count_df[race_order]
    else:
        count_df = count_df[count_df.sum(axis=0).sort_values(ascending=False).index]

    # Percent table
    pct_df = count_df.div(count_df.sum(axis=1), axis=0) * 100.0

    # Plot
    ax = pct_df.plot(kind="bar", stacked=True, figsize=figsize)
    ax.set_ylabel("Ratio (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.set_xlabel("")
    plt.xticks(rotation=0)

    # Legend on right
    ax.legend(title="", bbox_to_anchor=(1.02, 0.5), loc="center left")

    # Add labels inside each segment: "x% (n)"
    # ax.containers correspond to columns in pct_df (same order)
    for col_idx, (container, race_name) in enumerate(zip(ax.containers, pct_df.columns)):
        for row_idx, patch in enumerate(container):
            h = patch.get_height()          # percent height
            if h < min_pct_to_label:
                continue

            n = int(count_df.iloc[row_idx, col_idx])
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_y() + h / 2

            label = f"{h:.0f}% ({n})"
            ax.text(x, y, label, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()

    return pct_df, count_df

class CSVMerger:
    def __init__(self, root: str | Path, *, seed: int = 42):
        self.root = Path(root)
        self.seed = seed

    def _read(self, csv_path: Path, split: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df["split"] = split
        return df

    def build_full(self, train_csv: str, val_csv: str, out_csv: str = "full.csv") -> Path:
        train_path = self.root / train_csv
        val_path   = self.root / val_csv
        assert train_path.exists(), f"Missing: {train_path}"
        assert val_path.exists(), f"Missing: {val_path}"

        train_df = self._read(train_path, "train")
        val_df   = self._read(val_path, "val")

        full_df = pd.concat([train_df, val_df], ignore_index=True)

        # random shuffle (reproducible)
        full_df = full_df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        out_path = self.root / out_csv
        full_df.to_csv(out_path, index=False)
        
        return out_path

# # ---------- USAGE ----------
# root = "/path/to/dataset"  # <- change this
# out = CSVMerger(root, seed=42).build_full("train.csv", "val.csv", "full.csv")
# print("Wrote:", out)

def plot_agebin_race_gender_distributions(
    df: pd.DataFrame,
    *,
    age_col: str = "age",
    race_col: str = "race",
    gender_col: str = "gender",
    normalize: bool = False,  # if True: show counts + % labels + % right y-axis
    figsize: tuple[int, int] = (18, 4),
):
    def _counts(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna("NA").str.strip().value_counts(dropna=False)

    def _colors(n: int, cmap_name: str = "tab20"):
        cmap = plt.get_cmap(cmap_name)
        return [cmap(i % cmap.N) for i in range(n)]

    def _annotate_pct(ax, counts: pd.Series):
        total = float(counts.sum()) if counts.sum() else 1.0
        for i, c in enumerate(counts.values):
            pct = 100.0 * (float(c) / total)
            ax.annotate(
                f"{int(c)}\n{pct:.1f}%",
                xy=(i, float(c)),
                xytext=(0, 4),                 # 4 points above the bar
                textcoords="offset points",
                ha="center",
                va="bottom",
                clip_on=False,
            )

    def _add_pct_axis(ax, total_count: int):
        ax2 = ax.twinx()
        y0, y1 = ax.get_ylim()
        denom = max(total_count, 1)
        ax2.set_ylim(100.0 * y0 / denom, 100.0 * y1 / denom)
        ax2.set_ylabel("%")
        return ax2

    def _add_headroom(ax, y_max: float, headroom: float = 0.18):
        if y_max <= 0:
            return
        ax.set_ylim(0, y_max * (1.0 + headroom))

    # --- Age bin ordering ---
    age_order = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
    age_s = df[age_col].astype("string").fillna("NA").str.strip()
    age_counts = age_s.value_counts()
    remaining = [b for b in age_counts.index.tolist() if b not in age_order]
    age_counts = age_counts.reindex(age_order + remaining).dropna()

    race_counts = _counts(df[race_col])
    gender_counts = _counts(df[gender_col])

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1) Age bins
    axes[0].bar(age_counts.index.astype(str), age_counts.values, color=_colors(len(age_counts)))
    axes[0].set_title("Age-bin distribution", pad=14)
    axes[0].set_xlabel(age_col)
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", labelrotation=45)
    plt.setp(axes[0].get_xticklabels(), ha="right")
    _add_headroom(axes[0], float(age_counts.max()))
    if normalize:
        _annotate_pct(axes[0], age_counts)
        _add_pct_axis(axes[0], int(age_counts.sum()))

    # 2) Race
    axes[1].bar(race_counts.index.astype(str), race_counts.values, color=_colors(len(race_counts)))
    axes[1].set_title("Race distribution", pad=14)
    axes[1].set_xlabel(race_col)
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", labelrotation=45)
    plt.setp(axes[1].get_xticklabels(), ha="right")
    _add_headroom(axes[1], float(race_counts.max()))
    if normalize:
        _annotate_pct(axes[1], race_counts)
        _add_pct_axis(axes[1], int(race_counts.sum()))

    # 3) Gender
    axes[2].bar(gender_counts.index.astype(str), gender_counts.values, color=_colors(len(gender_counts)))
    axes[2].set_title("Gender distribution", pad=14)
    axes[2].set_xlabel(gender_col)
    axes[2].set_ylabel("Count")
    axes[2].tick_params(axis="x", labelrotation=45)
    plt.setp(axes[2].get_xticklabels(), ha="right")
    _add_headroom(axes[2], float(gender_counts.max()))
    if normalize:
        _annotate_pct(axes[2], gender_counts)
        _add_pct_axis(axes[2], int(gender_counts.sum()))

    fig.tight_layout()
    return fig, axes


@dataclass
class FairFaceSplitter:
    seed: int = 42

    def _make_key(self, df: pd.DataFrame, cols: list[str]) -> pd.Series:
        return df[cols].astype("string").fillna("NA").agg("|".join, axis=1)

    def _stratify_key(self, df: pd.DataFrame) -> pd.Series | None:
        # try strongest stratification first, then fall back
        candidates = [
            ["age", "gender", "race"],
            ["gender", "race"],
            ["race"],
            ["gender"],
        ]
        for cols in candidates:
            if all(c in df.columns for c in cols):
                key = self._make_key(df, cols)
                if key.value_counts().min() >= 2:
                    return key
        return None

    def split(
        self,
        df: pd.DataFrame,
        *,
        test_size: float = 0.15,
        val_size: float = 0.15,
        out_dir: str | Path = ".",
        prefix: str = "fairface_",
        split_col: str = "split",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits df into train/val/test.
        - test_size: fraction of full data to allocate to test
        - val_size: fraction of full data to allocate to val
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df = df.copy().reset_index(drop=True)

        # 1) full -> (trainval, test)
        strat = self._stratify_key(df)
        trainval_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=self.seed,
            shuffle=True,
            stratify=strat,
        )

        # 2) trainval -> (train, val)
        trainval_df = trainval_df.reset_index(drop=True)
        strat2 = self._stratify_key(trainval_df)
        val_frac_of_trainval = val_size / (1.0 - test_size)

        train_df, val_df = train_test_split(
            trainval_df,
            test_size=val_frac_of_trainval,
            random_state=self.seed,
            shuffle=True,
            stratify=strat2,
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df[split_col] = "train"
        val_df[split_col] = "val"
        test_df[split_col] = "test"

        train_df.to_csv(out_dir / f"{prefix}train.csv", index=False)
        val_df.to_csv(out_dir / f"{prefix}val.csv", index=False)
        test_df.to_csv(out_dir / f"{prefix}test.csv", index=False)

        return train_df, val_df, test_df
    
    
class SplitLeakageChecker:
    @staticmethod
    def assert_file_disjoint(df: pd.DataFrame, *, file_col: str = "file", split_col: str = "split") -> None:
        # each file should belong to exactly one split
        splits_per_file = df.groupby(file_col)[split_col].nunique()
        bad = splits_per_file[splits_per_file > 1]
        assert len(bad) == 0, f"Leakage: {len(bad)} files appear in multiple splits. Example:\n{bad.head(10)}"

    @staticmethod
    def assert_no_duplicate_files(df: pd.DataFrame, *, file_col: str = "file") -> None:
        dup = df[file_col].duplicated().sum()
        assert dup == 0, f"Found {dup} duplicate file rows in the dataframe."
