import pandas as pd
import matplotlib.pyplot as plt

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