"""
Script visualizing the difference in event distributions between training and test splits.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

import argparse
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None


def get_split_sums(ec_split: pd.DataFrame, i: int) -> pd.DataFrame:
    sums = pd.DataFrame(
        index=["tr", "va", "te"], columns=ec_split.columns, dtype=float
    ).fillna(0)
    sums = sums.T
    ec_split = ec_split.T

    # Training split
    ixs = list((j + i) % 10 for j in range(5))
    sums["tr"] += ec_split.iloc[:, ixs].sum(axis=1)
    # Validation split
    sums["va"] += ec_split.iloc[:, [(5 + i) % 10]].sum(axis=1)
    # Test split
    ixs = list((6 + j + i) % 10 for j in range(4))
    sums["te"] += ec_split.iloc[:, ixs].sum(axis=1)

    return sums.T


def plot_event_lines(val_df: pd.DataFrame, out_path: Union[Path, str]):
    nonempty = (val_df != 0).any(axis=0)
    tiny = val_df.abs().sum(axis=0) < 0.01
    small = val_df.max(axis=0) < 0.95

    unused = val_df.loc[:, tiny | small]
    val_df = val_df.loc[:, nonempty & ~tiny & ~small]

    fig, ax = plt.subplots(figsize=tuple(map(lambda x: x * 0.95, (6, 4))))
    for idx, column in enumerate(unused.columns):
        unused[column].plot(ax=ax, color="gray", alpha=0.1, label="_nolegend_")

    colormap = plt.colormaps.get_cmap("tab10")
    for idx, column in enumerate(val_df.columns):
        val_df[column].plot(ax=ax, color=colormap(idx), label=column)

    ax.set_xlabel("Fold (CV iteration)")
    ax.set_xticks(range(10))
    ax.set_ylabel("Difference in Normalized Event Counts")
    ax.legend(title="Event", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        help="Dataset split data csv directories",
        nargs="+",
    )
    args = parser.parse_args()

    for data_dir in args.data_dir:
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            continue

        event_counts_normal = pd.read_csv(data_dir / "normal.csv")
        event_counts_anomaly = pd.read_csv(data_dir / "anomaly.csv")

        values = {}
        values_anomaly = {}
        for i in range(10):
            split_normal = get_split_sums(event_counts_normal, i)
            split_anomaly = get_split_sums(event_counts_anomaly, i)

            # Normalize by split ratio
            ratio = np.array([5, 1, 4])
            split_normal = (split_normal.T / ratio).T
            split_anomaly = (split_anomaly.T / ratio).T

            # Normalize by column
            split_normal = split_normal / split_normal.max(axis=0)
            split_anomaly = split_anomaly / split_anomaly.max(axis=0)

            # Normalize by row
            # split_normal = (split_normal.T / split_normal.sum(axis=1)).T.fillna(0)
            # split_anomaly = (split_anomaly.T / split_anomaly.sum(axis=1)).T.fillna(0)

            for col in split_normal.columns:
                if col not in values:
                    values[col] = []

                test = split_normal.loc["te", col]
                train = split_normal.loc["tr", col]
                values[col].append(test - train)

                if col not in values_anomaly:
                    values_anomaly[col] = []
                test = split_anomaly.loc["te", col]
                train = split_anomaly.loc["tr", col]
                values_anomaly[col].append(test - train)

        values = pd.DataFrame(values)
        print((values > 0.95).sum(axis=1))
        plot_event_lines(values, data_dir / "normal_split_ed.pdf")

        values_anomaly = pd.DataFrame(values_anomaly)
        plot_event_lines(values_anomaly, data_dir / "anomaly_split_ed.pdf")
