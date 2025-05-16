"""
Plot the event distribution in a label-aware manner.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

import argparse
from pathlib import Path
from typing import Union

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None


def plot_anomaly_event_distribution(
    event_counts_normal: pd.DataFrame,
    event_counts_anomaly: pd.DataFrame,
    output_dir: Union[Path, str],
    out_name: str,
):
    """
    Plot the event distribution in a label-aware manner.
    two heatmaps are created, one for normal events and one for anomaly events.

    """
    vmin = min(event_counts_normal.min().min(), event_counts_anomaly.min().min())
    vmax = max(event_counts_normal.max().max(), event_counts_anomaly.max().max())
    vmin = max(1, vmin)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(9, 6.5))
    ax1 = fig.add_subplot(211)
    im1 = ax1.imshow(
        event_counts_normal, cmap="plasma", interpolation="nearest", norm=norm
    )
    ax1.set_title("Normal Sessions")
    ax1.set_xlabel("Events")
    ax1.set_ylabel("Data Segments")
    if event_counts_normal.shape[1] < 100:
        ax1.set_yticks(range(event_counts_normal.shape[0]))
        ax1.set_xticks(range(event_counts_normal.shape[1]))

    ax2 = fig.add_subplot(212)
    ax2.imshow(event_counts_anomaly, cmap="plasma", interpolation="nearest", norm=norm)
    ax2.set_title("Anomalous Sessions")
    ax2.set_xlabel("Events")
    ax2.set_ylabel("Data Segments")
    if event_counts_normal.shape[1] < 100:
        ax2.set_yticks(range(event_counts_anomaly.shape[0]))
        ax2.set_xticks(range(event_counts_anomaly.shape[1]))

    cbar = fig.colorbar(
        im1, ax=[ax1, ax2], orientation="vertical", fraction=0.02, pad=0.04
    )
    cbar.set_label("Event Counts")

    plt.savefig(f"{output_dir}/{out_name}", dpi=300, bbox_inches="tight")
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
        plot_anomaly_event_distribution(
            event_counts_normal, event_counts_anomaly, data_dir, "normal_anomaly"
        )

        cnt = 5
        uncommon_events = (event_counts_normal < cnt).sum(axis=0) > 8
        uncommon_events &= (event_counts_anomaly < cnt).sum(axis=0) > 8
        print(uncommon_events.sum())
        event_counts_normal = event_counts_normal.loc[:, ~uncommon_events]
        event_counts_anomaly = event_counts_anomaly.loc[:, ~uncommon_events]
        plot_anomaly_event_distribution(
            event_counts_normal,
            event_counts_anomaly,
            data_dir,
            "normal_anomaly_uncommon.pdf",
        )
