"""
Plot session length histograms.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm


def secs_to_str(secs):
    """
    Convert seconds to a string in the format HH:MM:SS.
    """
    parts = []
    days = secs // 86400
    secs %= 86400
    if days > 0:
        parts.append(f"{days:2}d")

    hours = secs // 3600
    secs %= 3600
    if hours > 0:
        parts.append(f"{hours:2}h")

    mins = secs // 60
    secs %= 60
    if mins > 0:
        parts.append(f"{mins:2}m")

    if secs > 0 or len(parts) == 0:
        parts.append(f"{secs:2}s")
    return ":".join(parts)


def plot_session_length(ax, data, norm, labely=True):
    ax.set_xlabel("Lines")
    if labely:
        ax.set_ylabel("Time")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index.map(secs_to_str))
    return ax.imshow(data, cmap="plasma", interpolation="nearest", norm=norm)


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
        hist_all = pd.read_csv(data_dir / "hist2d.csv", index_col=0, header=0)
        hist_anomaly = pd.read_csv(
            data_dir / "hist2d_anomalies.csv", index_col=0, header=0
        )

        # Drop empty columns
        nonempty = (hist_all != 0).any(axis=0) | (hist_anomaly != 0).any(axis=0)
        hist_all = hist_all.loc[:, nonempty]
        hist_anomaly = hist_anomaly.loc[:, nonempty]

        # Drop empty rows
        nonempty = (hist_all != 0).any(axis=1) | (hist_anomaly != 0).any(axis=1)
        hist_all = hist_all.loc[nonempty, :]
        hist_anomaly = hist_anomaly.loc[nonempty, :]

        hist_normal = hist_all - hist_anomaly

        # Calculate global min and max for LogNorm
        mins, maxs = [], []
        for hist in [hist_all, hist_normal, hist_anomaly]:
            tmp = hist[hist > 0]
            mins.append(tmp.min().min())
            maxs.append(tmp.max().max())
        vmin, vmax = min(mins), max(maxs)
        vmin = max(vmin, 0.01)  # Avoid log(0)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        # Draw heatmap
        fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

        axs[0].set_title("All Sequences")
        im = plot_session_length(axs[0], hist_all, norm)
        axs[1].set_title("Normal Sequences")
        plot_session_length(axs[1], hist_normal, norm, labely=False)
        axs[2].set_title("Anomalous Sequences")
        plot_session_length(axs[2], hist_anomaly, norm, labely=False)
        fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.02, pad=0.04)

        plt.savefig(f"{data_dir}/hist2d.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        # Draw only all sessions histogram
        data = hist_all.T
        fig, ax = plt.subplots(figsize=(data.shape[1] * 1.5, 4))
        ax.set_title("All Sequences")
        ax.set_xlabel("Time")
        ax.set_ylabel("Lines")
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(data.columns.map(secs_to_str))
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels(data.index)
        im = ax.imshow(data, cmap="plasma", interpolation="nearest", norm=norm)
        fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)

        plt.savefig(f"{data_dir}/hist2d_all.pdf", dpi=300, bbox_inches="tight")
        plt.close()
