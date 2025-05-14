"""
Show anomalies per second as a histogram.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        help="Dataset split data csv directories",
        nargs="+",
    )
    args = parser.parse_args()

    hists = {}
    for data_dir in args.data_dir:
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            continue
        hist = pd.read_csv(data_dir / "anomaly_per_session.csv", index_col=0, header=0)
        hist = hist["0"]
        hists[data_dir.name] = hist

        plt.figure(figsize=(5, 4))
        plt.bar(
            hist.index.astype(int),
            hist.values,
            color="blue",
            alpha=0.7,
            edgecolor="black",
        )
        plt.title("Anomalies Per Session")
        plt.xlabel("Anomalies")
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.75)

        plt.savefig(data_dir / "anomaly_per_session.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Total anomalies in {data_dir}: {hist.sum()}")
        hist2d = pd.read_csv(data_dir / "hist2d.csv", index_col=0, header=0)
        print("Total sessions in hist2d:", hist2d.sum().sum())
        hist2d = pd.read_csv(data_dir / "hist2d_anomalies.csv", index_col=0, header=0)
        print("Total sessions in hist2d anomalies:", hist2d.sum().sum())

        # Get the mean and std of the anomaly count per session
        count_idx = hist.index.to_numpy().astype(int)
        mean = (count_idx * hist.values).sum() / hist.values.sum()
        std = ((count_idx - mean) ** 2 * hist.values).sum() / hist.values.sum()
        std = std**0.5
        print(f"Mean anomalies per session in {data_dir}: {mean:.2f} +- {std:.2f}")
