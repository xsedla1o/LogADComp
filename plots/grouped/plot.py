"""
Plotting script for grouped boxplots of F1 scores across multiple classifiers and datasets.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

import os
import os.path
from collections import defaultdict
from pathlib import Path
from typing import Union, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_missing = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))


def note_missing(model_name, dataset_name):
    """
    Note that a model was missing for a dataset.
    """
    rng = "Shuffled" in dataset_name
    ratio = dataset_name.split("_")[-1]
    dataset = dataset_name.split("_")[0]
    if rng:
        dataset = dataset.replace("Shuffled", "")

    _missing[rng][dataset][ratio].add(model_name)


def load_model_dfs(model_output_dir):
    model_dfs = []
    if not os.path.exists(model_output_dir):
        print(f"Directory {model_output_dir} does not exist")
        return None
    if not list(n for n in os.listdir(model_output_dir) if n.startswith("metrics")):
        print(f"Directory {model_output_dir} does not contain metrics files")
        return None
    for i in range(10):
        offset = i / 10
        metrics_path = f"{model_output_dir}/metrics_{offset}.csv"
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            model_dfs.append(metrics_df)
        else:
            print(f"File {metrics_path} not found")
    if model_dfs:
        return pd.concat(model_dfs).reset_index().drop(columns="index")
    else:
        return None


def plot_grouped_f1_boxplot(
    base_output_dir: str,
    out_path: Union[str, Path],
    datasets: Dict[str, str],
    model_names: list[str],
    order_by_median: bool = False,
):
    """
    Plots a grouped boxplot of test-split F1 scores for multiple classifiers across multiple datasets.

    Parameters
    ----------
    base_output_dir : str
        Root directory containing one subfolder per dataset.
    out_path : str or Path
        Path to save the output plot.
    datasets : mapping of str to str
        Dataset labels and their corresponding subfolder names.
    model_names : list of str
        Names of the model subfolders under each dataset folder.
    order_by_median : bool, default=False
        If True, orders classifiers left→right by their overall median F1.

    """
    records = []
    # Collect all (dataset, classifier, f1) triples
    for label, ds in datasets.items():
        ds_dir = os.path.join(base_output_dir, ds)
        for model in model_names:
            model_dir = os.path.join(ds_dir, model)
            df = load_model_dfs(model_dir)
            if df is None:
                note_missing(model, ds)
                continue
            # only test‐split F1
            f1s = df.loc[df["split"] == "test", "f1"]
            for f in f1s:
                records.append({"dataset": label, "classifier": model, "f1": f})

    plot_df = pd.DataFrame.from_records(records)
    if plot_df.empty:
        print("No data found—check your paths or dataset/model lists.")
        return

    # Determine x-order
    if order_by_median:
        order = (
            plot_df.groupby("classifier")["f1"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        order = model_names

    print(out_path)
    print(plot_df.groupby("classifier")["f1"].median().sort_values(ascending=False))
    print(plot_df.groupby("classifier")["f1"].mean().sort_values(ascending=False))

    # Make the plot
    fig, ax = plt.subplots(figsize=(0.9 * len(model_names) + 2, 5))
    sns.boxplot(
        data=plot_df,
        x="classifier",
        y="f1",
        hue="dataset",
        order=order,
        palette="Set2",
        width=0.7,
        ax=ax,
    )

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

    ax.set_xlabel("Classifier")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    # plt.xticks()
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.legend(title="Dataset", loc="lower left")
    plt.tight_layout()

    # Save and close
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved grouped F1 boxplot to {out_path}")


base_dir = "outputs"
out_path = Path(__file__).parent
classifiers = [
    "NeuralLog",
    "LogRobust",
    "SVM",
    "LogBERT",
    "LogAnomaly",
    "DeepLog",
    "LogCluster",
    "SemPCA",
    "PCA",
]

# Base comparison plot
datasets = {
    "HDFS": "HDFSLogHubShuffled_0.5",
    "BGL": "BGL40Shuffled_0.5",
    "TBird": "TBirdShuffled_0.5",
}
plot_grouped_f1_boxplot(
    base_output_dir=base_dir,
    out_path=out_path / "grouped_f1_boxplot.pdf",
    datasets=datasets,
    model_names=classifiers,
    order_by_median=False,
)

# HDFS case study: train ratio
datasets = {
    "HDFS 0.50": "HDFSLogHubShuffled_0.5",
    "HDFS 0.10": "HDFSLogHubShuffled_0.1",
    "HDFS 0.01": "HDFSLogHubShuffled_0.01",
}
plot_grouped_f1_boxplot(
    base_output_dir=base_dir,
    out_path=out_path / "grouped_f1_hdfs_ratios.pdf",
    datasets=datasets,
    model_names=classifiers,
    order_by_median=False,
)

# BGL case study: train ratio
datasets = {
    "BGL 0.50": "BGL40Shuffled_0.5",
    "BGL 0.10": "BGL40Shuffled_0.1",
    "BGL 0.01": "BGL40Shuffled_0.01",
}
plot_grouped_f1_boxplot(
    base_output_dir=base_dir,
    out_path=out_path / "grouped_f1_bgl_ratios.pdf",
    datasets=datasets,
    model_names=classifiers,
    order_by_median=False,
)

# HDFS case study: Preprocessing
datasets = {
    "HDFS LogHub": "HDFSLogHubShuffled_0.5",
    "HDFS Fixed": "HDFSFixedShuffled_0.5",
    "HDFS Xu": "HDFSShuffled_0.5",
}
plot_grouped_f1_boxplot(
    base_output_dir=base_dir,
    out_path=out_path / "grouped_f1_hdfs_preprocessing.pdf",
    datasets=datasets,
    model_names=classifiers,
    order_by_median=False,
)

# BGL case study: Preprocessing
datasets = {
    "BGL 40l+60s": "BGL40Shuffled_0.5",
    "BGL 120l+60s": "BGL120Shuffled_0.5",
    "BGL 120l+Components": "BGLComponentShuffled_0.5",
}
plot_grouped_f1_boxplot(
    base_output_dir=base_dir,
    out_path=out_path / "grouped_f1_bgl_preprocessing.pdf",
    datasets=datasets,
    model_names=classifiers,
    order_by_median=False,
)

# Shuffling? Sequentially sampled datasets
datasets = {
    "HDFS": "HDFSLogHub_0.5",
    "BGL": "BGL40_0.5",
    "TBird": "TBird_0.5",
}
plot_grouped_f1_boxplot(
    base_output_dir=base_dir,
    out_path=out_path / "grouped_f1_seq.pdf",
    datasets=datasets,
    model_names=classifiers,
    order_by_median=False,
)

# Train ratio reduction without shuffling
datasets = {
    "HDFS 0.50": "HDFSLogHub_0.5",
    "HDFS 0.10": "HDFSLogHub_0.1",
    "HDFS 0.01": "HDFSLogHub_0.01",
}
plot_grouped_f1_boxplot(
    base_output_dir=base_dir,
    out_path=out_path / "grouped_f1_hdfs_seq_ratios.pdf",
    datasets=datasets,
    model_names=classifiers,
    order_by_median=False,
)

for rng, datasets in _missing.items():
    for dataset, ratios in datasets.items():
        for ratio, models in ratios.items():
            print(
                f"{dataset:30s} {('Shuffled' if rng else ''):10s} {ratio:8s}: {models}"
            )
