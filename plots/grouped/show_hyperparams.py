"""
Script to show hyperparameters of models across datasets.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

import json
import os
import os.path
from collections import defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd

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


def load_model_hyperparams(model_output_dir):
    if not os.path.exists(model_output_dir):
        print(f"Directory {model_output_dir} does not exist")
        return None
    if not list(
        n for n in os.listdir(model_output_dir) if n.endswith("hyperparameters.json")
    ):
        print(f"Directory {model_output_dir} does not contain hyperparam files")
        return None
    hyper = os.path.join(model_output_dir, "hyperparameters.json")
    if os.path.exists(hyper):
        with open(hyper) as f:
            hyper_data = json.load(f)
    else:
        print(f"File {hyper} not found")
        hyper_data = None

    train_hyper = os.path.join(model_output_dir, "train_hyperparameters.json")
    if os.path.exists(train_hyper):
        with open(train_hyper) as f:
            train_hyper_data = json.load(f)
    else:
        train_hyper_data = None

    if train_hyper_data is None:
        train_hyper_data = hyper_data
    if hyper_data == {}:
        hyper_data = train_hyper_data

    return hyper_data


def show_hyperparms(
    base_output_dir: str,
    datasets: Dict[str, str],
    model_names: list[str],
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
    """
    records = []
    # Collect all (dataset, classifier, f1) triples
    for label, ds in datasets.items():
        ds_dir = os.path.join(base_output_dir, ds)
        for model in model_names:
            model_dir = os.path.join(ds_dir, model)
            params = load_model_hyperparams(model_dir)
            if params is None or not params:
                continue
            records.append({"name": model, "dataset": label, **params})

    if len(records) == 0:
        print("No data found—check your paths or dataset/model lists.")
        return

    df = pd.DataFrame.from_records(records).set_index(["name", "dataset"]).sort_index()
    for name in df.index.get_level_values(0).unique():
        if name in ["LogRobust", "LogAnomaly", "DeepLog"]:
            df_subset = df.loc[name, :].reset_index().dropna(axis=1)
            df_subset["name"] = name

            # If present, rename "learning_rate" to "lr" and "learning_rate_decay" to "lr_decay"
            if "learning_rate" in df_subset.columns:
                df_subset.rename(columns={"learning_rate": "lr"}, inplace=True)
            if "learning_rate_decay" in df_subset.columns:
                df_subset.rename(
                    columns={"learning_rate_decay": "lr_decay"}, inplace=True
                )
            if "num_epochs" in df_subset.columns:
                df_subset.rename(columns={"num_epochs": "epochs"}, inplace=True)

            for x in ["hidden_size", "num_layers", "batch_size", "epochs"]:
                if x in df_subset.columns:
                    df_subset[x] = df_subset[x].astype(int)

            if "lr_decay" not in df_subset.columns:
                df_subset["lr_decay"] = pd.NA

            df_subset = df_subset[
                [
                    "name",
                    "dataset",
                    "hidden_size",
                    "num_layers",
                    "epochs",
                    "batch_size",
                    "lr",
                    "lr_decay",
                ]
            ]

            df_subset = df_subset.set_index(["name", "dataset"])
            ltx_str = df_subset.to_latex(float_format="%.3f")
            ltx_str = ltx_str.replace("_", "\\_")
            ltx_str = ltx_str.replace("\\multirow[t]", "\\multirow[c]")
            ltx_str = ltx_str.replace("NaN", "-")
            ltx_str = ltx_str.replace("dataset", name)
            lines = ltx_str.splitlines()
            lines = [line for line in lines if not line.startswith("name")]
            print("\n".join(lines) + "\n")
        else:
            # Reset the index and drop columns with NaN values
            df_subset = df.loc[name, :].reset_index().dropna(axis=1)
            df_subset = df_subset.T.reset_index()

            # rename columns based on the first row
            df_subset.rename(columns=df_subset.iloc[0], inplace=True)

            df_subset["name"] = name
            # df_subset.rename(columns={"dataset": "params"}, inplace=True)
            # Check if 'name' exists as a column before calling set_index
            df_subset = df_subset.loc[1:, :]
            df_subset = df_subset.set_index(["name", "dataset"])

            # Convert to LaTeX format
            ltx_str = df_subset.to_latex(float_format="%.3f")
            ltx_str = ltx_str.replace("_", "\\_")
            ltx_str = ltx_str.replace("dataset", "Dataset")
            ltx_str = ltx_str.replace("\\cline{1-", "\\cline{2-")
            ltx_str = ltx_str.replace("\\multirow[t]", "\\multirow[c]")
            lines = ltx_str.splitlines()
            lines = [line for line in lines if not line.startswith("name")]
            print("\n".join(lines) + "\n")


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
show_hyperparms(base_output_dir=base_dir, datasets=datasets, model_names=classifiers)

for rng, datasets in _missing.items():
    for dataset, ratios in datasets.items():
        for ratio, models in ratios.items():
            print(
                f"{dataset:30s} {('Shuffled' if rng else ''):10s} {ratio:8s}: {models}"
            )
