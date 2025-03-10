import argparse
import os.path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import tomli

from adapters import model_adapters
from dataloader import dataloaders


def get_dataset_paths(dir_config, dataset):
    dataset_dir = dir_config["datasets"]

    return {
        "dataset_dir": dataset_dir,
        "dataset": f"{dataset_dir}/{dataset}/{dataset}.log",
        "labels": f"{dataset_dir}/{dataset}/anomaly_label.csv",
        "processed_labels": f"{dataset_dir}/{dataset}/label.csv",
        "embeddings": f"{dataset_dir}/glove.6B.300d.txt",
        "loglizer_seqs": f"{dataset_dir}/{dataset}/{dataset}.seqs.csv",
    }


def get_model_paths(config_dict, dataset, model_name):
    train_ratio = 0.5
    output_dir = f"{config_dict['outputs']}/{dataset}_{train_ratio}/{model_name}"
    return {
        "output_dir": output_dir,
        "trials_output": f"{output_dir}/trials.csv",
        "hyperparameters": f"{output_dir}/hyperparameters.json",
    }


def load_model_dfs(config_dict):
    model_dfs = []
    for i in range(10):
        offset = i / 10
        metrics_path = f"{config_dict['output_dir']}/metrics_{offset}.csv"
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            model_dfs.append(metrics_df)
        else:
            print(f"File {metrics_path} not found")
    if model_dfs:
        return pd.concat(model_dfs).reset_index().drop(columns="index")
    else:
        return None


def plot_single(paths_conf, dataset, model_name):
    config_dict = get_model_paths(paths_conf, dataset, model_name)

    metrics_df = load_model_dfs(config_dict)
    if metrics_df is None:
        print(f"No data found for {model_name}")
        return

    data = metrics_df[metrics_df["split"] == "test"][["f1"]]
    fig, ax = plt.subplots()
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.boxplot(data, tick_labels=[model_name])
    plt.savefig(f"{config_dict['output_dir']}/f1.png")

    data = metrics_df[metrics_df["split"] == "test"]
    print(data)
    print(data.describe().drop("count"))


def plot_all(config: dict, dataset):
    d_name = dataset
    train_ratio = 0.5
    plot_dir = f"{config['outputs']}/{d_name}_{train_ratio}"

    plot_data: Dict[str, List[float]] = {}
    for model in model_adapters:
        config_dict = get_model_paths(config, d_name, model)
        model_df = load_model_dfs(config_dict)
        if model_df is not None:
            model_data = model_df[model_df["split"] == "test"][["f1"]]
            plot_data[model] = model_data.values.flatten().tolist()

            data = model_df[model_df["split"] == "test"]
            print(data)
            print(data.describe().drop("count"))

    fig, ax = plt.subplots()
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.boxplot(plot_data.values(), tick_labels=plot_data.keys())
    plt.savefig(f"{plot_dir}/f1.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset to use for anomaly detection, choices: " + ", ".join(dataloaders),
        choices=list(dataloaders),
    )
    parser.add_argument(
        "model",
        nargs="?",
        type=str,
        help="Model to use for anomaly detection, choices: "
        + ", ".join(model_adapters)
        + ", default=all",
        choices=list(model_adapters) + ["all"],
        default="all",
    )
    args = parser.parse_args()

    with open("paths.toml", "rb") as f:
        dir_config = tomli.load(f)

    config_dict = get_dataset_paths(dir_config, args.dataset) | dir_config

    if args.model == "all":
        plot_all(config_dict, args.dataset)
    else:
        plot_single(config_dict, args.dataset, args.model)
