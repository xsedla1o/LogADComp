import argparse
import os
import os.path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


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


def plot_f1_boxplot(model_output_dir, metrics_df, model_name):
    data = metrics_df[metrics_df["split"] == "test"][["f1"]]
    fig, ax = plt.subplots()
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.boxplot(data, tick_labels=[model_name])
    plt.savefig(f"{model_output_dir}/f1.png")
    plt.close()


def plot_split_performance(model_output_dir, metrics_df):
    test_data = metrics_df[metrics_df["split"] == "test"]
    val_data = metrics_df[metrics_df["split"] == "val"]
    train_data = metrics_df[metrics_df["split"] == "train"]

    test_x, test_y = test_data["offset"], test_data["f1"]
    val_x, val_y = val_data["offset"], val_data["f1"]
    train_x, train_y = train_data["offset"], train_data["f1"]

    fig, ax = plt.subplots()
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.plot(test_x, test_y, label="test")
    ax.plot(val_x, val_y, label="val")
    ax.plot(train_x, train_y, label="train")
    ax.set_xticks(test_x)
    ax.set_xlabel("Cross validation offset")
    plt.legend()
    plt.savefig(f"{model_output_dir}/split_performance.png")
    plt.close()


def plot_test_split_performance(output_dir: str, models):
    fig, ax = plt.subplots()
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)

    for model in models:
        model_df = load_model_dfs(f"{output_dir}/{model}")
        if model_df is not None:
            test_data = model_df[model_df["split"] == "test"]
            test_x, test_y = test_data["offset"], test_data["f1"]
            ax.plot(test_x, test_y, label=model)

    ax.set_xlabel("Cross validation offset")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.savefig(f"{output_dir}/test_split_performance.png", bbox_inches="tight")
    plt.close()


def plot_single(
    model_output_dir,
    model_name,
    verbose=False,
):
    metrics_df = load_model_dfs(model_output_dir)
    if metrics_df is None:
        print(f"No data found for {model_name}")
        return

    data = metrics_df[metrics_df["split"] == "test"]
    data.drop(columns=["split"]).to_csv(f"{model_output_dir}/test.csv", index=False)

    if verbose:
        print(data)
        print(data.describe().drop("count"))

    plot_f1_boxplot(model_output_dir, metrics_df, model_name)
    plot_split_performance(model_output_dir, metrics_df)


def plot_all(output_dir, models, verbose=False):
    plot_data: Dict[str, List[float]] = {}
    for model in models:
        model_df = load_model_dfs(f"{output_dir}/{model}")
        if model_df is not None:
            model_data = model_df[model_df["split"] == "test"][["f1"]]
            plot_data[model] = model_data.values.flatten().tolist()

            if verbose:
                data = model_df[model_df["split"] == "test"]
                print(model)
                print(data)
                print(data.describe().drop("count"))

    if not plot_data:
        print(f"No data found for {output_dir}")
        return

    fig, ax = plt.subplots(figsize=(2 + 1.1 * len(plot_data), 5))
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    try:
        ax.boxplot(plot_data.values(), tick_labels=plot_data.keys())
        plt.savefig(f"{output_dir}/f1.png", bbox_inches="tight")
    except ValueError as e:
        print(f"Error plotting boxplot for {output_dir}: {e}")
    plt.close()

    # Save mean and median values to CSV
    mean_data = pd.DataFrame(plot_data).mean(axis=0)
    median_data = pd.DataFrame(plot_data).median(axis=0)
    csv_df = pd.concat([mean_data, median_data], axis=1)
    csv_df.columns = ["mean", "median"]
    csv_df.to_csv(f"{output_dir}/mean_med_f1.csv", index=True)

    # Plot a mean of all models per split
    mean_data = pd.DataFrame(plot_data).mean(axis=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel("Split")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.plot(mean_data.index, mean_data.values, label="Mean")
    ax.set_xticks(mean_data.index)

    plt.savefig(f"{output_dir}/mean_f1.pdf", bbox_inches="tight", dpi=300)
    plt.close()


def recurse(output_dirs, verbose=False):
    for output_dir in output_dirs:
        models = [
            m
            for m in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, m))
        ]
        plot_all(output_dir, sorted(models), verbose=verbose)
        plot_test_split_performance(output_dir, sorted(models))
        for model in models:
            plot_single(
                f"{output_dir}/{model}",
                model,
                verbose=verbose,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dirs",
        nargs="+",
        help="Directories with model output results, for example outputs/${DATASET_ID}/",
        type=str,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
        default=False,
    )
    args = parser.parse_args()

    recurse(args.input_dirs, args.verbose)
