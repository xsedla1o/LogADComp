import os.path

import matplotlib.pyplot as plt
import pandas as pd
import tomli

if __name__ == "__main__":
    with open("paths.toml", "rb") as f:
        dir_config = tomli.load(f)

    d_name = "HDFS"
    model = "SVM"
    n_trials = 20
    train_ratio = 0.5
    dataset_dir = dir_config["datasets"]
    output_dir = f"{dir_config['outputs']}/{d_name}_{train_ratio}/{model}"

    config_dict = {
        "dataset_dir": dataset_dir,
        "dataset": f"{dataset_dir}/{d_name}/{d_name}.log",
        "labels": f"{dataset_dir}/{d_name}/anomaly_label.csv",
        "processed_labels": f"{dataset_dir}/{d_name}/label.csv",
        "embeddings": f"{dataset_dir}/glove.6B.300d.txt",
        "loglizer_seqs": f"{dataset_dir}/{d_name}/{d_name}.seqs.csv",
        "output_dir": output_dir,
        "trials_output": f"{output_dir}/trials.csv",
        "hyperparameters": f"{output_dir}/hyperparameters.json",
    }

    dfs = []
    for i in range(10):
        offset = i / 10
        metrics_path = f"{config_dict['output_dir']}/metrics_{offset}.csv"
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            dfs.append(metrics_df)
        else:
            print(f"File {metrics_path} not found")
    metrics_df = pd.concat(dfs).reset_index().drop(columns="index")

    data = metrics_df[metrics_df["split"] == "test"][["f1"]]
    labels = ["SVM"]
    fig, ax = plt.subplots()
    ax.set_ylabel("F1-score")
    ax.boxplot(data, tick_labels=labels)
    plt.savefig(f"{config_dict['output_dir']}/f1.png")

    data = metrics_df[metrics_df["split"] == "test"][["precision", "recall", "f1"]]
    print(data.describe())
