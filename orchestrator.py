import argparse
import csv
import gc
import json
import os

import optuna
import pandas as pd
import tomli

from adapters import model_adapters
from dataloader import dataloaders
from sempca.preprocessing import DataPaths
from utils import Timed, calculate_metrics

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None


def exists_and_not_empty(file_path: str) -> bool:
    return (
        os.path.exists(file_path)
        and os.path.isfile(file_path)
        and os.path.getsize(file_path) > 0
    )


def get_HDFS(config: dict):
    if all(exists_and_not_empty(config[x]) for x in ("dataset", "labels")):
        print("files already downloaded")
    else:
        os.system(f"bash scripts/download.sh HDFS {config['dataset_dir']}")

    if exists_and_not_empty(config["processed_labels"]):
        print("labels already processed")
    else:
        with (
            open(config["labels"], "r") as f,
            open(config["processed_labels"], "w") as out_f,
        ):
            csv_reader = csv.reader(f)
            csv_writer = csv.writer(out_f)
            for i, row in enumerate(csv_reader):
                if i == 0:
                    continue
                row[1] = "1" if row[1] == "Anomaly" else "0"
                csv_writer.writerow(row)


def get_BGL(config: dict):
    if exists_and_not_empty(config["dataset"]):
        print("files already downloaded")
    else:
        os.system(f"bash scripts/download.sh BGL {config['dataset_dir']}")


def get_embeddings(config: dict):
    if exists_and_not_empty(config["embeddings"]):
        print("embeddings already downloaded")
    else:
        os.system(f"bash scripts/download.sh embeddings {config['dataset_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset to use for anomaly detection, choices: " + ", ".join(dataloaders),
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model to use for anomaly detection, choices: "
        + ", ".join(model_adapters),
    )
    args = parser.parse_args()

    with open("paths.toml", "rb") as f:
        dir_config = tomli.load(f)

    d_name = args.dataset
    model_name = args.model
    n_trials = 100
    train_ratio = 0.5
    val_ratio = 0.1
    dataset_dir = dir_config["datasets"]
    output_dir = f"{dir_config['outputs']}/{d_name}_{train_ratio}/{model_name}"

    config_dict = {
        "dataset_dir": dataset_dir,
        "dataset": f"{dataset_dir}/{d_name}/{d_name}.log",
        "labels": f"{dataset_dir}/{d_name}/anomaly_label.csv",
        "processed_labels": f"{dataset_dir}/{d_name}/label.csv",
        "embeddings": f"{dataset_dir}/glove.6B.300d.txt",
        "loglizer_seqs": f"{dataset_dir}/{d_name}/{d_name}.seqs.csv",
        "word_vec_npz": f"{dataset_dir}/{d_name}/{d_name}.word_vec.npz",
        "ecv_npz": f"{dataset_dir}/{d_name}/{d_name}.ecv.npz",
        "output_dir": output_dir,
        "trials_output": f"{output_dir}/trials.csv",
        "hyperparameters": f"{output_dir}/hyperparameters.json",
    }

    # get abspath of this script
    root_dir = os.path.dirname(os.path.abspath(__file__))
    paths = DataPaths(
        dataset_name=d_name,
        project_root=root_dir,
        dataset_dir=f"{config_dict['dataset_dir']}/{d_name}",
        label_file=config_dict["processed_labels"],
    )
    print(paths)

    dataloader = dataloaders[d_name](config_dict, paths)
    dataloader.get()

    model = model_adapters[model_name]()
    xs, ys = model.transform_representation(dataloader)

    if all(
        exists_and_not_empty(config_dict[x])
        for x in ["trials_output", "hyperparameters"]
    ):
        print("Found hyperparameters and trials output")
        with open(config_dict["hyperparameters"], "r") as in_f:
            best_params = json.load(in_f)
    else:
        with Timed("Data loaded"):
            (x_train, y_train), (x_val, y_val), (x_test, y_test) = dataloader.split(
                xs, ys, train_ratio=train_ratio, val_ratio=val_ratio, offset=0.0
            )

        with Timed("Fit feature extractor and transform data"):
            x_train, x_val, x_test = model.preprocess_split(x_train, x_val, x_test)

        # Hyperparameter tuning
        study = optuna.create_study(direction="maximize")
        with Timed("Optimize hyperparameters"):
            study.optimize(
                model.get_trial_objective(x_train, y_train, x_val, y_val),
                n_trials=n_trials,
            )

        print(study.trials_dataframe(attrs=("number", "value", "params", "state")))
        print("Best F1 value", study.best_value)

        os.makedirs(config_dict["output_dir"], exist_ok=True)
        study.trials_dataframe().to_csv(config_dict["trials_output"])
        with open(config_dict["hyperparameters"], "w") as out_f:
            json.dump(study.best_params, out_f)

        best_params = study.best_params

        del x_train, y_train, x_val, y_val, x_test, y_test
        gc.collect()

    print("Best params", best_params)

    # Float offsets from 0.0 to 0.9 in steps of 0.1
    for offset in range(0, 10):
        offset /= 10

        if exists_and_not_empty(f"{config_dict['output_dir']}/metrics_{offset}.csv"):
            print(f"Found metrics for offset {offset}")
            continue
        else:
            print(f"Evaluating split with offset {offset}")

        (x_train, y_train), (x_val, y_val), (x_test, y_test) = dataloader.split(
            xs, ys, train_ratio=train_ratio, val_ratio=val_ratio, offset=offset
        )

        with Timed("Fit feature extractor and transform data"):
            x_train, x_val, x_test = model.preprocess_split(x_train, x_val, x_test)

        model.set_params(**best_params)

        with Timed("Fit model"):
            model.fit(x_train, y_train)

        metrics = []

        with Timed("Train validation"):
            y_pred = model.predict(x_train)
            meta = {"offset": offset, "split": "train"}
            metrics.append(calculate_metrics(y_train, y_pred) | meta)

        with Timed("Validation validation"):
            y_pred = model.predict(x_val)
            meta = {"offset": offset, "split": "val"}
            metrics.append(calculate_metrics(y_val, y_pred) | meta)

        with Timed("Test validation"):
            y_pred = model.predict(x_test)
            meta = {"offset": offset, "split": "test"}
            metrics.append(calculate_metrics(y_test, y_pred) | meta)

        metrics_df = pd.DataFrame(
            data=metrics,
        )
        metrics_df.to_csv(
            f"{config_dict['output_dir']}/metrics_{offset}.csv", index=False
        )
        print(metrics_df)

        del x_train, y_train, x_val, y_val, x_test, y_test, metrics
        gc.collect()

    dfs = []
    for i in range(10):
        offset = i / 10
        metrics_df = pd.read_csv(f"{config_dict['output_dir']}/metrics_{offset}.csv")
        dfs.append(metrics_df)
    metrics_df = pd.concat(dfs).reset_index().drop(columns=["index"])
    print(metrics_df)
