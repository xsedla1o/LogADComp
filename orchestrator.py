import argparse
import csv
import gc
import json
import os

import optuna
import pandas as pd
import tomli

from adapters import model_adapters, DualTrialAdapter
from dataloader import dataloaders
from sempca.preprocessing import DataPaths
from utils import Timed, calculate_metrics, get_memory_usage

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None


def exists_and_not_empty(file_path: str) -> bool:
    return (
        os.path.exists(file_path)
        and os.path.isfile(file_path)
        and os.path.getsize(file_path) > 0
    )


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
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of trials for hyperparameter optimization",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.5,
        help="Ratio of training data to use",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of validation data to use",
    )
    args = parser.parse_args()

    with open("paths.toml", "rb") as f:
        dir_config = tomli.load(f)

    d_name = args.dataset
    model_name = args.model
    n_trials = args.n_trials
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
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
        "train_hyperparameters": f"{output_dir}/train_hyperparameters.json",
        "hyperparameters": f"{output_dir}/hyperparameters.json",
    }

    # get abspath of this script
    root_dir = os.path.dirname(os.path.abspath(__file__))
    paths = DataPaths(
        dataset_name=d_name,
        project_root=root_dir,
        datasets_dir=config_dict["dataset_dir"],
        label_file=config_dict["processed_labels"],
    )
    print(paths)
    print(f"Base usage {get_memory_usage()}")

    dataloader = dataloaders[d_name](config_dict, paths)
    dataloader.get()

    model = model_adapters[model_name]()
    xs, ys = model.transform_representation(dataloader)
    print(f"Transformed usage {get_memory_usage()}")

    if exists_and_not_empty(config_dict["train_hyperparameters"]):
        print("Found train hyperparameters")
        with open(config_dict["train_hyperparameters"], "r") as in_f:
            best_train_params = json.load(in_f)
    elif isinstance(model, DualTrialAdapter):
        with Timed("Data loaded"):
            (x_train, y_train), (x_val, y_val), (x_test, y_test) = dataloader.split(
                xs, ys, train_ratio=train_ratio, val_ratio=val_ratio, offset=0.0
            )

        with Timed("Fit feature extractor and transform data"):
            x_train, x_val, x_test = model.preprocess_split(x_train, x_val, x_test)

        # Check if adapter requires training hyperparameters
        study = optuna.create_study(direction="minimize")
        with Timed("Optimize training hyperparameters"):
            study.optimize(
                model.get_training_trial_objective(x_train, y_train, x_val, y_val),
                n_trials=n_trials,
            )

        print(study.trials_dataframe(attrs=("number", "value", "params", "state")))
        print("Best loss value", study.best_value)

        os.makedirs(config_dict["output_dir"], exist_ok=True)
        study.trials_dataframe().to_csv(config_dict["train_hyperparameters"])
        with open(config_dict["train_hyperparameters"], "w") as out_f:
            json.dump(study.best_params, out_f)

        best_train_params = study.best_params

        print(f"Pre GC usage {get_memory_usage()}")
        del x_train, y_train, x_val, y_val, x_test, y_test
        gc.collect()
        print(f"Post GC usage {get_memory_usage()}")
    else:
        best_train_params = {}

    print("Best train params", best_train_params)
    print(f"Post train params usage {get_memory_usage()}")

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
                model.get_trial_objective(
                    x_train, y_train, x_val, y_val, best_train_params
                ),
                n_trials=n_trials,
            )

        print(study.trials_dataframe(attrs=("number", "value", "params", "state")))
        print("Best F1 value", study.best_value)

        os.makedirs(config_dict["output_dir"], exist_ok=True)
        study.trials_dataframe().to_csv(config_dict["trials_output"])
        with open(config_dict["hyperparameters"], "w") as out_f:
            json.dump(study.best_params, out_f)

        best_params = study.best_params

        print(f"Pre GC usage {get_memory_usage()}")
        del x_train, y_train, x_val, y_val, x_test, y_test
        gc.collect()
        print(f"Post GC usage {get_memory_usage()}")

    print("Best params", best_params)
    print(f"Post params usage {get_memory_usage()}")

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

        model.set_params(**best_train_params, **best_params)

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

        print(f"Pre GC usage {get_memory_usage()}")
        del x_train, y_train, x_val, y_val, x_test, y_test, metrics
        gc.collect()
        print(f"Post GC usage {get_memory_usage()}")

    dfs = []
    for i in range(10):
        offset = i / 10
        metrics_df = pd.read_csv(f"{config_dict['output_dir']}/metrics_{offset}.csv")
        dfs.append(metrics_df)
    metrics_df = pd.concat(dfs).reset_index().drop(columns=["index"])
    print(metrics_df)
