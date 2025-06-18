"""
Orchestrator for the LogADComp pipeline.

This script is responsible for orchestrating the entire LogADComp pipeline,
including data loading, preprocessing, hyperparameter tuning, and evaluation.

It uses Optuna for hyperparameter optimization and supports multiple datasets
and models.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>

See README.md for instructions and script help for usage.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Generator

import numpy as np
import optuna
import pandas as pd
import tomli
from optuna.pruners import PatientPruner, HyperbandPruner
from optuna.samplers import TPESampler

from adapters import model_adapters, DualTrialAdapter, ModelPaths, LogADCompAdapter
from dataloader import dataloaders
from sempca.const import SESSION
from sempca.preprocessing import DataPaths
from utils import Timed, calculate_metrics, get_memory_usage, seed_everything

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None


def exists_and_not_empty(file_path: str) -> bool:
    return (
        os.path.exists(file_path)
        and os.path.isfile(file_path)
        and os.path.getsize(file_path) > 0
    )


def parse_splits(splits: str) -> list:
    """
    Parses a string of comma-separated intervals and returns a list of integers.
    :param splits: A string of comma-separated intervals (e.g., "0-9,10-19").
    :return: A list of integers representing the parsed intervals.
    """
    result = []
    for part in splits.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            if start > end:
                raise ValueError(f"Invalid interval: {part}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    for i in result:
        if i < 0 or i > 9:
            raise ValueError(f"Invalid split offset: {i}")
    return result


class ModelPathsManager:
    def __init__(self, base: ModelPaths, target: LogADCompAdapter):
        self.target = target
        self.base = base
        self.curr = None
        self._current_split = None

    def set_split(self, split: int) -> ModelPaths:
        if split < 0 or split > 9:
            raise ValueError(f"Invalid split offset: {split}")

        if self.curr is not None and self._current_split == split:
            return self.curr

        self._current_split = split
        self.curr = ModelPaths(
            cache=self.base.cache / str(split),
            artefacts=self.base.artefacts / str(split),
        )

        self.target.set_paths(self.curr)
        return self.curr

    def with_splits(self, splits: Iterable[int]) -> Generator[int, None, None]:
        for split in splits:
            self.set_split(split)
            yield split

    def clear_split_cache(self):
        if self.curr is not None:
            self.curr.clear_split_cache()


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
        "--splits",
        type=str,
        default="0-9",
        help="Splits to use for evaluation, "
        "a comma separated list supporting intervals, default=0-9",
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
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before splitting",
        default=False,
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save model predictions as artefacts",
        default=False,
    )
    parser.add_argument(
        "--ignore-cache", "-i",
        action="store_true",
        help="Ignore the cache when evaluating methods",
        default=False,
    )
    args = parser.parse_args()

    try:
        split_offsets = parse_splits(args.splits)
        if len(split_offsets) == 0:
            print("Warning: No splits provided.")
            sys.exit(1)
    except (ValueError, TypeError) as e:
        print(f"Error parsing splits: {e}", file=sys.stderr)
        exit(1)

    with open("paths.toml", "rb") as f:
        dir_config = tomli.load(f)

    d_name = args.dataset
    model_name = args.model
    n_trials = args.n_trials
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    dataset_dir = dir_config["datasets"]
    suffix = "Shuffled" if args.shuffle else ""
    d_id = f"{d_name}{suffix}_{train_ratio}"

    dataset_srcdir = f"{dataset_dir}/{d_name}"
    output_dir = f"{dir_config['outputs']}/{d_id}/{model_name}"

    for d in [dataset_srcdir, output_dir]:
        os.makedirs(d, exist_ok=True)

    config_dict = {
        "dataset_dir": dataset_dir,
        "dataset": f"{dataset_srcdir}/{d_name}.log",
        "labels": f"{dataset_srcdir}/anomaly_label.csv",
        "processed_labels": f"{dataset_srcdir}/label.csv",
        "embeddings": f"{dataset_dir}/glove.6B.300d.txt",
        "loglizer_seqs": f"{dataset_srcdir}/{d_name}.seqs.csv",
        "word_vec_npz": f"{dataset_srcdir}/{d_name}.word_vec.npz",
        "ecv_npz": f"{dataset_srcdir}/{d_name}.ecv.npz",
        "t_seq_npz": f"{dataset_srcdir}/{d_name}.t_seq.npz",
        "embed_seq_npz": f"{dataset_srcdir}/{d_name}.embed_seq.npz",
        "random_indices": f"{dataset_srcdir}/{d_name}.random_indices.out",
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
        sequence_file=f"{dataset_srcdir}/raw_log_seqs.txt",
    )

    base_path = Path(dir_config["outputs"]) / ".."

    cache_base = dir_config.get("cache", base_path / "cache")
    cache_base = Path(cache_base).resolve()
    if "cache" not in dir_config:
        print(
            f"Warning: No cache directory found in paths.toml, "
            f"defaulting to {cache_base}"
        )

    artefacts_base = dir_config.get("artefacts", base_path / "artefacts")
    artefacts_base = Path(artefacts_base).resolve()
    if "artefacts" not in dir_config:
        print(
            f"Warning: No artefacts directory found in paths.toml, "
            f"defaulting to {artefacts_base}"
        )

    m_paths = ModelPaths(
        cache=(cache_base / d_id / model_name).resolve(),
        artefacts=(artefacts_base / SESSION / d_id / model_name).resolve(),
    )
    print(paths)
    print(m_paths)
    print(f"Base usage {get_memory_usage()}")

    # Download and extract the dataset if not already done
    dataloader = dataloaders[d_name](config_dict, paths)
    dataloader.get()

    model = model_adapters[model_name]()

    path_manager = ModelPathsManager(base=m_paths, target=model)

    # Transform the representation to match the model
    xs, ys = model.transform_representation(dataloader)
    print(f"Transformed usage {get_memory_usage()}")

    # Perform training hyperparameter tuning if needed
    if exists_and_not_empty(config_dict["train_hyperparameters"]):
        print("Found train hyperparameters")
        with open(config_dict["train_hyperparameters"], "r") as in_f:
            best_train_params = json.load(in_f)
    elif isinstance(model, DualTrialAdapter):
        path_manager.set_split(0)
        with Timed("Data loaded"):
            seed_everything()
            (x_train, y_train), (x_val, y_val), (x_test, y_test) = dataloader.split(
                xs,
                ys,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                offset=0.0,
                shuffle=args.shuffle,
            )

        with Timed("Fit feature extractor and transform data"):
            x_train, x_val, x_test = model.preprocess_split(x_train, x_val, x_test)

        # Training hyperparameter tuning
        seed_everything()
        study = optuna.create_study(
            study_name="training_hyperparameters",
            direction="minimize",
            sampler=TPESampler(),
            pruner=PatientPruner(HyperbandPruner(), patience=10),
        )
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
        with open(
            path_manager.curr.artefacts / "train_hyperparameters.json", "w"
        ) as out_f:
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

    # Hyperparameter tuning
    if all(
        exists_and_not_empty(config_dict[x])
        for x in ["trials_output", "hyperparameters"]
    ):
        print("Found hyperparameters and trials output")
        with open(config_dict["hyperparameters"], "r") as in_f:
            best_params = json.load(in_f)
    else:
        path_manager.set_split(0)
        with Timed("Data loaded"):
            seed_everything()
            (x_train, y_train), (x_val, y_val), (x_test, y_test) = dataloader.split(
                xs,
                ys,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                offset=0.0,
                shuffle=args.shuffle,
            )

        with Timed("Fit feature extractor and transform data"):
            x_train, x_val, x_test = model.preprocess_split(x_train, x_val, x_test)

        # Hyperparameter tuning
        seed_everything()
        study = optuna.create_study(
            study_name="hyperparameters",
            direction="maximize",
            sampler=TPESampler(),
            pruner=PatientPruner(HyperbandPruner(), patience=10),
        )
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
        with open(path_manager.curr.artefacts / "hyperparameters.json", "w") as out_f:
            json.dump(study.best_params, out_f)

        best_params = study.best_params

        del x_train, y_train, x_val, y_val, x_test, y_test
        gc.collect()
        print(f"Memory usage {get_memory_usage()}")

    print("Best params", best_params)
    print(f"Post params usage {get_memory_usage()}")

    # Evaluate the model - 10-fold cross-validation
    for offset in path_manager.with_splits(split_offsets):
        offset /= 10

        if exists_and_not_empty(f"{config_dict['output_dir']}/metrics_{offset}.csv"):
            if not args.ignore_cache:
                print(f"Found metrics for offset {offset}")
                continue
        else:
            print(f"Evaluating split with offset {offset}")

        seed_everything()
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = dataloader.split(
            xs,
            ys,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            offset=offset,
            shuffle=args.shuffle,
        )

        with Timed("Fit feature extractor and transform data"):
            x_train, x_val, x_test = model.preprocess_split(x_train, x_val, x_test)

        seed_everything()
        model.set_params(**best_train_params, **best_params)

        with Timed("Fit model"):
            model.fit(x_train, y_train, x_val, y_val)

        metrics = []
        predictions = None
        if args.save_predictions:
            predictions = {}

        with Timed("Train validation"):
            y_pred = model.predict(x_train)
            meta = {"offset": offset, "split": "train"}
            metrics.append(calculate_metrics(y_train, y_pred) | meta)
            if predictions is not None:
                predictions["train_pred"] = y_pred.astype(np.int8)

        with Timed("Validation validation"):
            y_pred = model.predict(x_val)
            meta = {"offset": offset, "split": "val"}
            metrics.append(calculate_metrics(y_val, y_pred) | meta)
            if predictions is not None:
                predictions["val_pred"] = y_pred.astype(np.int8)

        with Timed("Test validation"):
            y_pred = model.predict(x_test)
            meta = {"offset": offset, "split": "test"}
            metrics.append(calculate_metrics(y_test, y_pred) | meta)
            if predictions is not None:
                predictions["test_pred"] = y_pred.astype(np.int8)

        metrics_df = pd.DataFrame(
            data=metrics,
        )
        metrics_df.to_csv(
            f"{config_dict['output_dir']}/metrics_{offset}.csv", index=False
        )
        metrics_df.to_csv(path_manager.curr.artefacts / "metrics.csv", index=False)
        print(metrics_df)

        if predictions is not None:
            np.savez(
                path_manager.curr.artefacts / "predictions.npz",
                **predictions,
            )
            del predictions

        del x_train, y_train, x_val, y_val, x_test, y_test, metrics
        gc.collect()
        print(f"Memory usage {get_memory_usage()}")

    dfs = []
    for i in split_offsets:
        offset = i / 10
        metrics_df = pd.read_csv(f"{config_dict['output_dir']}/metrics_{offset}.csv")
        dfs.append(metrics_df)
    metrics_df = pd.concat(dfs).reset_index().drop(columns=["index"])
    print(metrics_df)
