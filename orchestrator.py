import csv
import gc
import json
import os

import optuna
import pandas as pd
import tomli

from loglizer.loglizer import preprocessing
from loglizer.loglizer.dataloader import load_HDFS
from loglizer.loglizer.models import SVM
from sempca.preprocessing import DataPaths, Preprocessor
from sempca.representations import TemplateTfIdf
from utils import Timed

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


def get_embeddings(config: dict):
    if exists_and_not_empty(config["embeddings"]):
        print("embeddings already downloaded")
    else:
        os.system(f"bash scripts/download.sh embeddings {config['dataset_dir']}")


def to_loglizer_seqs(loader, output_csv, drop_ids=None):
    if drop_ids is None:
        drop_ids = set()

    with open(output_csv, "w") as out_f:
        writer = csv.writer(
            out_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(("BlockId", "Label", "EventSequence"))

        for block, sequence in loader.block2eventseq.items():
            seq = " ".join(str(x) for x in sequence if x not in drop_ids)
            label_id = loader.label2id[loader.block2label[block]]
            writer.writerow((block, label_id, seq))


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

    get_HDFS(config_dict)
    get_embeddings(config_dict)

    # get abspath of this script
    root_dir = os.path.dirname(os.path.abspath(__file__))
    paths = DataPaths(
        dataset_name="HDFS",
        project_root=root_dir,
        dataset_dir=f"{config_dict['dataset_dir']}/HDFS",
        label_file=config_dict["processed_labels"],
    )
    print(paths)

    # instances = preprocessor.generate_instances(dataloader, drop_ids={m_id})

    if exists_and_not_empty(config_dict["loglizer_seqs"]):
        print("loglizer seqs already processed")
    else:
        preprocessor = Preprocessor()
        t_encoder = TemplateTfIdf()
        dataloader = preprocessor.get_dataloader("HDFS")(
            paths=paths, semantic_repr_func=t_encoder.present
        )

        dataloader.parse_by_drain(core_jobs=min(os.cpu_count() // 2, 8))

        # Drop malformed template
        m_id = None
        for t_id, template in dataloader.templates.items():
            if template == "such file or directory":
                m_id = t_id
                break

        to_loglizer_seqs(dataloader, config_dict["loglizer_seqs"], drop_ids={m_id})

        del preprocessor, t_encoder, dataloader
        gc.collect()

    if all(exists_and_not_empty(config_dict[x]) for x in ["trials_output", "hyperparameters"]):
        print("Found hyperparameters and trials output")
        with open(config_dict["hyperparameters"], "r") as in_f:
            best_params = json.load(in_f)
    else:
        with Timed("Data loaded"):
            (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_HDFS(
                config_dict["loglizer_seqs"],
                window="session",
                train_ratio=train_ratio,
                split_type="sequential_validation",
            )

        num_train = x_train.shape[0]
        num_val = x_val.shape[0]
        num_test = x_test.shape[0]
        num_total = num_train + num_test + num_val
        num_train_pos = sum(y_train)
        num_val_pos = sum(y_val)
        num_test_pos = sum(y_test)
        num_pos = num_train_pos + num_test_pos + num_val_pos
        print(f"Train: {num_train:10d} ({num_train_pos:10d})")
        print(f"Val:   {num_val:10d} ({num_val_pos:10d})")
        print(f"Test:  {num_test:10d} ({num_test_pos:10d})")
        print(f"Total: {num_total:10d} ({num_pos:10d})")

        feature_extractor = preprocessing.FeatureExtractor()
        with Timed("Fit feature extractor and transform data"):
            x_train = feature_extractor.fit_transform(x_train, term_weighting="tf-idf")
            x_val = feature_extractor.transform(x_val)
            x_test = feature_extractor.transform(x_test)


        # Hyperparameter tuning
        def objective(trial: optuna.Trial):
            model = SVM(
                penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
                tol=trial.suggest_float("tol", 1e-4, 1e-1, log=True),
                C=trial.suggest_float("C", 1e-3, 1e3, log=True),
                class_weight=trial.suggest_categorical(
                    "class_weight", [None, "balanced"]
                ),
                max_iter=trial.suggest_int("max_iter", 100, 1000, step=100),
                dual=False,
            )
            model.fit(x_train, y_train)
            _precision, _recall, f1 = model.evaluate(x_val, y_val)
            return f1


        study = optuna.create_study(direction="maximize")
        with Timed("Optimize hyperparameters"):
            study.optimize(objective, n_trials=n_trials)

        print(study.trials_dataframe(attrs=("number", "value", "params", "state")))
        print("Best params", study.best_params)
        print("Best F1 value", study.best_value)

        os.makedirs(config_dict["output_dir"], exist_ok=True)
        study.trials_dataframe().to_csv(config_dict["trials_output"])
        with open(config_dict["hyperparameters"], "w") as out_f:
            json.dump(study.best_params, out_f)

        best_params = study.best_params

        del x_train, y_train, x_val, y_val, x_test, y_test
        gc.collect()

    # Float offsets from 0.0 to 0.9 in steps of 0.1
    for offset in range(0, 10):
        offset /= 10

        if exists_and_not_empty(f"{config_dict['output_dir']}/metrics_{offset}.csv"):
            print(f"Found metrics for offset {offset}")
            continue
        else:
            print(f"Evaluating split with offset {offset}")

        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_HDFS(
            config_dict["loglizer_seqs"],
            window="session",
            train_ratio=train_ratio,
            split_type="sequential_validation",
            offset=offset,
        )

        feature_extractor = preprocessing.FeatureExtractor()
        with Timed("Fit feature extractor and transform data"):
            x_train = feature_extractor.fit_transform(x_train, term_weighting="tf-idf")
            x_val = feature_extractor.transform(x_val)
            x_test = feature_extractor.transform(x_test)

        model = SVM(
            **best_params,
            dual=False,
        )

        with Timed("Fit model"):
            model.fit(x_train, y_train)

        metrics = []

        with Timed("Train validation"):
            p, r, f1 = model.evaluate(x_train, y_train)
            metrics.append([p, r, f1, offset, "train"])

        with Timed("Validation validation"):
            p, r, f1 = model.evaluate(x_val, y_val)
            metrics.append([p, r, f1, offset, "val"])

        with Timed("Test validation"):
            p, r, f1 = model.evaluate(x_test, y_test)
            metrics.append([p, r, f1, offset, "test"])

        metrics_df = pd.DataFrame(
            columns=["precision", "recall", "f1", "offset", "split"],
            data=metrics,
        )
        metrics_df.to_csv(f"{config_dict['output_dir']}/metrics_{offset}.csv")
        print(metrics_df)

        del x_train, y_train, x_val, y_val, x_test, y_test, model, metrics
        gc.collect()

    dfs = []
    for i in range(10):
        offset = i / 10
        metrics_df = pd.read_csv(f"{config_dict['output_dir']}/metrics_{offset}.csv")
        dfs.append(metrics_df)
    metrics_df = pd.concat(dfs).reset_index().drop(columns="index")
    print(metrics_df)