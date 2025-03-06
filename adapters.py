import csv
import gc
import os
from abc import abstractmethod, ABC
from functools import wraps
from typing import Union, List, Callable, Type, Dict, TypeVar

import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from loglizer.loglizer.dataloader import _split_data, load_HDFS
from loglizer.loglizer.models import SVM
from sempca.models import PCAPlusPlus
from sempca.preprocessing import Preprocessor, DataPaths
from sempca.representations import (
    TemplateTfIdf,
    SequentialAdd,
)

from preprocess import EventCounter, Normalizer

T = TypeVar("T", bound=np.ndarray)


def exists_and_not_empty(file_path: str) -> bool:
    return (
        os.path.exists(file_path)
        and os.path.isfile(file_path)
        and os.path.getsize(file_path) > 0
    )


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


def skip_when_present(key: Union[str, List[str]], load: Callable = None):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if exists_and_not_empty(self.config[key]):
                print(f"Skipping as already processed: {self.config[key]}")
                return load(self) if load is not None else None
            return func(self, *args, **kwargs)

        return wrapper

    def list_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if all(exists_and_not_empty(self.config[k]) for k in key):
                print(
                    f"Skipping as already processed: , ".join(
                        self.config[k] for k in key
                    )
                )
                return load(self) if load is not None else None
            return func(self, *args, **kwargs)

        return wrapper

    if isinstance(key, str):
        return decorator
    elif isinstance(key, list):
        return list_decorator
    else:
        raise ValueError("key should be either str or list of str")


class HDFS:
    def __init__(self, config: dict):
        self.config = config

    def get(self):
        self._get_dataset()
        self._preprocess_labels()

    @skip_when_present(["dataset", "labels"])
    def _get_dataset(self):
        os.system(f"bash scripts/download.sh HDFS {self.config['dataset_dir']}")

    @skip_when_present("processed_labels")
    def _preprocess_labels(self):
        with (
            open(self.config["labels"], "r") as f,
            open(self.config["processed_labels"], "w") as out_f,
        ):
            csv_reader = csv.reader(f)
            csv_writer = csv.writer(out_f)
            for i, row in enumerate(csv_reader):
                if i == 0:
                    continue
                row[1] = "1" if row[1] == "Anomaly" else "0"
                csv_writer.writerow(row)


class BGL:
    def __init__(self, config: dict):
        self.config = config

    @skip_when_present("dataset")
    def get(self):
        os.system(f"bash scripts/download.sh BGL {self.config['dataset_dir']}")


class LogADCompAdapter(ABC):
    def __init__(self, config: dict, paths: DataPaths, **kwargs):
        self._model = None
        self.config = config
        self.paths = paths

    @abstractmethod
    def data_preprocessing(self):
        pass

    @staticmethod
    @abstractmethod
    def preprocess_split(x_train: T, x_val: T, x_test: T) -> tuple[T, T, T]:
        pass

    @abstractmethod
    def data_load(self):
        pass

    @abstractmethod
    def load_split(self, train_ratio: float, val_ratio: float, offset: float):
        pass

    @staticmethod
    @abstractmethod
    def get_trial_objective(x_train, y_train, x_val, y_val):
        pass

    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    def predict(self, x_test):
        return self._model.predict(x_test)

    @abstractmethod
    def set_params(self, **kwargs):
        pass


class SemPCAAdapter(LogADCompAdapter):
    def __init__(self, config: dict, paths: DataPaths):
        super().__init__(config, paths)
        self.xs = None
        self.ys = None
        self._model = PCAPlusPlus()
        self.threshold_mult = 1.0

    def set_params(
        self,
        n_components: Union[int, float] = 0.95,
        c_alpha: float = 3.2905,
        threshold: float = None,
        threshold_mult: float = 1.0,
    ):
        self._model = PCAPlusPlus(
            n_components=n_components, c_alpha=c_alpha, threshold=threshold
        )
        self.threshold_mult = threshold_mult

    def fit(self, x_train, _y_train=None):
        self._model.fit(x_train)
        self._model.threshold *= self.threshold_mult

    def data_load(self):
        loaded = np.load(self.config["sempca_sem_npz"])
        self.xs, self.ys = loaded["xs"], loaded["ys"]

    @skip_when_present("sempca_sem_npz", load=data_load)
    def data_preprocessing(self):
        preprocessor = Preprocessor()
        t_encoder = TemplateTfIdf()
        dataloader = preprocessor.get_dataloader("HDFS")(
            paths=self.paths, semantic_repr_func=t_encoder.present
        )

        dataloader.parse_by_drain(core_jobs=min(os.cpu_count() // 2, 8))

        # Drop malformed template
        m_id = None
        for t_id, template in dataloader.templates.items():
            if template == "such file or directory":
                m_id = t_id
                break

        instances = preprocessor.generate_instances(dataloader, drop_ids={m_id})
        ys = np.asarray(
            [int(preprocessor.label2id[inst.label]) for inst in instances], dtype=int
        )

        seqential_encoder = SequentialAdd(preprocessor.embedding)
        xs = seqential_encoder.transform(instances)

        np.savez_compressed(self.config["sempca_sem_npz"], xs=xs, ys=ys)

        del preprocessor, t_encoder, dataloader, xs, ys
        gc.collect()

    @staticmethod
    def preprocess_split(x_train: T, x_val: T, x_test: T) -> tuple[T, T, T]:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        return x_train, x_val, x_test

    @staticmethod
    def get_trial_objective(x_train, y_train, x_val, y_val):
        def objective(trial: optuna.Trial):
            # Getting a fixed threshold to work for all splits is hard,
            # maybe relying on Q statistics with a learned multiplier is better
            threshold_mult = trial.suggest_float("threshold_mult", 0.1, 2.0)
            model = PCAPlusPlus(
                n_components=trial.suggest_int(
                    "n_components", 1, x_train.shape[1] // 10, log=True
                ),
                c_alpha=trial.suggest_categorical("c_alpha", [3.8906]),
            )
            model.fit(x_train)
            model.threshold *= threshold_mult
            _precision, _recall, f1 = model.evaluate(x_val, y_val)
            return f1

        return objective

    def load_split(self, train_ratio: float, val_ratio: float, offset: float):
        return _split_data(
            self.xs,
            self.ys,
            train_ratio=train_ratio,
            split_type="sequential_validation",
            val_ratio=val_ratio,
            offset=offset,
        )


class SVMAdapter(LogADCompAdapter):
    def __init__(self, config: dict, paths: DataPaths, **kwargs):
        super().__init__(config, paths, **kwargs)
        self._model = SVM()

    @skip_when_present("loglizer_seqs")
    def data_preprocessing(self):
        preprocessor = Preprocessor()
        t_encoder = TemplateTfIdf()
        dataloader = preprocessor.get_dataloader("HDFS")(
            paths=self.paths, semantic_repr_func=t_encoder.present
        )

        dataloader.parse_by_drain(core_jobs=min(os.cpu_count() // 2, 8))

        # Drop malformed template
        m_id = None
        for t_id, template in dataloader.templates.items():
            if template == "such file or directory":
                m_id = t_id
                break

        to_loglizer_seqs(dataloader, self.config["loglizer_seqs"], drop_ids={m_id})

        del preprocessor, t_encoder, dataloader
        gc.collect()

    @staticmethod
    def preprocess_split(x_train: T, x_val: T, x_test: T) -> tuple[T, T, T]:
        extractor = Pipeline(
            [
                ("seq2vec", EventCounter()),
                ("norm", Normalizer(term_weighting="tf-idf")),
            ]
        )
        extractor.fit(x_train)
        x_train = extractor.transform(x_train)
        x_val = extractor.transform(x_val)
        x_test = extractor.transform(x_test)
        return x_train, x_val, x_test

    @staticmethod
    def get_trial_objective(x_train, y_train, x_val, y_val):
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

        return objective

    def load_split(self, train_ratio: float, val_ratio: float, offset: float):
        return load_HDFS(
            self.config["loglizer_seqs"],
            window="session",
            train_ratio=train_ratio,
            # val_ratio=val_ratio,
            offset=offset,
            split_type="sequential_validation",
        )

    def data_load(self):
        pass  # A noop for now

    def set_params(
        self, penalty="l1", tol=0.1, C=1, dual=False, class_weight=None, max_iter=100
    ):
        self._model = SVM(
            penalty=penalty,
            tol=tol,
            C=C,
            dual=dual,
            class_weight=class_weight,
            max_iter=max_iter,
        )

    def fit(self, x_train, y_train):
        self._model.fit(x_train, y_train)


model_adapters: Dict[str, Type[LogADCompAdapter]] = {
    "SemPCA": SemPCAAdapter,
    "SVM": SVMAdapter,
}
