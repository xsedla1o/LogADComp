import csv
import gc
import os
from abc import ABC, abstractmethod
from functools import wraps
from logging import Logger
from typing import Tuple, Union, List, Callable, Dict, Type

import numpy as np

from preprocess import EventCounter
from sempca.preprocessing import DataPaths, Preprocessor, BasicDataLoader, BGLLoader
from sempca.representations import SequentialAdd, TemplateTfIdf
from sempca.utils import get_logger

NdArr = np.ndarray
NdArrPair = Tuple[np.ndarray, np.ndarray]


def exists_and_not_empty(file_path: str) -> bool:
    return (
        os.path.exists(file_path)
        and os.path.isfile(file_path)
        and os.path.getsize(file_path) > 0
    )


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
                    f"Skipping as already processed: "
                    + ", ".join(self.config[k] for k in key)
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


def cyclic_read(data: np.ndarray, samples: int, offset: int) -> Tuple[np.ndarray, int]:
    end = offset + samples
    if end <= data.shape[0]:
        return data[offset:end], end
    else:
        # join the end of the dataset to the beginning
        end = end - data.shape[0]
        if end > data.shape[0]:
            raise ValueError(
                f"The number of samples {samples} is too large "
                f"for the dataset {data.shape[0]}"
            )
        return np.append(data[offset:], data[:end], axis=0), end


class DataLoader(ABC):
    log: Logger = None

    def __init__(self, config: dict, paths: DataPaths):
        self.config = config
        self.paths = paths

    @abstractmethod
    def get(self):
        """Download the dataset"""

    @abstractmethod
    def _parse(self) -> Tuple[Preprocessor, BasicDataLoader, set]:
        """Parse the dataset and return the preprocessor, dataloader and drop ids"""

    def load_ecv_representation(self):
        loaded = np.load(self.config["ecv_npz"])
        return loaded["xs"], loaded["ys"]

    @skip_when_present("ecv_npz", load=load_ecv_representation)
    def get_ecv_representation(self) -> Tuple[NdArr, NdArr]:
        """Load the ECV representation of the dataset for loglizer models"""
        preprocessor, dataloader, drop_ids = self._parse()

        xs, ys = [], []
        for block, sequence in dataloader.block2eventseq.items():
            xs.append([int(x) - 1 for x in sequence if x not in drop_ids])
            ys.append(dataloader.label2id[dataloader.block2label[block]])

        xs, ys = np.asarray(xs, dtype=object), np.asarray(ys, dtype=int)
        xs = EventCounter().fit(xs).transform(xs)

        np.savez_compressed(self.config["ecv_npz"], xs=xs, ys=ys)

        del preprocessor, dataloader
        gc.collect()

        return xs, ys

    def load_word_vec_representation(self):
        loaded = np.load(self.config["word_vec_npz"])
        return loaded["xs"], loaded["ys"]

    @skip_when_present("word_vec_npz", load=load_word_vec_representation)
    def get_word_vec_representation(self) -> Tuple[NdArr, NdArr]:
        """Load the word vector representation of the dataset for SemPCA models"""
        preprocessor, dataloader, drop_ids = self._parse()

        instances = preprocessor.generate_instances(dataloader, drop_ids=drop_ids)
        seqential_encoder = SequentialAdd(preprocessor.embedding)
        xs = seqential_encoder.transform(instances)
        ys = np.asarray(
            [int(preprocessor.label2id[inst.label]) for inst in instances], dtype=int
        )

        del preprocessor, dataloader
        gc.collect()

        np.savez_compressed(self.config["word_vec_npz"], xs=xs, ys=ys)

        return xs, ys

    def split(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        train_ratio: float,
        val_ratio: float,
        offset: float,
    ) -> Tuple[NdArrPair, NdArrPair, NdArrPair]:
        num_train = int(train_ratio * xs.shape[0])
        num_validation = int(val_ratio * xs.shape[0])
        num_test = xs.shape[0] - num_train - num_validation

        train_begin = int(offset * xs.shape[0])

        x_train, train_end = cyclic_read(xs, num_train, train_begin)
        x_validation, validation_end = cyclic_read(xs, num_validation, train_end)
        x_test, _ = cyclic_read(xs, num_test, validation_end)

        y_train, _ = cyclic_read(ys, num_train, train_begin)
        y_validation, _ = cyclic_read(ys, num_validation, train_end)
        y_test, _ = cyclic_read(ys, num_test, validation_end)

        self.log_split(x_train, y_train, x_validation, y_validation, x_test, y_test)
        return (x_train, y_train), (x_validation, y_validation), (x_test, y_test)

    def log_split(self, x_train, y_train, x_val, y_val, x_test, y_test):
        num_train = x_train.shape[0]
        num_val = x_val.shape[0]
        num_test = x_test.shape[0]
        num_total = num_train + num_test + num_val

        num_train_pos = sum(y_train)
        num_val_pos = sum(y_val)
        num_test_pos = sum(y_test)
        num_pos = num_train_pos + num_test_pos + num_val_pos

        self.log.debug(f"Split  {'samples':>10s}  {'anomalies':>10s}")
        self.log.debug(f"Train: {num_train:10d} ({num_train_pos:10d})")
        self.log.debug(f"Val:   {num_val:10d} ({num_val_pos:10d})")
        self.log.debug(f"Test:  {num_test:10d} ({num_test_pos:10d})")
        self.log.debug(f"Total: {num_total:10d} ({num_pos:10d})")


class HDFS(DataLoader):
    log = get_logger("DataLoader.HDFS")

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

    def _parse(self) -> Tuple[Preprocessor, BasicDataLoader, set]:
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
        drop_ids = {m_id}

        return preprocessor, dataloader, drop_ids


class BGL(DataLoader):
    log = get_logger("DataLoader.BGL")

    @skip_when_present("dataset")
    def get(self):
        os.system(f"bash scripts/download.sh BGL {self.config['dataset_dir']}")

    def _parse(self) -> Tuple[Preprocessor, BasicDataLoader, set]:
        preprocessor = Preprocessor()
        t_encoder = TemplateTfIdf()
        dataloader = BGLLoader(
            paths=self.paths,
            semantic_repr_func=t_encoder.present,
            group_component=False,
            win_secs=60,
            win_lines=120,
            win_kind="tumbling",
        )

        dataloader.parse_by_drain(core_jobs=min(os.cpu_count() // 2, 8))

        return preprocessor, dataloader, set()


dataloaders: Dict[str, Type[DataLoader]] = {
    "HDFS": HDFS,
    "BGL": BGL,
}
