import csv
import gc
import os
import sys
from abc import ABC, abstractmethod
from functools import wraps
from logging import Logger
from pathlib import Path
from typing import Tuple, Union, List, Callable, Dict, Type

import numpy as np

from preprocess import EventCounter
from sempca.preprocessing import DataPaths, Preprocessor, BasicDataLoader, BGLLoader
from sempca.representations import SequentialAdd, TemplateTfIdf
from sempca.utils import get_logger

sys.path.append(str(Path(__file__).parent / "neurallog.d"))

from neurallog import data_loader as nl_loader

NdArr = np.ndarray
NdArrPair = Tuple[np.ndarray, np.ndarray]


def exists_and_not_empty(file_path: Union[Path, str]) -> bool:
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
                    + ", ".join(str(self.config[k]) for k in key)
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
        self.config["embeddings"] = paths.word2vec_file

    def get(self):
        """Download the dataset"""
        self._get_embeddings()

    @skip_when_present("embeddings")
    def _get_embeddings(self):
        os.system(f"bash scripts/download.sh embeddings {self.config['dataset_dir']}")

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

    def load_t_seq_representation(self):
        loaded = np.load(self.config["t_seq_npz"], allow_pickle=True)
        return loaded["xs"], loaded["ys"]

    @skip_when_present("t_seq_npz", load=load_t_seq_representation)
    def get_t_seq_representation(self) -> Tuple[NdArr, NdArr]:
        """Load the T-Seq representation of the dataset for loglizer models"""
        x_instances, ys = self.get_instances()

        xs = np.vectorize(lambda x: x.sequence, otypes=[object])(x_instances)
        np.savez_compressed(self.config["t_seq_npz"], xs=xs, ys=ys)

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

    def load_embedding_sequences(self):
        loaded = np.load(self.config["embed_seq_npz"], allow_pickle=True)
        return loaded["block_embed_seqs"], loaded["block_labels"]

    @skip_when_present("embed_seq_npz", load=load_embedding_sequences)
    def get_bert_embedding_sequences(self):
        _, sempca_dataloader, drop_ids = self._parse()
        block2lines = sempca_dataloader.block2seqs
        block2eventseq = sempca_dataloader.block2eventseq

        E, content2content_id, line2content_id = self.load_neurallog()

        content_id2content = {
            c_id: content for content, c_id in content2content_id.items()
        }
        content_id2embedding = {
            c_id: E[content_id2content[c_id]] for c_id in content_id2content.keys()
        }

        if drop_ids is None:
            drop_ids = set()

        print("Start generating instances.")

        block_ids = []
        block_seqs = []
        block_labels = []
        # Prepare semantic embedding sequences for instances.
        for block, label in sempca_dataloader.block2label.items():
            if block in block2eventseq:
                block_embeddings = []

                len_lines = len(block2lines[block])
                len_events = len(block2eventseq[block])
                if len_lines != len_events:
                    print(
                        f"{block}: Different lengths of lines - {len_lines}"
                        f" and events - {len_events}"
                    )

                for x, y in zip(block2lines[block], block2eventseq[block]):
                    if y in drop_ids:
                        continue
                    c_id = line2content_id.get(x, None)
                    if c_id is None:
                        print(f"{block}: Line not found {x}")
                        continue
                    emb = content_id2embedding.get(c_id, None)
                    if emb is None:
                        print(f"{block}: Content not found {c_id}")
                        continue
                    block_embeddings.append(emb)

                block_ids.append(block)
                block_labels.append(0 if label == "Normal" else 1)
                block_seqs.append(block_embeddings)
            else:
                print(f"Found mismatch block: {block}. Please check.")

        block_ids_np = np.array(block_ids, dtype=str)
        block_labels_np = np.array(block_labels, dtype=int)
        block_seqs_np = np.array(block_seqs, dtype=object)
        np.savez_compressed(
            self.config["embed_seq_npz"],
            block_ids=block_ids_np,
            block_labels=block_labels_np,
            block_embed_seqs=block_seqs_np,
        )

        del E, content2content_id, line2content_id
        del block2lines, block_ids, block_labels, block_seqs
        gc.collect()

        return block_seqs_np, block_labels_np

    @abstractmethod
    def load_neurallog(self) -> Tuple[dict, dict, dict]:
        """Parse files using NeuralLog and return required structures"""

    def get_embedding_and_instances(self):
        preprocessor, dl, d_ids = self._parse()
        return dl.id2embed, self._get_instances(preprocessor, dl, d_ids)

    def get_instances(self):
        preprocessor, dataloader, drop_ids = self._parse()
        return self._get_instances(preprocessor, dataloader, drop_ids)

    def _get_instances(self, preprocessor, dataloader, drop_ids):
        instances = preprocessor.generate_instances(dataloader, drop_ids=drop_ids)
        ys = np.asarray(
            [int(preprocessor.label2id[inst.label]) for inst in instances], dtype=int
        )

        return np.asarray(instances, dtype=object), ys

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
        super().get()
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

    def load_neurallog(self) -> Tuple[dict, dict, dict]:
        """Parse files using NeuralLog and return required structures"""
        _data, E, content2content_id, line2content_id = nl_loader.load_HDFS_file(
            self.config["dataset"],
            "bert",
            skip_multi_blk=False,
        )
        return E, content2content_id, line2content_id


class HDFSFixed(HDFS):
    """
    This class parses the HDFS dataset, but merges all templates matching
    `BLOCK* ask [IPANDPORT] to delete .*` into a single template.
    """

    log = get_logger("DataLoader.HDFSFixed")

    def __init__(self, config: dict, paths: DataPaths):
        for key in ["dataset", "labels", "processed_labels"]:
            config[key] = config[key].replace("HDFSFixed", "HDFS")
        config["dataset_dir_orig"] = config["dataset_dir"].replace("HDFSFixed", "HDFS")

        paths.in_file = config["dataset"]
        paths.label_file = config["processed_labels"]

        paths.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.log.debug("%s", config)
        self.log.debug("%s", paths)
        super().__init__(config, paths)

    @skip_when_present(["dataset", "labels"])
    def _get_dataset(self):
        os.system(f"bash scripts/download.sh HDFS {self.config['dataset_dir_orig']}")

    def _matches(self, template: str) -> bool:
        """
        Check if the template matches the pattern for merging.
        """
        return "BLOCK* ask" in template and "to delete" in template

    def _parse(self) -> Tuple[Preprocessor, BasicDataLoader, set]:
        p, dl, d_ids = super()._parse()

        self.log.info("Pre-merge: %s templates", len(dl.templates))

        # Find the templates to merge
        t_ids = [k for k, t in dl.templates.items() if self._matches(t)]
        assert len(t_ids) >= 1, "Should have found at least one template to merge"

        # Find the global replacement template
        t_lens = [len(dl.templates[t_id]) for t_id in t_ids]
        shortest_template_id = t_ids[np.argmin(t_lens)]
        self.log.debug(
            "Merging templates %s into %s - %s",
            ", ".join(map(str, t_ids)),
            shortest_template_id,
            dl.templates[shortest_template_id],
        )

        # Replace the templates
        t_id_set = set(t_ids)
        for block, seq in dl.block2eventseq.items():
            for i, e_id in enumerate(seq):
                if e_id in t_id_set:
                    seq[i] = shortest_template_id
        for line, e_id in dl.log2temp.items():
            if e_id in t_id_set:
                dl.log2temp[line] = shortest_template_id

        # Remove the old templates
        for t_id in t_ids:
            del dl.templates[t_id]
            if t_id in dl.id2embed:
                del dl.id2embed[t_id]

        self.log.info("Post-merge: %s templates", len(dl.templates))
        return p, dl, d_ids


class BGL(DataLoader):
    log = get_logger("DataLoader.BGL")

    def get(self):
        super().get()
        self._get_dataset()

    @skip_when_present("dataset")
    def _get_dataset(self):
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

    def load_neurallog(self) -> Tuple[dict, dict, dict]:
        """Parse files using NeuralLog and return required structures"""
        _data, E, content2content_id, line2content_id = (
            nl_loader.load_supercomputers_file(
                self.config["dataset"],
                "bert",
            )
        )
        return E, content2content_id, line2content_id


dataloaders: Dict[str, Type[DataLoader]] = {
    "HDFS": HDFS,
    "HDFSFixed": HDFSFixed,
    "BGL": BGL,
}
