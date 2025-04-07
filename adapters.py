import os
import shutil
import sys
import time
from abc import abstractmethod, ABC
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Type
from typing import Union, Callable

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset, Dataset

from dataloader import NdArr, DataLoader
from loglizer.loglizer.models import SVM, LogClustering
from preprocess import Normalizer
from sempca.const import device
from sempca.entities import Instance
from sempca.models import LogAnomaly, LogRobust, DeepLog
from sempca.models import PCAPlusPlus, PCA
from sempca.module import Optimizer, Vocab
from sempca.utils import (
    get_logger,
    update_sequences,
    update_instances,
)
from sempca.utils import tqdm
from utils import calculate_metrics, get_memory_usage, Timed
from utils import log_gpu_memory_usage

sys.path.append(str(Path(__file__).parent / "logbert"))

from bert_pytorch.dataset import LogDataset, WordVocab
from bert_pytorch.predict_log import compute_anomaly_bool, evaluate_threshold
from bert_pytorch.dataset.sample import fixed_window_data
from bert_pytorch.model import BERTLog
from bert_pytorch import Predictor, Trainer


@dataclass
class CachePaths:
    """
    DataPaths is a dataclass that contains all the paths to the files and directories
    """

    split: Union[Path, str]

    @staticmethod
    def to_path(path: Union[str, Path, None]) -> Union[Path, None]:
        if path is not None:
            return Path(path)
        return None

    def __post_init__(self):
        self.split = self.to_path(self.split)
        os.makedirs(self.split, exist_ok=True)
        self._subdirs = []

    def register_subdir(self, subdir: Union[str, Path]) -> None:
        """Registers a permanent subdir under the split directory."""
        subdir = self.to_path(subdir).resolve()
        assert str(subdir).startswith(str(self.split)), "Subdir must be under split"
        subdir.mkdir(parents=True, exist_ok=True)
        self._subdirs.append(subdir)

    def clear_split_cache(self):
        """Recursively deletes all contents of the split cache dir."""
        for file in self.split.glob("*"):
            print("Deleting file:", file)
            if file.is_file():
                file.unlink()
            else:
                shutil.rmtree(file)
        # Recreate the split directory
        for subdir in self._subdirs:
            subdir.mkdir(parents=True, exist_ok=True)


class LogADCompAdapter(ABC):
    def __init__(self, paths: CachePaths):
        self._model = None
        self.paths = paths

    @staticmethod
    @abstractmethod
    def transform_representation(loader: DataLoader) -> Tuple[NdArr, NdArr]:
        """Call the loader's method to get the required representation"""

    @staticmethod
    @abstractmethod
    def preprocess_split(
        x_train: NdArr, x_val: NdArr, x_test: NdArr
    ) -> tuple[NdArr, NdArr, NdArr]:
        """Preprocess the splits in ways specific to the model"""

    @staticmethod
    @abstractmethod
    def get_trial_objective(x_train, y_train, x_val, y_val, prev_params: dict = None):
        """Optuna objective function to optimize hyperparameters"""

    @abstractmethod
    def set_params(self, **kwargs):
        """Set the current wrapped model's hyperparameters"""

    @abstractmethod
    def fit(self, x_train, y_train, x_val, y_val):
        """Fit the model on the training data, allowing for validation"""

    def predict(self, x_test):
        """Predict on the test data"""
        return self._model.predict(x_test)


class DualTrialAdapter(LogADCompAdapter):
    @staticmethod
    @abstractmethod
    def get_training_trial_objective(x_train, y_train, x_val, y_val):
        """Optuna objective function to optimize training hyperparameters"""


class PCAAdapter(LogADCompAdapter):
    def __init__(self, paths: CachePaths):
        super().__init__(paths)
        self._model = PCA()
        self.threshold_mult = 1.0

    @staticmethod
    def transform_representation(loader: DataLoader) -> Tuple[NdArr, NdArr]:
        return loader.get_ecv_representation()

    @staticmethod
    def preprocess_split(
        x_train: NdArr, x_val: NdArr, x_test: NdArr
    ) -> Tuple[NdArr, NdArr, NdArr]:
        norm = Normalizer(term_weighting="tf-idf", normalization="zero-mean")
        x_train = norm.fit_transform(x_train)
        x_val = norm.transform(x_val)
        x_test = norm.transform(x_test)
        return x_train, x_val, x_test

    @staticmethod
    def get_trial_objective(x_train, y_train, x_val, y_val, prev_params: dict = None):
        def objective(trial: optuna.Trial):
            threshold_mult = trial.suggest_float("threshold_mult", 0.1, 2.0)
            model = PCA(
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

    def set_params(
        self,
        n_components: Union[int, float] = 0.95,
        c_alpha: float = 3.2905,
        threshold: float = None,
        threshold_mult: float = 1.0,
    ):
        self._model = PCA(
            n_components=n_components, threshold=threshold, c_alpha=c_alpha
        )
        self.threshold_mult = threshold_mult

    def fit(self, x_train, *args, **kwargs):
        self._model.fit(x_train)
        self._model.threshold *= self.threshold_mult


class SemPCAAdapter(PCAAdapter):
    def __init__(self, paths: CachePaths):
        super().__init__(paths)
        self._model = PCAPlusPlus()
        self.threshold_mult = 1.0

    @staticmethod
    def transform_representation(loader: DataLoader) -> Tuple[NdArr, NdArr]:
        return loader.get_word_vec_representation()

    @staticmethod
    def preprocess_split(
        x_train: NdArr, x_val: NdArr, x_test: NdArr
    ) -> tuple[NdArr, NdArr, NdArr]:
        return x_train, x_val, x_test

    @staticmethod
    def get_trial_objective(x_train, y_train, x_val, y_val, prev_params: dict = None):
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


class SVMAdapter(LogADCompAdapter):
    def __init__(self, paths: CachePaths):
        super().__init__(paths)
        self._model = SVM()

    @staticmethod
    def transform_representation(loader: DataLoader) -> Tuple[NdArr, NdArr]:
        return loader.get_ecv_representation()

    @staticmethod
    def preprocess_split(
        x_train: NdArr, x_val: NdArr, x_test: NdArr
    ) -> Tuple[NdArr, NdArr, NdArr]:
        norm = Normalizer(term_weighting="tf-idf")
        x_train = norm.fit_transform(x_train)
        x_val = norm.transform(x_val)
        x_test = norm.transform(x_test)
        return x_train, x_val, x_test

    @staticmethod
    def get_trial_objective(x_train, y_train, x_val, y_val, prev_params: dict = None):
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

    def fit(self, x_train, y_train, *args, **kwargs):
        self._model.fit(x_train, y_train)


class LogClusterAdapter(LogADCompAdapter):
    def __init__(self, paths: CachePaths):
        super().__init__(paths)
        self._model = LogClustering()

    @staticmethod
    def transform_representation(loader: DataLoader) -> Tuple[NdArr, NdArr]:
        return loader.get_ecv_representation()

    @staticmethod
    def preprocess_split(
        x_train: NdArr, x_val: NdArr, x_test: NdArr
    ) -> Tuple[NdArr, NdArr, NdArr]:
        return x_train, x_val, x_test

    @staticmethod
    def get_trial_objective(x_train, y_train, x_val, y_val, prev_params: dict = None):
        def objective(trial: optuna.Trial):
            model = LogClustering(
                max_dist=trial.suggest_float("max_dist", 0.3, 0.8),
                anomaly_threshold=trial.suggest_float("anomaly_threshold", 0.3, 0.9),
                num_bootstrap_samples=trial.suggest_int(
                    "num_bootstrap_samples", 500, 5000, step=500
                ),
            )
            model.fit(x_train[y_train == 0, :])
            _precision, _recall, f1 = model.evaluate(x_val, y_val)
            return f1

        return objective

    def set_params(
        self, max_dist=0.3, anomaly_threshold=0.3, num_bootstrap_samples=1000
    ):
        self._model = LogClustering(
            max_dist=max_dist,
            anomaly_threshold=anomaly_threshold,
            num_bootstrap_samples=num_bootstrap_samples,
        )

    def fit(self, x_train, y_train, *args, **kwargs):
        self._model.fit(x_train[y_train == 0, :])


class SemPCALSTMAdapter(DualTrialAdapter, ABC):
    def __init__(self, paths: CachePaths, window=10):
        super().__init__(paths)
        self.log = get_logger("SemPCALSTMAdapter")
        self.window = window

    def get_event2index(self, x_train, x_test):
        """
        Calculate unique events in pre & post for event count vector calculation.
        :param x_train: pre data, including training set and validation set(if has)
        :param x_test: post data, mostly testing set
        :return: mappings
        """
        self.log.info("Getting train instances' event-idx mapping.")

        train_event2idx = {}
        test_event2idx = {}

        events = set()
        for inst in x_train:
            events.update((int(event) for event in inst.sequence))
        train_events = sorted(list(events))

        embed_size = len(train_events)
        self.log.info("Embed size: %d in train dataset." % embed_size)
        for idx, event in enumerate(train_events):
            train_event2idx[event] = idx

        self.log.info("Getting test instances' event-idx mapping.")

        events = set()
        for inst in x_test:
            events.update((int(event) for event in inst.sequence))
        test_events = sorted(list(events))

        base = len(train_events)
        increment = 0
        for event in test_events:
            if event not in train_events:
                train_events.append(event)
                test_event2idx[event] = base + increment
                increment += 1
            else:
                test_event2idx[event] = train_event2idx[event]
        embed_size = len(train_events)
        self.log.info("Embed size: %d in test dataset." % embed_size)
        return train_event2idx, test_event2idx

    def get_sliding_window_dataset(
        self,
        instances: np.ndarray,
        pad_token: int,
        normal_only: bool = False,
        step: int = 1,
        dtype: torch.dtype = torch.long,
        input_dim: bool = False,
    ) -> Tuple[TensorDataset, List[int]]:
        """
        Generate sliding windows and corresponding labels from a list of instances.

        Parameters:
          instances: List of Instance objects (each with a .sequence attribute).
          pad_token: The token used for padding (e.g. self.vocab.PAD or -1).
          normal_only: If True, only process instances with label "Normal".
          step: Step size between windows.
          dtype: The desired torch dtype (e.g. torch.long for LogAnomaly, torch.float for DeepLog).
          input_dim: Whether to unsqueeze windows along the last dimension.

        Returns:
          A TensorDataset of windows and labels, and a list of window counts per instance.
        """
        windows_list = []
        labels_list = []
        window_counts = []
        num_sessions = 0

        for instance in instances:
            if normal_only and instance.label != "Normal":
                continue
            num_sessions += 1

            seq = instance.sequence
            pad_length = self.window + 1 - len(seq)
            if pad_length > 0:
                seq = seq + [pad_token] * pad_length

            # Convert to tensor using the specified dtype, shape: [L]
            seq_tensor = torch.tensor(seq, dtype=dtype)
            # Create sliding windows with the provided step, shape: [L - window + 1, window]
            windows = seq_tensor.unfold(0, self.window, step)[:-1]
            # labels -> shape (L - window,) == (num_windows,)
            labels = seq_tensor[self.window :]

            if input_dim:
                windows = windows.unsqueeze(-1)  # e.g., to add a channel dimension

            windows_list.append(windows)
            labels_list.append(labels)
            window_counts.append(windows.shape[0])

        all_windows = torch.cat(windows_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        all_labels = all_labels.to(torch.long)
        self.log.debug(
            "Number of sessions: %d, windows: %d", num_sessions, len(all_windows)
        )
        return TensorDataset(all_windows, all_labels), window_counts


class DeepLogAdapter(SemPCALSTMAdapter):
    def __init__(self, paths: CachePaths, window=10, in_size=1):
        super().__init__(paths, window)
        self.log = get_logger("DeepLogAdapter")
        self._model: Optional[DeepLog] = None
        self.pad = 0

        self.window = window
        self.in_size = in_size

    @staticmethod
    def transform_representation(loader: DataLoader) -> Tuple[NdArr, NdArr]:
        return loader.get_instances()

    def preprocess_split(
        self, x_train: NdArr, x_val: NdArr, x_test: NdArr
    ) -> Tuple[NdArr, NdArr, NdArr]:
        train_e2i, _test_e2i = self.get_event2index(
            np.concatenate((x_train, x_val), axis=0), x_test
        )
        train_e2i = {k: v + 1 for k, v in train_e2i.items()}
        train_e2i["PAD"] = 0
        self.num_classes = len(train_e2i)
        self.log.info("Num classes after padding %d", self.num_classes)

        update_sequences(x_train, train_e2i)
        update_sequences(x_val, train_e2i)
        update_sequences(x_test, train_e2i)

        return x_train, x_val, x_test

    def get_training_trial_objective(self, x_train, y_train, x_val, y_val):
        """Optuna objective function to optimize training hyperparameters."""
        assert self.num_classes is not None, "Call split preprocessing first"
        train_set, _ = self.get_sliding_window_dataset(
            x_train, self.pad, normal_only=True, dtype=torch.float32
        )
        val_set, _ = self.get_sliding_window_dataset(
            x_val, self.pad, normal_only=True, dtype=torch.float32
        )

        def objective(trial: optuna.Trial):
            hidden_size = trial.suggest_categorical("hidden_size", [64])
            num_layers = trial.suggest_categorical("num_layers", [2])
            self.num_epochs = trial.suggest_categorical("num_epochs", [10])
            self.batch_size = trial.suggest_categorical(
                "batch_size", [32, 64, 128, 512]
            )
            self.lr = trial.suggest_float("lr", 1e-4, 1e-2, step=1e-4)

            self.num_candidates = self.num_classes  # dummy value
            model = DeepLog(
                input_dim=self.in_size,
                hidden=hidden_size,
                layer=num_layers,
                num_classes=self.num_classes,
            )
            val_loader = TorchDataLoader(
                val_set, batch_size=self.batch_size, shuffle=False
            )

            def show_memory_usage(_m, e):
                self.log.debug("Memory usage at epoch %d: %s", e, get_memory_usage())

            def pruning_callback(mod: DeepLog, epoch: int):
                val_loss = self.get_val_loss(mod, val_loader)
                self.log.info("Validation loss: %.4f", val_loss)

                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            self.train(
                model,
                train_set,
                callbacks=[
                    show_memory_usage,
                    lambda _, _b: log_gpu_memory_usage(self.log),
                    pruning_callback,
                ],
            )
            val_loss = self.get_val_loss(model, val_loader)
            return val_loss

        return objective

    def get_trial_objective(
        self, x_train, y_train, x_val, y_val, prev_params: dict = None
    ):
        """Objective function for tuning evaluation-specific hyperparameters."""
        assert self.num_classes is not None, "Call split preprocessing first"

        self.set_params(**(prev_params or {}))
        train_set, _ = self.get_sliding_window_dataset(
            x_train, self.pad, normal_only=True, dtype=torch.float32
        )
        self.train(self._model, train_set)

        def objective(trial: optuna.Trial):
            self.num_candidates = trial.suggest_int(
                "num_candidates", 1, self.num_classes
            )

            y_pred = self.predict(x_val)
            m = calculate_metrics(y_val, y_pred)
            return m["f1"]

        return objective

    def set_params(
        self,
        input_dim: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_candidates: int = 5,
        num_epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
    ):
        assert self.num_classes is not None, "Call split preprocessing first"
        self._model = DeepLog(input_dim, hidden_size, num_layers, self.num_classes)
        self.num_candidates = num_candidates
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

    def fit(self, x_train, *args, **kwargs):
        train_set, _ = self.get_sliding_window_dataset(
            x_train, self.pad, normal_only=True, dtype=torch.float32
        )
        self.train(self._model, train_set)

    def train(
        self,
        model: DeepLog,
        train_set: TensorDataset,
        model_save_path: str = None,
        callbacks: Optional[List[Callable[[DeepLog, int], None]]] = None,
    ):
        self.log.info(
            "Starting training with window size: %d, batch size: %d, num_epochs: %d",
            self.window,
            self.batch_size,
            self.num_epochs,
        )
        self.log.info("Model: %s", model)
        self.log.info("Number of candidates: %d", self.num_candidates)
        train_loader = TorchDataLoader(
            train_set, batch_size=self.batch_size, shuffle=False
        )
        model = model.to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        total_step = len(train_loader)
        start_time = time.time()
        for epoch in range(self.num_epochs):
            model.train()
            start = time.strftime("%H:%M:%S")
            self.log.info(
                "Starting epoch: %d | phase: train | start time: %s | learning rate: %f"
                % (epoch + 1, start, optimizer.param_groups[0]["lr"])
            )
            train_loss = 0
            for seq, label in tqdm(train_loader):
                # Forward pass
                seq = seq.view(-1, self.window, self.in_size).to(device)
                output = model(seq)
                loss = criterion(output, label.to(device))
                # Backward
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            self.log.info(
                "Epoch [{}/{}], train_loss: {:.4f}".format(
                    epoch + 1, self.num_epochs, train_loss / total_step
                )
            )
            elapsed_time = time.time() - start_time
            self.log.info("elapsed_time: {:.3f}s".format(elapsed_time))

            if model_save_path is not None:
                torch.save(model.model.state_dict(), model_save_path)
            if callbacks is not None:
                for callback in callbacks:
                    callback(model, epoch)
        self.log.info("Finished Training")

    def get_val_loss(self, model, val_loader):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss = 0
        with torch.no_grad():
            for seq, label in val_loader:
                seq = seq.view(-1, self.window, 1).to(device)
                output = model(seq)
                loss = criterion(output, label.to(device))
                val_loss += loss.item()
        return val_loss

    def predict(self, x_test):
        return self._predict(self._model, x_test)

    def _predict(self, model: DeepLog, x_test: np.ndarray):
        model.to(device)
        with torch.no_grad():
            dataset, window_counts = self.get_sliding_window_dataset(
                x_test, pad_token=self.pad, dtype=torch.float, input_dim=True
            )
            loader = TorchDataLoader(dataset, batch_size=1024, shuffle=False)

            outputs = []
            for seqs, labels in loader:
                seqs, labels = seqs.to(device), labels.to(device)
                # (batch_size, num_classes)
                predictions = model(seqs)
                # (batch_size, num_classes) -> (batch_size, num_candidates)
                topk_indices = torch.topk(
                    predictions, self.num_candidates, dim=1
                ).indices
                # (batch_size,)
                matches = (topk_indices == labels.unsqueeze(1)).any(dim=1)
                outputs.append(matches.cpu())

            log_gpu_memory_usage(self.log)

            matches = torch.cat(outputs, dim=0)

            # Reassemble the per-instance results using the window_counts.
            y_pred = []
            start_idx = 0
            for count in window_counts:
                seq_matches = matches[start_idx : start_idx + count]
                sample_pred = 0 if seq_matches.all().item() else 1
                y_pred.append(sample_pred)
                start_idx += count

        return np.asarray(y_pred)


class LogAnomalyAdapter(SemPCALSTMAdapter):
    def __init__(self, paths: CachePaths, window=10):
        super().__init__(paths, window)
        self.log = get_logger("LogAnomalyAdapter")
        self._model: Optional[LogAnomaly] = None
        self.vocab: Optional[Vocab] = None
        self.num_classes = None  # Must be set via preprocessing splits.
        self.num_candidates = None
        self.epochs = None
        self.batch_size = None
        self.learning_rate = None

    def transform_representation(self, loader: DataLoader) -> tuple:
        embedding, instances = loader.get_embedding_and_instances()
        self.vocab = Vocab()
        self.vocab.load_from_dict(embedding)
        return instances

    def preprocess_split(
        self, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
    ) -> tuple:
        """
        For LogAnomaly, update the training and test splits.
        (Assumes update_instances returns (x_train, x_test, _))
        """
        train_e2i, _test_e2i = self.get_event2index(
            np.concatenate((x_train, x_val), axis=0), x_test
        )
        self.num_classes = len(train_e2i)

        x_train, x_test, _ = update_instances(x_train, x_test)
        x_train, x_val, _ = update_instances(x_train, x_val)

        return x_train, x_val, x_test

    def get_training_trial_objective(self, x_train, y_train, x_val, y_val):
        """Optuna objective function to optimize training hyperparameters"""
        assert self.num_classes is not None, "Call split preprocessing first"
        train_set, _ = self.get_sliding_window_dataset(
            x_train, self.vocab.PAD, normal_only=True
        )
        val_set, _ = self.get_sliding_window_dataset(
            x_val, self.vocab.PAD, normal_only=True
        )

        def objective(trial: optuna.Trial):
            """Return the validation loss for a given set of hyperparameters."""
            hidden_size = trial.suggest_categorical("hidden_size", [128])
            _num_layers = trial.suggest_categorical("num_layers", [2])
            self.epochs = trial.suggest_categorical("epochs", [10])
            self.batch_size = trial.suggest_categorical(
                "batch_size", [128, 512, 1024, 2048]
            )
            self.learning_rate = trial.suggest_categorical(
                "learning_rate", [1e-4, 1e-3, 2e-3, 5e-3, 1e-2]
            )
            self.learning_rate_decay = trial.suggest_float(
                "learning_rate_decay", 0.7, 0.99, step=0.01
            )

            model = LogAnomaly(self.vocab, hidden_size, self.vocab.vocab_size, device)
            val_loader = TorchDataLoader(
                val_set, batch_size=self.batch_size, shuffle=False
            )

            def show_memory_usage(_m, e):
                self.log.debug("Memory usage at epoch %d: %s", e, get_memory_usage())

            def pruning_callback(mod: LogAnomaly, epoch: int):
                val_loss = self.get_val_loss(mod, val_loader)
                self.log.info("Validation loss: %.4f", val_loss)

                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            self.train(
                model,
                train_set,
                callbacks=[
                    show_memory_usage,
                    lambda _, _b: log_gpu_memory_usage(self.log),
                    pruning_callback,
                ],
            )
            val_loss = self.get_val_loss(model, val_loader)
            return val_loss

        return objective

    def get_trial_objective(
        self, x_train, y_train, x_val, y_val, prev_params: dict = None
    ):
        """Objective function for tuning evaluation-specific hyperparameters."""
        assert self.num_classes is not None, "Call split preprocessing first"

        self.set_params(**(prev_params or {}))
        train_set, _ = self.get_sliding_window_dataset(
            x_train, self.vocab.PAD, normal_only=True
        )
        self.train(self._model, train_set)

        def objective(trial):
            self.num_candidates = trial.suggest_int(
                "num_candidates", 1, self.num_classes
            )

            y_pred = self.predict(x_val)
            metrics = calculate_metrics(y_val, y_pred)
            return metrics["f1"]

        return objective

    def set_params(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_candidates: int = None,
        epochs: int = 5,
        batch_size: int = 2048,
        learning_rate: float = 0.001,
        learning_rate_decay: float = 0.75,
    ):
        """
        Set hyperparameters and instantiate the LogAnomaly model.
        It is assumed that self.vocab and self.num_classes have been set prior to this call.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

        self.num_candidates = num_candidates

        self._model = LogAnomaly(self.vocab, hidden_size, self.vocab.vocab_size, device)
        self.log.info("LogAnomaly instantiated: %s", self._model.model)

    def fit(self, x_train: np.ndarray, *args, **kwargs):
        """
        Fit the model using the provided training instances.
        """
        train_set, _ = self.get_sliding_window_dataset(
            x_train, self.vocab.PAD, normal_only=True
        )
        self.train(self._model, train_set)

    def train(
        self,
        model: LogAnomaly,
        train_set: TensorDataset,
        model_save_path: str = None,
        callbacks: Optional[List[Callable[[LogAnomaly, int], None]]] = None,
    ):
        """
        Train the LogAnomaly model.
        """
        self.log.info(
            "Starting training for %d epochs with batch size %d, num_epochs: %d",
            self.epochs,
            self.batch_size,
            self.epochs,
        )
        self.log.info("Model: %s", model)
        self.log.info("Number of candidates: %s", self.num_candidates)

        train_loader = TorchDataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )
        vocab_size = self.vocab.vocab_size

        optimizer = Optimizer(
            filter(lambda p: p.requires_grad, model.model.parameters()),
            lr=self.learning_rate,
            lr_decay=self.learning_rate_decay,
            lr_decay_step=len(train_loader),
        )
        self.log.debug("Scheduler decay %s", self.learning_rate_decay)

        global_step = 0
        batch_step_coef = 2048 // self.batch_size  # 2048 is the default/max batch size
        for epoch in range(self.epochs):
            model.model.train()
            start = time.strftime("%H:%M:%S")
            self.log.info(
                "Epoch %d starting at %s with learning rate: %s",
                epoch + 1,
                start,
                optimizer.lr,
            )

            batch_iter = 0
            total_loss = 0
            last_printed_batch = 0
            for seq, label in tqdm(train_loader):
                seq = seq.to(device)
                qual = F.one_hot(seq, vocab_size).sum(dim=1).float().to(device)
                loss = model.forward((seq, qual, None), label.to(device))

                total_loss += loss.item()
                loss.backward()

                if (batch_iter / batch_step_coef) % 100 == 0:
                    self.log.info(
                        "Step:%d, Epoch:%d, Batch:%d, avg loss:%.2f",
                        global_step,
                        epoch + 1,
                        batch_iter,
                        total_loss / (batch_iter - last_printed_batch + 1),
                    )
                    total_loss = 0
                    last_printed_batch = batch_iter

                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.model.parameters()),
                    max_norm=1,
                )
                optimizer.step()
                optimizer.zero_grad()

                batch_iter += 1
                global_step += 1
            self.log.info("Epoch %d finished.", epoch + 1)
            if model_save_path is not None:
                torch.save(model.model.state_dict(), model_save_path)
            if callbacks is not None:
                for callback in callbacks:
                    callback(model, epoch)
        self.log.info("Training complete.")

    def get_val_loss(self, model, val_loader: TorchDataLoader):
        """
        Compute the validation loss for the given model.
        """
        model.model.eval()
        with torch.no_grad():
            total_loss = 0
            for seq, label in val_loader:
                seq = seq.to(device)
                qual = (
                    F.one_hot(seq, self.vocab.vocab_size).sum(dim=1).float().to(device)
                )
                loss = model.forward((seq, qual, None), label.to(device))
                total_loss += loss.item()
            return total_loss

    def predict(self, x_test):
        return self._predict(self._model, x_test)

    def _predict(self, model: LogAnomaly, x_test: np.ndarray):
        vocab_size = self.vocab.vocab_size

        model.model.eval()
        with torch.no_grad():
            dataset, window_counts = self.get_sliding_window_dataset(
                x_test, pad_token=self.vocab.PAD, dtype=torch.long
            )

            loader = TorchDataLoader(dataset, batch_size=1024, shuffle=False)

            outputs = []
            for seqs, labels in loader:
                seqs, labels = seqs.to(device), labels.to(device)
                qual = F.one_hot(seqs, vocab_size).sum(dim=1).float().to(device)

                predictions = model.model((seqs, qual, None))

                # (batch_size, num_classes) -> (batch_size, num_candidates)
                topk_indices = torch.topk(
                    predictions, self.num_candidates, dim=1
                ).indices
                # (batch_size,)
                matches = (topk_indices == labels.unsqueeze(1)).any(dim=1)

                outputs.append(matches.cpu())

            matches = torch.cat(outputs, dim=0)

            log_gpu_memory_usage(self.log)

            # Reassemble the per-instance results using the window_counts.
            y_pred = []
            start_idx = 0
            for count in window_counts:
                seq_matches = matches[start_idx : start_idx + count]
                sample_pred = 0 if seq_matches.all().item() else 1
                y_pred.append(sample_pred)
                start_idx += count
        return np.asarray(y_pred)


class InstanceDataset(Dataset):
    def __init__(self, instances: np.ndarray, labels: np.ndarray = None):
        self.instances = instances
        if labels is not None:
            self.labels = labels
        else:
            self.labels = np.zeros(len(instances), dtype=int)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx], self.labels[idx]


class LogRobustAdapter(LogADCompAdapter):
    def __init__(self, paths: CachePaths):
        super().__init__(paths)
        self.log = get_logger("LogRobustAdapter")
        self._model: Optional[LogRobust] = None
        self.vocab: Optional[Vocab] = None
        self.num_classes = None  # Must be set via preprocessing splits.
        self.epochs = None
        self.batch_size = None
        self.learning_rate = None
        self.hidden_size = None
        self.num_layers = None

    def transform_representation(self, loader: DataLoader) -> tuple:
        """
        Use the DataLoader's unified method to obtain instances and labels.
        (Expects the DataLoader to implement a 'get_embedding_and_instances' method.)
        """
        embedding, instances = loader.get_embedding_and_instances()
        self.vocab = Vocab()
        self.vocab.load_from_dict(embedding)
        return instances

    def preprocess_split(
        self, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
    ) -> tuple:
        """
        For LogRobust, the event indices are calculated using the training set
        """
        return x_train, x_val, x_test

    def set_params(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        learning_rate_decay: float = 0.75,
    ):
        """
        Set hyperparameters and instantiate the LogRobust model.
        It is assumed that self.vocab has been set prior to this call.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self._model = LogRobust(self.vocab, hidden_size, num_layers, device)
        self.log.info("LogRobust instantiated: %s", self._model.model)

    def _collate_fn(
        self, batch: List[Tuple[Instance, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pads the sequences to the maximum length within the batch and creates a mask.
        """
        instances, labels = zip(*batch)

        seqs = [
            torch.tensor(
                [self.vocab.word2id(event) for event in inst.sequence], dtype=torch.long
            )
            for inst in instances
        ]
        padded_seqs = pad_sequence(seqs, batch_first=True, padding_value=self.vocab.PAD)

        lengths = torch.tensor([len(seq) for seq in seqs])
        mask = (
            torch.arange(padded_seqs.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        ).to(torch.long)

        labels = torch.tensor(labels, dtype=torch.long)
        return padded_seqs, mask, labels

    def train(
        self,
        model: LogRobust,
        x_train: np.ndarray,
        y_train: np.ndarray,
        callbacks: Optional[List[Callable[[LogRobust, int], None]]] = None,
        model_save_path: str = None,
    ):
        """
        Train the LogRobust model.
        """
        self.log.info(
            "Starting training for %d epochs with batch size %d",
            self.epochs,
            self.batch_size,
        )

        train_dataset = InstanceDataset(x_train, y_train)
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

        optimizer = Optimizer(
            filter(lambda p: p.requires_grad, model.model.parameters()),
            lr=self.learning_rate,
            lr_decay=self.learning_rate_decay,
            lr_decay_step=len(train_loader),
        )

        global_step = 0

        for epoch in range(self.epochs):
            model.model.train()
            start = time.strftime("%H:%M:%S")
            self.log.info(
                "Starting epoch: %d | phase: train | start time: %s | learning rate: %s"
                % (epoch + 1, start, optimizer.lr)
            )

            for batch_idx, (inputs, masks, labels) in enumerate(tqdm(train_loader)):
                inputs = inputs.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                loss = model.forward((inputs, masks), labels)
                loss.backward()

                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.model.parameters()),
                    max_norm=1,
                )
                optimizer.step()
                model.model.zero_grad()
                global_step += 1

                if batch_idx % 100 == 0:
                    self.log.info(
                        "Epoch: %d, Batch: %d, Loss: %.4f",
                        epoch + 1,
                        batch_idx,
                        loss.item(),
                    )

            if callbacks is not None:
                for callback in callbacks:
                    callback(model, epoch)

            self.log.info("Training epoch %d finished." % epoch)
            if model_save_path:
                torch.save(model.model.state_dict(), model_save_path)
        self.log.info("Training complete.")

    def fit(self, x_train, y_train, *args, **kwargs):
        """
        Fit the model using the provided training instances.
        """
        self.train(self._model, x_train, y_train)

    def predict(self, x_test):
        return self._predict(self._model, x_test)

    def _predict(self, model: LogRobust, x_test: np.ndarray):
        model.model.eval()

        test_dataset = InstanceDataset(x_test)
        test_loader = TorchDataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

        all_predictions = []

        with torch.no_grad():
            for inputs, mask, _labels in test_loader:
                inputs = inputs.to(device)
                mask = mask.to(device)

                tag_logits = model.model((inputs, mask))
                pred_tags = torch.argmax(tag_logits, dim=1).detach().cpu().numpy()

                all_predictions.extend(pred_tags)

        return np.array(all_predictions)

    def get_trial_objective(
        self, x_train, y_train, x_val, y_val, prev_params: dict = None
    ):
        """Optuna objective function to optimize training hyperparameters."""

        def objective(trial: optuna.Trial):
            """Return the validation loss for a given set of hyperparameters."""
            hidden_size = trial.suggest_categorical("hidden_size", [128])
            num_layers = trial.suggest_categorical("num_layers", [2])
            self.epochs = trial.suggest_categorical("epochs", [40])
            self.batch_size = trial.suggest_categorical(
                "batch_size", [32, 64, 128, 256, 512, 1024]
            )
            self.learning_rate = trial.suggest_categorical(
                "learning_rate", [1e-4, 1e-3, 2e-3, 5e-3, 1e-2]
            )
            self.learning_rate_decay = trial.suggest_float(
                "learning_rate_decay", 0.7, 0.99, step=0.01
            )

            model = LogRobust(self.vocab, hidden_size, num_layers, device)

            def show_memory_usage(_m, e):
                self.log.debug("Memory usage at epoch %d: %s", e, get_memory_usage())

            def val_callback(model: LogRobust, epoch: int):
                with torch.no_grad():
                    y_pred = self._predict(model, x_val)

                metrics = calculate_metrics(y_val, y_pred)
                self.log.info("Validation F1: %.4f", metrics["f1"])

                trial.report(metrics["f1"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.set_params(
                hidden_size=hidden_size,
                num_layers=num_layers,
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
            )

            self.train(
                model, x_train, y_train, callbacks=[show_memory_usage, val_callback]
            )
            y_pred = self._predict(model, x_val)
            metrics = calculate_metrics(y_val, y_pred)
            self.log.info("Final validation F1: %.4f", metrics["f1"])
            return metrics["f1"]

        return objective


CACHED_DATASET_KEYS = [
    "train_path",
    "valid_path",
    "test_normal_path",
    "test_anomaly_path",
]
CACHED_PATH_KEYS = CACHED_DATASET_KEYS + ["vocab_path"]


class LogBERTAdapter(LogADCompAdapter):
    def __init__(self, paths: CachePaths):
        super().__init__(paths)
        self.log = get_logger("LogBERTAdapter")
        self._configure_options()
        self.vocab_size = None
        self.threshold = None

    def _configure_options(self):
        """Configure the options dictionary based on the dataset."""
        self.o = {}  # Options
        self.o["device"] = device
        self.o["output_dir"] = self.paths.split

        model_dir = self.o["output_dir"] / "bert"
        self.paths.register_subdir(model_dir)
        self.o["model_dir"] = model_dir

        self.o["model_path"] = self.o["model_dir"] / "best_bert.pth"
        self.o["train_vocab"] = self.o["output_dir"] / "train"

        self.o["vocab_path"] = self.o["output_dir"] / "vocab.pkl"  # pickle file

        self.o["scale_path"] = self.o["model_dir"] / "scale.pkl"

        self.o["train_path"] = self.o["output_dir"] / "train"
        self.o["valid_path"] = self.o["output_dir"] / "valid"
        self.o["test_normal_path"] = self.o["output_dir"] / "test_normal"
        self.o["test_anomaly_path"] = self.o["output_dir"] / "test_anomaly"

        # Ensure directories exist
        # Convert paths to strings to keep LogBERT happy
        for k, path in self.o.items():
            if isinstance(path, Path):
                if path.is_dir():
                    self.o[k] = str(path) + "/"
                else:
                    self.o[k] = str(path)

        # For the skip_when_present decorator
        self.config = {k: self.o[k] for k in CACHED_PATH_KEYS}

    @staticmethod
    def transform_representation(loader: DataLoader) -> Tuple[NdArr, NdArr]:
        """Adapts the data from the loader."""
        return loader.get_t_seq_representation()

    def preprocess_split(
        self, x_train: NdArr, x_val: NdArr, x_test: NdArr
    ) -> Tuple[NdArr, NdArr, NdArr]:
        """Preprocesses and writes the instances in the LogBERT format."""
        self.log.info("Preprocessing the instances")
        self._prepare_vocab(x_train, x_val)
        return x_train, x_val, x_test

    def _prepare_vocab(self, x_train, x_val):
        vocab = WordVocab(np.concatenate([x_train, x_val], axis=0))
        self.log.debug("Vocab size: %d", len(vocab))
        self.log.debug("Saving to: %s", self.o["vocab_path"])
        self.vocab_size = len(vocab)
        vocab.save_vocab(self.o["vocab_path"])

    @staticmethod
    def _write_to_file(seqs, out_f):
        """Write a list of instances to a file in LogBERT format."""
        with open(out_f, "w") as out_f:
            for seq in seqs:
                out_f.write(" ".join(map(str, seq)))
                out_f.write("\n")

    def set_params(self, threshold: float = 0.2, num_candidates: int = 6):
        """No hyperparameters to set directly for LogBERT, as they are managed internally."""
        self.o["window_size"] = 128
        self.o["adaptive_window"] = True
        self.o["seq_len"] = 512
        self.o["max_len"] = 512  # for position embedding
        self.o["min_len"] = 0
        self.o["mask_ratio"] = 0.65
        # sample ratio
        self.o["train_ratio"] = 1
        self.o["valid_ratio"] = 0.1
        self.o["test_ratio"] = 1

        # features
        self.o["is_logkey"] = True
        self.o["is_time"] = False

        self.o["hypersphere_loss"] = True
        self.o["hypersphere_loss_test"] = False

        self.o["scale"] = None  # MinMaxScaler()

        # model
        self.o["hidden"] = 256  # embedding size
        self.o["layers"] = 4
        self.o["attn_heads"] = 4

        self.o["epochs"] = 200
        self.o["warm_up_epochs"] = 10
        self.o["n_epochs_stop"] = 10
        self.o["batch_size"] = 32

        self.o["corpus_lines"] = None
        self.o["on_memory"] = True
        self.o["num_workers"] = min(5, int(os.getenv("PBS_NCPUS", os.cpu_count())))
        self.o["lr"] = 1e-3
        self.o["adam_beta1"] = 0.9
        self.o["adam_beta2"] = 0.999
        self.o["adam_weight_decay"] = 0.00
        self.o["with_cuda"] = True
        self.o["cuda_devices"] = None
        self.o["log_freq"] = None

        # predict
        self.o["num_candidates"] = num_candidates
        self.o["gaussian_mean"] = 0
        self.o["gaussian_std"] = 1

        self.threshold = threshold

    def get_trial_objective(
        self, x_train, y_train, x_val, y_val, prev_params: dict = None
    ):
        """Tune evaluation-specific hyperparameters.
        - num_candidates: number of candidates to consider for anomaly detection
        - threshold: threshold of missed predictions to trigger anomaly
        """
        assert self.vocab_size is not None, "Call split preprocessing first"
        self.set_params(**prev_params)
        self.fit(x_train, y_train, x_val, y_val)

        x_train_norm = x_train[y_train == 0]
        x_val_norm = x_val[y_val == 0]

        x_train_anomaly = x_train[y_train == 1]
        x_val_anomaly = x_val[y_val == 1]

        x_norm = np.concatenate((x_train_norm, x_val_norm), axis=0)
        x_anomaly = np.concatenate((x_train_anomaly, x_val_anomaly), axis=0)

        def objective(trial: optuna.Trial):
            threshold = trial.suggest_float("threshold", 0.01, 0.3, step=0.01)
            n_candidates = trial.suggest_int("num_candidates", 1, self.vocab_size)

            new_opts = self.o.copy()
            new_opts["num_candidates"] = n_candidates
            with torch.no_grad():
                p, model = self._load_predictor_model(new_opts)
                vocab = WordVocab.load_vocab(p.vocab_path)

                normal_results, _, _ = self._predict_helper(
                    p, model, x_norm, vocab, scale=None
                )
                anomaly_results, _, _ = self._predict_helper(
                    p, model, x_anomaly, vocab, scale=None
                )

                res = evaluate_threshold(
                    normal_results, anomaly_results, p.get_params(), threshold
                )

            if res is None:
                self.log.debug("%s -> N/A", trial.params)
                return 0.0

            FP, TP, TN, FN, P, R, F1 = res
            self.log.debug("%s -> FP: %d, FN: %d, F1: %.4f", trial.params, FP, FN, F1)
            return F1

        return objective

    def fit(self, x_train, y_train, x_val, y_val):
        """Trains the LogBERT model or loads an existing one if available."""
        t = Trainer(self.o)

        x_train_norm = x_train[y_train == 0]
        x_val_norm = x_val[y_val == 0]

        train_seq_x, train_tim_x, _ = self.fixed_windows_from_sequences(
            x_train_norm, t.window_size, t.adaptive_window, t.seq_len, t.min_len
        )
        val_seq_x, val_tim_x, _ = self.fixed_windows_from_sequences(
            x_val_norm, t.window_size, t.adaptive_window, t.seq_len, t.min_len
        )

        t.train_on(train_seq_x, train_tim_x, val_seq_x, val_tim_x)

    def _load_predictor_model(self, options: dict = None) -> Tuple[Predictor, BERTLog]:
        """Load the LogBERT model."""
        if options is None:
            options = self.o
        p = Predictor(options)
        model: BERTLog = torch.load(p.model_path, weights_only=False)
        model.to(p.device)
        model.eval()

        if p.hypersphere_loss:
            center_dict = torch.load(p.model_dir + "best_center.pt", weights_only=False)
            p.center = center_dict["center"]
            p.radius = center_dict["radius"]
        return p, model

    def predict(self, x_test):
        """Makes predictions using LogBERT."""
        with torch.no_grad():
            p, model = self._load_predictor_model()

            vocab = WordVocab.load_vocab(p.vocab_path)

            scale = None
            with Timed("Running model to get anomaly scores"):
                test_normal_results, test_normal_errors, seq_split_cnts = (
                    self._predict_helper(p, model, x_test, vocab, scale)
                )

            with Timed("Detecting anomalies based on threshold"):
                sample_y_pred = compute_anomaly_bool(
                    test_normal_results, p.get_params(), self.threshold
                )

            y_pred = []
            start_idx = 0
            for count in seq_split_cnts:
                sample_pred = sample_y_pred[start_idx : start_idx + count]
                seq_pred = 1 if sample_pred.any() else 0
                y_pred.append(seq_pred)
                start_idx += count
            return np.array(y_pred, dtype=int)

    @staticmethod
    def fixed_windows_from_sequences(
        sequences, window_size, adaptive_window, seq_len, min_len
    ):
        """
        Generate log_seqs and tim_seqs directly from a list of instances without
        converting the sequences to a string.

        Each instance is expected to have a 'sequence' attribute that is a list of tokens.
        Tokens should be either a single value (log key) or a two-element structure [log_key, timestamp].

        :param sequences: List of sequences.
        :param window_size: Window size for segmentation.
        :param adaptive_window: Boolean flag for adaptive windowing.
        :param seq_len: Maximum number of tokens per session.
        :param scale: Optional scaler for time sequences (if needed).
        :param min_len: Minimum sequence length required.
        :return: Tuple (log_seqs, tim_seqs) sorted by descending sequence length.
        """
        log_seqs = []
        tim_seqs = []
        seq_split_cnts = []
        skipped = 0

        for seq in sequences:
            log_seq, tim_seq, split_cnt = fixed_window_data(
                seq,
                window_size,
                adaptive_window=adaptive_window,
                seq_len=seq_len,
                min_len=min_len,
            )
            if split_cnt == 0:
                skipped += 1
                continue

            log_seqs += log_seq
            tim_seqs += tim_seq
            seq_split_cnts.append(split_cnt)

        # Convert to numpy arrays (using dtype=object to accommodate variable lengths).
        log_seqs = np.array(log_seqs, dtype=object)
        tim_seqs = np.array(tim_seqs, dtype=object)

        # Sort sequences by their length in descending order.
        lengths = np.array(list(map(len, log_seqs)))
        sorted_indices = np.argsort(-lengths)
        log_seqs = log_seqs[sorted_indices]
        tim_seqs = tim_seqs[sorted_indices]

        print(f"Processed {len(log_seqs)} sequences, skipped {skipped}")
        assert len(log_seqs) == len(tim_seqs)
        assert len(log_seqs) == len(seq_split_cnts)
        return log_seqs, tim_seqs, seq_split_cnts

    @staticmethod
    def _logkey_detection(p: Predictor, mask_lm_output, bert_labels, masks):
        # shape (num_masked, 2) where each row is [batch_index, token_index]
        m_idxs = torch.nonzero(masks, as_tuple=False)
        # extracts all masked token outputs, flattening to (num_masked, V)
        flat_masked_output = mask_lm_output[m_idxs[:, 0], m_idxs[:, 1]]
        flat_masked_labels = bert_labels[m_idxs[:, 0], m_idxs[:, 1]]

        # Compute how many masked tokens each batch element has
        B = mask_lm_output.size(0)
        batch_idx = m_idxs[:, 0]
        counts = torch.bincount(batch_idx, minlength=B)  # shape: (B,)
        max_count = counts.max().item()

        # Compute the position (within each batch) for each masked token.
        offsets = torch.zeros_like(counts)
        if B > 0:
            offsets[1:] = torch.cumsum(counts, dim=0)[:-1]
        # Each token's position is its overall index minus the offset for its batch.
        flat_0_positions = torch.arange(m_idxs.size(0), device=batch_idx.device)
        positions = flat_0_positions - offsets[batch_idx]

        # Allocate padded tensors, shape (B, max_count, V), (B, max_count)
        padded_masked_output = torch.zeros(
            (B, max_count, mask_lm_output.size(-1)), device=mask_lm_output.device
        )
        padded_masked_labels = torch.full(
            (B, max_count), -1, device=flat_masked_labels.device
        )

        # Use advanced indexing to scatter each token into the proper position
        padded_masked_output[batch_idx, positions] = flat_masked_output
        padded_masked_labels[batch_idx, positions] = flat_masked_labels

        # Create valid mask to disregard padding entries
        valid_mask = padded_masked_labels != -1

        undetected_tokens, _ = p.detect_logkey_anomaly_vectorized(
            padded_masked_output, padded_masked_labels, valid_mask
        )
        return undetected_tokens

    def _predict_helper(
        self, p: Predictor, model: BERTLog, sequences, vocab, scale=None
    ):
        total_results = []
        output_cls = []
        logkey_test, time_test, seq_split_cnts = self.fixed_windows_from_sequences(
            sequences, p.window_size, p.adaptive_window, p.seq_len, p.min_len
        )

        seq_dataset = LogDataset(
            logkey_test,
            time_test,
            vocab,
            seq_len=p.seq_len,
            corpus_lines=p.corpus_lines,
            on_memory=p.on_memory,
            predict_mode=True,
            mask_ratio=p.mask_ratio,
        )

        # use large batch size in test data
        data_loader = TorchDataLoader(
            seq_dataset,
            batch_size=p.batch_size,
            num_workers=p.num_workers,
            collate_fn=seq_dataset.collate_fn,
        )
        telem = defaultdict(float)
        t_begin = time.time()
        for idx, data in enumerate(tqdm(data_loader, desc="Predicting")):
            data = {key: value.to(p.device) for key, value in data.items()}
            ts = time.time()
            result = model(data["bert_input"], data["time_input"])
            telem["forward"] += time.time() - ts

            # mask_lm_output, mask_tm_output: batch_size x session_size x vocab_size
            # cls_output: batch_size x hidden_size
            # bert_label, time_label: batch_size x session_size
            # in session, some logkeys are masked

            mask_lm_output, mask_tm_output, bert_labels = (
                result["logkey_output"],
                result["time_output"],
                data["bert_label"],
            )
            output_cls += result["cls_output"].tolist()

            tsv = time.time()
            undetected_tokens = []
            svdd_labels = []
            total_logkeys = torch.sum(data["bert_input"] > 0, dim=1).tolist()

            masks = bert_labels > 0
            masked_tokens = torch.sum(masks, dim=1).tolist()

            if p.is_logkey:
                undetected_tokens = self._logkey_detection(
                    p, mask_lm_output, bert_labels, masks
                )

            if p.hypersphere_loss_test:
                # detect by deepSVDD distance
                assert result["cls_output"][0].size() == p.center.size()
                dist = torch.sqrt(
                    torch.sum((result["cls_output"] - p.center) ** 2, dim=1)
                )
                svdd_labels = (dist > p.radius).long().tolist()

            total_results.extend(
                (
                    {
                        "num_error": 0,
                        "undetected_tokens": undetected_tokens[j].item()
                        if p.is_logkey
                        else 0,
                        "masked_tokens": masked_tokens[j],
                        "total_logkey": total_logkeys[j],
                        "deepSVDD_label": svdd_labels[j]
                        if p.hypersphere_loss_test
                        else 0,
                    }
                    for j in range(len(bert_labels))
                )
            )
            telem["vectorized detection"] += time.time() - tsv

        t_total = time.time() - t_begin
        telem["unaccounted"] = t_total - sum(telem.values())
        prof_line = ", ".join(
            f"{i + 1}. {k}: {v:.3f}s" for i, (k, v) in enumerate(telem.items())
        )
        print("Profiled: " + prof_line)

        # for hypersphere distance
        return total_results, output_cls, seq_split_cnts


model_adapters: Dict[str, Type[LogADCompAdapter]] = {
    "PCA": PCAAdapter,
    "SemPCA": SemPCAAdapter,
    "SVM": SVMAdapter,
    "LogCluster": LogClusterAdapter,
    "DeepLog": DeepLogAdapter,
    "LogAnomaly": LogAnomalyAdapter,
    "LogRobust": LogRobustAdapter,
    "LogBERT": LogBERTAdapter,
}
