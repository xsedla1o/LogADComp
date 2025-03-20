import time
from abc import abstractmethod, ABC
from typing import Union, Type, Dict, Tuple, List, Optional, Callable

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
from sempca.models import PCAPlusPlus, PCA, DeepLog, LogAnomaly, LogRobust
from sempca.module import Optimizer, Vocab
from sempca.utils import (
    get_logger,
    update_sequences,
    update_instances,
)
from sempca.utils import tqdm
from utils import calculate_metrics, get_memory_usage


class LogADCompAdapter(ABC):
    def __init__(self):
        self._model = None

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
    def fit(self, x_train, y_train):
        """Fit the model on the training data"""

    def predict(self, x_test):
        """Predict on the test data"""
        return self._model.predict(x_test)


class DualTrialAdapter(LogADCompAdapter):
    @staticmethod
    @abstractmethod
    def get_training_trial_objective(x_train, y_train, x_val, y_val):
        """Optuna objective function to optimize training hyperparameters"""


class PCAAdapter(LogADCompAdapter):
    def __init__(self):
        super().__init__()
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

    def fit(self, x_train, _y_train=None):
        self._model.fit(x_train)
        self._model.threshold *= self.threshold_mult


class SemPCAAdapter(PCAAdapter):
    def __init__(self):
        super().__init__()
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
    def __init__(self):
        super().__init__()
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

    def fit(self, x_train, y_train):
        self._model.fit(x_train, y_train)


class LogClusterAdapter(LogADCompAdapter):
    def __init__(self):
        super().__init__()
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

    def fit(self, x_train, y_train):
        self._model.fit(x_train[y_train == 0, :])


class DeepLogAdapter(DualTrialAdapter):
    def __init__(self, window=10):
        super().__init__()
        self.log = get_logger("DeepLogAdapter")
        self.window = window
        # self.last_model_output = self.data_paths
        self._model = None  # DeepLog()

    @staticmethod
    def transform_representation(loader: DataLoader) -> Tuple[NdArr, NdArr]:
        return loader.get_instances()

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

    def preprocess_split(
        self, x_train: NdArr, x_val: NdArr, x_test: NdArr
    ) -> Tuple[NdArr, NdArr, NdArr]:
        train_e2i, _test_e2i = self.get_event2index(
            np.concatenate((x_train, x_val), axis=0), x_test
        )
        self.num_classes = len(train_e2i)

        update_sequences(x_train, train_e2i)
        update_sequences(x_val, train_e2i)
        update_sequences(x_test, train_e2i)

        return x_train, x_val, x_test

    def generate_inputs_by_instances(self, instances, window, step=1):
        """
        Generate batched inputs by given instances.
        Parameters
        ----------
        instances: input insances for training.
        window: windows size for sliding window in DeepLog
        step: step size in DeepLog

        Returns
        -------
        TensorDataset of training inputs and labels.
        """
        num_sessions = 0
        inputs = []
        outputs = []
        for inst in instances:
            if inst.label == "Normal":
                num_sessions += 1
                event_list = tuple(map(int, inst.sequence))
                for i in range(0, len(event_list) - window, step):
                    inputs.append(event_list[i : i + window])
                    outputs.append(event_list[i + window])
        self.log.debug("Number of sessions: %s", num_sessions)
        self.log.debug("Number of seqs: %s", len(inputs))
        dataset = TensorDataset(
            torch.tensor(inputs, dtype=torch.float32),
            torch.tensor(outputs, dtype=torch.long),
        )
        return dataset

    def train(
        self,
        x_train,
        _y_train,
        model: DeepLog,
        num_epochs=5,
        batch_size=32,
        lr=0.001,
        window_size=10,
        input_size=1,
        callbacks: Optional[List[Callable[[DeepLog, int], None]]] = None,
    ):
        self.log.info(
            "Starting training with window size: %d, batch size: %d, num_epochs: %d",
            window_size,
            batch_size,
            num_epochs,
        )
        self.log.info("Model: %s", model)
        self.log.info("Number of candidates: %d", self.num_candidates)
        x_train = self.generate_inputs_by_instances(x_train, window=window_size)
        train_loader = TorchDataLoader(x_train, batch_size=batch_size, shuffle=False)
        model = model.to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        total_step = len(train_loader)
        start_time = time.time()
        for epoch in range(num_epochs):
            model.train()
            start = time.strftime("%H:%M:%S")
            self.log.info(
                "Starting epoch: %d | phase: train | start time: %s | learning rate: %f"
                % (epoch + 1, start, optimizer.param_groups[0]["lr"])
            )
            train_loss = 0
            for seq, label in tqdm(train_loader):
                # Forward pass
                seq = seq.view(-1, window_size, input_size).to(device)
                output = model(seq)
                loss = criterion(output, label.to(device))
                # Backward
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            self.log.info(
                "Epoch [{}/{}], train_loss: {:.4f}".format(
                    epoch + 1, num_epochs, train_loss / total_step
                )
            )
            elapsed_time = time.time() - start_time
            self.log.info("elapsed_time: {:.3f}s".format(elapsed_time))

            if callbacks is not None:
                for callback in callbacks:
                    callback(model, epoch)

            # torch.save(model.state_dict(), last_model_output)
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

    def _predict(self, model: DeepLog, x_test: List[Instance]):
        model.to(device)
        with torch.no_grad():
            windows_list = []
            labels_list = []
            window_counts = []

            for instance in x_test:
                seq = instance.sequence.copy()
                pad_length = self.window + 1 - len(seq)
                if pad_length > 0:
                    seq = seq + [-1] * pad_length

                # shape: [L]
                seq_tensor = torch.tensor(seq, dtype=torch.float)
                # Get sliding windows shape: [L - window + 1, window]
                windows = seq_tensor.unfold(0, self.window, 1)[:-1]
                # labels -> shape (L - window,) == (num_windows,)
                labels = seq_tensor[self.window :]

                # (window, 1) -> (num_windows, window, 1)
                windows_list.append(windows.unsqueeze(-1))
                labels_list.append(labels)  # shape: (num_windows,)
                window_counts.append(windows.shape[0])

            # (total_windows, window, 1)
            all_windows = torch.cat(windows_list, dim=0)
            # (total_windows,)
            all_labels = torch.cat(labels_list, dim=0)

            batch_size = 1024
            outputs_list = []
            for i in range(0, all_windows.size(0), batch_size):
                batch_windows = all_windows[i : i + batch_size].to(device)
                batch_outputs = model(batch_windows)  # (batch_size, num_classes)
                outputs_list.append(batch_outputs)

            all_outputs = torch.cat(outputs_list, dim=0)
            # (total_windows, num_classes) -> (total_windows, num_candidates)
            topk_indices = torch.topk(all_outputs, self.num_candidates, dim=1).indices
            # (total_windows,)
            topk_indices = topk_indices.to(all_labels.device)
            matches = (topk_indices == all_labels.unsqueeze(1)).any(dim=1)

            if torch.cuda.is_available():
                allocd = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                total = torch.cuda.get_device_properties(0).total_memory
                self.log.debug(
                    "GPU usage: %d (%d) / %d MB - %.2f%%",
                    allocd // 1024**2,
                    reserved // 1024**2,
                    total // 1024**2,
                    allocd / total * 100,
                )

            # Reassemble the per-instance results using the window_counts.
            y_pred = []
            start_idx = 0
            for count in window_counts:
                seq_matches = matches[start_idx : start_idx + count]
                sample_pred = 0 if seq_matches.all().item() else 1
                y_pred.append(sample_pred)
                start_idx += count
        return np.asarray(y_pred)

    def get_training_trial_objective(self, x_train, y_train, x_val, y_val):
        """Optuna objective function to optimize training hyperparameters."""
        assert self.num_classes is not None, "Call split preprocessing first"
        val_set = self.generate_inputs_by_instances(x_val, self.window)

        def objective(trial: optuna.Trial):
            hidden_size = trial.suggest_categorical("hidden_size", [64])
            num_layers = trial.suggest_categorical("num_layers", [2])
            num_epochs = trial.suggest_categorical("num_epochs", [10])
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 512])
            lr = trial.suggest_float("lr", 1e-4, 1e-2, step=1e-4)

            self.num_candidates = self.num_classes  # dummy value
            model = DeepLog(
                input_dim=1,
                hidden=hidden_size,
                layer=num_layers,
                num_classes=self.num_classes,
            )

            val_loader = TorchDataLoader(val_set, batch_size=batch_size, shuffle=False)

            def show_memory_usage(_m, e):
                self.log.debug("Memory usage at epoch %d: %s", e, get_memory_usage())

            def pruning_callback(mod: DeepLog, epoch: int):
                val_loss = self.get_val_loss(mod, val_loader)
                self.log.info("Validation loss: %.4f", val_loss)

                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            self.train(
                x_train,
                y_train,
                model=model,
                num_epochs=num_epochs,
                batch_size=batch_size,
                lr=lr,
                callbacks=[show_memory_usage, pruning_callback],
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
        self.train(
            x_train,
            y_train,
            model=self._model,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
        )

        def objective(trial: optuna.Trial):
            self.num_candidates = trial.suggest_int(
                "num_candidates", 1, self.num_classes
            )

            model = self._model  # Reuse already trained model if available
            y_pred = self._predict(model, x_val)
            m = calculate_metrics(y_val, y_pred)
            return m["f1"]

        return objective

    def set_params(
        self,
        input_dim: int = 1,
        hidden_size: int = 6,
        num_layers: int = 2,
        num_candidates: int = 5,
        num_epochs: int = 5,
        batch_size: int = 32,
        lr: float = 0.001,
    ):
        assert self.num_classes is not None, "Call split preprocessing first"
        self._model = DeepLog(input_dim, hidden_size, num_layers, self.num_classes)
        self.num_candidates = num_candidates
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

    def fit(self, x_train, y_train):
        self.train(
            x_train,
            y_train,
            model=self._model,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
        )


class LogAnomalyAdapter(DualTrialAdapter):
    def __init__(self, window=10):
        super().__init__()
        self.log = get_logger("LogAnomalyAdapter")
        self.window = window
        self._model: Optional[LogAnomaly] = None
        self.vocab: Optional[Vocab] = None
        self.num_classes = None  # Must be set via preprocessing splits.
        self.num_candidates = None
        self.epochs = None
        self.batch_size = None
        self.learning_rate = None

    def transform_representation(self, loader: DataLoader) -> tuple:
        """
        Use the DataLoader's unified method to obtain instances and labels.
        (Expects the DataLoader to implement a 'get_instances' method.)
        """
        embedding, instances = loader.get_embedding_and_instances()
        self.vocab = Vocab()
        self.vocab.load_from_dict(embedding)
        return instances

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
            val_set = self.generate_inputs_by_instances(x_val, self.window)
            val_loader = TorchDataLoader(
                val_set, batch_size=self.batch_size, shuffle=False
            )

            def show_memory_usage(_m, e):
                self.log.debug("Memory usage at epoch %d: %s", e, get_memory_usage())

            def pruning_callback(mod: LogAnomaly, epoch: int):
                with torch.no_grad():
                    val_loss = self.get_val_loss(mod, val_loader)

                self.log.info("Validation loss: %.4f", val_loss)

                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            self.train(model, x_train, callbacks=[show_memory_usage, pruning_callback])
            val_loss = self.get_val_loss(model, val_loader)
            return val_loss

        return objective

    def get_trial_objective(
        self, x_train, y_train, x_val, y_val, prev_params: dict = None
    ):
        """
        Return an objective function for hyperparameter tuning via optuna.
        This implementation reuses the adapter instance and only instantiates a new underlying
        LogAnomaly model when set_params is called.
        """

        self.set_params(**(prev_params or {}))
        self.train(self._model, x_train)

        def objective(trial):
            # Hyperparameter suggestions.
            self.num_candidates = trial.suggest_int(
                "num_candidates", 1, self.num_classes
            )

            # Evaluate on the validation set.
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

    def generate_inputs_by_instances(self, instances, window, step=1) -> TensorDataset:
        """
        Generate batched inputs by given instances.
        Parameters
        ----------
        instances: input insances for training.
        window: windows size for sliding window in DeepLog
        step: step size in DeepLog

        Returns
        -------
        TensorDataset of training inputs and labels.
        """
        num_sessions = 0
        inputs = []
        outputs = []
        for inst in instances:
            if inst.label == "Normal":
                num_sessions += 1
                event_list = tuple(map(int, inst.sequence))
                for i in range(0, len(event_list) - window, step):
                    inputs.append(event_list[i : i + window])
                    outputs.append(event_list[i + window])
        self.log.debug("Number of sessions: %s", num_sessions)
        self.log.debug("Number of seqs: %s", len(inputs))
        dataset = TensorDataset(
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(outputs, dtype=torch.long),
        )
        return dataset

    def train(
        self,
        model: LogAnomaly,
        x_train: np.ndarray[Instance],
        model_save_path: str = None,
        callbacks: Optional[List[Callable[[LogAnomaly, int], None]]] = None,
    ):
        """
        Train the LogAnomaly model.
        For each batch:
          - Extract sequential data from instances.
          - Use the feature extractor to obtain quantity features.
          - Attach the computed quantities to each instance.
          - Generate training inputs/targets.
          - Compute loss, backpropagate, apply gradient clipping, and update the model.
        """
        self.log.info(
            "Starting training for %d epochs with batch size %d",
            self.epochs,
            self.batch_size,
        )

        train_set = self.generate_inputs_by_instances(x_train, self.window)
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
                batch_iter += 1
                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.model.parameters()),
                    max_norm=1,
                )
                optimizer.step()
                optimizer.zero_grad()
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

    def fit(self, x_train, y_train):
        """
        Fit the model using the provided training instances.
        (Note: For LogAnomaly, y_train is not used directly.)
        """
        self.train(self._model, x_train)

    def predict(self, x_test):
        return self._predict(self._model, x_test)

    def _generate_windows_and_labels(self, instances: np.ndarray[Instance]) -> tuple:
        windows_list = []
        labels_list = []
        window_counts = []

        for instance in instances:
            seq = instance.sequence.copy()
            pad_length = self.window + 1 - len(seq)
            if pad_length > 0:
                seq = seq + [self.vocab.PAD] * pad_length

            # shape: [L]
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            # Get sliding windows shape: [L - window + 1, window]
            windows = seq_tensor.unfold(0, self.window, 1)[:-1]
            # labels -> shape (L - window,) == (num_windows,)
            labels = seq_tensor[self.window :]

            # (window,) -> (num_windows, window)
            windows_list.append(windows)
            labels_list.append(labels)  # shape: (num_windows,)
            window_counts.append(windows.shape[0])

        # (total_windows, window)
        all_windows = torch.cat(windows_list, dim=0)
        # (total_windows,)
        all_labels = torch.cat(labels_list, dim=0)

        return all_windows, all_labels, window_counts

    def _predict(self, model: LogAnomaly, x_test: np.ndarray[Instance]):
        vocab_size = self.vocab.vocab_size

        model.model.eval()
        with torch.no_grad():
            all_windows, all_labels, window_counts = self._generate_windows_and_labels(
                x_test
            )

            batch_size = 1024
            outputs_list = []
            for i in range(0, all_windows.size(0), batch_size):
                batch_windows = all_windows[i : i + batch_size].to(device)
                qual = (
                    F.one_hot(batch_windows, vocab_size).sum(dim=1).float().to(device)
                )
                batch_outputs = model.model((batch_windows, qual, None))
                outputs_list.append(batch_outputs)

            all_outputs = torch.cat(outputs_list, dim=0)
            # (total_windows, num_classes) -> (total_windows, num_candidates)
            topk_indices = torch.topk(all_outputs, self.num_candidates, dim=1).indices
            # (total_windows,)
            topk_indices = topk_indices.to(all_labels.device)
            matches = (topk_indices == all_labels.unsqueeze(1)).any(dim=1)

            if torch.cuda.is_available():
                allocd = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                total = torch.cuda.get_device_properties(0).total_memory
                self.log.debug(
                    "GPU usage: %d (%d) / %d MB - %.2f%%",
                    allocd // 1024**2,
                    reserved // 1024**2,
                    total // 1024**2,
                    allocd / total * 100,
                )

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
    def __init__(self, instances: np.ndarray[Instance], labels: np.ndarray[int] = None):
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
    def __init__(self):
        super().__init__()
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
        x_train: np.ndarray[Instance],
        y_train: np.ndarray[int],
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

    def fit(self, x_train, y_train):
        """
        Fit the model using the provided training instances.
        """
        self.train(self._model, x_train, y_train)

    def predict(self, x_test):
        return self._predict(self._model, x_test)

    def _predict(self, model: LogRobust, x_test: np.ndarray[Instance]):
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
            self.epochs = trial.suggest_categorical("epochs", [20])
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
            y_pred = self.predict(x_val)
            metrics = calculate_metrics(y_val, y_pred)
            return metrics["f1"]

        return objective


model_adapters: Dict[str, Type[LogADCompAdapter]] = {
    "PCA": PCAAdapter,
    "SemPCA": SemPCAAdapter,
    "SVM": SVMAdapter,
    "LogCluster": LogClusterAdapter,
    "DeepLog": DeepLogAdapter,
    "LogAnomaly": LogAnomalyAdapter,
    "LogRobust": LogRobustAdapter,
}
