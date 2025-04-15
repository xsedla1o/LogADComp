import time
from typing import Callable
from typing import Optional, Tuple, List

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from dataloader import NdArr, DataLoader
from sempca.const import device
from sempca.models import DeepLog
from sempca.utils import get_logger, update_sequences
from sempca.utils import tqdm
from utils import calculate_metrics, get_memory_usage
from utils import log_gpu_memory_usage
from .sempca_lstm import SemPCALSTMAdapter


class DeepLogAdapter(SemPCALSTMAdapter):
    def __init__(self, window=10, in_size=1):
        super().__init__(window)
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
