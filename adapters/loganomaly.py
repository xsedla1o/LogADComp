import time
from typing import Callable
from typing import Optional, List

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from .sempca_lstm import SemPCALSTMAdapter

from dataloader import DataLoader

from sempca.const import device
from sempca.models import LogAnomaly
from sempca.module import Optimizer, Vocab
from sempca.utils import get_logger, update_sequences
from sempca.utils import tqdm
from utils import calculate_metrics, get_memory_usage
from utils import log_gpu_memory_usage


class LogAnomalyAdapter(SemPCALSTMAdapter):
    def __init__(self, window=10):
        super().__init__(window)
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
        train_e2i = {k: v + 1 for k, v in train_e2i.items()}
        train_e2i["PAD"] = 0
        self.num_classes = len(train_e2i)
        self.log.info("Num classes after padding %d", self.num_classes)

        update_sequences(x_train, train_e2i)
        update_sequences(x_val, train_e2i)
        update_sequences(x_test, train_e2i)

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
                "learning_rate", [1e-3, 2e-3, 5e-3]
            )
            self.learning_rate_decay = trial.suggest_float(
                "learning_rate_decay", 0.85, 0.99, step=0.01
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
