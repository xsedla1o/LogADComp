"""
Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

import time
from typing import Callable
from typing import Optional, List

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from dataloader import DataLoader
from sempca.const import device
from sempca.models import LogAnomaly
from sempca.module import Optimizer, Vocab
from sempca.utils import get_logger
from sempca.utils import tqdm
from utils import get_memory_usage
from utils import log_gpu_memory_usage
from .base import ModelPaths
from .sempca_lstm import SemPCALSTMAdapter


class LogAnomalyAdapter(SemPCALSTMAdapter):
    """
    Wrapper for the LogAnomaly model.
    """

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

    def set_paths(self, paths: ModelPaths):
        self._artefact_dir = paths.artefacts
        super().set_paths(paths)

    def transform_representation(self, loader: DataLoader) -> tuple:
        embedding, instances = loader.get_embedding_and_instances()
        self.vocab = Vocab()
        self.vocab.load_from_dict(embedding)
        return instances

    def get_training_trial_objective(self, x_train, y_train, x_val, y_val):
        """Optuna objective function to optimize training hyperparameters"""
        assert self.num_classes is not None, "Call split preprocessing first"
        train_set, _ = self.get_sliding_window_dataset(
            x_train, self.vocab.PAD, normal_only=True
        )
        val_set, _ = self.get_sliding_window_dataset(
            x_val, self.vocab.PAD, normal_only=True
        )

        batch_sizes = [64, 128, 256, 512, 1024, 2048]
        if self.num_classes > 128:
            batch_sizes.pop()
        if self.num_classes > 256:
            batch_sizes.pop()
        if self.num_classes > 512:
            batch_sizes.pop()
        if self.num_classes > 1024:
            batch_sizes.pop()
        if self.num_classes > 2048:
            batch_sizes.pop()
        self.log.debug(
            "N classes: %d, selecting batch size from %s", self.num_classes, batch_sizes
        )

        def objective(trial: optuna.Trial):
            """Return the validation loss for a given set of hyperparameters."""
            hidden_size = trial.suggest_categorical("hidden_size", [128])
            _num_layers = trial.suggest_categorical("num_layers", [2])
            self.epochs = trial.suggest_categorical("epochs", [10, 5])
            self.batch_size = trial.suggest_categorical("batch_size", batch_sizes)
            self.learning_rate = trial.suggest_categorical(
                "learning_rate", [2e-3, 1e-3, 5e-4]
            )
            self.learning_rate_decay = trial.suggest_float(
                "learning_rate_decay", 0.74, 1.0, step=0.01
            )

            model = LogAnomaly(self.vocab, hidden_size, self.vocab.vocab_size, device)
            val_loader = TorchDataLoader(
                val_set, batch_size=self.batch_size, shuffle=False
            )

            def show_memory_usage(_m, e):
                self.log.debug("Memory usage at epoch %d: %s", e, get_memory_usage())

            self.train(
                model,
                train_set,
                val_set,
                run_suffix=f"trial_{trial.number}",
                callbacks=[show_memory_usage],
            )
            val_loss = self.get_val_loss(model, val_loader)
            return val_loss

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

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """
        Fit the model using the provided training instances.
        """
        train_set, _ = self.get_sliding_window_dataset(
            x_train, self.vocab.PAD, normal_only=True
        )
        val_set, _ = self.get_sliding_window_dataset(
            x_val, self.vocab.PAD, normal_only=True
        )
        self.train(self._model, train_set, val_set)
        self._find_num_candidates(x_val, y_val)

    def train(
        self,
        model: LogAnomaly,
        train_set: TensorDataset,
        val_set: TensorDataset = None,
        run_suffix: str = "",
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
        self.training_log = []

        train_loader = TorchDataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )
        n_batches = len(train_loader)
        if val_set is not None:
            val_loader = TorchDataLoader(
                val_set, batch_size=self.batch_size, shuffle=False
            )
        else:
            val_loader = None
        vocab_size = self.vocab.vocab_size

        optimizer = Optimizer(
            filter(lambda p: p.requires_grad, model.model.parameters()),
            lr=self.learning_rate,
            lr_decay=self.learning_rate_decay,
            lr_decay_step=len(train_loader),
        )
        self.log.debug("Scheduler decay %s", self.learning_rate_decay)

        for epoch in range(self.epochs):
            model.model.train()
            start = time.strftime("%H:%M:%S")
            start_lr = optimizer.lr[0]
            self.log.info(
                "Epoch %d starting at %s with learning rate: %s",
                epoch + 1,
                start,
                optimizer.lr,
            )

            total_loss = 0
            for seq, label in tqdm(train_loader):
                seq = seq.to(device)
                qual = F.one_hot(seq, vocab_size).sum(dim=1).float().to(device)
                loss = model.forward((seq, qual, None), label.to(device))

                total_loss += loss.item()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.model.parameters()),
                    max_norm=1,
                )
                optimizer.step()
                optimizer.zero_grad()

            self.log.info("Epoch %d finished.", epoch + 1)
            if model_save_path is not None:
                torch.save(model.model.state_dict(), model_save_path)

            if callbacks is not None:
                for callback in callbacks:
                    callback(model, epoch)

            avg_train_loss = total_loss / n_batches
            self._training_log_epoch(
                epoch, start, model, start_lr, avg_train_loss, val_loader
            )

        self._training_log_save(run_suffix)
        self.log.info("Training complete.")

    def _training_log_epoch(
        self, epoch, start_t, model, lr, avg_train_loss, val_loader
    ):
        if val_loader is not None:
            avg_valid_loss = self.get_val_loss(model, val_loader)
        else:
            avg_valid_loss = None
        self.log.debug(
            "Epoch %d loss - train: %.2f, valid: %.2f",
            epoch + 1,
            avg_train_loss,
            avg_valid_loss,
        )
        self.training_log.append((epoch, start_t, lr, avg_train_loss, avg_valid_loss))

    def _training_log_save(self, run_suffix: str = ""):
        file = "training_log" + (f"_{run_suffix}" if run_suffix else "") + ".csv"
        pd.DataFrame(
            self.training_log,
            columns=["epoch", "time", "lr", "train_loss", "valid_loss"],
        ).to_csv(self._artefact_dir / file)

    def get_val_loss(self, model: LogAnomaly, val_loader: TorchDataLoader):
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
            return total_loss / len(val_loader)

    def predict(self, x_test):
        return self._predict(self._model, x_test)

    def _predict(self, model: LogAnomaly, x_test: np.ndarray):
        vocab_size = self.vocab.vocab_size

        model.model.eval()
        with torch.no_grad():
            dataset, window_counts = self.get_sliding_window_dataset(
                x_test, pad_token=self.vocab.PAD, dtype=torch.long
            )

            loader = TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=False)

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

        return self.reassemble_instances(matches, window_counts)
