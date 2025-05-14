"""
Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

import time
from typing import Callable
from typing import Optional, Tuple, List

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader as TorchDataLoader, Dataset

from dataloader import DataLoader
from sempca.const import device
from sempca.entities import Instance
from sempca.models import LogRobust
from sempca.module import Optimizer, Vocab
from sempca.utils import get_logger
from sempca.utils import tqdm
from utils import calculate_metrics, get_memory_usage
from .base import LogADCompAdapter


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
    """
    Wrapper for the LogRobust model.
    """

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
            self.epochs = trial.suggest_categorical("epochs", [20])
            self.batch_size = trial.suggest_categorical(
                "batch_size", [32, 64, 128, 256, 512, 1024]
            )
            self.learning_rate = trial.suggest_categorical(
                "learning_rate", [1e-3, 2e-3, 5e-3]
            )
            self.learning_rate_decay = trial.suggest_float(
                "learning_rate_decay", 0.85, 0.99, step=0.01
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
