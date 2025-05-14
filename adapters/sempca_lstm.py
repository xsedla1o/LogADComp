"""
An abstract base adapter class extracting common code SemPCA-LSTM models.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

from abc import ABC
from typing import Tuple, List

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from dataloader import NdArr
from sempca.utils import get_logger
from sempca.utils import update_sequences
from utils import calculate_metrics
from .base import DualTrialAdapter, ModelPaths


class SemPCALSTMAdapter(DualTrialAdapter, ABC):
    def __init__(self, window=10):
        super().__init__()
        self.log = get_logger("SemPCALSTMAdapter")
        self.window = window

    def set_paths(self, paths: ModelPaths):
        """Set the paths for the adapter."""
        self.threshold_trials_path = paths.artefacts / "training_trial_results.csv"

    def get_event2index(self, x_train, x_test):
        """
        Calculate unique events in pre & post for event count vector calculation.

        Args:
            x_train (list): Pre data, including training set and validation set (if any).
            x_test (list): Post data, mostly testing set.

        Returns:
            tuple: Mappings for train and test events.
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
        train_e2i = {k: v + 1 for k, v in train_e2i.items()}
        train_e2i["PAD"] = 0
        self.num_classes = len(train_e2i)
        self.log.info("Num classes after padding %d", self.num_classes)

        update_sequences(x_train, train_e2i)
        update_sequences(x_val, train_e2i)
        update_sequences(x_test, train_e2i)

        return x_train, x_val, x_test

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

    @staticmethod
    def reassemble_instances(matches: torch.Tensor, window_counts: List[int]) -> NdArr:
        """Reassemble the per-instance results using the window_counts."""
        y_pred = []
        start_idx = 0
        for count in window_counts:
            seq_matches = matches[start_idx : start_idx + count]
            sample_pred = 0 if seq_matches.all().item() else 1
            y_pred.append(sample_pred)
            start_idx += count
        return np.asarray(y_pred)

    def _find_num_candidates(self, x_val, y_val):
        """Tune `num_candidates`."""
        assert self.num_classes is not None, "Call split preprocessing first"
        assert self._model is not None, "Model must be trained first"
        trial_results = []

        def objective(trial: optuna.Trial):
            self.num_candidates = trial.suggest_int(
                "num_candidates", 1, self.num_classes
            )

            y_pred = self.predict(x_val)
            metrics = calculate_metrics(y_val, y_pred)
            trial_results.append({"num_candidates": self.num_candidates} | metrics)
            return metrics["f1"]

        study = optuna.create_study(study_name="N Candidates", direction="maximize")
        study.optimize(objective, n_trials=20)
        results_df = pd.DataFrame(trial_results)
        results_df.to_csv(self.threshold_trials_path, index=False)
        self.log.info("Results saved to %s", self.threshold_trials_path)

        n_candidates = results_df.loc[results_df.f1.idxmax(), "num_candidates"]
        self.log.info("Best num_candidates: %s", n_candidates)
        self.num_candidates = n_candidates

    @staticmethod
    def get_trial_objective(x_train, y_train, x_val, y_val, prev_params: dict = None):
        """No params, dummy function."""

        def objective(trial: optuna.Trial):
            return 0.0

        return objective
