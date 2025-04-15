from abc import ABC
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import TensorDataset

from sempca.utils import get_logger
from .base import DualTrialAdapter


class SemPCALSTMAdapter(DualTrialAdapter, ABC):
    def __init__(self, window=10):
        super().__init__()
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
