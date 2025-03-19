import os
from time import time
from typing import ContextManager, Union, Dict

import numpy as np
import psutil


class Timed(ContextManager):
    """Timer context manager

    Usage:
    with Timed():
        # code to be timed

    or

    output_dict = {}
    with Timed("job_name", output_dict):
        # code to be timed
    """

    def __init__(self, label: str = None, output_to: dict = None):
        """
        Args:
            label (str): Label to be printed or stored in the output_dict
            output_to (dict): Dictionary to store the time taken, if None,
                the time is printed
        """
        self.label = label or "Time taken"
        self.output_dict = output_to

        if label is not None and output_to is not None:
            self.print = False
        else:
            self.print = True

        self.start = None
        self.end = None

    def go(self):
        self.start = time()
        return self

    def stop(self):
        self.end = time()
        if self.print:
            seconds = self.end - self.start
            if seconds < 60:
                print(f"{self.label}: {seconds:.2f}s ")
            else:
                print(f"{self.label}: {int(seconds / 60)}m {seconds % 60:.2f}s ")
        else:
            self.output_dict[self.label] = self.end - self.start

    def __enter__(self):
        return self.go()

    def __exit__(self, *args):
        self.stop()
        return False


def get_process_memory():
    process = psutil.Process(os.getpid())
    mi = process.memory_info()
    return mi.rss, mi.vms, mi.shared


def format_bytes(bytes):
    if abs(bytes) < 1000:
        return str(bytes) + "B"
    elif abs(bytes) < 1e6:
        return str(round(bytes / 1e3, 2)) + "kB"
    elif abs(bytes) < 1e9:
        return str(round(bytes / 1e6, 2)) + "MB"
    else:
        return str(round(bytes / 1e9, 2)) + "GB"


def get_memory_usage():
    rss, vms, shared = get_process_memory()
    return {
        "rss": format_bytes(rss),
        "vms": format_bytes(vms),
        "shared": format_bytes(shared),
    }


def calculate_metrics(y_true, y_pred) -> Dict[str, Union[int, float]]:
    """
    Calculate evaluation metrics dictionary for precision, recall, f1, tnr, and acc.

    Parameters
    ----------
        y_pred: ndarry, the predicted result list
        y_true: ndarray, the ground truth label list

    Returns
    -------
        dict: dictionary containing evaluation metrics
    """
    y_true = np.array(y_true, copy=False)
    y_pred = np.array(y_pred, copy=False)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum(y_true == 0) - TN
    FN = np.sum(y_true == 1) - TP

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    tnr = TN / (TN + FP + 1e-8)
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tnr": round(tnr, 4),
        "acc": round(acc, 4),
    }
