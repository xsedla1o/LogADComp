"""
Provides a dictionary of model adapters for different log analysis methods.

As multiple unrelated models are used, the adapters are loaded lazily to avoid unnecessary imports.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

import sys
from pathlib import Path
from typing import Dict, Type

from .base import LogADCompAdapter, DualTrialAdapter, ModelPaths

sys.path.append(str(Path(__file__).parent.parent))


def _import_neurallog():
    from .neurallog import NeuralLogAdapter

    return NeuralLogAdapter


def _import_logbert():
    from .logbert import LogBERTAdapter

    return LogBERTAdapter


def _import_logrobust():
    from .logrobust import LogRobustAdapter

    return LogRobustAdapter


def _import_loganomaly():
    from .loganomaly import LogAnomalyAdapter

    return LogAnomalyAdapter


def _import_deeplog():
    from .deeplog import DeepLogAdapter

    return DeepLogAdapter


def _import_sempca():
    from .sempca import SemPCAAdapter

    return SemPCAAdapter


def _import_pca():
    from .sempca import PCAAdapter

    return PCAAdapter


def _import_logcluster():
    from .loglizer import LogClusterAdapter

    return LogClusterAdapter


def _import_svm():
    from .loglizer import SVMAdapter

    return SVMAdapter


adapter_loaders = {
    "SVM": _import_svm,
    "LogCluster": _import_logcluster,
    "SemPCA": _import_sempca,
    "PCA": _import_pca,
    "DeepLog": _import_deeplog,
    "LogAnomaly": _import_loganomaly,
    "LogRobust": _import_logrobust,
    "LogBERT": _import_logbert,
    "NeuralLog": _import_neurallog,
}


class AdapterDict(dict):
    """
    Dictionary to hold model adapters with lazy loading.
    """

    def __getitem__(self, key):
        if key not in self.__dict__ and key in adapter_loaders:
            self.__dict__[key] = adapter_loaders[key]()
        return self.__dict__[key]

    def keys(self):
        return adapter_loaders.keys()

    def items(self):
        return {k: self.__getitem__(k) for k in self.keys()}.items()

    def __iter__(self):
        return iter(self.keys())


model_adapters: Dict[str, Type[LogADCompAdapter]] = AdapterDict()
