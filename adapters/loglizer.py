"""
Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

from typing import Tuple

import optuna
from loglizer.loglizer.models import SVM, LogClustering

from dataloader import NdArr, DataLoader
from preprocess import Normalizer
from .base import LogADCompAdapter


class SVMAdapter(LogADCompAdapter):
    """Wrapper fror the loglizer SVM model, which is a wrapper for sklearn.LinearSVC."""

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

    def fit(self, x_train, y_train, *args, **kwargs):
        self._model.fit(x_train, y_train)


class LogClusterAdapter(LogADCompAdapter):
    """Wrapper for the loglizer LogClustering model."""

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
        norm = Normalizer(term_weighting="tf-idf")
        x_train = norm.fit_transform(x_train)
        x_val = norm.transform(x_val)
        x_test = norm.transform(x_test)
        return x_train, x_val, x_test

    @staticmethod
    def get_trial_objective(x_train, y_train, x_val, y_val, prev_params: dict = None):
        def objective(trial: optuna.Trial):
            max_dist = trial.suggest_float("max_dist", 0.01, 0.65, step=0.01)
            model = LogClustering(
                max_dist=max_dist,
                anomaly_threshold=trial.suggest_float(
                    "anomaly_threshold", max_dist, max_dist + 0.1, step=0.01
                ),
                num_bootstrap_samples=trial.suggest_int(
                    "num_bootstrap_samples", 3000, 5000, step=500
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
