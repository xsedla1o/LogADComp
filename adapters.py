import csv
import os
from abc import abstractmethod, ABC
from typing import Union, Type, Dict, Tuple

import optuna

from dataloader import NdArr, DataLoader
from loglizer.loglizer.models import SVM
from preprocess import Normalizer
from sempca.models import PCAPlusPlus


def exists_and_not_empty(file_path: str) -> bool:
    return (
        os.path.exists(file_path)
        and os.path.isfile(file_path)
        and os.path.getsize(file_path) > 0
    )


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
    def get_trial_objective(x_train, y_train, x_val, y_val):
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


class SemPCAAdapter(LogADCompAdapter):
    def __init__(self):
        super().__init__()
        self.xs = None
        self.ys = None
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
    def get_trial_objective(x_train, y_train, x_val, y_val):
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

    def fit(self, x_train, _y_train=None):
        self._model.fit(x_train)
        self._model.threshold *= self.threshold_mult


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
    def get_trial_objective(x_train, y_train, x_val, y_val):
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


model_adapters: Dict[str, Type[LogADCompAdapter]] = {
    "SemPCA": SemPCAAdapter,
    "SVM": SVMAdapter,
}
