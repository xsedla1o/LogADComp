"""
The interface definition for all the adapter classes.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

import os
import shutil
from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Callable
from typing import Union

import optuna

from dataloader import NdArr, DataLoader


@dataclass
class ModelPaths:
    """
    ModelPaths is a dataclass that contains all the paths to the files and directories

    Used to specify the paths to the cache and artefacts directories for the model.
    """

    cache: Union[Path, str]
    artefacts: Union[Path, str]

    @staticmethod
    def to_path(path: Union[str, Path]) -> Union[Path]:
        if path is not None:
            path = Path(path)
            os.makedirs(path, exist_ok=True)
            return path
        raise ValueError("Path must be a string or Path object")

    def __post_init__(self):
        self.cache = self.to_path(self.cache)
        self.artefacts = self.to_path(self.artefacts)
        self._subdirs = []

    def register_subdir(self, subdir: Union[str, Path]) -> None:
        """Registers a permanent subdir under the split directory."""
        subdir = self.to_path(subdir).resolve()
        assert str(subdir).startswith(str(self.cache)), "Subdir must be under cache"
        subdir.mkdir(parents=True, exist_ok=True)
        self._subdirs.append(subdir)

    def clear_split_cache(self):
        """Recursively deletes all contents of the split cache dir."""
        for file in self.cache.glob("*"):
            print("Deleting file:", file)
            if file.is_file():
                file.unlink()
            else:
                shutil.rmtree(file)
        # Recreate the split directory
        for subdir in self._subdirs:
            subdir.mkdir(parents=True, exist_ok=True)


class LogADCompAdapter(ABC):
    """
    Base class for all the adapters.

    An adapter is a wrapper around a model that allows for easy integration with the
    LogADComp pipeline. It provides a common interface for all the models and handles
    the data loading, preprocessing, and training.
    """

    def __init__(self):
        self._model = None

    def set_paths(self, paths: ModelPaths):
        """Set/Update paths to cache files of the model (none by default)"""
        pass

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
    def get_trial_objective(
        x_train, y_train, x_val, y_val, prev_params: dict = None
    ) -> Callable[[optuna.Trial], float]:
        """Optuna objective function to optimize hyperparameters"""

    @abstractmethod
    def set_params(self, **kwargs):
        """Set the current wrapped model's hyperparameters"""

    @abstractmethod
    def fit(self, x_train: NdArr, y_train: NdArr, x_val: NdArr, y_val: NdArr):
        """Fit the model on the training data, allowing for validation"""

    def predict(self, x_test: NdArr) -> NdArr:
        """Predict on the test data"""
        return self._model.predict(x_test)


class DualTrialAdapter(LogADCompAdapter):
    """
    Base class for all the adapters a dual trial to optimize hyperparameters.
    """

    @staticmethod
    @abstractmethod
    def get_training_trial_objective(x_train, y_train, x_val, y_val):
        """Optuna objective function to optimize training hyperparameters"""
