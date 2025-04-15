import os
import shutil
from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from typing import Union

from dataloader import NdArr, DataLoader


@dataclass
class ModelPaths:
    """
    ModelPaths is a dataclass that contains all the paths to the files and directories
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
    def get_trial_objective(x_train, y_train, x_val, y_val, prev_params: dict = None):
        """Optuna objective function to optimize hyperparameters"""

    @abstractmethod
    def set_params(self, **kwargs):
        """Set the current wrapped model's hyperparameters"""

    @abstractmethod
    def fit(self, x_train, y_train, x_val, y_val):
        """Fit the model on the training data, allowing for validation"""

    def predict(self, x_test):
        """Predict on the test data"""
        return self._model.predict(x_test)


class DualTrialAdapter(LogADCompAdapter):
    @staticmethod
    @abstractmethod
    def get_training_trial_objective(x_train, y_train, x_val, y_val):
        """Optuna objective function to optimize training hyperparameters"""
