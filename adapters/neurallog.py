import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import optuna
import tensorflow as tf
from official.nlp import optimization
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import Sequence

from dataloader import NdArr, DataLoader
from .base import LogADCompAdapter, ModelPaths

p = str(Path(__file__).parent.parent / "neurallog.d")
if p not in sys.path:
    sys.path.append(p)

from neurallog.models import NeuralLog

NTHREADS = int(os.getenv("PBS_NCPUS", os.cpu_count() // 2))
# Set the number of threads used within an individual op for parallelism.
tf.config.threading.set_intra_op_parallelism_threads(NTHREADS)

# Set the number of threads used for parallelism across independent operations.
tf.config.threading.set_inter_op_parallelism_threads(NTHREADS)
print("Setting intra_op parallelism threads to %d" % NTHREADS)


class BatchGenerator(Sequence):
    def __init__(self, X, Y, batch_size, max_len=75, embed_dim=768):
        assert len(X) == len(Y), "X and Y must have the same number of samples"

        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.max_len = max_len
        self.embed_dim = embed_dim

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        x = self.X[
            idx * self.batch_size : min((idx + 1) * self.batch_size, len(self.X))
        ]
        X = np.zeros((len(x), self.max_len, self.embed_dim))
        Y = np.zeros((len(x), 2))
        item_count = 0
        for i in range(
            idx * self.batch_size, min((idx + 1) * self.batch_size, len(self.X))
        ):
            x = self.X[i]
            if len(x) > self.max_len:
                x = x[-self.max_len :]
            x = np.pad(
                np.array(x),
                pad_width=((self.max_len - len(x), 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            X[item_count] = np.reshape(x, [self.max_len, self.embed_dim])
            Y[item_count] = self.Y[i]
            item_count += 1
        return X[:], Y[:, 0]


class NeuralLogAdapter(LogADCompAdapter):
    def __init__(self):
        super().__init__()
        self._model: Model = None
        self._fitted = False

    def set_paths(self, paths: ModelPaths):
        self._model_path = str(paths.cache / "neural_log.hdf5")

    @staticmethod
    def transform_representation(loader: DataLoader) -> Tuple[NdArr, NdArr]:
        return loader.get_bert_embedding_sequences()

    @staticmethod
    def preprocess_split(
        x_train: NdArr, x_val: NdArr, x_test: NdArr
    ) -> tuple[NdArr, NdArr, NdArr]:
        """No split normalization is applied."""
        return x_train, x_val, x_test

    @staticmethod
    def get_trial_objective(x_train, y_train, x_val, y_val, prev_params: dict = None):
        """No hyperparameters for now."""

        def objective(trial: optuna.Trial) -> float:
            return 0.0

        return objective

    def set_params(
        self,
        embed_dim: int = 768,
        ff_dim: int = 2048,
        max_len: int = 75,
        num_heads: int = 12,
        dropout: float = 0.1,
        train_batch_size: int = 256,
        train_epochs: int = 10,
        test_batch_size: int = 1024,
    ):
        """
        Args:
            embed_dim: dimensionality of token (line) embeddings
            ff_dim: hidden layer size in feed forward network in transformer block
            max_len: max sequence length in the dataset (lines)
            num_heads: number of attention heads in the transformer block
            dropout: dropout rate for the transformer block
            train_batch_size: batch size for training
            train_epochs: number of epochs for training
            test_batch_size: batch size for testing
        """
        self._model = NeuralLog(embed_dim, ff_dim, max_len, num_heads, dropout)
        self.train_batch_size = train_batch_size
        self.train_epochs = train_epochs
        self.test_batch_size = test_batch_size
        self._fitted = False

    def fit(self, x_train, y_train, x_val, y_val):
        """Fit the model on the training data, allowing for validation"""
        batch_size = self.train_batch_size

        training_generator = BatchGenerator(x_train, y_train, batch_size)
        validate_generator = BatchGenerator(x_val, y_val, batch_size)

        self.train(
            training_generator,
            validate_generator,
            num_train_samples=len(x_train),
            num_val_samples=len(x_val),
            batch_size=batch_size,
            epoch_num=self.train_epochs,
            model_name=self._model_path,
        )
        self._fitted = True

    def train(
        self,
        training_generator,
        validate_generator,
        num_train_samples,
        num_val_samples,
        batch_size,
        epoch_num,
        model_name=None,
    ):
        epochs = epoch_num
        steps_per_epoch = num_train_samples
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.1 * num_train_steps)

        init_lr = 3e-4
        optimizer = optimization.create_optimizer(
            init_lr=init_lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            optimizer_type="adamw",
        )

        loss_object = SparseCategoricalCrossentropy()

        self._model.compile(loss=loss_object, metrics=["accuracy"], optimizer=optimizer)

        print(self._model.summary())

        # checkpoint
        filepath = model_name
        checkpoint = ModelCheckpoint(
            filepath,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            mode="max",
            save_weights_only=True,
        )
        early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=5,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        callbacks_list = [checkpoint, early_stop]

        self._model.fit(
            training_generator,
            epochs=epoch_num,
            verbose=1,
            validation_data=validate_generator,
            workers=min(16, int(os.getenv("PBS_NCPUS", os.cpu_count()))),
            max_queue_size=32,
            callbacks=callbacks_list,
            shuffle=True,
        )

    def predict(self, x_test):
        """Predict on the test data"""
        if not self._fitted:
            self._model.load_weights(self._model_path)
            self._fitted = True

        batch_size = self.test_batch_size
        x, y = x_test, np.zeros(len(x_test))

        test_loader = BatchGenerator(x, y, batch_size)
        prediction = self._model.predict(
            test_loader,
            workers=16,
            max_queue_size=32,
            verbose=1,
        )
        prediction = np.argmax(prediction, axis=1)

        return prediction
