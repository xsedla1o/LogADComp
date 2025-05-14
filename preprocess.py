"""
Preprocessing classes for PCA, SVM and LogCluster models.

This file is based on the original code from the Loglizer project,
`loglizer.preprocessing.FeatureExtractor` class.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

from collections import Counter

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import TransformerMixin, BaseEstimator

from sempca.utils import get_logger


class EventCounter(BaseEstimator, TransformerMixin):
    """
    Extracts event counts from log sequences.
    """
    logger = get_logger("EventCountVectorExtractor", "StaticLogger.log")

    def __init__(self, oov=False, min_count=1):
        """
        Perform event counting on log sequences to produce event count vectors.

        Args:
            oov (bool): Whether to use out-of-vocabulary (OOV) events.
            min_count (int): The minimal occurrence of events. Only valid when `oov` is True. Defaults to 1.
        """
        self.events = None
        self.oov = oov
        self.min_count = min_count

    def fit(self, X: np.ndarray, y=None):
        """
        Fits the EventCounter on the data matrix.

        Args:
            X (ndarray): Log sequences matrix.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            EventCounter: The fitted instance.
        """
        X_counts = [Counter(seq) for seq in X]
        X_df = pd.DataFrame(X_counts).fillna(0)
        X_df = X_df.fillna(0)

        self.events = list(X_df.columns)
        X = X_df.values

        if self.oov:
            if self.min_count > 1:
                idx = np.sum(X > 0, axis=0) >= self.min_count
                self.events = np.array(X_df.columns)[idx].tolist()

        self.fitted_ = True  # Make sklearn happy
        return self

    def transform(self, X: np.ndarray, y=None):
        """
        Transforms the data matrix with trained parameters.

        Args:
            X (ndarray): Log sequences matrix.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            ndarray: The transformed data matrix.
        """
        self.logger.debug("====== Transformed test data summary ======")

        X_counts = [Counter(seq) for seq in X]
        X_df = pd.DataFrame(X_counts).fillna(0)

        empty_events = set(self.events) - set(X_df.columns)
        if empty_events:
            self.logger.debug("Empty events: %s", empty_events)
            empty_df = pd.DataFrame({event: [0] * len(X_df) for event in empty_events})
            X_df = pd.concat([X_df, empty_df], axis=1)
        X = X_df[self.events].values

        if self.oov:
            oov_cols = list(set(X_df.columns) - set(self.events))
            if oov_cols:
                oov_vec = np.sum(X_df[oov_cols].values > 0, axis=1)
            else:
                oov_vec = np.zeros(X_df.shape[0])
            X = np.hstack([X, oov_vec.reshape(X.shape[0], 1)])

        self.logger.debug("Data shape: {}-by-{}".format(X.shape[0], X.shape[1]))
        return X


class Normalizer(BaseEstimator, TransformerMixin):
    """
    Enables term weighting and normalization of log sequences.
    """
    logger = get_logger("Normalizer", "StaticLogger.log")

    def __init__(self, term_weighting=None, normalization=None):
        """
        Initializes the Normalizer.

        Args:
            term_weighting (str, optional): Term weighting method. Can be 'tf-idf' or None. Defaults to None.
            normalization (str, optional): Normalization method. Can be 'zero-mean', 'sigmoid', or None. Defaults to None.
        """
        self.idf_vec = None
        self.mean_vec = None
        self.term_weighting = term_weighting
        self.normalization = normalization

    def fit(self, X: np.ndarray, y=None):
        """
        Fits the Normalizer on the training data and stores parameters.

        Args:
            X (ndarray): Log sequences matrix.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            Normalizer: The fitted instance.
        """
        self.logger.debug("====== Fitting normalization on train data ======")

        num_instance, num_event = X.shape
        X_fit = X.copy()  # work on a copy so as not to modify the original X

        if self.term_weighting == "tf-idf":
            df_vec = np.sum(X_fit > 0, axis=0)
            self.idf_vec = np.log(num_instance / (df_vec + 1e-8))
            X_fit = X_fit * self.idf_vec

        if self.normalization == "zero-mean":
            mean_vec = X_fit.mean(axis=0)
            self.mean_vec = mean_vec.reshape(1, num_event)
        elif self.normalization == "sigmoid":
            X_fit[X_fit != 0] = expit(X_fit[X_fit != 0])

        self.fitted_ = True  # Make sklearn happy
        return self

    def transform(self, X: np.ndarray, y=None):
        """
        Transforms the data matrix with trained parameters.

        Args:
            X (ndarray): Log sequences matrix.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            ndarray: The transformed data matrix.
        """
        self.logger.debug("====== Normalizing data ======")

        if self.term_weighting == "tf-idf":
            X = X * self.idf_vec
        if self.normalization == "zero-mean":
            X = X - self.mean_vec
        elif self.normalization == "sigmoid":
            X[X != 0] = expit(X[X != 0])

        self.logger.debug("Data shape: {}-by-{}".format(X.shape[0], X.shape[1]))
        return X
