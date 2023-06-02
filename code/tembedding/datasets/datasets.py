# -*- coding: utf-8 -*
##########################################################################
# NSAp - Copyright (C) CEA, 2022
# Olivier Cornelis
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that contains the implemented datasets.
"""

# Imports
import time
import joblib
import numpy as np
import multiprocessing
from sklearn import datasets
from types import SimpleNamespace
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedShuffleSplit, RepeatedStratifiedKFold)
from tembedding.color_utils import print_yellow, print_white, print_cyan
from .dfc_synthetic import make_dfc_synth
from .monkey_data import make_dfc_anesthesia


class Dataset(object):
    """ Load datasets.
    """
    def __init__(self, name="synthetic", test_size=0.1, decimate_ratio=0,
                 random_state=42, **kwargs):
        """ Init class.

        Parameters
        ----------
        name: str, default 'synthetic'
            the name of the dataset: 'synthetic', 'anesthesia'.
        test_size: float, default 0.1
            should be between 0.0 and 1.0 and represent the proportion of the
            dataset to include in the test split.
        decimate_ratio: float, default 1
            optionnaly reduce the dataset size by applying this ratio.
        random_state: int, default 42
            control the randomness for reproducibility.
        kwargs: dict
            contains 'synthetic' and 'anesthesia' datasets locations.
        """
        print_cyan("Loading dataset:")
        tic = time.time()
        n_samples = 1500
        np.random.seed(random_state)
        meta = None
        if name == "synthetic":
            X, y, states, meta = make_dfc_synth(**kwargs)
            dataset = (X, y)
            n_classes = meta.n_classes
            n_samples = len(X)
        elif name == "anesthesia":
            X, y, states, meta = make_dfc_anesthesia(**kwargs)
            dataset = (X, y)
            n_classes = meta.n_classes
            n_samples = len(X)
        else:
            raise NotImplementedError(f"Invalid 'dataset' name: {name}.")
        X, y = dataset
        self.random_state = random_state
        self.n_classes = n_classes
        self.meta = meta
        print_white(f"  number of classes: {n_classes}")
        y = self._sanitize_labels(y)
        if name == "synthetic":
            n_test_subjects = int(meta.n_subjects * test_size)
            cut_index = meta.n_wins * n_test_subjects
            print_white(f"  number of subjects in test set: {n_test_subjects}")
            print_white(f"  cut index: {cut_index}")
            train_index = np.arange(cut_index, len(X))
            test_index = np.arange(cut_index)
        else:
            train_index  = np.where(meta.train_indices)[0]
            test_index = np.where(meta.test_indices)[0]
        if decimate_ratio > 0:
            train_index, test_index = self._decimate([train_index, test_index],
                                                     decimate_ratio)
        self.train_index = train_index
        self.test_index = test_index
        self.n_samples = len(train_index) + len(test_index)
        self.scaler = StandardScaler() # TODO MinMaxScaler or None
        X_train = self.scaler.fit_transform(X[train_index])
        X_test = self.scaler.transform(X[test_index])
        self.dataset = SimpleNamespace(
            X_train=X_train, X_test=X_test,
            y_train=y[train_index], y_test=y[test_index])
        print_white(f"  duration: {int(time.time()-tic)}s")

    def _sanitize_labels(self, y):
        """ Sanitize the input labels: reassign -1.
        """
        max_y = np.max(y)
        y[y == -1] = max_y + 1
        return y

    def _decimate(self, datasets, ratio):
        """ Decimate the input datasets.
        """
        dec_datasets = []
        for arr in datasets:
            n_samples = len(arr)
            indices = np.random.choice(
                n_samples, int((1 - ratio) * n_samples), replace=False)
            dec_datasets.append(arr[indices])
        return dec_datasets

    def get_cv(self, X, y, n_repeats=1):
        """ Get the cross-validation indices.

        Parameters
        ----------
        X: numpy array (n_samples, n_features)
            the data.
        y: numpy array (n_samples, )
            the labels.
        n_repeats: int, default 1
            the number of times cross-validator needs to be repeated.

        Returns
        -------
        cv: list of 2-uplet
            the training/validation indices associated to each fold.
        """
        print_yellow("  Cross-validation indices...")
        skf = RepeatedStratifiedKFold(
            n_splits=5, n_repeats=n_repeats, random_state=self.random_state)
        cv = [item for item in skf.split(X, y)]
        print_white(f"  number of folds: {len(cv)}")
        set_sizes = [(len(train_indices), len(val_indices))
                     for train_indices, val_indices in cv]
        print_white(f"  sets sizes: {set_sizes}")
        return cv

    def get_train_data(self):
        """ Get the train data.

        Returns
        -------
        X: numpy array (n_samples, n_features)
            the data.
        y: numpy array (n_samples, )
            the labels.
        """
        X, y = self.dataset.X_train, self.dataset.y_train
        print_yellow("  train data:")
        print_white(f"  X: {X.shape}")
        print_white(f"  y: {y.shape}")
        return X, y

    def get_test_data(self):
        """ Get the test data.

        Returns
        -------
        X: numpy array (n_samples, n_features)
            the data.
        y: numpy array (n_samples, )
            the labels.
        """
        X, y = self.dataset.X_test, self.dataset.y_test
        print_yellow("  test data:")
        print_white(f"  X: {X.shape}")
        print_white(f"  y: {y.shape}")
        return X, y

    @classmethod
    def get_embeddings(cls, X, n_dim=2, method="pca", cachedir=None,
                       **kwargs):
        """ Create n-d embeddings of the input data.

        Parameters
        ----------
        X: numpy array (n_samples, n_features) or (n_samples, n_samples)
            the data or precomputed distances.
        n_dim: int, default 2
            the output features' dimensions.
        method: str
            the method used to create the embeddings.
        cachedir: str, default None
            optionnaly use smart caching.
        kwargs: dict
            parameters of the reduction method.

        Returns
        -------
        X: numpy array (n_samples, n_dim)
            the associated embeddings.
        """
        print_yellow(f"  {method.upper()}:")
        mem = joblib.Memory(cachedir, verbose=0)
        tic = time.time()
        if method == "pca":
            embeddings, _, _ = cls._pca(
                X, n_components=n_dim, mem=mem, **kwargs)
        elif method == "pca_auto":
            embeddings, _, _ = cls._pca_auto(
                X, n_components=n_dim, mem=mem, **kwargs)
        elif method == "umap":
            embeddings, _ = cls._umap(
                X, n_components=n_dim, mem=mem, **kwargs)
        else:
            raise NotImplementedError(
                f"The '{method}' strategy not yet implemented.")       
        print_white(f"  embeddings: {embeddings.shape}")
        print_white(f"  duration: {int(time.time()-tic)}s")
        return embeddings

    @classmethod
    def _pca(cls, X, n_components, mem, **kwargs):
        """ Use PCA for dimension reduction.
        """
        print_white(f" kwargs: {kwargs}")
        embedder = PCA(n_components=n_components, **kwargs)
        embeddings = mem.eval(embedder.fit_transform, X)
        evar = round(sum(embedder.explained_variance_ratio_) * 100)
        print_white(f"  explained variance: {evar}%")
        return embeddings, evar, embedder

    @classmethod
    def _pca_auto(cls, X, n_components, mem, **kwargs):
        """ Use PCA for dimension reduction, with multiple trials to keep
        the desired estimated variance ratio.
        """
        kwargs["svd_solver"] = "full"
        print_white(f" kwargs: {kwargs}")
        assert n_components > 0 and n_components < 1, (
            "'n_components' must be in ]0, 1[ to  select the number of "
            "components such that the amount of variance that needs to be "
            "explained is greater than the percentage specified by "
            "'n_components'.")
        return cls._pca(X, n_components=n_components, mem=mem, **kwargs)

    @classmethod
    def _umap(self, dists, n_components, mem, **kwargs):
        """ Use UMAP for dimension reduction.
        """
        import umap
        kwargs["metric"] = "precomputed"
        kwargs["n_jobs"] = multiprocessing.cpu_count() - 2
        print_white(f" kwargs: {kwargs}")
        embedder = umap.UMAP(n_components=n_components, **kwargs)
        embeddings = mem.eval(embedder.fit_transform, dists)
        return embeddings, embedder
