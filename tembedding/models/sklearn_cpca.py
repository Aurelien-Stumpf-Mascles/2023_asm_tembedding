# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
# From: https://github.com/abidlabs/contrastive
##########################################################################


# Imports
import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.metrics import calinski_harabasz_score
from sklearn.model_selection import StratifiedKFold
from .cpca import CPCA


class SKCPCA(BaseEstimator):
    """ Contrastive PCA with a scikit-learn API.
    """
    def __init__(self, X_background, n_components=2, standardize=True,
                 alpha=1):
        """ Init class.

        Parameters
        ----------
        X_background: array with shape (n_data_points, n_features)
            background dataset in which the interesting directions that we
            would like to discover are absent or unenriched.
        n_components: int, default 2
            the number of contrastive components.
        standardize: bool, default True
            wether to standardize the foreground and background datasets.
        alpha: flait, default 1
            hyper-parameter to constrast the foreground and background
            datasets.

        """
        self.n_components = n_components
        self.standardize = standardize
        self.alpha = alpha
        self.X_background = X_background

    def fit(self, X, y=None):
        """ Fit the model with X.

        Parameters
        ----------
        X: dict of array with shape (n_data_points, n_features)
            foreground dataset in which the interesting directions that we
            would like to discover are present or enriched.
            background: 
        y: Ignored
            ignored.

        Returns
        -------
        self: object
            returns the instance itself.
        """
        self.model = CPCA(n_components=self.n_components,
                          standardize=self.standardize)
        self.model.fit(X, self.X_background)
        return self

    def transform(self, X):
        """ Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            new data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_transformed: array of shape (n_samples, n_components)
            projection of X in the first principal components, where
            `n_samples` is the number of samples and `n_components` is the
            number of the components.
        """
        X_transformed = self.model.cpca_alpha(X, self.alpha)
        return X_transformed

    def fit_transform(self, X, y=None):
        """ Fit the model with X and apply the dimensionality reduction on the
        foreground.

        Parameters
        ----------
        X: dict of array with shape (n_data_points, n_features)
            foreground: dataset in which the interesting directions that we
            would like to discover are present or enriched.
            background: dataset in which the interesting directions that we
            would like to discover are absent or unenriched.
        y: Ignored
            ignored.

        Returns
        -------
        X_transformed: array of shape (n_samples, n_components)
            transformed values.
        """
        self.fit(X, y)
        return self.transform(X)

    def score(self, X, y=None):
        """ Return the Calinski and Harabasz score.

        The score is defined as ratio of the sum of between-cluster
        dispersion and of within-cluster dispersion.

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            the data.
        y: array of shape (n_samples, )
            some labels.

        Returns
        -------
        ll : float
            average log-likelihood of the samples under the current model.
        """
        X_new = self.transform(X)
        metrics = []
        for proj in X_new.T:
            proj.shape += (1, )
            metrics.append(calinski_harabasz_score(proj, y))
        return max(metrics)


if __name__ == "__main__":

    # Create dataset
    N = 400
    D = 30
    gap = 3
    # > In B, all the data pts are from the same distribution, which has
    # different variances in three subspaces
    B = np.zeros((N, D))
    B[:, 0:10] = np.random.normal(0, 10, (N, 10))
    B[:, 10:20] = np.random.normal(0, 3, (N, 10))
    B[:, 20:30] = np.random.normal(0, 1, (N, 10))
    # > In A there are four clusters.
    A = np.zeros((N, D))
    A[:, 0:10] = np.random.normal(0, 10, (N, 10))
    # group 1
    A[0:100, 10:20] = np.random.normal(0, 1, (100, 10))
    A[0:100, 20:30] = np.random.normal(0, 1, (100, 10))
    # group 2
    A[100:200, 10:20] = np.random.normal(0, 1, (100, 10))
    A[100:200, 20:30] = np.random.normal(gap, 1, (100, 10))
    # group 3
    A[200:300, 10:20] = np.random.normal(2 * gap, 1, (100, 10))
    A[200:300, 20:30] = np.random.normal(0, 1, (100, 10))
    # group 4
    A[300:400, 10:20] = np.random.normal(2 * gap, 1, (100, 10))
    A[300:400, 20:30] = np.random.normal(gap, 1, (100, 10))
    A_labels = [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100

    # Train model
    foreground_data, background_data = A, B
    foreground_labels = A_labels
    print(f"foreground: {foreground_data.shape}")
    print(f"backgroud: {background_data.shape}")
    cpca = SKCPCA(background_data, n_components=2, standardize=False,
                  alpha=7.8)
    X_new = cpca.fit_transform(foreground_data)
    print(f"Embedding: {X_new.shape}")

    # Plot result
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.scatter(X_new[:, 0], X_new[:, 1], c=foreground_labels, alpha=0.5)
    plt.show()

    # Run a grid search on alpha
    max_log_alpha = 3
    param_grid = {"alpha": [0] + np.logspace(-1, max_log_alpha, 20).tolist()}
    distributions = {"alpha": uniform(loc=0, scale=10**max_log_alpha)}
    estimator = SKCPCA(background_data, n_components=2, standardize=False)
    cv = StratifiedKFold(n_splits=5)
    gs = GridSearchCV(
        estimator=estimator, param_grid=param_grid, cv=cv, n_jobs=5, verbose=5)
    rs = RandomizedSearchCV(
        estimator=estimator, param_distributions=distributions, random_state=0,
        n_iter=1000, cv=cv, n_jobs=5, verbose=5)
    for search in (gs, ): #, rs):
        print("=" * 6)
        print(search.__class__.__name__)
        print("=" * 6)
        search.fit(foreground_data, y=foreground_labels)
        results = pd.DataFrame(search.cv_results_)
        print(results)
        print(search.best_params_, search.best_score_)
        best_estimator = search.best_estimator_





