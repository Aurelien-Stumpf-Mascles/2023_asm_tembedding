# -*- coding: utf-8 -*-
###############################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
###############################################################################

"""
Module that simplifies static functional connectivity computation.
"""

# Imports
import os
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import LedoitWolf
from sklearn.covariance import OAS
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
from sklearn.covariance import EmpiricalCovariance
from nilearn.connectome import GroupSparseCovarianceCV
from joblib import Memory
from joblib import Parallel, delayed
from statsmodels.stats.moment_helpers import cov2corr


def connectivity(timeseries, outdir, kind="correlation", vectorize=False,
                 alpha=None, njobs=1, verbose=0): 
    """ Estimates static functional connectivity (on the whole time series)
    using several estimation models.

    Parameters
    ----------
    timeseries: array (n_subjects, n_times, n_regions)
        array of BOLD time-series.
    outdir: str
        the destination folder.
    kind: str, optional
        the matrix kind: 'tangent','correlation','partial correlation' or
        'covariance'.
    vectorize: bool, optional
        if True, connectivity matrices are reshaped into 1D arrays and only
        their flattened lower triangular parts are returned: vectorized
        connectivity coefficients do not include the matrices diagonal
        elements.
    alpha: float, default None
        if sparse inverse covariance estimation with an l1-penalized estimator
        is selected, define the regularization parameter (the higher alpha,
        the more regularization, the sparser the inverse covariance), otherwise
        use a cross validation strategy to define this parameter.
    njobs: int, default 1
        the number of parallel jobs.
    verbose: int, optional
        indicate the level of verbosity. By default, nothing is printed.
    
    Returns
    ---------
    conn_static: array (n_subjects, n_regions, n_regions)
        subject-level static functional connectivity matrix.
    conn_static_mean: array (n_regions, n_regions)
        group-level static functional connectivity matrix.        
    """ 
    cache_dir = os.path.join(outdir, "cachedir")
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    mem = Memory(cache_dir, verbose=verbose)
    connectivity_cached = mem.cache(_connectivity)
    return connectivity_cached(timeseries, cache_dir, kind, vectorize, njobs,
                               verbose)


def _connectivity(timeseries, cache_dir, kind="correlation", vectorize=False,
                  alpha=None, njobs=1, verbose=0):
    """ See 'connectivity' documentation.
    """
    # Computing individual functional connectivity
    #cov_estimator = LedoitWolf(
    #    assume_centered=True, 
    #    store_precision=True)
    if kind == "lasso":
        mem = Memory(cache_dir, verbose=verbose)
        graphlasso_cached = mem.cache(_graphlasso)
        conn_static = Parallel(n_jobs=njobs, verbose=verbose)(delayed(
            graphlasso_cached)(ts, alpha=alpha) for ts in timeseries)
        #covariance_estimator = GraphicalLassoCV(cv=5, max_iter=200)
        #conn_static = []
        #for idx, ts in enumerate(timeseries):
        #    covariance_estimator.fit(ts)
        #    cov = covariance_estimator.covariance_
        #    conn_static.append(cov2corr(cov))
        #    print('Covariance matrix has shape {0}.'.format(connectivity.shape))
        conn_static = np.asarray(conn_static)
    else:
        conn_measure = ConnectivityMeasure(
            kind=kind, 
            vectorize=vectorize,
            discard_diagonal=True)
        conn_static = conn_measure.fit_transform(timeseries)
    
    # Computing group functional connectivity
    if kind == "tangent":
        conn_static_mean =  conn_measure.mean_
    else:
        conn_static_mean = conn_static.mean(axis=0)
            
    return conn_static, conn_static_mean


def _graphlasso(ts, alpha=None):
    """ Compute covariance matrix using Graph Lasso.
    If alpha is not provided use Cross-validation to set alpha.
    Finally convert the covariance matrix to obtain a corrrelation matrix.
    """
    if alpha is None:
        covariance_estimator = GraphicalLassoCV(cv=5, max_iter=200)
    else:
        covariance_estimator = GraphicalLasso(alpha=alpha)
    cov = covariance_estimator.fit(ts)
    cov = covariance_estimator.covariance_
    return cov2corr(cov)
