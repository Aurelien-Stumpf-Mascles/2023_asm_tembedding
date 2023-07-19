# -*- coding: utf-8 -*-
###############################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
###############################################################################

"""
Module that simplifies dynamic functional connectivity computation.
"""

# Imports
import os
import scipy
import numpy as np
from joblib import Memory
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
import statsmodels.stats.weightstats as sms
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from scipy.stats import zscore
from scipy.cluster.hierarchy import linkage, cut_tree
from nicon.utils import fisher_zscore


def sliding_window(timeseries, win_size, outdir, sliding_step=1, verbose=0): 
    """ Computes sliding-window time-series per subject per region.
    Applies a Tukey window for tapering (to smooth edge effects).

    Parameters
    ----------
    timeseries: array (n_subjects, n_times, n_regions)
        array of BOLD time-series.
    win_size: int
        the window size.
    outdir: str
        the destination folder.
    sliding_step: int, optional
        the sliding step size.
    verbose: int, optional
        indicate the level of verbosity. By default, nothing is printed.

    Returns
    -------
    timeseries_split: array (n_subjects, n_windows, win_size, n_regions)
        array of splitted/smoothed BOLD time-series.
    """
    cache_dir = os.path.join(outdir, "cachedir")
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    mem = Memory(cache_dir, verbose=verbose)
    sliding_window_cached = mem.cache(_sliding_window)
    return sliding_window_cached(timeseries, win_size, sliding_step)


def _sliding_window(timeseries, win_size, sliding_step=1, with_tapering=True):
    """ See 'sliding_window' documentation.
    """
    n_sub, n_vol, n_rois = timeseries.shape
    # n_windows = int(np.ceil((n_vol - win_size) // sliding_step))
    # missing a +1 no ?! if n_vol = win_size, n_windows should be 1, not 0 !
    # n_vol should be >= win_size
    n_windows = int(np.ceil((n_vol - win_size) // sliding_step))+1
    timeseries_split = np.zeros((n_sub, n_windows, win_size, n_rois))
    fulltime_win = np.arange(n_vol)
    slid_wins = sliding_window_1d(
        fulltime_win=fulltime_win,
        n_windows=n_windows,
        win_size=win_size,
        sliding_step=sliding_step)
    for idx1, signal in enumerate(timeseries):
        for idx2, cur_win in enumerate(slid_wins):
            cur_timeserie = signal[cur_win, :]
            if with_tapering:
                for idx3 in range(n_rois):
                    timeseries_split[idx1][idx2][:, idx3] = taper(
                        window=cur_timeserie[:, idx3],
                        win_size=win_size)
            else:
                timeseries_split[idx1][idx2] = cur_timeserie
    return timeseries_split    


def sliding_window_1d(fulltime_win, n_windows, win_size=60, sliding_step=1):
    """ Slide a window along the samples ignoring leftover samples.

    Parameters
    ----------
    fulltime_win: array (n_times)
        the volume indices.
    win_size: int
        the window size.
    sliding_step: int, optional
        the sliding step size.
    
    Returns
    ----------
    out: array (n_windows, win_size)
        array with all sliding windows
    """ 
    out = np.ndarray((n_windows, win_size), dtype=fulltime_win.dtype)
    for idx in range(n_windows):
        start = idx * sliding_step
        stop = start + win_size
        out[idx] = fulltime_win[start: stop]
    return out


def taper(window, win_size=60):
    """ Apply a Tukey window, also known as a tapered cosine window to smooth
    edge effects.

    Removing the average value before tapering to avoid spurious correlations
    at the beginning and end of the windows.

    Parameters
    ----------
    window : array
        the sliding window.
    win_size: int
        the window size.
    
    Returns
    -------
    out: array
        the transformed window.
    """
    taper = scipy.signal.tukey(win_size, alpha=0.5, sym=True)
    # taper = scipy.signal.gaussian(35, std=6, sym=True)
    window -= np.mean(window)
    return window * taper


def cluster_states(conn, n_states, outdir, init=None, ctype="kmeans",
                   return_raw=False, verbose=0):
    """ Dynamic connectivity states estimator using two passes Kmeans.

    Parameters
    ----------
    conn: array (n_subjects, n_windows, n_regions, n_regions)
        array with connectivities.
    n_states: int
        the desired number of states.
    outdir: str
        the destination folder.
    init: int, default None
        initialize cluster centroids using a first clustering on the
        high variance connectivities for each subject.
    ctype: str, default 'kmeans'
        the clustering algorithm: 'kmeans' or 'agglomerative'.
    # njobs: int, default 1
    #     the number of parallel jobs. --> #n_jobs input argument has been deprecated 
    return_raw: bool, default False
        if set return the raw clustering data.
    verbose: int, optional
        indicate the level of verbosity. By default, nothing is printed.

    Returns
    -------
    states: arr (n_states, n_regions, n_regions)
        the computed states.
    labels: arr (n_subjects, n_windows)
        the clustering labels.
    linked: arr
        the hierarchical clustering encoded as a linkage matrix.
    metrics: dict
        some metrics that charterize the partitioning.
    """
    cache_dir = os.path.join(outdir, "cachedir")
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    mem = Memory(cache_dir, verbose=verbose)
    n_subjects, n_windows, n_rois = conn.shape[:-1]
    conn_mix = conn.reshape(-1, n_rois, n_rois)
    iu = np.triu_indices(n_rois, k=1)
    conn_flat = np.asarray([arr[iu] for arr in conn_mix])
    metrics = {}
    if ctype == "kmeans":
        linked = None
        kmeans_cached = mem.cache(_kmeans)
        batch_kmeans_cached = mem.cache(_batch_kmeans)
        if init is not None:
            #conn_highvariance_cached = mem.cache(conn_highvariance)
            conn_highvariance_cached = mem.cache(conn_examplars)
            conn_highvar = conn_highvariance_cached(conn, iu)
            if verbose > 0:
                print("[info] Examplars: ", conn_highvar.shape)
            conn_highvar_flat = np.asarray([arr[iu] for arr in conn_highvar])
            if verbose > 0:
                print("[info] Examplars reshape: ", conn_highvar_flat.shape)
            filter_examplars_cached = mem.cache(filter_examplars)
            conn_highvar_flat = filter_examplars_cached(conn_highvar_flat)
            if verbose > 0:
                print("[info] Examplars filtered: ", conn_highvar_flat.shape)
            cluster_centers, _ = kmeans_cached(
                data=conn_highvar_flat,
                n_clusters=n_states,
                init="k-means++",
                n_init=init,
                max_iter=300,
                # n_jobs=njobs,
                verbose=verbose)
            if verbose > 0:
                print("[info] Clusters init: ", cluster_centers.shape)
            cluster_centers, labels = kmeans_cached(
                data=conn_flat,
                n_clusters=n_states,
                init=cluster_centers,
                n_init=1,
                max_iter=300,
                # n_jobs=njobs,
                verbose=verbose)
        else:
            cluster_centers, labels = kmeans_cached(
                data=conn_flat,
                n_clusters=n_states,
                init="k-means++",
                n_init=10,
                max_iter=300, 
                verbose=verbose)
    elif ctype == "agglomerative":
        linkage_cached = mem.cache(_linkage)
        linked = linkage_cached(
            data=conn_flat,
            #metric="cityblock",
            method="centroid")
        labels = cut_tree(linked, n_clusters=n_states).squeeze()
        cluster_centers = []
        for label in sorted(np.unique(labels)):
            _arr = conn_flat[np.argwhere(labels == label)].squeeze()
            cluster_centers.append(np.mean(_arr, axis=0))
        cluster_centers = np.asarray(cluster_centers)
    else:
        raise ValueError("Undefined '{0}' clustering algorithm.".format(ctype))
    if verbose > 0:
        print("[info] Clusters: ", cluster_centers.shape)
    silhouette_score_cached = mem.cache(silhouette_score)
    metrics["silhouette"] = silhouette_score_cached(
        conn_flat, labels, metric="euclidean")
    calinski_harabasz_score_cached = mem.cache(calinski_harabasz_score)
    metrics["calinski_harabasz"] = calinski_harabasz_score_cached(
        conn_flat, labels)
    metrics["elbow"] = elbow(
        conn_flat, cluster_centers, labels)
    if not return_raw:
        states = np.asarray([_unupper_tri(vec, n_rois, iu)
                             for vec in cluster_centers])
        labels = labels.reshape(n_subjects, n_windows)
    else:
        states = np.asarray(cluster_centers)
        linked = conn_flat
    return states, labels, linked, metrics


def elbow(data, centroids, labels):
    """ Elbow method: calculate square of Euclidean distance of each sample
    from its cluster center and add to current WSS.
    """
    sse = 0  
    for idx in range(len(data)):
      state = centroids[labels[idx]]
      sse += np.sum((data[idx] - state) ** 2)
    return sse


def _batch_kmeans(data, *args, **kwargs):
    """ See 'sklearn.cluster.KMeans' for the documentation.
    """
    clustering = MiniBatchKMeans(*args, **kwargs)
    clustering.fit(data)
    return clustering.cluster_centers_, clustering.labels_

def _kmeans(data, *args, **kwargs):
    """ See 'sklearn.cluster.KMeans' for the documentation.
    """
    clustering = KMeans(*args, **kwargs)
    clustering.fit(data)
    return clustering.cluster_centers_, clustering.labels_


def _linkage(data, *args, **kwargs):
    """ See 'scipy.cluster.hierarchy.linkage' for the documentation.
    """
    linked = linkage(data, *args, **kwargs)
    return linked


def _agglomerative(data, *args, **kwargs):
    """ See 'sklearn.cluster.AgglomerativeClustering' for the documentation.
    """
    clustering = AgglomerativeClustering(*args, **kwargs)
    clustering.fit_predict(data)
    return clustering.labels_


def _unupper_tri(vec, n_rois, iu):
    arr = np.zeros((n_rois, n_rois), dtype=vec.dtype)
    arr[iu] = vec
    arr += arr.T
    return arr


def conn_highvariance(conn):
    """ Identify windows with high variance in connectivity for each subject:

    - calculate the average connectitivy (average of all edges).
    - define a 95% confidence interval on this average.
    - select data points outside (higher values).
    
    Parameters
    ----------
    conn: array (n_subjects, n_windows, n_regions, n_regions)
        array with connectivities.
    
    Returns
    ----------
    highvar_conn: array (n_samples, n_regions, n_regions)
        all windows of high variance.
    """
    conn_mean = conn.mean(axis=(-1, -2))
    highvar_conn = []
    for idx, sub_conn_mean in enumerate(conn_mean):
        stat_desc = sms.DescrStatsW(sub_conn_mean)
        lower, upper = stat_desc.tconfint_mean(
            alpha=0.95, alternative="two-sided")
        ind_highvar = np.argwhere(sub_conn_mean > upper)
        highvar_conn.append(conn[idx][ind_highvar][:, 0])
    print([item.shape for item in highvar_conn])
    return np.vstack(highvar_conn)


def conn_examplars(conn, iu):
    """ Identify examplars as windows with local maxima in FC variance.

    - compute the zscores associated to the correlations std.
    - compute peaks where the absolute normalized variance was higher than 0.5
    - compute one-intervals (from the square signal) middle indices.

    Parameters
    ----------
    conn: array (n_subjects, n_windows, n_regions, n_regions)
        array with connectivities.
    iu: array
        upper triangular indices.

    
    Returns
    -------
    highvar_conn: array (n_samples, n_regions, n_regions)
        all windows of high variance.
    """
    highvar_conn = []
    for idx, sub_conn in enumerate(conn):
        sub_conn_flat = np.asarray([arr[iu] for arr in sub_conn])
        std = np.std(sub_conn_flat, axis=1)
        zscore = scipy.stats.zscore(std)
        square = (zscore > 0.5).astype(int)
        intervals = [[]]
        for _idx, value in enumerate(square):
            if value == 1:
                intervals[-1].append(_idx)
            elif len(intervals[-1]) != 0:
                intervals.append([])
        if len(intervals[-1]) == 0:
            intervals.pop()
        ind_highvar = [int(np.mean(item)) for item in intervals]
        highvar_conn.append(conn[idx][ind_highvar])
    return np.vstack(highvar_conn)


def filter_examplars(conn):
    """ Filter examplars.

    Parameters
    ----------
    conn: array (n_items, n_samples)
        array with connectivities.

    Returns
    -------
    conn: array (n_filtered_items, n_samples)
        array with connectivities.
    """
    av_dist = pairwise_distances(conn, metric="cityblock")
    av_dist = np.mean(av_dist, axis=1)
    av_dist = np.abs(zscore(av_dist))
    return conn[av_dist < 3] 


def dynamic_connectivity(timeseries, cond, N_ROIS, win_size,
                         sliding_step, OUTDIR):
    """
    Compute dynamic connectivity

    Parameters
    ----------
    timeseries : array (n_subjects, n_times, n_regions)
       array of BOLD time-series.
    cond : str, optional
        Acquisition condition.
    N_ROIS : int, 
        Number of region.
    win_size : int
        the window size.
    sliding_step : int
        the sliding step size.
    OUTDIR : str
        the destination folder.

    Returns
    -------
    None.

    """
    if cond is not None:
        print("--- cond ---", cond)
        # Cut into sliding windows the signal
        windows_filename = os.path.join(OUTDIR,
                                        "windows_step{0}TR_{1}.npy".format(
                                            sliding_step, cond))
    else:
        windows_filename = os.path.join(OUTDIR,
                                        "windows_step{0}TR.npy".format(
                                            sliding_step))
    if os.path.isfile(windows_filename):
        ref_windows = np.load(windows_filename)
    else:
        if not timeseries.shape[2] == N_ROIS:
            print("transpose timeserie")
            timeseries = timeseries.transpose(0, 2, 1)
        ref_windows = _sliding_window(timeseries, win_size,
                                      sliding_step=sliding_step, with_tapering=False)
        np.save(windows_filename, ref_windows)
    print("Windows: {0}".format(ref_windows.shape))
    
    windows = ref_windows.reshape(-1, win_size, N_ROIS)
    windows = windows.transpose(0, 2, 1)
    print("Functional windowed time series: {0}".format(windows.shape))
    
    # Compute correlation
    dFC_fisher = np.zeros((len(windows), N_ROIS, N_ROIS))
    for idx in range(len(windows)):
        dFC = np.corrcoef(windows[idx])
        #transformed correlations to a Fisher-Z score, as the bounded nature of
        # Pearson correlation violates certain statistical assumptions
        dFC_fisher[idx] = fisher_zscore(dFC, None, cache=False, verbose=5)
    print("dFC :", dFC_fisher.shape)
    if cond is not None:
        np.save(os.path.join(OUTDIR,"dFCs_{0}.npy".format(cond)), dFC_fisher)
    else:
        np.save(os.path.join(OUTDIR,"dFCs.npy"), dFC_fisher)
