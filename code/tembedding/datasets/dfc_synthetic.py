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
Dynamic functional connectivity (dFC) analysis of resting-state fMRI
data is commonly performed by calculating sliding-window correlations
(SWC). The multivariate timeseries consisting of time points and brain
regions were converted into a series of FC matrices using SWC, as follows
A window is used to select a short segment of the timeseries for all nodes.
The window is then shifted in time by a given step size to extract
overlapping segments, of the same length, for the whole timeseries of a
given subject. We measure FC in each window by estimating covariance from
the precision matrix, regularised with the L1-norm, where the
regularisation parameter is estimated for each subject using
cross-validation.
"""

# Imports
import os
import numpy as np
from types import SimpleNamespace
from tembedding.color_utils import print_yellow, print_white


def make_dfc_synth(dataset_path, dfc_key, label_key):
    """ Generate dFC synthetic dataset.

    Parameters
    ----------
    dataset_path: str
        the path to the numpy compressed array that contains the
        data.
    dfc_key: str
        the key associated to the desired datasets: 'dfc<win_size>'.
    label_key:
        the key associated to the conrresponding labels: 'labels<win_size'.

    Returns
    -------
    X: array (n_samples, n_features)
        the generated samples.
    y: array (n_samples, )
        the integer labels for cluster membership of each sample.
    states: array (n_states, n_rois, n_rois)
        the true clustering centroids.
    meta: SimpleNamespace
        dataset associated meta information.
    """
    print_yellow("  Loading synthetic dataset:")
    print_white(f"  path: {dataset_path}")
    print_white(f"  DFC key: {dfc_key}")
    print_white(f"  label key: {label_key}")
    data = np.load(dataset_path, mmap_mode="r")
    try:
        dfc = data[dfc_key]
        label = data[label_key]
        states = data["states"]
    except:
        keys = list(data.keys())
        raise Exception(f"Invalid keys, must be in {keys}.")
    print_white(f"  DFC: {dfc.shape}")
    print_white(f"  label: {label.shape}")
    print_white(f"  states: {states.shape}")
    n_classes = len(states)
    n_subjects, n_wins, n_rois, _ = dfc.shape
    dfc = dfc.reshape(-1, n_rois, n_rois).astype(float)
    triu_indices = np.triu_indices(n_rois, k=1)
    dfc = dfc.T[triu_indices].T
    win_size = label.shape[-1]
    label = label.reshape(-1, win_size).astype(int)
    label = np.max(label, axis=1)
    print_white(f"  flatten DFC: {dfc.shape}")
    print_white(f"  flatten label: {label.shape}")
    meta = SimpleNamespace(
        n_classes=n_classes, n_subjects=n_subjects, n_wins=n_wins,
        n_rois=n_rois, win_size=win_size)
    return dfc, label, states, meta
