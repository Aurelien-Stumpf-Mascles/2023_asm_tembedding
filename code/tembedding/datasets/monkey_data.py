# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Simple loader for the monkey anesthesia and DBS datasets.
"""

# Imports
import os
import numpy as np
import pandas as pd
from types import SimpleNamespace
from tembedding.color_utils import print_yellow, print_white, print_green


def make_dfc_anesthesia(datasetdir):
    """ Generate dFC anesthesia dataset.

    Parameters
    ----------
    datasetdir: str
        path to the dataset location.

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
    print_yellow("  Loading anesthesia dataset:")
    print_white(f"  path: {datasetdir}")
    dfc_file = os.path.join(datasetdir, "inputs.npy")
    meta_file = os.path.join(datasetdir, "metadata.tsv")
    print_white(f"  DFC file: {dfc_file}")
    print_white(f"  description file: {meta_file}")
    train_monkeys = ["almira", "khali", "kimiko", "rana"]
    test_monkey = ["jade"]
    meta_df = pd.read_csv(meta_file, sep="\t")
    print_green(meta_df)
    meta_df["condition"].replace("ketmine", "ketamine", inplace=True)
    train_mask = meta_df["monkey"].isin(train_monkeys).values
    test_mask = meta_df["monkey"].isin(test_monkey).values
    label = meta_df["matlab_kmeans_labels"].values.astype(int)
    dfc = np.load(dfc_file).astype(float).squeeze()
    states = []
    for idx in np.unique(label):
        mask = (label == idx)
        states.append(dfc[mask].mean(axis=0))
    states = np.asarray(states)
    print_white(f"  DFC: {dfc.shape}")
    print_white(f"  label: {label.shape}")
    print_white(f"  states: {states.shape}")
    n_samples, n_rois, _ = dfc.shape
    triu_indices = np.triu_indices(n_rois, k=1)
    dfc = dfc.T[triu_indices].T
    print_white(f"  flatten DFC: {dfc.shape}")
    n_classes = len(states)
    subjects = meta_df["monkey"].unique() 
    n_subjects = subjects.size
    n_wins = (meta_df["monkey"] == subjects[0]).sum()
    meta = SimpleNamespace(
        n_classes=n_classes, n_subjects=n_subjects, n_wins=n_wins,
        n_rois=n_rois, win_size=35, train_indices=train_mask,
        test_indices=test_mask, df=meta_df)
    return dfc, label, states, meta
