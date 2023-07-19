# -*- coding: utf-8 -*-
###############################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
###############################################################################

import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from utils import extract_centroids
from utils import extract_signal
from utils import similarity
from utils import fisher_zscore
from static_connectivity import connectivity
from dynamic_connectivity import sliding_window
from dynamic_connectivity import cluster_states
import plotting


n_states = 7
njobs = 40
tr = 1.25
outdir = "/neurospin/nsap/processed/dynamic_networks/data"
atlas = "/neurospin/lbi/monkeyfmri/tmp/Resting_state/data_test/atlas_cocomac/rrMNI.nii"
structural = "/neurospin/nsap/processed/dynamic_networks/data/structural.txt"
labels = ["roi{0}".format(idx) for idx in range(1, 83)]
images = [
    glob.glob("/neurospin/lbi/monkeyfmri/DICOM/3T_dicom/*/*/*/*/resting_state/"
              "awake_bold/run*_awake_bold/sMNI*.nii"),
    glob.glob("/neurospin/lbi/monkeyfmri/DICOM/3T_dicom/*/*/*/*/resting_state/"
              "stim_off/run*_stim_off/sMNI*.nii"),
    glob.glob("/neurospin/lbi/monkeyfmri/DICOM/3T_dicom/*/*/*/*/resting_state/"
              "stim_on_3v/run*_stim_on_3v/sMNI*.nii"),
    glob.glob("/neurospin/lbi/monkeyfmri/DICOM/3T_dicom/*/*/*/*/resting_state/"
              "stim_on_5v/run*_stim_on_5v/sMNI*.nii")
]
names = ["awake", "anesthetize", "3v", "5v"]
motions = []
for _images in images:
    print(len(_images))
    print(set([elem.split(os.sep)[-8] for elem in _images]))
    _motions = []
    for path in _images:
        _motions.append(glob.glob(os.path.join(
            os.path.dirname(path), "cmcrps_*.txt"))[0])
    motions.append(_motions)


timeseries_split, conn_static_mean = [], []
for _images, _motions in zip(images, motions):
    _timeseries = extract_signal(
        _images,
        atlas,
        outdir,
        tr,
        low_pass=0.15,
        high_pass=0.01,
        smoothing_fwhm=None,
        masker_type="label",
        confounds=_motions,
        verbose=5)
    print(_timeseries.shape)
    _conn_static, _conn_static_mean = connectivity(
        _timeseries,
        outdir=outdir,
        kind="covariance",
        verbose=5)
    print(_conn_static.shape, _conn_static_mean.shape)
    _timeseries_split = sliding_window(
        _timeseries,
        win_size=35,
        outdir=outdir,
        sliding_step=30,
        verbose=5)
    print(_timeseries_split.shape)
    timeseries_split.append(_timeseries_split)
    conn_static_mean.append(_conn_static_mean)
plotting.plot_mosaic(
    data=conn_static_mean,
    names=names,
    vmax=1, vmin=-1)


timeseries_split = np.concatenate(tuple(timeseries_split), axis=0)
timeseries_shape = timeseries_split.shape
print(timeseries_shape)
timeseries_split = timeseries_split.reshape(-1, *timeseries_shape[-2:])
print(timeseries_split.shape)
conn_static, _ = connectivity(
    timeseries_split,
    outdir=outdir,
    kind="tangent",
    verbose=5)
conn_shape = conn_static.shape
print(conn_shape)
conn_static = conn_static.reshape(*timeseries_shape[:-2], *conn_shape[-2:])
print(conn_static.shape)
conn_static = fisher_zscore(conn_static)
states, win_labels, linked = cluster_states(
    conn=conn_static,
    n_states=n_states,
    outdir=outdir,
    init=False,
    #ctype="agglomerative",
    njobs=njobs,
    verbose=5)
print(states.shape, win_labels.shape)
if linked is not None:
    plotting.plot_dendogram(linked, threshold=None, n_leafs=10)


structural = np.loadtxt(structural)
structural = np.expand_dims(structural, axis=0)
states = np.concatenate((states, structural), axis=0)
np.save("/neurospin/nsap/processed/dynamic_networks/data/states_4cond.npy", states)
similarity_matrix = similarity(states)
print(similarity_matrix.shape)
state_labels = ["state{0}".format(idx) for idx in range(1, n_states + 1)]
plotting.plot_matrix(similarity_matrix, labels=state_labels, reorder=False)
structural_similarities = similarity_matrix[:-1, n_states]
structural_order = np.argsort(structural_similarities)
print(structural_order)


win_labels_split = []
offset = 0
for _images in images:
    print(len(_images), offset)
    win_labels_split.append(win_labels[offset: offset + len(_images)])
    offset += len(_images)


labels = ["state{0}".format(idx) for idx in range(1, n_states + 1)]
def scale(a):
    return (a - np.min(a))/np.ptp(a)
states = [scale(arr) for arr in states]
plotting.plot_mosaic(data=states, names=labels)
plotting.plot_mosaic(
    data=[states[idx] for idx in structural_order] + [states[-1]],
    names=[labels[idx] for idx in structural_order] + ["ref"])
x = np.arange(n_states)
width = 0.8 / len(names)
data = np.zeros((n_states, len(win_labels_split)), dtype=int)
for idx1 in range(n_states):
    sorted_indx = structural_order[idx1]
    for idx2 in range(len(win_labels_split)):
        data[idx1, idx2] = win_labels_split[idx2].flatten().tolist().count(
            sorted_indx)
fig, ax = plt.subplots()
nb_rect = data.shape[1]
for idx in range(nb_rect):
    ax.bar(x - width * idx, data[:, idx], width, label=names[idx])
ax.set_ylabel("Scores")
ax.set_title("States repartition")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plotting.show()





