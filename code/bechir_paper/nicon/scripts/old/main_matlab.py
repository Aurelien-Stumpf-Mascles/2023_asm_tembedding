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
from matplotlib import gridspec
from utils import extract_centroids
from utils import extract_signal
from utils import similarity
from utils import fisher_zscore
from static_connectivity import connectivity
from dynamic_connectivity import sliding_window
from dynamic_connectivity import cluster_states
import plotting
import hdf5storage


n_states = 7
njobs = 40
tr = 1.25
outdir = "/neurospin/nsap/processed/dynamic_networks/data"
atlas = "/neurospin/lbi/monkeyfmri/tmp/Resting_state/data_test/atlas_cocomac/rrMNI.nii"
structural = "/neurospin/nsap/processed/dynamic_networks/data/structural.txt"
labels = ["roi{0}".format(idx) for idx in range(1, 83)]
images = [
    glob.glob("/neurospin/lbi/monkeyfmri/resting_state/results/timeseries/timeseries_*awake_bold.mat"),
    glob.glob("/neurospin/lbi/monkeyfmri/resting_state/results/timeseries/timeseries_*stim_off.mat"),
    glob.glob("/neurospin/lbi/monkeyfmri/resting_state/results/timeseries/timeseries_*stim_on_3v.mat"),
    glob.glob("/neurospin/lbi/monkeyfmri/resting_state/results/timeseries/timeseries_*stim_on_5v.mat")
]
names = ["awake", "anesthetize", "3v", "5v"]
for _images in images:
    print(len(_images))
    print(set([elem.split(os.sep)[-1].split("_")[1] for elem in _images]))

timeseries_split, conn_static_mean = [], []
for _images in images:
    _timeseries = []
    for path in _images:
        mat = hdf5storage.loadmat(path)["scans"]
        _timeseries.append(mat)
    _timeseries = np.asarray(_timeseries)
    print(_timeseries.shape)
    _conn_static, _conn_static_mean = connectivity(
        _timeseries,
        outdir=outdir,
        kind="correlation",
        verbose=5)
    print(_conn_static.shape, _conn_static_mean.shape)
    _timeseries_split = sliding_window(
        _timeseries,
        win_size=35,
        outdir=outdir,
        sliding_step=1,
        verbose=5)
    print(_timeseries_split.shape)
    timeseries_split.append(_timeseries_split)
    conn_static_mean.append(_conn_static_mean)
plotting.plot_mosaic(
    data=conn_static_mean,
    names=names,
    vmax=1, vmin=-1)
#for mat in conn_static_mean:
#    plotting.plot_matrix(mat, vmin=-1, vmax=1, labels=labels, reorder=False)


timeseries_split = np.concatenate(tuple(timeseries_split), axis=0)
timeseries_shape = timeseries_split.shape
print(timeseries_shape)
timeseries_split = timeseries_split.reshape(-1, *timeseries_shape[-2:])
print(timeseries_split.shape)
conn_static, _ = connectivity(
    timeseries_split,
    outdir=outdir,
    kind="correlation",
    alpha=0.1,
    njobs=njobs,
    verbose=200)
conn_shape = conn_static.shape
print(conn_shape)
conn_static = conn_static.reshape(*timeseries_shape[:-2], *conn_shape[-2:])
print(conn_static.shape)
conn_static = fisher_zscore(conn_static)

if 0:
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 3, wspace=0.2, hspace=0.2)
    for cnt, n_clusters in enumerate(range_n_clusters):
        states, win_labels, data = cluster_states(
            conn=conn_static,
            n_states=n_clusters,
            outdir=outdir,
            init=500,
            njobs=njobs,
            return_raw=True,
            verbose=2)
        print(n_clusters, win_labels.shape, data.shape)
        row, col = divmod(cnt, 3)
        plotting.plot_silhouette(
            n_clusters=n_clusters,
            labels=win_labels,
            data=data,
            outdir=outdir,
            figure=fig,
            subplot_spec=gs[cnt],
            verbose=2)

states, win_labels, linked = cluster_states(
    conn=conn_static,
    n_states=n_states,
    outdir=outdir,
    init=500,
    njobs=njobs,
    verbose=2)
print(states.shape, win_labels.shape)
if linked is not None:
    plotting.plot_dendogram(linked, threshold=None, n_leafs=10)


structural = np.loadtxt(structural)
structural = np.expand_dims(structural, axis=0)
states = np.concatenate((states, structural), axis=0)
np.save("/neurospin/nsap/processed/dynamic_networks/data/states_matlab.npy", states)
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
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, wspace=0.5, hspace=0.2)
plotting.plot_mosaic(
    data=states,
    names=labels,
    figure=fig,
    subplot_spec=gs[0])
plotting.plot_mosaic(
    data=[states[idx] for idx in structural_order] + [states[-1]],
    names=[labels[idx] for idx in structural_order] + ["ref"],
    figure=fig,
    subplot_spec=gs[1])
x = np.arange(n_states)
width = 0.8 / len(names)
data = np.zeros((n_states, len(win_labels_split)), dtype=float)
for idx1 in range(n_states):
    sorted_indx = structural_order[idx1]
    for idx2 in range(len(win_labels_split)):
        data[idx1, idx2] = win_labels_split[idx2].flatten().tolist().count(
            sorted_indx)
for idx2, _run_labels in enumerate(win_labels_split):
    nb_runs = len(_run_labels)
    win_size = len(_run_labels[0])
    print(nb_runs, win_size)
    print(data[:, idx2])
    data[:, idx2] = data[:, idx2] / (nb_runs * win_size)
    print(data[:, idx2])
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





