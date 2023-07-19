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
names = ["awake", "anesthetize", "3v", "5v"]
if 1:
    images = [
        glob.glob("/neurospin/lbi/monkeyfmri/resting_state/results/timeseries/temporal/corrtemporal_*awake_bold.mat"),
        glob.glob("/neurospin/lbi/monkeyfmri/resting_state/results/timeseries/temporal/corrtemporal_*stim_off.mat"),
        glob.glob("/neurospin/lbi/monkeyfmri/resting_state/results/timeseries/temporal/corrtemporal_*stim_on_3v.mat"),
        glob.glob("/neurospin/lbi/monkeyfmri/resting_state/results/timeseries/temporal/corrtemporal_*stim_on_5v.mat")
    ]
    for _images in images:
        print(len(_images))
        print(set([elem.split(os.sep)[-1].split("_")[2] for elem in _images]))
    timeseries_split = []
    for _images in images:
        _timeseries = []
        for path in _images:
            mat = hdf5storage.loadmat(path)["C"]
            _timeseries.append(mat.T.reshape(-1, 82, 82))
        _timeseries = np.asarray(_timeseries)
        print(_timeseries.shape)
        timeseries_split.append(_timeseries)
    conn_filtered = np.concatenate(tuple(timeseries_split), axis=0)
    conn_filtered = 0.5 * np.log((1 + conn_filtered) / (1 - conn_filtered))
    conn_filtered[np.isnan(conn_filtered)] = 0 
    conn_filtered[np.isinf(conn_filtered)] = 0
    print(conn_filtered.shape)

else:
    mat = hdf5storage.loadmat(
        "/neurospin/lbi/monkeyfmri/resting_state/results/"
        "results_awake_bold_stim_off_stim_on_3v_stim_on_5v/20191115/kmeans/city/"
        "data_kmeans_before_clustering_7state_500replicates.mat")
    print(mat["cpt1"])
    print(mat["LLL"])
    conn_filtered = mat["D"].T.reshape(-1, 82, 82)
    conn_filtered = np.expand_dims(conn_filtered, axis=1)
    print(conn_filtered.shape)

states, win_labels, linked = cluster_states(
    conn=conn_filtered,
    n_states=n_states,
    outdir=outdir,
    init=500,
    #ctype="agglomerative",
    njobs=njobs,
    verbose=5)
print(states.shape, win_labels.shape)
if linked is not None:
    plotting.plot_dendogram(linked, threshold=None, n_leafs=100)


structural = np.loadtxt(structural)
structural = np.expand_dims(structural, axis=0)
states = np.concatenate((states, structural), axis=0)
np.save("/neurospin/nsap/processed/dynamic_networks/data/states_matlab2.npy", states)
similarity_matrix = similarity(states)
print(similarity_matrix.shape)
state_labels = ["state{0}".format(idx) for idx in range(1, n_states + 1)]
plotting.plot_matrix(similarity_matrix, labels=state_labels, reorder=False)
structural_similarities = similarity_matrix[:-1, n_states]
structural_order = np.argsort(structural_similarities)
print(structural_order)
print([structural_similarities[idx] for idx in structural_order])

win_labels_split = []
offset = 0
if 1:
    for _images in images:
        print(len(_images), offset)
        win_labels_split.append(win_labels[offset: offset + len(_images)])
        offset += len(_images)
else:
    _win_labels_split = []
    for nb_win in mat["LLL"][0]:
        nb_win = int(nb_win)
        _win_labels_split.append(win_labels[offset:offset + nb_win, 0])
        offset += nb_win
    offset = 0
    for cut in (47, 38, 36, 38):
        values = []
        for item in _win_labels_split[offset:offset + cut]:
            values.extend(item)
        win_labels_split.append(np.asarray(values))
        offset += cut
print(len(win_labels_split))
print([item.shape for item in win_labels_split])


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
    data[:, idx2] = data[:, idx2] / (nb_runs * win_size)
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





