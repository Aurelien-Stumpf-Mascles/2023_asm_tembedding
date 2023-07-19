# -*- coding: utf-8 -*-
###############################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
###############################################################################

import os
from collections import OrderedDict
from pprint import pprint
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
import pandas as pd


#n_states = 7
njobs = 40
tr = 1.25
outdir = "/neurospin/nsap/processed/dynamic_networks/data"
atlas = "/neurospin/lbi/monkeyfmri/tmp/Static_RS_CIVM_R/atlas/rCIVM_R_labels_MNIspace_1iso.nii"
# structural = "/neurospin/nsap/processed/dynamic_networks/data/structural.txt"
images = [
    glob.glob("/neurospin/lbi/monkeyfmri/resting_state/JS_JT_results/timeseries_CIVM_R/timeseries_*DBS*awake_bold*.mat"),
    glob.glob("/neurospin/lbi/monkeyfmri/resting_state/JS_JT_results/timeseries_CIVM_R/timeseries_*DBS*stim_off*.mat"),
    glob.glob("/neurospin/lbi/monkeyfmri/resting_state/JS_JT_results/timeseries_CIVM_R/timeseries_*DBS*stim_on_3v*.mat"),
    glob.glob("/neurospin/lbi/monkeyfmri/resting_state/JS_JT_results/timeseries_CIVM_R/timeseries_*DBS*stim_on_5v*.mat"),
    glob.glob("/neurospin/lbi/monkeyfmri/resting_state/JS_JT_results/timeseries_CIVM_R/timeseries_*DBS*stim_cont_on_3v*.mat"),
    glob.glob("/neurospin/lbi/monkeyfmri/resting_state/JS_JT_results/timeseries_CIVM_R/timeseries_*DBS*stim_cont_on_5v*.mat")
]
names = ["awake", "anesthesia", "CMT DBS 3V", "CMT DBS 5V", "VLT DBS 3V", "VLT DBS 5V"]
for _images in images:
    print(len(_images))
    print(set([elem.split(os.sep)[-1].split("_")[1] for elem in _images]))
atlas_meta = pd.read_excel("/neurospin/lbi/monkeyfmri/tmp/Static_RS_CIVM_R/List_GNWnodes_fromCIVM_R_labels.xlsx")
def filter_meta(df, key):
    fdf = df[df["GNW"] == key]
    labels = {
        "lh": set(fdf["CIVM_R_left"].values - 1),
        "rh": set(fdf["CIVM_R_right"].values - 1)}
    return labels
def average_conn(conn, key, labels, avg_conn, hemi="rh"):
    print(conn.shape)
    if hemi + key not in avg_conn:
        conn_avg[hemi + key] = OrderedDict()
    _data = conn[:, tuple(labels[key][hemi])]
    _data = np.mean(_data, axis=1)
    _data.shape += (1, )
    for _hemi in ("lh", "rh"):
        for _key, _value in labels.items():
            if _key == key:
                continue
            avg_conn[hemi + key].setdefault(_hemi + _key, []).append(np.mean(_data[tuple(labels[_key][_hemi]), :]))
    return avg_conn
data = {}
for key in set(atlas_meta["GNW"]):
    if not isinstance(key, str) or key == "":
        continue
    data[key] = filter_meta(atlas_meta, key)
pprint(data)
labels = OrderedDict()
for index in ("CIVM_R_left", "CIVM_R_right"):
    for label in set(atlas_meta[index]):
        if label == 0:
            continue
        df = atlas_meta[atlas_meta[index] == label]
        if label not in labels:
            if label % 2 == 0:
                hemi = "R-"
            else:
                hemi = "L-"
            labels[label] = hemi + "_".join([str(elem) for elem in set(df["ABBREVIATIONS CIVM_R"].values)])
pprint(labels)


conn_static_mean = []
conn_avg = {}
for _images in images:
    _timeseries = []
    for path in _images:
        mat = hdf5storage.loadmat(path)["scans"]
        #if mat.shape[1] == 222:
        #    mat = np.delete(mat, 21, axis=1)
        print(path, mat.shape)
        _timeseries.append(mat)
    _timeseries = np.asarray(_timeseries)
    print(_timeseries.shape)
    _conn_static, _conn_static_mean = connectivity(
        _timeseries,
        outdir=outdir,
        kind="correlation",
        verbose=5)
    _conn_static_mean = fisher_zscore(_conn_static_mean)
    for key in data.keys():
        for hemi in ("lh", "rh"):
            conn_avg = average_conn(_conn_static_mean, key, data, conn_avg, hemi=hemi)
    conn_static_mean.append(_conn_static_mean)
pprint(conn_avg)
for key in conn_avg.keys():
    _conn_avg = conn_avg[key]
    conn_avg_arr = np.asarray(list(_conn_avg.values()))
    print(conn_avg_arr.shape)
    # R- L-
    df = pd.DataFrame(data=conn_avg_arr, index=[elem.replace("rh", "R-").replace("lh", "L-") for elem in _conn_avg.keys()], columns=names)
    plotting.plot_array(df, vmin=-0.6, vmax=0.6, title=key.replace("rh", "R-").replace("lh", "L-"), square=False, auto=False)
#plotting.plot_matrix(conn_avg_arr, vmin=-0.6, vmax=0.6, reorder=False, labels=list() + names, title="R-MD")
#plotting.plot_mosaic(
#    data=conn_static_mean,
#    names=names,
#    vmax=1, vmin=-1)
# Sort L then R
for mat, title in zip(conn_static_mean, names):
    sortmat = np.concatenate((mat[::2], mat[1::2]), axis=0)
    sortmat = np.concatenate((sortmat[:, ::2], sortmat[:, 1::2]), axis=1)
    df = pd.DataFrame(data=sortmat, index=labels.values(), columns=labels.values())
    plotting.plot_array(df, vmin=-0.6, vmax=0.6, title=title)
for cnt, fig in enumerate([plt.figure(n) for n in plt.get_fignums()]):
    fig.tight_layout()
    fig.savefig("/neurospin/lbi/monkeyfmri/tmp/Static_RS_CIVM_R/snaps/{0}.png".format(cnt))
#plt.show()
