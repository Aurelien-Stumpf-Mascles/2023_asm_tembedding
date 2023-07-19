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
from utils import extract_centroids
from utils import extract_signal
from utils import similarity
from static_connectivity import connectivity
from dynamic_connectivity import sliding_window
from dynamic_connectivity import cluster_states
import plotting



# Filter the BOLD signals between 0.01 and 0.2 Hz
# import scipy.signal as spsg
#n_order = 3
#Nyquist_freq = 0.5 / TR
#low_f = 0.01 / Nyquist_freq
#high_f = 0.2 / Nyquist_freq
#b,a = spsg.iirfilter(n_order, [low_f,high_f], btype='bandpass', ftype='butter')
#filtered_ts_emp = spsg.filtfilt(b,a,ts_emp, axis=-1)

njobs = 40
tr = 1.25
outdir = "/neurospin/nsap/processed/dynamic_networks/data"
atlas = "/neurospin/lbi/monkeyfmri/tmp/Resting_state/data_test/atlas_cocomac/rrMNI.nii"
structural = "/neurospin/nsap/processed/dynamic_networks/data/structural.txt"
labels = ["roi{0}".format(idx) for idx in range(1, 83)]
#awake_images = glob.glob(
#    "/neurospin/lbi/monkeyfmri/tmp/Resting_state/data_test/raw/awake/*.nii")
#anesthetized_images = glob.glob(
#    "/neurospin/lbi/monkeyfmri/tmp/Resting_state/data_test/raw/anesthetized/*.nii")
anesthetized_images = glob.glob(
    "/neurospin/lbi/monkeyfmri/DICOM/3T_dicom/*/*/*/*/resting_state/stim_off/run*_stim_off/sMNI*.nii")
awake_images = glob.glob(
    "/neurospin/lbi/monkeyfmri/DICOM/3T_dicom/*/*/*/*/resting_state/awake_bold/run*_awake_bold/sMNI*.nii")
#stimon5v = glob.glob(
#    "/neurospin/lbi/monkeyfmri/DICOM/3T_dicom/*/*/*/*/resting_state/stim_ON_5v/run*_stim_ON_5v/sMNI*.nii")
#stimon3v = glob.glob(
#    "/neurospin/lbi/monkeyfmri/DICOM/3T_dicom/*/*/*/*/resting_state/stim_ON_3v/run*_stim_ON_3v/sMNI*.nii")
#stimoncon5v = glob.glob(
#    "/neurospin/lbi/monkeyfmri/DICOM/3T_dicom/*/*/*/*/resting_state/stim_ON_con_5v/run*_stim_ON_con_5v/sMNI*.nii")
#stimoncon3v = glob.glob(
#    "/neurospin/lbi/monkeyfmri/DICOM/3T_dicom/*/*/*/*/resting_state/stim_ON_con_3v/run*_stim_ON_con_3v/sMNI*.nii")
print(len(awake_images), len(anesthetized_images))
awake_motions = []
for path in awake_images:
    awake_motions.append(glob.glob(
        os.path.join(os.path.dirname(path), "cmcrps_*.txt"))[0])
anesthetized_motions = []
for path in anesthetized_images:
    anesthetized_motions.append(
        glob.glob(os.path.join(os.path.dirname(path), "cmcrps_*.txt"))[0])
for images in [awake_images, anesthetized_images]:
    print(set([elem.split(os.sep)[-8] for elem in images]))


affine = np.diag([3., 2.7, 3.3, 1.])
affine[:3, 3] = [-55., -75., -45]
atlas_centroids = extract_centroids(atlas, affine=affine)

awake_timeseries = extract_signal(
    awake_images,
    atlas,
    outdir,
    tr,
    low_pass=0.05,
    high_pass=0.025,
    smoothing_fwhm=None,
    masker_type="label",
    confounds=awake_motions,
    verbose=5)
print(awake_timeseries.shape)
#plotting.plot_timeseries(awake_timeseries[:, :, 5: 10])

awake_conn_static, awake_conn_static_mean = connectivity(
    awake_timeseries,
    outdir=outdir,
    kind="covariance",
    verbose=5)
print(awake_conn_static.shape, awake_conn_static_mean.shape)
#plotting.plot_matrix(awake_conn_static_mean, vmax=1, vmin=-1, labels=labels, reorder=False)
#plotting.plot_connectome(awake_conn_static_mean, atlas_centroids, edge_threshold="98.5%")


anesthetized_timeseries = extract_signal(
    anesthetized_images,
    atlas,
    outdir,
    tr,
    low_pass=0.05,
    high_pass=0.025,
    smoothing_fwhm=None,
    masker_type="label",
    confounds=anesthetized_motions,
    verbose=5)
print(anesthetized_timeseries.shape)
#plotting.plot_timeseries(anesthetized_timeseries[:, :, 5: 10])

anesthetized_conn_static, anesthetized_conn_static_mean = connectivity(
    anesthetized_timeseries,
    outdir=outdir,
    kind="covariance",
    verbose=5)
print(anesthetized_conn_static.shape, anesthetized_conn_static_mean.shape)
#plotting.plot_matrix(anesthetized_conn_static_mean, vmax=1, vmin=-1, labels=labels, reorder=False)
#plotting.plot_connectome(anesthetized_conn_static_mean, atlas_centroids, edge_threshold="98.5%")


plotting.plot_mosaic(
    data=[awake_conn_static_mean, anesthetized_conn_static_mean],
    names=["awake", "anesthetized"],
    vmax=1, vmin=-1)


awake_timeseries_split = sliding_window(
    awake_timeseries,
    win_size=35,
    outdir=outdir,
    sliding_step=1,
    verbose=5)
anesthetized_timeseries_split = sliding_window(
    anesthetized_timeseries,
    win_size=35,
    outdir=outdir,
    sliding_step=1,
    verbose=5)
print(awake_timeseries_split.shape, anesthetized_timeseries_split.shape)
timeseries_split = np.concatenate(
    (awake_timeseries_split, anesthetized_timeseries_split), axis=0)
timeseries_shape = timeseries_split.shape
timeseries_split = timeseries_split.reshape(-1, *timeseries_shape[-2:])
print(timeseries_split.shape)
#plotting.plot_timeseries(timeseries_split[:9, :, 5: 10])

# It is also common to transform the correlations to a Fisher-Z score, as the
# bounded nature of Pearson correlation violates certain statistical
# assumptions.
# https://www.frontiersin.org/articles/10.3389/fnins.2019.00685/full
conn_static, _ = connectivity(
    timeseries_split,
    outdir=outdir,
    kind="correlation",
    verbose=5)
conn_shape = conn_static.shape
print(conn_static.shape)
conn_static = conn_static.reshape(*timeseries_shape[:-2], *conn_shape[-2:])
print(conn_static.shape)
# Transfrom the correlation values to Fisher z-scores    
# conn_static = np.arctanh(conn_static)


# Cluster validation can be done using Dunn's index, since it aims at the
# identification of 'compact and well separated clusters'
n_states = 7
states, win_labels, linked = cluster_states(
    conn=conn_static,
    n_states=n_states,
    outdir=outdir,
    init=False,
    ctype="agglomerative",
    njobs=njobs,
    verbose=5)
print(states.shape, win_labels.shape)
if linked is not None:
    plotting.plot_dendogram(linked, threshold=2200, n_leafs=10)
#for conn in states:
#    plotting.plot_matrix(conn, labels=labels, reorder=False)


structural = np.loadtxt(structural)
#np.fill_diagonal(structural, 0)
#plotting.plot_matrix(structural, vmax=3, vmin=0, labels=labels, reorder=False)
structural = np.expand_dims(structural, axis=0)
states = np.concatenate((states, structural), axis=0)
np.save("/neurospin/nsap/processed/dynamic_networks/data/states.npy", states)


similarity_matrix = similarity(states)
print(similarity_matrix.shape)
state_labels = ["state{0}".format(idx) for idx in range(1, len(similarity_matrix) + 1)]
plotting.plot_matrix(similarity_matrix, labels=state_labels, reorder=False)
structural_similarities = similarity_matrix[:-1, n_states]
structural_order = np.argsort(structural_similarities)
print(structural_order)

awake_win_labels = win_labels[:awake_timeseries_split.shape[0]]
anesthetized_win_labels = win_labels[awake_timeseries_split.shape[0]:]
print(awake_win_labels.shape, anesthetized_win_labels.shape)

import matplotlib.pyplot as plt
if 0:
    plt.figure()
    plt.hist(awake_win_labels.flatten(), bins=n_states)
    plt.figure()
    plt.hist(anesthetized_win_labels.flatten(), bins=n_states)

if 1:
    labels = ["state{0}".format(idx) for idx in range(1, n_states + 1)]
    def scale(a):
        return (a - np.min(a))/np.ptp(a)
    states = [scale(arr) for arr in states]
    plotting.plot_mosaic(data=states, names=labels)
    plotting.plot_mosaic(
        data=[states[idx] for idx in structural_order] + [states[-1]],
        names=[labels[idx] for idx in structural_order] + ["ref"])
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    data = np.zeros((n_states, 2), dtype=int)
    for idx in range(n_states):
        sorted_indx = structural_order[idx]
        data[idx, 0] = awake_win_labels.flatten().tolist().count(sorted_indx)
        data[idx, 1] = anesthetized_win_labels.flatten().tolist().count(sorted_indx)
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, data[:, 0], width, label='Awake')
    rects2 = ax.bar(x + width/2, data[:, 1], width, label='Anesthetized')
    ax.set_ylabel('Scores')
    ax.set_title('States repartition')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

plotting.show()





