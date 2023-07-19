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


#n_states = 7
njobs = 40
tr = 1.25
outdir = "/neurospin/nsap/processed/dynamic_networks/data"
atlas = "/neurospin/lbi/monkeyfmri/tmp/Static_RS_CIVM_R/atlas/rCIVM_R_labels_MNIspace_1iso.nii"
#structural = "/neurospin/nsap/processed/dynamic_networks/data/structural.txt"
labels = ["roi{0}".format(idx) for idx in range(1, 223)]
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
        kind="correlation",
        verbose=5)
    print(_conn_static.shape, _conn_static_mean.shape)
    conn_static_mean.append(_conn_static_mean)
plotting.plot_mosaic(
    data=conn_static_mean,
    names=names,
    vmax=0.6, vmin=-0.6)
plt.show()
