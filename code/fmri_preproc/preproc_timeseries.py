# -*- coding: utf-8 -*-
###############################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
###############################################################################


"""
Perform timeseries pre-processings.
"""

# Imports
import os
import pandas as pd
from pprint import pprint
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import nilearn.signal
import nilearn.image
import nilearn.masking
from nilearn.input_data import NiftiLabelsMasker
import scipy.signal

# Global parameters
test = True
meta_file = "/neurospin/lbi/monkeyfmri/deepstim/database/ANESTHETIC_database/participants.tsv"
# preproc = "pypreclin"
preproc = "nsm"
data_dir = "/neurospin/lbi/monkeyfmri/deepstim/database/ANESTHETIC_database/derivatives/{0}".format(preproc)
outdir_name = "{0}_timeseries".format(preproc)
mask_file = "/neurospin/lbi/monkeyfmri/images/reference/mni-resampled_1by1by1.nii"
atlas = None #"/neurospin/lbi/monkeyfmri/resting_state/references/rRM_F99_ROItemplate_MNI.nii"  # cocomac
add_derivates = False
add_compor = False
fwhm = None  #3.
tr = 2.4
n_dummy_trs = 4
low_pass = 0.005
high_pass = 0.0025
locs = [(48, 35, 22), (21, 58, 33)]
n_jobs = 10


def preproc_timeseries(image_file, mask_file, confounds_file, tr, output_dir,
                       n_dummy_trs=4, add_derivates=True, add_compor=True,
                       low_pass=0.05, high_pass=0.0025, fwhm=None, atlas=None,
                       locs=None):
    """ Apply the Yeo 2011 timeseries pre-processing schema.

    This function applies on the input timeseries as follows:

    - drop dummy TRs
    - detrend
    - low- and high-pass filters
    - remove confounds
    - standardize

    Drop dummy TRs:
    During the initial stages of a functional scan there is a strong signal
    decay artifact, thus the first 4ish (by default) or so TRs are very high
    intensity signals that don't reflect the rest of the scan. Therefore we
    drop these timepoints.

    The filtering stage is composed of:

    - low pass filter out high frequency signals from the data (upper than
    0.05 Hz by default). fMRI signals are slow evolving processes, any high
    frequency signals are likely due to noise.
    - high pass filter out any very low frequency signals (below 0.0025 Hz by
    default), which may be due to intrinsic scanner instabilities.

    Confound regressors are composed of:

    - 6 motion parameters (X, Y, Z, RotX, RotY, RotZ)
    - global signal (GlobalSignal)
    - 2 Largest Principal components of non-grey matter (aCompCor01, aCompCor02)

    This is a total of 9 base confound regressor variables. Finally we
    optionally add temporal derivatives of each of these signals as well
    (1 temporal derivative for each). This results in 18 confound regressors.

    According to Lindquist et al. (2018), removal of confounds will be done
    orthogonally to temporal filters (low- and/or high-pass filters), if both
    are specified.

    Parameters
    ----------
    image_file: str
        4-dimensional image with timeseries in the last dimension of length N.
    mask_file: str
        signal is only cleaned from voxels inside the mask. If
        mask is provided, it should have same shape and affine as the input
        image file.
    confounds_file: str
        confounds timeseries of size (N, 7) containing the 6 motion parameters
        and the global signal.
    tr: float
        repetition time in seconds (sampling period).
    output_dir : str
        output directory where image clean.nii.gz will be saved
    n_dummy_trs: int, default 4
        the number of dummy TRs that will be removed at the begining of each
        timeserie.
    add_derivates: bool, default True
        optionally add temporal derivatives of each confound.
    add_compor: bool, default True
        optionally add compcor confounds.
    low_pass: float, default 0.05
        low cutoff frequencies in Hz.
    high_pass: float, default 0.0025
        high cutoff frequencies in Hz.
    fwhm: float, default None
        smoothing strength as a Full-Width at Half Maximum (FWHM) in
        millimeters.
    atlas: str, default None
        label region definitions.
    locs: list of 3-uplet, default None
        location to display signal preprocessing results (in voxels).
    """
    # Load data
    im = nibabel.load(image_file)
    mask_im = nibabel.load(mask_file)
    mask_arr = mask_im.get_data()
    if len(np.unique(mask_arr)) not in (1, 2):
        mask_arr[mask_arr > 0] = 1
        mask_im = nibabel.Nifti1Image(mask_arr, mask_im.affine)
    confounds = np.loadtxt(confounds_file)
    basename = os.path.join(output_dir, os.path.basename(image_file)[:-4])
    clean_basename = basename
    timeserie_basename = basename.replace("bold", "timeseries")

    # Compute confounds
    if add_compor:
        compcor_confounds = nilearn.image.high_variance_confounds(
            im, n_confounds=2, percentile=2.)
        confounds = np.concatenate((confounds, compcor_confounds), axis=1)
    signals = nilearn.masking.apply_mask(im, mask_im)
    global_signal = np.mean(signals, axis=1)
    global_signal = np.expand_dims(global_signal, axis=1)
    confounds = np.concatenate((confounds, global_signal), axis=1)
    if add_derivates:
        dt_confounds = np.diff(confounds, axis=0, prepend=0)
        confounds = np.concatenate((confounds, dt_confounds), axis=1)

    # Drop dummy volumes
    im = im.slicer[..., n_dummy_trs:]
    confounds = confounds[n_dummy_trs:]
    signals = signals[n_dummy_trs:]

    # Clean signal
    #clean_signals = nilearn.signal.clean(
    #    signals, detrend=True, standardize="zscore",
    #    confounds=confounds, standardize_confounds=True,
    #    low_pass=low_pass, high_pass=high_pass, t_r=tr)

    # Reshape data
    #clean_im = nilearn.masking.unmask(clean_signals, mask_im)

    clean_signals = nilearn.signal.clean(
        signals, detrend=True, standardize=False, high_pass = 0.05, t_r=tr)
    clean_im = nilearn.masking.unmask(clean_signals, mask_im)

    # Smooth the data
    if fwhm is not None:
        clean_im = nilearn.image.smooth_img(clean_im, fwhm)

    # Save the result
    nibabel.save(clean_im, clean_basename + ".nii.gz")

    if locs is not None:

        # QC
        fig, axs = plt.subplots(2 * len(locs))
        im_arr = im.get_data()
        clean_arr = clean_im.get_data()
        print(im_arr.shape, clean_arr.shape)
        for cnt, (x, y, z) in enumerate(locs):
            axs[2 * cnt].plot(im_arr[x, y, z], color="r")
            axs[2 * cnt + 1].plot(clean_arr[x, y, z], color="g")
            axs[2 * cnt].set_xlim(0, im_arr[x, y, z].shape[0])
            axs[2 * cnt].set_xticks([0, im_arr[x, y, z].shape[0]])
            axs[2 * cnt].tick_params(axis="x", labelsize=7)
            axs[2 * cnt + 1].set_xlim(0, clean_arr[x, y, z].shape[0])
            axs[2 * cnt + 1].set_xticks([0, clean_arr[x, y, z].shape[0]])
            axs[2 * cnt + 1].tick_params(axis="x", labelsize=7)
        plt.savefig(clean_basename + ".png")

    if atlas is not None:

        # Extract average timeseries
        masker = NiftiLabelsMasker(labels_img=atlas, mask_img=mask_im)
        time_series = masker.fit_transform(clean_basename + ".nii.gz")

        # Save the result
        np.savetxt(timeserie_basename + ".txt", time_series)
        np.save(timeserie_basename + ".npy", time_series)
    
        print(timeserie_basename + ".npy")

    # QC
    plt.figure()
    vmin = time_series.min()
    vmax = np.percentile(time_series, 99)
    plt.imshow(time_series.T, cmap="jet", vmin=vmin, vmax=vmax)
    plt.savefig(timeserie_basename + ".png")


# Load data
df = pd.read_csv(meta_file, sep="\t", dtype=str)
df = df.rename(
    columns={'monkey': 'sub', 'session': 'ses', 'condition': 'cond'})
print(df)
basename1 = "sub-{0}_ses-{1}_task-rest_space-mni_run-{2}_desc-smask_bold.nii"
basename2 = "sub-{0}_ses-{1}_run-{2}_confounds.par"
dataset = []
for index, row in df.iterrows():
    fmri_file = os.path.join(
        data_dir, "sub-{0}".format(row["sub"]), "ses-{0}".format(row["ses"]),
        "func", basename1.format(row["sub"], row["ses"], row["run"]))
    confounds_file = os.path.join(
        data_dir, "sub-{0}".format(row["sub"]), "ses-{0}".format(row["ses"]),
        "func", basename2.format(row["sub"], row["ses"], row["run"]))
    skip = False
    for path in (fmri_file, confounds_file):
        if not os.path.isfile(path):
            print(path)
            skip = True
    if skip:
        continue

    """
    outdir = os.path.dirname(fmri_file).replace(preproc, outdir_name)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    dataset.append((fmri_file, confounds_file, outdir))
    if test:
        break
    """

    outdir = "/neurospin/lbi/monkeyfmri/deepstim/workspace/2023_ASM_tembedding/gitproject/2023_asm_tembedding/data/TimeSeries"
    dataset.append((fmri_file, confounds_file, outdir))

print("nb runs: {0} / {1}".format(len(dataset), len(df)))


# Parallel call
Parallel(n_jobs=n_jobs, verbose=20)(delayed(preproc_timeseries)(
    fmri_file, mask_file, confounds_file, tr=tr, output_dir=outdir,
    n_dummy_trs=n_dummy_trs, add_derivates=add_derivates, low_pass=low_pass,
    high_pass=high_pass, fwhm=fwhm, atlas=atlas, locs=locs, add_compor=add_compor)
        for fmri_file, confounds_file, outdir in dataset[:2])

#for fmri_file, confounds_file, outdir in dataset :
#    preproc_timeseries(fmri_file, mask_file, confounds_file, tr=tr, output_dir=outdir,
#    n_dummy_trs=n_dummy_trs, add_derivates=add_derivates, low_pass=low_pass,
#    high_pass=high_pass, fwhm=fwhm, atlas=atlas, locs=locs, add_compor=add_compor)