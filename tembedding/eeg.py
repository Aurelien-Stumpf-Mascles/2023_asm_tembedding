# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


# Imports
import os
import shutil
import numpy as np
import pandas as pd
import mne
from nice import Markers
from nice.markers import (PowerSpectralDensity,
                          KolmogorovComplexity,
                          PermutationEntropy,
                          SymbolicMutualInformation,
                          PowerSpectralDensitySummary,
                          PowerSpectralDensityEstimator)


def eeg_markers(epochs_file, outdir, basename, njobs=1):
    """ Compute EEG markers.

    Parameters
    ----------
    epochs_file: str
        the preprocessed epochs file.
    outdir: str
        the destunation folder.
    basename: str
        the name of the generated file.
    njobs: int, default 1
        the number of parallel jobs when applicable.

    References
    ----------
    Robust EEG-based cross-site and cross-protocol classification of states
    of consciousness, Brain, 2018.
    """
    epochs = mne.read_epochs(epochs_file)
    backend = "python"
    # > we define one base estimator to avoid recomputation when looking up
    # markers
    psds_params = dict(n_fft=4096, n_overlap=100, n_jobs=njobs, nperseg=128)
    base_psd = PowerSpectralDensityEstimator(
        psd_method="welch", tmin=None, tmax=None, fmin=1., fmax=45.,
        psd_params=psds_params, comment="default")
    # > here are the resting-state compatible markers
    markers = Markers([
        PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,
                             normalize=False, comment="delta"),
        PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,
                             normalize=True, comment="deltan"),
        PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,
                             normalize=False, comment="theta"),
        PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,
                             normalize=True, comment="thetan"),
        PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12.,
                             normalize=False, comment="alpha"),
        PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12.,
                             normalize=True, comment="alphan"),
        PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                             normalize=False, comment="beta"),
        PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                             normalize=True, comment="betan"),
        PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,
                             normalize=False, comment="gamma"),
        PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,
                             normalize=True, comment="gamman"),
        PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=45.,
                             normalize=False, comment="summary_se"),
        PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                    percentile=.5, comment="summary_msf"),
        PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                    percentile=.9, comment="summary_sef90"),
        PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                    percentile=.95, comment="summary_sef95"),
        PermutationEntropy(tmin=None, tmax=0.6, backend=backend),
        # csd needs to be skipped
        SymbolicMutualInformation(
            tmin=None, tmax=0.6, method="weighted", backend=backend,
            method_params={"nthreads": "auto", "bypass_csd": True},
            comment="weighted"),
        KolmogorovComplexity(tmin=None, tmax=0.6, backend=backend,
                             method_params={"nthreads": "auto"}),
    ])

    # Prepare reductions.
    # > keep in mind that this is BCI, we have some localized effects.
    # Therefore we will consider the standard deviation across channels.
    # Contraty to the paper, this is a single subject analysis. We therefore do
    # not pefrorm a full reduction but only compute one statistic
    # per marker and per epoch. In the paper, instead, we computed summaries over
    # epochs and sensosrs, yielding one value per marker per EEG recoding.
    epochs_fun = np.mean
    channels_fun = np.std
    reduction_params = {
        "PowerSpectralDensity": {
            "reduction_func": [
                {"axis": "frequency", "function": np.sum},
                {"axis": "epochs", "function": epochs_fun},
                {"axis": "channels", "function": channels_fun}]
        },
        "PowerSpectralDensitySummary": {
            "reduction_func": [
                {"axis": "epochs", "function": epochs_fun},
                {"axis": "channels", "function": channels_fun}]
        },
        "SymbolicMutualInformation": {
            "reduction_func": [
                {"axis": "epochs", "function": epochs_fun},
                {"axis": "channels", "function": channels_fun},
                {"axis": "channels_y", "function": channels_fun}]
        },
        "PermutationEntropy": {
            "reduction_func": [
                {"axis": "epochs", "function": epochs_fun},
                {"axis": "channels", "function": channels_fun}]
        },
        "KolmogorovComplexity": {
            "reduction_func": [
                {"axis": "epochs", "function": epochs_fun},
                {"axis": "channels", "function": channels_fun}]
        }
    }

    # Generate markers
    X = np.empty((len(epochs), len(markers)))
    for ii in range(len(epochs)):
        markers.fit(epochs[ii])
        X[ii, :] = markers.reduce_to_scalar(marker_params=reduction_params)
        # XXX hide this inside code
        for marker in markers.values():
            delattr(marker, "data_")
        delattr(base_psd, "data_")

    # Saving markers
    df = pd.DataFrame(
        X, columns=["-".join(item.split("/")[2:]) for item in markers.keys()])
    df.to_csv(os.path.join(outdir, f"{basename}_makers.tsv"), sep="\t",
              index=False)        
