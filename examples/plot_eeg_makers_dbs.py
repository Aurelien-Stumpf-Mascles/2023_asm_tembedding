# -*- coding: utf-8 -*-
"""
EEG markers on DBS dataset
==========================

Credit: A Grigis

The goal here is to reproduce the results of the following paper:

Deep brain stimulation of the thalamus restores signatures of consciousness
in a nonhuman primate model, ScienceAdvances 2022.
"""

import os
import glob
import mne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tembedding.eeg import eeg_markers


############################################################################
# Fetch the data
# --------------
#
# The data were preprocessed as follows:
# Following the MR Ballisto-CardioGram and motion artifacts cleaning, the
# signal was filtered between 1 Hz (IIR Butterworth high-pass zero-phase
# two-pass forward and reverse noncausal filter, order 12—effective, after
# forward-backward) and 25 Hz (IIR  Butterworth  lowpass  zero-phase
# noncausal filter, order 16—effective, after forward-backward) and
# downsampled to 250-Hz sampling rate. The data were cut 15 s after the
# start of the stimulation until 15 s before the end of the scanning. The
# remaining time series was cut into epochs of 0.8 s, with a random jitter
# ranging from  0.55  to  0.85 s. For the cleaning of artifacted epochs or
# channels, the Python package Autoreject was used with the number of
# channel interpolations equal to 1, 2, 4, or 8. Average EEG reference
# projection was applied.

datadir = os.getenv("TEMBEDDING_DIR")
if datadir is None:
    raise ValueError("Please specify the dataset directory in the "
                     "TEMBEDDING_DIR variable.")
eegdir = os.path.join(datadir, "derivatives", "eeg_preproc")
data = {"subject": [], "session": [], "condition": [], "path": []}
for path in glob.glob(os.path.join(eegdir, "*", "*", "*.fif")):
    split = path.split(os.sep)
    data["condition"].append(split[-3])
    sid, ses = split[-2].split("_")[:2]
    data["subject"].append(sid)
    data["session"].append(ses)
    data["path"].append(path)
data = pd.DataFrame.from_dict(data)
print(data)
print(data.groupby("subject").describe())


############################################################################
# Compute the dynamic EEG markers
# -------------------------------
#


############################################################################
# Plot the EEG markers
# --------------------
#
