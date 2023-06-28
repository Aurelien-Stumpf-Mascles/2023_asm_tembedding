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
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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
data = {"subject": [], "session": [], "run": [], "condition": [], "path": []}
for path in glob.glob(os.path.join(eegdir, "*", "*", "*.fif")):
    split = path.split(os.sep)
    data["condition"].append(split[-3].replace(
        "_results", "").replace("_", ""))
    sid, ses = split[-2].split("_")[:2]
    data["subject"].append(sid)
    data["session"].append(ses)
    data["run"].append(split[-2].split("_")[-1])
    data["path"].append(path)
data = pd.DataFrame.from_dict(data)
print(data)
print(data.groupby("subject").describe())
print(f"conditions: {set(data.condition.values)}")


############################################################################
# Compute the dynamic EEG markers
# -------------------------------
#
outdir = os.path.join(datadir, "derivatives", "eeg_markers")
if not os.path.isdir(outdir):
    os.mkdir(outdir)
if 0:
    Parallel(n_jobs=-2)(delayed(eeg_markers)(
            epochs_file=row.path, outdir=outdir, njobs=1,
            basename=(f"sub-{row.subject}_ses-{row.session}_cond-{row.condition}_"
                      f"run-{row.run}"))
        for _, row in data.iterrows())
data["marker_path"] = [os.path.join(
    outdir, (f"sub-{row.subject}_ses-{row.session}_cond-{row.condition}_"
             f"run-{row.run}_makers.tsv"))
    for _, row in data.iterrows()]
print(data)


############################################################################
# Plot the EEG markers
# --------------------
#

markers = {"condition": []}
for _, row in data.iterrows():
    _data = pd.read_csv(row.marker_path, sep="\t")
    for _key in _data.columns:
        _mean = np.mean(_data[_key].values)
        markers.setdefault(_key, []).append(_mean)
    markers["condition"].append(row.condition)
sns.set_style("white")
palette = "Set2"
marker_names = list(markers.keys())
marker_names.remove("condition")
for marker_name in marker_names:
    plt.figure()
    ax = sns.violinplot(x="condition", y=marker_name, data=markers,
                        hue="condition", dodge=False, palette=palette,
                        scale="width", inner=None)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height,
                             transform=ax.transData))
    sns.boxplot(x="condition", y=marker_name, data=markers, saturation=1,
                showfliers=False, width=0.3,
                boxprops={"zorder": 3, "facecolor": "none"}, ax=ax)
    old_len_collections = len(ax.collections)
    sns.stripplot(x="condition", y=marker_name, data=markers, hue="condition",
                  palette=palette, dodge=False, ax=ax)
    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend_.remove()
    ax.spines[["right", "left", "top"]].set_visible(False)
    ax.grid(axis="y")
    plt.savefig(os.path.join(outdir, f"{marker_name}.png"), dpi=150)


############################################################################
# Find the most important markers
# -------------------------------
#
