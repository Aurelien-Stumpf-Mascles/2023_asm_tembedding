# -*- coding: utf-8 -*-
###############################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
###############################################################################

"""
Build brain states using Kmeans: let on monkey out
"""

# Imports
import os
import glob
import json
import argparse
import collections
import multiprocessing
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import nicon.plotting as plotting
from nicon.utils import similarity
from nicon.dynamic_connectivity import cluster_states


# Arguments
parser = argparse.ArgumentParser(description="K-MEANS")
parser.add_argument(
    "--data", metavar="FILE",
    default="/neurospin/lbi/monkeyfmri/deepstim/database/ANESTHETIC_database/derivatives/reference_kmeans/inputs/inputs.npy",
    help="path to the file containing the dynamic connectivity data")
parser.add_argument(
    "--metadata", metavar="FILE",
    default="/neurospin/lbi/monkeyfmri/deepstim/database/ANESTHETIC_database/derivatives/reference_kmeans/inputs/metadata.tsv",
    help="path to the file containing the metadata informations.")
parser.add_argument(
    "--ref-conn", metavar="FILE",
    default="/neurospin/lbi/monkeyfmri/deepstim/database/ANESTHETIC_database/derivatives/reference_kmeans/inputs/structural.txt",
    help="path to the file containing the reference structural connectivity "
         "data")
parser.add_argument(
    "--n-windows", default=464, type=int, metavar="N",
    help="the number of sliding windows.")
parser.add_argument(
    "--states", type=int, default=7, metavar="N",
    help="the number of states.")
parser.add_argument(
    "--test-monkey", default="jade", metavar="SID",
    help="the subject to leave out during the training.")
parser.add_argument(
    "--outdir", metavar="DIR",
    help="the output directory path",
    default = "/neurospin/lbi/monkeyfmri/deepstim/workspace/2023_ASM_tembedding/gitproject/2023_asm_tembedding/data/BrainStates/kmeans")
args = parser.parse_args()


# Load data
outdir = os.path.join(args.outdir, "kmeans{0}".format(args.states))
if not os.path.isdir(outdir):
    os.mkdir(outdir)
conn_filtered = np.load(args.data).squeeze()
n_regions = conn_filtered.shape[-1]
print("- conn:", conn_filtered.shape)
labels = ["roi{0}".format(idx) for idx in range(1, n_regions + 1)]
metadata = pd.read_csv(args.metadata, sep="\t")
names = metadata["condition"].unique()
print("- conditions:", names)
train_monkeys = metadata["monkey"].values.tolist()
train_monkeys.remove(args.test_monkey)
test_monkey = [args.test_monkey]
train_mask = metadata["monkey"].isin(train_monkeys).values
test_mask = metadata["monkey"].isin(test_monkey).values
print(metadata)


# Clustering
#njobs = multiprocessing.cpu_count() - 1
train_conn_filtered = conn_filtered[train_mask]
test_conn_filtered = conn_filtered[test_mask]
train_conn_filtered = train_conn_filtered.reshape(
    -1, args.n_windows, n_regions, n_regions)
print("- train conn:", train_conn_filtered.shape)
states, win_labels, linked, metrics = cluster_states(
    conn=train_conn_filtered,
    n_states=args.states,
    outdir=outdir,
    init=500,
    ctype="kmeans",
    return_raw=False,
    verbose=5)
print("- states:", states.shape)
print("- labels:", win_labels.shape)
pprint(metrics)
with open(os.path.join(outdir, "metrics.json"), "wt") as of:
    json.dump(metrics, of, indent=4)
np.save(os.path.join(outdir, "states.npy"), states)
np.save(os.path.join(outdir, "labels.npy"), win_labels)


# Similarity with structural: sort
structural = np.loadtxt(args.ref_conn)
structural = np.expand_dims(structural, axis=0)
states_similarity = np.concatenate((np.abs(states), structural), axis=0)
states = np.concatenate((states, structural), axis=0)
np.save(os.path.join(outdir, "full_states.npy"), states)
similarity_matrix = similarity(states_similarity)
print("- similarity:", similarity_matrix.shape)
state_labels = ["state {0}".format(idx) for idx in range(1, args.states + 1)]
structural_similarities = similarity_matrix[:-1, args.states]
structural_order = np.argsort(structural_similarities)
print("- structural order:", structural_order)
np.save(os.path.join(outdir, "order.npy"), structural_order)
with open(os.path.join(outdir, "order.json"), "wt") as of:
    json.dump(structural_order.tolist(), of)
print([structural_similarities[idx] for idx in structural_order])
labels = ["state{0}".format(idx) for idx in range(1, args.states + 1)]
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
fig.savefig(os.path.join(outdir, "brain_states.png"))


# Create brain states histogram
data = {}
win_labels = win_labels.flatten()
for idx, (_, row) in enumerate(metadata[train_mask].iterrows()):
    label = win_labels[idx]
    data.setdefault(row["condition"], []).append(label)
print(data.keys())
hist_data = np.zeros((args.states, len(names)), dtype=float)
for idx1 in range(args.states):
    sorted_indx = structural_order[idx1]
    for idx2, cond in enumerate(names):
        hist_data[idx1, idx2] = data[cond].count(sorted_indx)
hist_data /= hist_data.sum(axis=0, keepdims=1)
np.save(os.path.join(outdir, "hist.npy"), hist_data)
fig = plotting.plot_brain_states_hist(hist_data, names)
fig.savefig(os.path.join(outdir, "brain_states_hist.png"))
# plt.show()


# Save test set
test_labels = []
metrics = {}
for matrix in test_conn_filtered:
    scores = []
    matrix = np.expand_dims(matrix, axis=0)
    for idx, state in enumerate(states[:-1]):
        state = np.expand_dims(state, axis=0)
        data = np.concatenate((matrix, state), axis=0)
        metric = similarity(data, metric="corrcoef")
        metrics.setdefault(idx, []).append(metric[0, 1])
        scores.append(metric[0, 1])
    test_labels.append(np.argmax(scores))
np.save(os.path.join(outdir, "test_labels.npy"), test_labels)
result = pd.DataFrame.from_dict(collections.OrderedDict([
    ("pred", test_labels),]))
for idx in metrics:
    result["prob_{0}".format(idx)] = metrics[idx]
result.to_csv(os.path.join(outdir, "test_labels.tsv"), sep="\t", index=False)
