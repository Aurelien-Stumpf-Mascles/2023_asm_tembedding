# -*- coding: utf-8 -*
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Olivier Cornelis
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Module to compute DFC distance auxiliary varaibles.
"""

# Imports
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from tembedding.metrics import batch_inv_structure_ssim
from tembedding.color_utils import print_yellow, print_white, print_cyan


def precompute_dfc_dist_exp(dfc_file, metadata_file, run_idx, outdir,
                            n_wins=464, n_rois=82):
    """ Precompute the distance between any two pair of dynamic functional
    connectivity matrices (dFCs).

    Parameters
    ----------
    dfc_file: str
        the path to the dFCs.
    metadata_file: str
        the path to the associated meta data.
    run_idx: int
        the run index to be processed.
    outdir: str
        the destination folder containing the computed distances.
    n_wins: int, default 464
        the number of sliding windows.
    n_rois: int, default 82
        the number of ROIs in the considered template.
    """
    print_yellow("Load input dFC and associated metadata...")
    dfc = np.load(dfc_file).astype(np.single)
    dfc = torch.from_numpy(dfc.reshape(-1, n_wins, n_rois, n_rois))
    n_runs = len(dfc)
    meta = pd.read_csv(metadata_file, sep="\t")
    print_white(f"- dFC: {dfc.shape}")
    print_white(meta)

    print_yellow("Normalize dFC...")
    dfc -= torch.mean(dfc, dim=(-2,-1)).reshape(-1, n_wins, 1, 1)

    print_yellow(f"Compute distances between dFCs for run {run_idx}...")
    dist = np.zeros((n_wins, n_runs, n_wins), dtype=np.single)
    for widx in tqdm(range(n_wins)):
        for ridx in range(n_runs):
            ssim = batch_inv_structure_ssim(
                dfc[run_idx, widx], dfc[ridx]).detach().numpy()
            dist[widx, ridx] = ssim
    filename = os.path.join(outdir, f"struct-dfc_run-{run_idx}_dists.npy")
    np.save(filename, dist)
    print_cyan(filename)
