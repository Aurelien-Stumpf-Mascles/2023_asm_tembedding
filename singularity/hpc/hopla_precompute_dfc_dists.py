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
import fire
import glob
import datetime
import collections
import pandas as pd
from hopla.converter import hopla


def run(dfc_file, metadata_file, outdir, simg_file, n_wins=464, n_rois=82,
        name="dfc_dists", process=False, njobs=10, use_pbs=False,
        test=False):
    """ Parse data and execute the processing with hopla.

    Parameters
    ----------
    dfc_file: str
        the path to the dFCs.
    metadata_file: str
        the path to the associated meta data.
    outdir: str
        the destination folder containing the computed distances.
    simg_file: str
        path to the 'tembedding' singularity image.
    n_wins: int, default 464
        the number of sliding windows.
    n_rois: int, default 82
        the number of ROIs in the considered template.
    name: str, default 'dfc_dists'
        the name of the current analysis.
    process: bool, default False
        optionally launch the process.
    njobs: int, default 10
        the number of parallel jobs.
    use_pbs: bool, default False
        optionnaly use PBSPRO batch submission system.
    test: bool, default False
        optionally, select only one run.
    """
    meta = pd.read_csv(metadata_file, sep="\t")
    n_dfcs = len(meta)
    n_runs = int(n_dfcs / n_wins)
    run_indices = []
    for idx in range(n_runs):
        dist_file = os.path.join(outdir, f"struct-dfc_run-{idx}_dists.npy")
        if not os.path.isfile(dist_file):
            run_indices.append(idx)
        else:
            print(f"'{dist_file}' already on disk.")
    if len(run_indices) == 0:
        raise RuntimeError("No data to process!")
    if test:
        run_indices = run_indices[:1]
    print(f"number of runs: {len(run_indices)}")

    if process:
        pbs_kwargs = {}
        if use_pbs:
            clusterdir = os.path.join(outdir, f"{name}_pbs")
            if not os.path.isdir(clusterdir):
                os.makedirs(clusterdir)
            pbs_kwargs = {
                "hopla_cluster": True,
                "hopla_cluster_logdir": clusterdir,
                "hopla_cluster_queue": "Nspin_long"}
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = os.path.join(outdir, "logs")
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
        logfile = os.path.join(logdir, f"{name}_{date}.log")
        cmd = (f"singularity run --bind /neurospin --cleanenv "
               f"{simg_file} precompute-dfc-dist")
        status, exitcodes = hopla(
            cmd,
            dfc_file=dfc_file,
            metadata_file=metadata_file,
            outdir=outdir,
            run_idx=run_indices,
            hopla_name_replace=True,
            hopla_iterative_kwargs=["run-idx"],
            hopla_optional=["dfc-file", "metadata-file", "run-idx"],
            hopla_cpus=njobs,
            hopla_logfile=logfile,
            hopla_use_subprocess=True,
            hopla_verbose=1,
            hopla_python_cmd=None,
            **pbs_kwargs)


if __name__ == "__main__":
    fire.Fire(run)
