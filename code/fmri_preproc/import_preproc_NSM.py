# -*- coding: utf-8 -*-
###############################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
###############################################################################


# Imports
import os
import glob
import shutil
from pprint import pprint
import progressbar
import pandas as pd


# Copy files preprocessed spatially with NSM and temporally by .mat missing (Filtered_4DVolume.nii)

# Global parameters
process = False
meta_file = "/neurospin/lbi/monkeyfmri/deepstim/database/ANESTHETIC_database/participants.tsv"
data_dir = "/neurospin/lbi/monkeyfmri/DICOM/3T_dicom/"
outdir = "/neurospin/lbi/monkeyfmri/deepstim/database/ANESTHETIC_database/derivatives/nsm_timeseries/matlab_extraction/"


# Get all preproc files
df = pd.read_csv(meta_file, sep="\t", dtype=str)
df = df.replace(regex={'sevoflurane4': 'sevoflurane_4',
                       'sevoflurane2': 'sevoflurane_2',
    'light-propofol': 'light_propofol',
    'deep-propofol': 'deep_propofol'})
print(df)
dataset = {}
meta = {}
metafiles = {}
with progressbar.ProgressBar(max_value=len(df)) as bar:
    for index, row in df.iterrows():

        unique_id = "{0}-{1}-{2}".format(row["sub"], row["ses"], row["run"])
        ses_id = "{0}-{1}".format(row["sub"], row["ses"])
        if unique_id in ("rana-20150626-019", "rana-20150626-020"):
            continue

        basedir = glob.glob(os.path.join(
            data_dir, row["sub"], row["ses"][:4], row["ses"][4:6], row["ses"],
            "resting_state", row["cond"], "run{0}_{1}_{2}_*{3}ABC_filtered").format(
                row["run"].lstrip('0'), row["ses"], row["sub"], row["cond"][-1]))
        assert len(basedir) == 1
        basedir = basedir[0]
        funcdir = os.path.join(
            outdir, "sub-{0}".format(row["sub"]), "ses-{0}".format(row["ses"]),
            "func")
        anatdir = os.path.join(
            outdir, "sub-{0}".format(row["sub"]), "ses-{0}".format(row["ses"]),
            "anat")
        for path in (funcdir, anatdir):
            if not os.path.isdir(path):
                os.makedirs(path)
        if ses_id not in meta:
            metafiles[ses_id] = os.path.join(
                os.path.dirname(funcdir), "sub-{0}_ses-{1}.tsv".format(
                    row["sub"], row["ses"]))
            meta[ses_id] = pd.DataFrame([[row["sub"], row["ses"], row["run"],
                                          row["cond"], basedir]])
            meta[ses_id].columns = ["sub", "ses", "run", "cond", "source"]
        else:
            meta[ses_id].loc[len(meta[ses_id])] = [
                row["sub"], row["ses"], row["run"], row["cond"], basedir]
        _data = {}

        _data["mni_masked_fmri_file"] = os.path.join(basedir, "Filtered_4DVolume.nii")
        _data["dest_mni_masked_fmri_file"] = os.path.join(
            funcdir, "sub-{0}_ses-{1}_task-rest_space-mni_run-{2}_desc-smask_bold.nii".format(
                row["sub"], row["ses"], row["run"]))

        path = _data["mni_masked_fmri_file"]
        assert os.path.isfile(path), path

        dataset[unique_id] = _data
        bar.update(index + 1)
pprint(dataset)
pprint(meta)


# Copy & rename files
with progressbar.ProgressBar(max_value=len(dataset)) as bar:
    for index, (unique_id, _data) in enumerate(dataset.items()):
        for _key, _path in _data.items():
            if _key.startswith("dest_"):
                continue
            _dest_path = _data["dest_" + _key]
            if not os.path.isfile(_dest_path):
                if process:
                    shutil.copy(_path, _dest_path)
                else:
                    print(_path, "->", _dest_path)
        bar.update(index)
            

# Save metadata
with progressbar.ProgressBar(max_value=len(dataset)) as bar:
    for index, (unique_id, _df) in enumerate(meta.items()):
        _path = metafiles[unique_id]
        if process:
            _df.to_csv(_path, sep="\t", index=False)
        else:
            print(_path)
        bar.update(index)

#%%
# Copy files only preprocessed spatially with NSM

# Global parameters
process = True
meta_file = "/neurospin/lbi/monkeyfmri/deepstim/database/ANESTHETIC_database/participants.tsv"
data_dir = "/neurospin/lbi/monkeyfmri/DICOM/3T_dicom/"
outdir = "/neurospin/lbi/monkeyfmri/deepstim/database/ANESTHETIC_database/derivatives/nsm"


# Get all preproc files
df = pd.read_csv(meta_file, sep="\t", dtype=str)
df = df.replace(regex={'sevoflurane4': 'sevoflurane_4',
                       'sevoflurane2': 'sevoflurane_2',
                       'light-propofol': 'light_propofol',
                       'deep-propofol': 'deep_propofol'})
print(df)
dataset = {}
meta = {}
metafiles = {}
with progressbar.ProgressBar(max_value=len(df)) as bar:
    for index, row in df.iterrows():

        unique_id = "{0}-{1}-{2}".format(row["sub"], row["ses"], row["run"])
        ses_id = "{0}-{1}".format(row["sub"], row["ses"])
        if unique_id in ("rana-20150626-019", "rana-20150626-020"):
            continue

        basedir = glob.glob(os.path.join(
            data_dir, row["sub"], row["ses"][:4], row["ses"][4:6], row["ses"],
            "resting_state", row["cond"], "run{0}_{1}_{2}_*{3}").format(
                row["run"].lstrip('0'), row["ses"], row["sub"], row["cond"][-1]))
        assert len(basedir) == 1
        basedir = basedir[0]
        funcdir = os.path.join(
            outdir, "sub-{0}".format(row["sub"]), "ses-{0}".format(row["ses"]),
            "func")
        anatdir = os.path.join(
            outdir, "sub-{0}".format(row["sub"]), "ses-{0}".format(row["ses"]),
            "anat")
        for path in (funcdir, anatdir):
            if not os.path.isdir(path):
                os.makedirs(path)
        if ses_id not in meta:
            metafiles[ses_id] = os.path.join(
                os.path.dirname(funcdir), "sub-{0}_ses-{1}.tsv".format(
                    row["sub"], row["ses"]))
            meta[ses_id] = pd.DataFrame([[row["sub"], row["ses"], row["run"],
                                          row["cond"], basedir]])
            meta[ses_id].columns = ["sub", "ses", "run", "cond", "source"]
        else:
            meta[ses_id].loc[len(meta[ses_id])] = [
                row["sub"], row["ses"], row["run"], row["cond"], basedir]
        _data = {}

        _data["mni_masked_fmri_file"] = glob.glob(os.path.join(basedir, "sMNI*.nii"))
        _data["dest_mni_masked_fmri_file"] = os.path.join(
            funcdir, "sub-{0}_ses-{1}_task-rest_space-mni_run-{2}_desc-smask_bold.nii".format(
                row["sub"], row["ses"], row["run"]))

        path = _data["mni_masked_fmri_file"]
        assert len(path) == 1
        _data["mni_masked_fmri_file"] = _data["mni_masked_fmri_file"][0]
        
        _data["confounds_file"] = os.path.join(basedir, "cmcrp*.txt")
        _data["dest_confounds_file"] = os.path.join(
            funcdir, "sub-{0}_ses-{1}_run-{2}_confounds.par".format(
                row["sub"], row["ses"], row["run"]))

        key ="confounds_file"
        regex = _data[key]
        _files = glob.glob(regex)
        assert len(_files) == 1, regex
        _data[key] = _files[0]

        dataset[unique_id] = _data
        bar.update(index + 1)
pprint(dataset)
pprint(meta)


# Copy & rename files
with progressbar.ProgressBar(max_value=len(dataset)) as bar:
    for index, (unique_id, _data) in enumerate(dataset.items()):
        for _key, _path in _data.items():
            if _key.startswith("dest_"):
                continue
            _dest_path = _data["dest_" + _key]
            if not os.path.isfile(_dest_path):
                if process:
                    shutil.copy(_path, _dest_path)
                else:
                    print(_path, "->", _dest_path)
        bar.update(index)
            

# Save metadata
with progressbar.ProgressBar(max_value=len(dataset)) as bar:
    for index, (unique_id, _df) in enumerate(meta.items()):
        _path = metafiles[unique_id]
        if process:
            _df.to_csv(_path, sep="\t", index=False)
        else:
            print(_path)
        bar.update(index)

