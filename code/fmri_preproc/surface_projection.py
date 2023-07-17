# -*- coding: utf-8 -*-
###############################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
###############################################################################

"""
Project data on template white surface and compute ROI connectivities.
"""

# Imports
import os
import nibabel
import pypreclin
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn import surface as nilearn_surface
from nilearn.connectome import ConnectivityMeasure
from nicon.dynamic_connectivity import _sliding_window


# Global parameters
test = False
meta_file = "/neurospin/lbi/monkeyfmri/deepstim/database/ANESTHETIC_database/participants.tsv"
data_dir = "/neurospin/lbi/monkeyfmri/deepstim/database/ANESTHETIC_database/derivatives/pypreclin_timeseries"
outdir_name = "pypreclin_surfaces"
mask_file = "/neurospin/lbi/monkeyfmri/images/reference/mni-resampled_1by1by1.nii"
surf_dir = os.path.join(
	os.path.dirname(pypreclin.__file__), "resources", "MNI", "result", "sub",
	"surf")
rh_white_file = os.path.join(surf_dir, "rh.white.native.gii")
rh_annot_file = os.path.join(surf_dir, "rh.cocomac.white.native.annot")
lh_white_file = os.path.join(surf_dir, "lh.white.native.gii")
lh_annot_file = os.path.join(surf_dir, "lh.cocomac.white.native.annot")
n_jobs = 10


def read_gifti(surf_file):
    image = nibabel.load(surf_file)
    nb_of_surfs = len(image.darrays)
    if nb_of_surfs != 2:
        raise ValueError(
            "'{0}' does not a contain a valid white mesh.".format(surf_file))
    vertices = image.darrays[0].data
    triangles = image.darrays[1].data
    return vertices, triangles


def surf_edges(shape, rh_vertices, lh_vertices):
    edge_arr = np.zeros(shape, dtype=int)
    for vertices in (rh_vertices, lh_vertices):
        indices = np.round(vertices).astype(int).T
        indices[0, np.where(indices[0] >= shape[0])] = 0
        indices[1, np.where(indices[1] >= shape[1])] = 0
        indices[2, np.where(indices[2] >= shape[2])] = 0
        edge_arr[indices.tolist()] = 1
    return edge_arr


# Project signal on template white matter surface
def wm_projection(fmri_file, mask_file, rh_white_file, lh_white_file, outdir,
                  debug=False):
    """ Project fmri BOLD signal on white matter mesh.

    All data must be in the same voxel space.

    Parameters
    ----------
    fmri_file: str
        the fmri volume to be projected.
    mask_file: str
        a brain mask.
    rh_white_file: str
        the white matter right hemisphere mesh.
    lh_white_file: str
        the white matter left hemisphere mesh.
    outdir: str
        the destination folder.

    Returns
    -------
    *h_texture_file: str
        the BOLD signal projected on the *h hemisphere white mesh.
    *h_win_texture_files: list of str
        the BOLD signal projected on the *h hemisphere white mesh using
        sliding windows.
    """
    im = nibabel.load(fmri_file)
    mask_im = nibabel.load(mask_file)
    if debug:
        print(im.affine, nibabel.orientations.aff2axcodes(im.affine))
        print(mask_im.affine, nibabel.orientations.aff2axcodes(mask_im.affine))
    im = nibabel.Nifti1Image(im.get_data(), np.eye(4))
    mask_im = nibabel.Nifti1Image(mask_im.get_data(), np.eye(4))
    if debug:
        rh_vertices, _ = read_gifti(rh_white_file)
        lh_vertices, _ = read_gifti(lh_white_file)
        edge_arr = surf_edges(im.shape, rh_vertices, lh_vertices)
        edge_im = nibabel.Nifti1Image(edge_arr, im.affine)
        nibabel.save(edge_im, os.path.join(outdir, "sub-template_white.nii.gz"))
    rh_texture = nilearn_surface.vol_to_surf(
        im, rh_white_file, radius=1., interpolation="linear",
        kind="line", mask_img=None)
    lh_texture = nilearn_surface.vol_to_surf(
        im, lh_white_file, radius=1., interpolation="linear",
        kind="line", mask_img=None)
    rh_texture = np.expand_dims(rh_texture, axis=1)
    lh_texture = np.expand_dims(lh_texture, axis=1)
    if debug:
        for data in (rh_texture, lh_texture):
            print("-- texture:", data.shape)
    basename = os.path.basename(fmri_file).rsplit("_", 1)[0]
    rh_texture_file = os.path.join(outdir, basename + "_hemi-rh_bold.npy")
    lh_texture_file = os.path.join(outdir, basename + "_hemi-lh_bold.npy")
    np.save(rh_texture_file, rh_texture)
    np.save(lh_texture_file, lh_texture)

    rh_win_texture_files = []
    lh_win_texture_files = []
    rh_texture = np.transpose(rh_texture, (0, 2, 1)) 
    lh_texture = np.transpose(lh_texture, (0, 2, 1)) 
    for win_size, sliding_step in ((100, 50), (33, 16)):
        rh_windows = _sliding_window(
            rh_texture, win_size, sliding_step, with_tapering=False).squeeze()
        lh_windows = _sliding_window(
            lh_texture, win_size, sliding_step, with_tapering=False).squeeze()
        if debug:
            for data in (rh_windows, lh_windows):
                print("-- windows:", data.shape)
        rh_win_texture_file = os.path.join(
            outdir, basename + "_win-{0}|{1}_hemi-rh_bold.npy".format(
                win_size, sliding_step))
        lh_win_texture_file = os.path.join(
            outdir, basename + "_win-{0}|{1}_hemi-lh_bold.npy".format(
                win_size, sliding_step))
        np.save(rh_win_texture_file, rh_windows)
        np.save(lh_win_texture_file, lh_windows)
        rh_win_texture_files.append(rh_win_texture_file)
        lh_win_texture_files.append(lh_win_texture_file)

    return (rh_texture_file, lh_texture_file, rh_win_texture_files,
            lh_win_texture_files)


# Project signal on white matter surface atlas
def wm_atlas_projection(rh_texture_file, rh_win_texture_files, rh_annot_file,
                        lh_texture_file, lh_win_texture_files, lh_annot_file,
                        outdir, debug=False):
    """ Intersect fmri BOLD signal on white matter mesh with an atlas
    and compute connectivity profiles using correlation.

    Parameters
    ----------
    *h_texture_file: str
        the BOLD signal projected on the *h hemisphere white mesh.
    *h_win_texture_files: list of str
        the BOLD signal projected on the *h hemisphere white mesh using
        sliding windows.
    *h_annot_file: str
        the corresponding atlas labels.
    outdir: str
        the destination folder.
    """
    # Load data
    rh_labels, rh_ctab, rh_names = nibabel.freesurfer.read_annot(rh_annot_file)
    lh_labels, lh_ctab, lh_names = nibabel.freesurfer.read_annot(lh_annot_file)
    # connectivity_measure = ConnectivityMeasure(kind="partial correlation")
    indices = {"rh": (rh_labels != 0), "lh": (lh_labels != 0)}

    # Go through all textures
    for _rh_file, _lh_file in zip([rh_texture_file] + rh_win_texture_files,
                                  [lh_texture_file] + lh_win_texture_files):
        rh_texture = np.load(_rh_file)
        lh_texture = np.load(_lh_file)
        if debug:
            print("-- textures:", rh_texture.shape, lh_texture.shape)
        n_windows = rh_texture.shape[1]
        basename = {
            "rh": os.path.basename(_rh_file).rsplit("_", 1)[0] + "_profiles.npy",
            "lh": os.path.basename(_lh_file).rsplit("_", 1)[0] + "_profiles.npy"}

        # Compute label wise average connectivity profiles
        mean_profiles = []
        for _labels, _texture in [(rh_labels, rh_texture),
                                  (lh_labels, lh_texture)]:
            for label in sorted(set(np.unique(_labels)) - {0}):
                mean_profiles.append(np.mean(_texture[_labels == label], axis=0))
        mean_profiles = np.asarray(mean_profiles)
        if debug:
            print("-- mean profiles:", mean_profiles.shape)
            assert not np.isnan(mean_profiles).any()

        # Compute vertex wise connectivity profiles
        con_profiles = {}
        for _hemi, _labels, _texture in [("rh", rh_labels, rh_texture),
                                         ("lh", lh_labels, lh_texture)]:
            for _windows, _label in zip(_texture, _labels):
                con = []
                for idx in range(n_windows):
                    _signal = _windows[idx]
                    if _label == 0:
                        con.append([np.nan] * len(mean_profiles))
                    else:
                        con.append([
                            np.correlate(_signal, _mean_signal, mode="valid")[0]
                            for _mean_signal in mean_profiles[:, idx]])
                        # _signal.shape += (1, )
                        # _data = np.concatenate((_signal.T, mean_profiles),
                        #                        axis=0).T
                        # con = connectivity_measure.fit_transform([_data])[0, 0, 1:]
                con_profiles.setdefault(_hemi, []).append(con)
            con_profiles[_hemi] = np.asarray(con_profiles[_hemi])
            if debug:
                print("-- con profile:", _hemi, con_profiles[_hemi].shape)
                assert not np.isnan(con_profiles[_hemi][indices[_hemi]]).any()
            profile_file = os.path.join(outdir, basename[_hemi])
            np.save(profile_file, con_profiles[_hemi])


def subject_projection(fmri_file, mask_file, rh_white_file, rh_annot_file,
                       lh_white_file, lh_annot_file, outdir, debug=False):
    (rh_texture_file, lh_texture_file, rh_win_texture_files,
     lh_win_texture_files) = wm_projection(
        fmri_file, mask_file, rh_white_file, lh_white_file, outdir, debug)
    wm_atlas_projection(
        rh_texture_file, rh_win_texture_files, rh_annot_file,
        lh_texture_file, lh_win_texture_files, lh_annot_file, outdir,
        debug)


# Load data
df = pd.read_csv(meta_file, sep="\t", dtype=str)
print(df)
basename = "sub-{0}_ses-{1}_task-rest_space-mni_run-{2}_desc-smask_bold.nii.gz"
dataset = []
for index, row in df.iterrows():
    fmri_file = os.path.join(
        data_dir, "sub-{0}".format(row["sub"]), "ses-{0}".format(row["ses"]),
        "func", basename.format(row["sub"], row["ses"], row["run"]))
    if not os.path.isfile(fmri_file):
        print(fmri_file)
        continue
    outdir = os.path.dirname(fmri_file).replace(
        "pypreclin_timeseries", outdir_name)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    dataset.append((fmri_file, outdir))
    if test:
        break
print("nb runs: {0} / {1}".format(len(dataset), len(df)))


# Parallel call
Parallel(n_jobs=n_jobs, verbose=20)(delayed(subject_projection)(
    fmri_file, mask_file, rh_white_file, rh_annot_file, lh_white_file,
    lh_annot_file, outdir, test)
        for fmri_file, outdir in dataset)
