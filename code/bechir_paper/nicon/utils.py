# -*- coding: utf-8 -*-
###############################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
###############################################################################

"""
Module that privides common tools.
"""

# Imports
import os
import nibabel
import numpy as np
from collections import OrderedDict
from nilearn.input_data import NiftiLabelsMasker
from nilearn.input_data import NiftiSpheresMasker
from nilearn.input_data import NiftiMapsMasker
from nilearn.regions import connected_label_regions
from nilearn.image import high_variance_confounds
from joblib import Memory
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


def fisher_zscore(conn, outdir, cache=True, verbose=0):
    """ Fisher-Z score.
    For any particular value of r, the Pearson product-moment correlation
    coefficient, this section will perform the Fisher r-to-z transformation
    according to the formula: zr = (1/2)[loge(1+r) - loge(1-r)]
    If a value of N is entered (optional), it will also calculate the standard
    error of zr as: SEzr = 1/sqrt[N-3]

    Parameters
    ----------
    conn: array
        array with connectivities.
    outdir: str
        the destination folder.
    cache: bool, default True
        use smart caching.
    verbose: int, optional
        indicate the level of verbosity. By default, nothing is printed.

    Returns
    -------
    conn_filtered: array
        transformed correlations to a Fisher-Z score, as the bounded nature of
        Pearson correlation violates certain statistical assumptions.
    """
    if cache:
        cache_dir = os.path.join(outdir, "cachedir")
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        mem = Memory(cache_dir, verbose=verbose)
        fisher_zscore_cached = mem.cache(_fisher_zscore)
        return fisher_zscore_cached(conn)
    else:
        return _fisher_zscore(conn)


def _fisher_zscore(conn):
    """ See documentation of fisher_zscore.
    """
    conn_filtered = 0.5 * np.log((1 + conn) / (1 - conn))
    conn_filtered[np.isnan(conn_filtered)] = 0
    conn_filtered[np.isinf(conn_filtered)] = 0
    return conn_filtered


def similarity(conn, mask=None, metric="corrcoef"):
    """ Compute the similarity metric chosen,
            Pearson correlation by default.

    Parameters
    ----------
    conn: array (n_samples, n_regions, n_regions)
        array with connectivities.
    metric: str, default 'corrcoef'
        the similarity metric.

    Returns
    -------
    similarity: array (n_samples, n_samples)
        the similarity matrix.
    """
    n_samples, n_rois = conn.shape[:-1]
    iu = np.triu_indices(n_rois, k=1)
    # iu = np.tril_indices(n_rois, k= -1)
    conn_flat = [arr[iu] for arr in conn]
    if mask is not None:
        mask = mask[iu].astype(int)
        conn_flat = [arr[np.where(mask == 1)] for arr in conn_flat]
    if metric == "corrcoef":
        similarity = np.corrcoef(conn_flat)
    elif metric == "cosine":
        similarity = cosine_similarity(conn_flat)
    elif metric == "euclidean":
        similarity = euclidean_distances(conn_flat)
    else:
        raise ValueError("Unknown metric.")

    return similarity


def extract_centroids(atlas, background_label=0, affine=None):
    """ Extracts the centroids of the atlas regions.

    Parameters
    ----------
    atlas: str
        region definitions, as one image of labels (3d image).
    background_label: int, optional
        label used in to represent background.
    affine: array (4, 4), optional
        transform the centroid by applying this transformation.

    Returns
    -------
    coords: array (n_rois, 3)
        the ROI center coordinates in world space.
    """
    im = nibabel.load(atlas)
    arr = im.get_data()
    labels = sorted(set(np.unique(arr)) - {background_label})
    centroids = [np.argwhere(arr == lab).mean(axis=0) for lab in labels]
    if affine is None:
        affine = np.eye(4)
    trf = np.dot(im.affine, affine)
    return apply_affine(np.asarray(centroids), trf)


def apply_affine(coords, affine):
    """ Apply an affine transformation on each coordiantes.

    Parameters
    ----------
    coords: array (n_points, 3)
        the coordiantes.
    affine: array (4, 4)
        an affine transformation to applied.

    Returns
    -------
    new_coords: array (n_points, 3)
        the new coordiantes.
    """
    nb_points, _ = coords.shape
    ones = np.ones((nb_points, 1), dtype=coords.dtype)
    homogenous_coords = np.concatenate((coords, ones), axis=1)
    new_coords = np.dot(affine, homogenous_coords.T).T[..., :3]
    return new_coords


def extract_signal(images, atlas, outdir, tr, low_pass=0.1, high_pass=0.01,
                   smoothing_fwhm=5, masker_type="label", confounds=None,
                   compcor=True, verbose=0):
    """ Extracts BOLD time-series from regions of interest using 'sphere',
    'map' or 'label' nilearn Nifti masker.

    Parameters
    ----------
    images: list of str
        a list of images in the same space as the atlas.
    atlas: str or list
        depending on the masking strategy:
        -label: region definitions, as one image of labels (3d image).
        -map: set of continuous maps (4d image).
        -sphere: seed definitionsas a list of coordinates in the same space as
         the images (list of triplet).
    outdir: str
        the destination folder.
    tr: float
        repetition time, in second (sampling period). Set to None if not
        specified. Mandatory if used together with low_pass or high_pass.
    low_pass, high_pass: float, optional
        respectively low and high cutoff frequencies, in Hertz.
    smoothing_fwhm: float, optional
        if smoothing_fwhm is not None, it gives the full-width half maximum in
        millimeters of the spatial smoothing to apply to the signal.
    masker_type: str, optional
        type of masker used to extract BOLD signals: 'sphere', 'map' or
        'label'.
    confounds: list of str, default None
        these files are passed to signal.clean, shape: (number of scans,
        number of confounds).
    compcor: bool, default True
        add a compinent based noise correction confound.
    verbose: int, optional
        indicate the level of verbosity. By default, nothing is printed.

    Returns
    -------
    timeseries: array (n_subjects, n_times, n_regions)
        array of BOLD time-series.
    """
    cache_dir = os.path.join(outdir, "cachedir")
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    mem = Memory(cache_dir, verbose=verbose)
    if masker_type == "sphere":
        masker = NiftiSpheresMasker(
            seeds=atlas,
            radius=4,
            smoothing_fwhm=smoothing_fwhm,
            standardize=True,
            detrend=True,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=tr,
            memory=mem,
            verbose=verbose)
    elif masker_type == "map":
        masker = NiftiMapsMasker(
            maps_img=atlas,
            smoothing_fwhm=smoothing_fwhm,
            standardize=True,
            detrend=True,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=tr,
            memory=mem,
            resampling_target=None,
            verbose=verbose)
    elif masker_type == "label":
        masker = NiftiLabelsMasker(
            labels_img=atlas,
            smoothing_fwhm=smoothing_fwhm,
            standardize=True,
            detrend=True,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=tr,
            memory=mem,
            resampling_target=None,
            verbose=verbose)
    else:
        raise ValueError("Please provide a valid masker type.")
    timeseries = []
    high_variance_confounds_cached = mem.cache(high_variance_confounds)
    confounds = confounds or [None] * len(images)
    for path, confound in zip(images, confounds):
        _confounds = []
        if compcor:
            _confounds.append(high_variance_confounds_cached(path))
        if confound is not None:
            _confounds.append(confound)
        if len(_confounds) == 0:
            _confounds = None
        timeseries.append(masker.fit_transform(path, confounds=_confounds))
    return np.asarray(timeseries)


def get_average_profile(conn, seed, seeds, profile, average=True):
    """ Compute the connectivity profile between one seed and all others.

    Parameters
    ----------
    conn: array
        array with connectivities.
    seed: str
        the seed used to compute the connectivity profile.
    seeds: dict
        the seeds name - label mapping.
    profile: dict
        the current average profile.
    average: bool, default True
        if set average connectivities.

    Returns
    -------
    profile: dict
        the updated average profile.
    """
    if seed not in profile:
        profile[seed] = OrderedDict()
    data = conn[:, tuple(seeds[seed])]
    data = np.mean(data, axis=1)
    data.shape += (1, )
    for key, value in seeds.items():
        if key == seed:
            continue
        if average:
            profile[seed].setdefault(key, []).append(
                np.mean(data[tuple(seeds[key]), :]))
        else:
            profile[seed].setdefault(key, []).append(
                np.mean(data[tuple(seeds[key]), :, 0], axis=0))
    return profile
