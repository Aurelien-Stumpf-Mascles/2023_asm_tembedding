# -*- coding: utf-8 -*-
###############################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
###############################################################################

"""
Module that provides common plots.
"""

# Imports
import os
import matplotlib
from matplotlib import gridspec
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
import seaborn as sns
from joblib import Memory
import numpy as np
import nilearn.plotting as plotting
from scipy.cluster.hierarchy import dendrogram
import torch
import torchvision
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import manifold
from nicon.utils import extract_centroids
import nilearn.plotting.glass_brain as glass_brain
from mne.viz import circular_layout, _plot_connectivity_circle
import pandas as pd


def show():
    """ Show all the figures generated.
    """
    if matplotlib.get_backend().lower() != "agg":  # avoid warnings
        plt.show()


def plot_brain_states_hist(hist_data, names):
    """ Plot the brain states histogram has defined in the article.

    Parameters
    ----------
    hist_data: array (n_states, n_conditions)
        the histogram to be displayed.
    names: list (n_conditions)
        the conditions names.
    """
    hist_data = hist_data.T
    x = np.arange(hist_data.shape[0])
    width = 0.8 / hist_data.shape[1]
    nb_rect = hist_data.shape[1]
    ticks = []
    ticks_names = []
    fig, ax = plt.subplots()
    for idx in range(nb_rect):
        tick = x - width * (hist_data.shape[1] - 1 - idx)
        ax.bar(tick, hist_data[:, idx], width,
               color=[str(c) for c in np.linspace(0.8, 0.2, hist_data.shape[0])],
               edgecolor="white")
        ticks.extend(tick.tolist())
        ticks_names.extend([str(idx + 1)] * len(x))
    custom_lines = []
    for color in np.linspace(0.8, 0.2, hist_data.shape[0]):
        custom_lines.append(Line2D([0], [0], color=str(color), lw=6))
    ax.set_ylabel("Probability")
    ax.set_title("Brain States", va = "center_baseline")
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks_names)
    # ax.legend(custom_lines, names)
    ax.legend(custom_lines, names, loc='best', fancybox=True, shadow=True, 
              bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
        
    return fig


def _get_civmr_json_and_transform(direction):
    """Returns the json filename and and an affine transform, which has
    been tweaked by hand to fit the CIVMR template.
    """
    from matplotlib import transforms
    direction_to_view_name = {'x': 'side',
                              'y': 'back',
                              'z': 'top',
                              'l': 'side',
                              'r': 'side'}
    direction_to_transform_params = {
        'x': [0.73, 0, 0, 0.7, -49, -19],
        'y': [0.7, 0, 0, 0.7, -28, -17],
        'z': [0.7, 0, 0, 0.72, -29, -36],
        'l': [0, 0, 0, 0, 0, 0],
        'r': [0, 0, 0, 0, 0, 0]}

    dirname = os.path.dirname(os.path.abspath(__file__))
    dirname = os.path.join(dirname, 'glass_brain_files')
    direction_to_filename = dict([
        (_direction, os.path.join(
            dirname,
            'brain_schematics_{0}.json'.format(view_name)))
        for _direction, view_name in direction_to_view_name.items()])

    direction_to_transforms = dict([
        (_direction, transforms.Affine2D.from_values(*params))
        for _direction, params in direction_to_transform_params.items()])

    direction_to_json_and_transform = dict([
        (_direction, (direction_to_filename[_direction],
                      direction_to_transforms[_direction]))
        for _direction in direction_to_filename])

    filename_and_transform = direction_to_json_and_transform.get(direction)

    if filename_and_transform is None:
        message = ("No glass brain view associated with direction '{0}'. "
                   "Possible directions are {1}").format(
                       direction,
                       list(direction_to_json_and_transform.keys()))
        raise ValueError(message)

    return filename_and_transform


def plot_network(con, atlas, labels, outdir, title=None, verbose=0):
    """ Diplay the connectivity network.
    """
    figure, ax = plt.subplots()
    cache_dir = os.path.join(outdir, "cachedir")
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    mem = Memory(cache_dir, verbose=verbose)
    glass_brain._get_json_and_transform = _get_civmr_json_and_transform
    extract_centroids_cached = mem.cache(extract_centroids)
    atlas_centroids = extract_centroids_cached(atlas, affine=None)

    try:
        from pyconnectome.plotting.network import plot_network
        import nibabel
        thr = sorted(con.flatten())[-100]
        edges = np.argwhere(con >= thr)
        edge_weights = con[con >= thr] * 255.
        weights = np.ones((len(atlas_centroids), )) * 5
        im = nibabel.load(atlas)
        arr = im.get_data()
        _atlas_centroids = atlas_centroids.copy()
        _atlas_centroids[:, 0] += arr.shape[0] // 2
        # TODO: +8 ??
        _atlas_centroids[:, 1] += (arr.shape[1] / 2 + 8)
        _atlas_centroids[:, 2] += arr.shape[2] // 2
        plot_network(nodes=_atlas_centroids, labels=labels, weights=weights,
                     edges=edges, mask=atlas, weight_node_by_size=True,
                     edge_weights=edge_weights, weight_edge_by_color=True,
                     interactive=False, snap=True, animate=False, outdir=outdir,
                     name=title + "_up", actor_ang=(0., 0., 0.))
        plot_network(nodes=_atlas_centroids, labels=labels, weights=weights,
                     edges=edges, mask=atlas, weight_node_by_size=True,
                     edge_weights=edge_weights, weight_edge_by_color=True,
                     interactive=False, snap=True, animate=False, outdir=outdir,
                     name=title + "_side1", actor_ang=(-90., 0., 90.))
        plot_network(nodes=_atlas_centroids, labels=labels, weights=weights,
                     edges=edges, mask=atlas, weight_node_by_size=True,
                     edge_weights=edge_weights, weight_edge_by_color=True,
                     interactive=False, snap=True, animate=False, outdir=outdir,
                     name=title + "_side2", actor_ang=(90., 180., 90.))
    except:
        pass

    plotting.plot_connectome(
        con, atlas_centroids, figure=figure, edge_threshold=thr, node_size=1)
    ax.set_title(title)
    ax.axis("off")
    # plt.show()
    # stop


def plot_timeseries(timeseries, ncols=3):
    """ Create a new figure with the timeseries displayed.

    Parameters
    ----------
    timeseries: array (n_subjects, n_times, n_regions)
        array of BOLD time-series.
    ncols: int, optional
        the number of columns in the created figure.
    """
    nb_subjects = len(timeseries)
    nrows, rest = divmod(nb_subjects, ncols)
    if rest > 0:
        nrows += 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    for idx in range(nb_subjects):
        axs[divmod(idx, ncols)].plot(timeseries[idx])


def plot_matrix(*args, **kwargs):
    """ Plot the given matrix using 'nilearn.plotting.plot_matrix'.
    """
    plotting.plot_matrix(*args, **kwargs)


def plot_array(arr, vmin=None, vmax=None, title=None, square=True, auto=True,
               cbar=True, figure=None, subplot_spec=None, cmap=None,
               mask=None, nonsym_cmap=False):
    """ Plot the given array using seaborn.
    """
    if subplot_spec is not None or figure is not None:
        ax = figure.add_subplot(subplot_spec)
    else:
        figure, ax = plt.subplots()
    kwargs = {"cbar": cbar}
    if not auto:
        kwargs["xticklabels"] = True
        kwargs["yticklabels"] = True
    else:
        kwargs["xticklabels"] = False
        kwargs["yticklabels"] = False
    if cmap is not None and nonsym_cmap:
        colors = plt.get_cmap(cmap, 101)(np.linspace(0, 1, 101))
        colors = np.concatenate((colors, colors), axis=0)
        cmap = matplotlib.colors.ListedColormap(colors)
    if cmap is not None:
        if cmap == "diverging":
            cmap = sns.diverging_palette(240, 10, n=30, as_cmap=True)
        kwargs["cmap"] = cmap
    else:
        kwargs["cmap"] = "jet"
    if mask is not None:
        kwargs["mask"] = mask
    ax = sns.heatmap(
        arr, 
        vmin=vmin, vmax=vmax, center=0,
        square=square,
        **kwargs
    )
    if title is not None:
        ax.set_title(title)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment="right"
    )
    #ax.axhline(arr.shape[0] // 2, c="w")
    #ax.axvline(arr.shape[1] // 2, c="w")


def plot_mosaic(data, names, figure=None, subplot_spec=None, *args, **kwargs):
    """ Plot the given matrices as a mosaic.
    """
    if subplot_spec is not None or figure is not None:
        ax = figure.add_subplot(subplot_spec)
    else:
        figure, ax = plt.subplots()
    _data = np.asarray(data)
    _data = np.expand_dims(_data, axis=1)
    tensor = torch.from_numpy(_data)
    out = torchvision.utils.make_grid(tensor)
    im = ax.imshow(out.numpy().transpose((1, 2, 0)), cmap="gray", *args, **kwargs)
    ax.set_title(names)
    figure.colorbar(im, ax=ax)


def plot_connectome(*args, **kwargs):
    """ Plot connectome on top of the brain glass schematics using 
    'nilearn.plotting.plot_connectome'.
    """
    plotting.plot_connectome(*args, **kwargs)


def plot_dendogram(linked, threshold=None, n_leafs=None):
    """ Display a dendogram as generated by scipy.

    Parameters
    ----------
    linked: array
        the hierarchical clustering encoded as a linkage matrix.
    threshold: float, default None
        a cuting threshold.
    n_leafs: int, default None
        display anly the N last fusion.
    """
    kwargs = {}
    if threshold is not None:
        kwargs["color_threshold"] = threshold
    if n_leafs is not None:
        kwargs["truncate_mode"] = "lastp"
        kwargs["p"] = n_leafs 
    dendrogram(linked,
               distance_sort="descending",
               **kwargs)
    if threshold is not None:
        plt.axhline(y=2200, c="grey", lw=1, linestyle="dashed")


def plot_silhouette(n_clusters, labels, data, outdir, figure=None,
                    subplot_spec=None, verbose=0):
    """ Silhouette analysis can be used to study the separation distance
    between the resulting clusters. The silhouette plot displays a measure
    of how close each point in one cluster is to points in the neighboring
    clusters and thus provides a way to assess parameters like number of
    clusters visually. This measure has a range of [-1, 1].

    Parameters
    ----------
    n_clusters: int
        the number of clusters.
    labels: arr (n_samples)
        the clustering labels.
    data: arr (n_samples, n_features)
        the clustering data.
    outdir: str
        the destination folder.
    figure: plt.Figure, default None
        a Matplotlib figure used, if None is given, a
        new figure is created.
    subplot_spec: matplotlib.gridspec.SubplotSpec, default None
        the axes that will be subdivided in 2.
    verbose: int, default 0
        indicate the level of verbosity. By default, nothing is printed.

    Returns
    -------
    fig: plt.Figure
        the generated figure.
    """
    # Some caching
    cache_dir = os.path.join(outdir, "cachedir")
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    mem = Memory(cache_dir, verbose=verbose)

    # Use a dimension reduction technique to generate a 2d display
    tsne_cached = mem.cache(_tsne)
    data_plane = tsne_cached(
        data=data,
        n_components=2,
        init="random",
        perplexity=50,
        n_iter=3000,
        verbose=verbose)
    print(data_plane.shape)

    # Create a figure with 1 row and 2 columns
    if subplot_spec is not None or figure is not None:
        nested_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=subplot_spec, wspace=0.2)
        fig = figure
        ax1 = plt.Subplot(fig, nested_gs[0])
        ax2 = plt.Subplot(fig, nested_gs[1])
        fig.add_subplot(ax1)
        fig.add_subplot(ax2)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
    ax1.set_title("Silhouette %d" % n_clusters, fontsize=14, fontweight="bold")
    ax2.set_title("t-SNE 75", fontsize=14, fontweight="bold")

    # The 1st subplot is the silhouette plot
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])
    silhouette_score_cached = mem.cache(silhouette_score)
    silhouette_avg = silhouette_score_cached(data, labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    silhouette_samples_cached = mem.cache(silhouette_samples)
    sample_silhouette_values = silhouette_samples_cached(data, labels)
    y_lower = 10
    for idx in range(n_clusters):
        ith_cluster_silhouette_values = (
            sample_silhouette_values[labels == idx])
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(idx) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(idx))
        y_lower = y_upper + 10 
    # ax1.set_xlabel("coefficient values")
    # ax1.set_ylabel("label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-1, 0, 1])

    # The 2nd subplot shows the actual clusters formed
    colors = cm.nipy_spectral(labels.astype(float) / n_clusters)
    ax2.scatter(data_plane[:, 0], data_plane[:, 1], marker=".", s=30, lw=0,
                alpha=0.7, c=colors, edgecolor="k")
    # ax2.set_xlabel("Feature space for the 1st feature")
    # ax2.set_ylabel("Feature space for the 2nd feature")

    return fig


def _tsne(data, *args, **kwargs):
    """ See 'sklearn.manifold.TSNE' for the documentation.
    """
    tsne = manifold.TSNE(*args, **kwargs)
    return tsne.fit_transform(data)


def plot_chord_diagram(con, rois_filename, title, outdir=None):
    """
    Visualize connectivity as a chord diagram (circular graph).

    Parameters
    ----------
    con : array of float (n_states, n_rois, n_rois)
        Connectivity score. Can be a square matrix or 
        an array of square matrices.
    rois_filename : str
        Path of the file with the node names.
    title : str
        The figure title.
    outdir : str
        Path of the directory to save the figure. Default :None.

    Raises
    ------
    AssertionError
        If more than 10 connectivity matrices to plot, return error.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure handle.

    """
    
    #Load rois names
    rois_df = pd.read_csv(rois_filename, sep="\t")
    rois = (rois_df["name"] + "-" + rois_df["hemi"].str[0]).tolist()
    # rois = rois_df["complete_label"]
    
    # reorder labels to get them in miroir
    hemi_size = len(rois) // 2
    node_order = rois[:hemi_size]
    node_order.extend(rois[hemi_size:][::-1])
    node_angles = circular_layout(rois, node_order, start_pos=90,
                                  group_boundaries=[0, hemi_size])
    
    if len(con.shape) == 2 :
        # select only the 5% strongest connections
        thr = np.percentile(np.abs(con), 98)
        n_lines = np.sum(con.flatten() >= thr)
        print("number of connections :", n_lines)
        fig, _ = plot_connectivity_circle(
            con, rois, 
            n_lines=n_lines,
            node_angles=node_angles,
            colorbar_size=0.3, padding=2, fontsize_title=14,
            title=title, show=False
            )
    if len(con.shape)>2 and len(con)<=10:
        # no_names = [''] * len(rois)
        for ii, state in enumerate(con):
            # select only the 5% strongest connections
            thr = np.percentile(np.abs(state), 98)
            n_lines = np.sum(state.flatten() >= thr)        
            print("number of connections :", n_lines)
            fig, _ = plot_connectivity_circle(state, rois, n_lines=n_lines,
                                     node_angles=node_angles,
                                     title=title.format(ii), fontsize_title=14,
                                     padding=2,
                                     colorbar_size=0.3, fontsize_colorbar=6,
                                     show=False
                                     )
            if outdir is not None:
                fig.savefig(os.path.join(outdir, 'state_{0}'.format(ii+1)))
        
    if len(con.shape)>2 and len(con)>10:
        raise AssertionError("Too many connectivity matrices")
        
    return fig

#########
# for chord diagram interactive plot (for one matrice only)
# cf. with holoviews in jupyter-notebook 'connectivity_chord'
########
