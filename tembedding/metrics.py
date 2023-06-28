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
Module to compute distance between dFCs
"""

# Imports
import math
import numpy as np
import torch
import torch.nn.functional as F


def batch_inv_structure_ssim(dfc, seq_dfcs):
    """ Compute the structure-SSIM between a dFC and a sequence of dFCs.

    Parameters
    ----------
    dfc: array (N, N)
        reference dFC.
    seq_dfcs: array (M, N, N)
        a sequence of M dFCs.

    Returns
    -------
    ssims: array (M, )
        the computed structure-SSIM.
    """
    return 1 - batch_structure_ssim(dfc, seq_dfcs)


def structure_ssim(img1, img2, window_size=11, window=None):
    """ Compute the structure-SSIM between two dFCs.

    Parameters
    ----------
    img1: array (N, N)
        a dFC.
    img2: array (N, N)
        another dFC.
    window_size: int, defaul 11
        the side-length of the sliding window used in comparison. Must be
        an odd value.
    window: , default None


    Returns
    -------
    ssim: float
        the computed structure-SSIM.
    """
    pad = int(window_size // 2)
    if len(img1.shape) == 2:
        height, width = img1.shape
        img1 = torch.unsqueeze(img1, dim=0)
        img2 = torch.unsqueeze(img2, dim=0)
        img1 = torch.unsqueeze(img1, dim=0)
        img2 = torch.unsqueeze(img2, dim=0)
    elif len(img1.shape) == 3:
        _, height, width = img1.shape
        img1 = torch.unsqueeze(img1, dim=0)
    else :
        batch, _, height, width = img1.shape
    img1 -= torch.mean(img1)
    img2 -= torch.mean(img2)

    # if window is not provided, init one: window should be atleast 11x11 
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=1)
    
    # calculating the mu parameter (locally) for both images using a
    # gaussian filter calculates the luminosity params
    mu1 = F.conv2d(input=img1, weight = window, padding=pad)
    mu2 = F.conv2d(input=img2, weight = window, padding=pad)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad) - mu2_sq
    sigma12 =  F.conv2d(img1 * img2, window, padding=pad) - mu12

    # Some constants for stability  
    C3 = (0.03 ) ** 2 / 2

    numerator =  sigma12 + C3
    denominator = np.sqrt(sigma1_sq*sigma2_sq) + C3 

    struct_score = numerator / denominator
    ret = struct_score.mean()
    
    return ret

def batch_structure_ssim(img1, img2, window_size=11, window=None):
    """ Compute the structure-SSIM between a dFC and a sequence of dFCs.

    Parameters
    ----------
    img1: array (N, N)
        reference dFC.
    img2: array (M, N, N)
        a sequence of M dFCs.
    window_size: int, defaul 11
        the side-length of the sliding window used in comparison. Must be
        an odd value.
    window: , default None

    Returns
    -------
    ssims: array (M, )
        the computed structure-SSIM.

    Notes
    -----
    Requires that img1 and img2 are normalized.
    """
    pad = int(window_size // 2)
    if len(img1.shape) == 2 :
        height, width = img1.shape
        img1 = torch.unsqueeze(img1,dim=0)
        img2 = torch.unsqueeze(img2,dim=1)
        img1 = torch.unsqueeze(img1,dim=0)
    elif len(img1.shape) == 3 :
        _, height, width = img1.shape
        img1 = torch.unsqueeze(img1,dim=0)
    else :
        batch, _, height, width = img1.shape

    # if window is not provided, init one: window should be atleast 11x11
    if window is None: 
        real_size = min(window_size, height, width) 
        window = create_window(real_size, channel=1)
    
    # calculating the mu parameter (locally) for both images using
    # a gaussian filter calculates the luminosity params
    mu1 = F.conv2d(input=img1, weight = window, padding=pad)
    mu2 = F.conv2d(input=img2, weight = window, padding=pad)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad) - mu2_sq
    sigma12 =  F.conv2d(img1 * img2, window, padding=pad) - mu12

    # Some constants for stability  
    C3 = (0.03 ) ** 2 / 2

    numerator =  sigma12 + C3
    denominator = np.sqrt(sigma1_sq*sigma2_sq) + C3 
    struct_score = numerator / denominator
    ret = struct_score.mean(1).mean(1).mean(1)
    
    return ret


def create_window(window_size, structure="gaussian", channel=1):
    """
    """
    if structure == "gaussian":
        # Generate an 1D tensor containing values sampled from a gaussian
        # distribution
        _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
        # Converting to 2D  
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.Tensor(_2d_window.expand(
            channel, 1, window_size, window_size).contiguous()).double()

    elif structure == "mean":
        window = (torch.ones(size=(1, 1, window_size, window_size)).double() *
                  1 / window_size**2)

    return window.type(torch.float32)


def gaussian(window_size, sigma):
    """ Generates a list of Tensor values drawn from a gaussian distribution
    with standard diviation sigma and sum of all elements to one.
    """    
    gauss =  torch.Tensor([
        math.exp(-(x - window_size//2)**2 / float(2 * sigma**2))
        for x in range(window_size)])
    return gauss / gauss.sum()
