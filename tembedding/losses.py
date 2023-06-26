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
Module that implements common losses.
"""

# Import
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def dot_similarity(ref: torch.Tensor, pos: torch.Tensor,
                   neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Cosine similarity the ref, pos and negative pairs.

    Parameters
    ----------
    ref: Tensor
        The reference samples of shape `(n, d)`.
    pos: Tensor
        The positive samples of shape `(n, d)`.
    neg: Tensor
        The negative samples of shape `(n, d)`.

    Returns
    -------
    sim_refpos: Tensor
        The similarity between reference samples and positive samples of
        shape `(n,)`
    sim_refneg: Tensor
        The similarities between reference samples and negative samples of
        shape `(n, n)`.
    """
    pos_dist = torch.einsum("ni,ni->n", ref, pos)
    neg_dist = torch.einsum("ni,mi->nm", ref, neg)
    return pos_dist, neg_dist


def euclidean_similarity(
        ref: torch.Tensor, pos: torch.Tensor,
        neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Negative L2 distance between the ref, pos and negative pairs.

    Parameters
    ----------
    ref: Tensor
        The reference samples of shape `(n, d)`.
    pos: Tensor
        The positive samples of shape `(n, d)`.
    neg: Tensor
        The negative samples of shape `(n, d)`.

    Returns
    -------
    sim_refpos: Tensor
        The similarity between reference samples and positive samples of
        shape `(n,)`
    sim_refneg: Tensor
        The similarities between reference samples and negative samples of
        shape `(n, n)`.
    """
    ref_sq = torch.einsum("ni->n", ref**2)
    pos_sq = torch.einsum("ni->n", pos**2)
    neg_sq = torch.einsum("ni->n", neg**2)

    pos_cosine, neg_cosine = dot_similarity(ref, pos, neg)
    pos_dist = -(ref_sq + pos_sq - 2 * pos_cosine)
    neg_dist = -(ref_sq[:, None] + neg_sq[None] - 2 * neg_cosine)

    return pos_dist, neg_dist


def infonce(pos_dist: torch.Tensor, neg_dist: torch.Tensor, beta) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
    """ InfoNCE implementation.

    See :py:class:`BaseInfoNCE` for reference.
    """
    align = (-pos_dist).mean()
    uniform = torch.logsumexp(neg_dist, dim=1).mean()
    return align + beta*uniform, align, uniform


class CosineInfoNCE(object):
    """ CosineInfoNCE base loss with a fixed temperature.

    Attributes
    ----------
    temperature: float
        The softmax temperature.
    """
    def __init__(self, temperature: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.beta = beta

    def __call__(self, ref: torch.Tensor, pos: torch.Tensor,
                 neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_dist, neg_dist = dot_similarity(ref, pos, neg)
        return infonce(pos_dist / self.temperature,
                       neg_dist / self.temperature, self.beta)


class EuclideanInfoNCE(object):
    """ EuclideanInfoNCE base loss with a fixed temperature.

    Attributes
    ----------
    temperature: float
        The softmax temperature.
    """
    def __init__(self, temperature: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.beta = beta

    def __call__(self, ref: torch.Tensor, pos: torch.Tensor,
                  neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_dist, neg_dist = euclidean_similarity(ref, pos, neg)
        return infonce(pos_dist / self.temperature,
                       neg_dist / self.temperature, self.beta)
