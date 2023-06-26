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
Module that implements the CEBRA base model.
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


class Squeeze(nn.Module):
    """ Squeeze 3rd dimension of input tensor, pass through otherwise.
    """
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """ Squeeze 3rd dimension of input tensor, pass through otherwise.

        Parameters
        ----------
        inp: Tensor
            a 3D input tensor.

        Returns
        -------
        sq_inp: Tensor
            If the third dimension of the input tensor can be squeezed,
            return the resulting 2D output tensor. If input is 2D or less,
            return the input.
        """
        if inp.dim() > 2:
            return inp.squeeze(2)
        return inp
 

class _Norm(nn.Module):
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return inp / torch.norm(inp, dim=1, keepdim=True)


class Model(nn.Module):
    """ The CEBRA base model.
    """
    def __init__(self, *layers, num_input=None, num_output=None,
                 normalize=True):
        super(Model,self).__init__()
        if num_input < 1:
            raise ValueError("Input dimension needs to be at least 1, but "
                             f"got {num_input}.")
        if num_output < 1:
            raise ValueError("Output dimension needs to be at least 1, but "
                             f"got {num_output}.")
        self.num_input: int = num_input
        self.num_output: int = num_output
        if normalize:
            layers += (_Norm(), )
        layers += (Squeeze(), )
        self.net = nn.Sequential(*layers)
        self.normalize = normalize
    
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """ Compute the embedding given the input signal.

        Based on the parameters used for initializing, the output embedding
        is normalized to the hypersphere (`normalize = True`).

        Parameters
        ----------
        inp: Tensor
            The input tensor of shape `num_samples x self.num_input x time`

        Returns
        -------
        out: Tensor
            The output tensor of shape `num_samples x self.num_output x
            (time - receptive field)`.
        """
        return self.net(inp)
