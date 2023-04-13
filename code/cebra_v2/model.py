'''Création du modèle'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable

class Squeeze(nn.Module):
    """Squeeze 3rd dimension of input tensor, pass through otherwise."""

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Squeeze 3rd dimension of input tensor, pass through otherwise.

        Args:
            inp: 1-3D input tensor

        Returns:
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
    def __init__(self,
                 *layers,
                 num_input=None,
                 num_output=None,
                 normalize=True):
        super(Model,self).__init__()

        if num_input < 1:
            raise ValueError(
                f"Input dimension needs to be at least 1, but got {num_input}.")
        if num_output < 1:
            raise ValueError(
                f"Output dimension needs to be at least 1, but got {num_output}."
            )
        self.num_input: int = num_input
        self.num_output: int = num_output

        if normalize:
            layers += (_Norm(),)
        layers += (Squeeze(),)
        self.net = nn.Sequential(*layers)
        self.normalize = normalize
    
    def forward(self, inp):
        """Compute the embedding given the input signal.

        Args:
            inp: The input tensor of shape `num_samples x self.num_input x time`

        Returns:
            The output tensor of shape `num_samples x self.num_output x (time - receptive field)`.

        Based on the parameters used for initializing, the output embedding
        is normalized to the hypersphere (`normalize = True`).
        """
        return self.net(inp)