'''Création de la distribution'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable

class Distribution:

    def __init__(self, T, data, offset, time_delta: int = 1): #discrete, continuous,
        self.T = T
        self.data = data
        #self.discrete = discrete
        #self.continuous = continuous 
        self.time_delta = time_delta #biggest time difference for positives
        self.offset = offset

    def sample_prior(self, num_samples: int) -> torch.Tensor:
        """Return indices from a uniform prior distribution.
        Prior distributions return a batch of indices. Indexing
        the dataset with these indices will return samples from
        the prior distribution.

        Args:
            num_samples: The number of samples

        Returns:
            The reference indices of shape ``(num_samples, )``.
        """
        return torch.randint(self.T - self.offset,size = (num_samples,))

    def sample_conditional(self, reference_idx: torch.Tensor, num_samples) -> torch.Tensor:
        """Return indices from the conditional distribution knowing a batch of reference indices
        Conditional distributions return a bTypeError: torch._VariableFunctionsClass.from_numpy() takes no keyword argumentsatch of indices. Indexing
        the dataset with these indices will return samples from
        the prior distribution.

        Args:
            reference_idx: The reference indices.

        Returns:
            The positive indices. The positive samples will match the reference
            samples in their discrete variable, and will otherwise be drawn from
            the :py:class:`.TimedeltaDistribution`.
        """

        if reference_idx.dim() != 1:
            raise ValueError(
                f"Reference indices have wrong shape: {reference_idx.shape}. "
                "Pass a 1D array of indices of reference samples.")
        
        diff_idx = torch.randint(self.time_delta, (num_samples,))
        
        return reference_idx + diff_idx