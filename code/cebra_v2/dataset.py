'''Définition du dataset du loader'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
import cebra_v2.distribution

class Batch:
    """A batch of reference, positive, negative samples and an optional index.

    Attributes:
        reference: The reference samples, typically sampled from the prior
            distribution
        positive: The positive samples, typically sampled from the positive
            conditional distribution depending on the reference samples
        negative: The negative samples, typically sampled from the negative
            conditional distribution depending (but often indepent) from
            the reference samples
        index: TODO(stes), see docs for multisession training distributions
        index_reversed: TODO(stes), see docs for multisession training distributions
    """

    __slots__ = ["reference", "positive", "negative", "index", "index_reversed"]

    def __init__(self,
                 reference,
                 positive,
                 negative,
                 index=None,
                 index_reversed=None):
        self.reference = reference
        self.positive = positive
        self.negative = negative

    def to(self, device):
        """Move all batch elements to the GPU."""
        self.reference = self.reference.to(device)
        self.positive = self.positive.to(device)
        self.negative = self.negative.to(device)

class BatchIndex:
    def __init__(self,positive,negative,reference):
        self.positive = positive
        self.negative = negative
        self.reference = reference

class TensorDataset(torch.utils.data.Dataset):
    """Discrete and/or continuously indexed dataset based on torch/numpy arrays.

    If dealing with datasets sufficiently small to fit :py:func:`numpy.array` or :py:class:`torch.Tensor`, this
    dataset is sufficient---the sampling auxiliary variable should be specified with a dataloader.
    Based on whether `continuous` and/or `discrete` auxiliary variables are provided, this class
    can be used with the discrete, continuous and/or mixed data loader classes.

    Args:
        neural:
            Array of dtype ``float`` or float Tensor of shape ``(N, D)``, containing neural activity over time.
        continuous:
            Array of dtype ```float`` or float Tensor of shape ``(N, d)``, containing the continuous behavior
            variables over the same time dimension.
        discrete:
            Array of dtype ```int64`` or integer Tensor of shape ``(N, d)``, containing the discrete behavior
            variables over the same time dimension.

    Example:

        >>> import cebra.data
        >>> import torch
        >>> data = torch.randn((100, 30))
        >>> index1 = torch.randn((100, 2))
        >>> index2 = torch.randint(0,5,(100, ))
        >>> dataset = cebra.d    print(dataset[torch.arange(len(dataset))])ata.datasets.TensorDataset(data, continuous=index1, discrete=index2)

    """

    def __init__(
        self,
        neural,
        continuous = None,
        discrete = None,
        offset: int = 1,
    ):
        super().__init__()
        self.neural = self._to_tensor(neural, torch.FloatTensor).float()
        if not(continuous is None and discrete is None):
            self.continuous = self._to_tensor(continuous, torch.FloatTensor)
            self.discrete = self._to_tensor(discrete, torch.LongTensor)
        self.offset = offset

    def _to_tensor(self, array, check_dtype=None):
        if array is None:
            return None
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)
        return array

    def input_dimension(self) -> int:
        return self.neural.shape[1]

    def __len__(self):
        return len(self.neural)

    def load_batch(self, index: BatchIndex) -> Batch:
        """Return the data at the specified index location."""
        return Batch(
            positive=self.neural[index.positive],
            negative=self.neural[index.negative],
            reference=self.neural[index.reference],
        )

    def __getitem__(self, index):
        batch = self.load_batch(index)
        return batch

class Loader(torch.utils.data.DataLoader): 
    """Dataloader class.

    Reference and negative samples will be drawn from a uniform prior
    distribution. Depending on the ``prior`` attribute, the prior
    will uniform over time-steps (setting ``empirical``), or be adjusted
    such that each discrete value in the dataset is uniformly distributed
    (setting ``uniform``).

    Positive samples are sampled according to the specified distribution 

    Args:
        See dataclass fields.

    Yields:
        Batches of the specified size from the given dataset object.

    Note:
        The ``__iter__`` method is non-deterministic, unless explicit seeding is implemented
        in derived classes. It is recommended to avoid global seeding in numpy
        and torch, and instead locally instantiate a ``Generator`` object for
        drawing samples.
    """

    def __init__(self,data,num_steps,batch_size,time_delta):
        super(Loader,self).__init__(dataset = data,batch_size = batch_size)
        self.num_steps = num_steps
        self.distribution = cebra_v2.distribution.Distribution(len(data), data, time_delta, time_delta)
        #    discrete=self.dindex,
        #    continuous=self.cindex,
        #    time_delta=self.time_offset)

    def get_indices(self, num_samples: int) -> BatchIndex:
        """Samples indices for reference, positive and negative examples.

        The reference and negative samples will be sampled uniformly from
        all available time steps.

        The positive samples will be sampled conditional on the reference
        samples according to the specified ``conditional`` distribution.

        Args:
            num_samples: The number of samples (batch size) of the returned
                :py:class:`cebra.data.datatypes.BatchIndex`.

        Returns:
            Indices for reference, positive and negatives samples.
        """
        reference_idx = self.distribution.sample_prior(num_samples * 2)
        negative_idx = reference_idx[num_samples:]
        reference_idx = reference_idx[:num_samples]

        positive_idx = self.distribution.sample_conditional(reference_idx,num_samples)

        return BatchIndex(reference=reference_idx,
                          positive=positive_idx,
                          negative=negative_idx)

    def __len__(self):
        """The number of batches returned when calling as an iterator."""
        return self.num_steps

    def __iter__(self) -> Batch:
        for i in range(self.num_steps):
            index = self.get_indices(num_samples=self.batch_size)
            yield i,self.dataset.load_batch(index)