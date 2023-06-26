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
Module that implements CEBRA dataset and loaders.
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
import tembedding.datasets.distributions as distributions


class Batch(object):
    """ A batch of reference, positive, negative samples and an optional index.

    Attributes
    ----------
    reference:
        The reference samples, typically sampled from the prior
        distribution.
    positive:
        The positive samples, typically sampled from the positive
        conditional distribution depending on the reference samples.
    negative:
        The negative samples, typically sampled from the negative
        conditional distribution depending (but often indepent) from
        the reference samples.
    index:
        See docs for multisession training distributions.
    index_reversed:
        See docs for multisession training distributions.
    """
    __slots__ = ["reference", "positive", "negative", "index",
                 "index_reversed"]

    def __init__(self, reference, positive, negative):
        self.reference = reference
        self.positive = positive
        self.negative = negative

    def to(self, device):
        """ Move all batch elements to the GPU.
        """
        self.reference = self.reference.to(device)
        self.positive = self.positive.to(device)
        self.negative = self.negative.to(device)


class BatchIndex(object):
    def __init__(self, positive, negative, reference, session_ref=None,
                 session_pos=None, session_neg=None):     
        self.reference = reference
        self.positive = positive
        self.negative = negative
        if session_ref is not None:
            self.session_ref = session_ref
        if session_pos is not None:
            self.session_pos = session_pos
        if session_neg is not None:
            self.session_neg = session_neg


class TensorDataset(torch.utils.data.Dataset):
    """ Discrete and/or continuously indexed dataset based on torch/numpy
    arrays.
    If dealing with datasets sufficiently small to fit :py:func:`numpy.array`
    or :py:class:`torch.Tensor`, this dataset is sufficient---the sampling
    auxiliary variable should be specified with a dataloader. Based on whether
    `continuous` and/or `discrete` auxiliary variables are provided, this class
    can be used with the discrete, continuous and/or mixed data loader classes.

    Parameters
    ----------
    neural:
        Array of dtype ``float`` or float Tensor of shape ``(N, D)``,
        containing neural activity over time.
    continuous:
        Array of dtype ```float`` or float Tensor of shape ``(N, d)``,
        containing the continuous behavior variables over the same time
        dimension.
    discrete:
        Array of dtype ```int64`` or integer Tensor of shape ``(N, d)``,
        containing the discrete behavior variables over the same time
        dimension.
    """
    def __init__(self, neural, continuous=None, discrete=None,
                offset: int = 1):
        super().__init__()
        self.neural = self._to_tensor(neural, torch.FloatTensor).float()
        if discrete is not None:
            self.discrete = self._to_tensor(discrete, torch.LongTensor)
        else :
            self.discrete = None
        if continuous is not None : 
            self.continuous = self._to_tensor(continuous, torch.FloatTensor)
        else : 
            self.continuous = None
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
        """ Return the data at the specified index location.
        """
        return Batch(
            positive=self.neural[index.positive],
            negative=self.neural[index.negative],
            reference=self.neural[index.reference])

    def __getitem__(self, index):
        batch = self.load_batch(index)
        return batch


class Loader(torch.utils.data.DataLoader): 
    """ Dataloader class.
    Reference and negative samples will be drawn from a uniform prior
    distribution. Depending on the ``prior`` attribute, the prior
    will uniform over time-steps (setting ``empirical``), or be adjusted
    such that each discrete value in the dataset is uniformly distributed
    (setting ``uniform``).
    Positive samples are sampled according to the specified distribution.

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

    def __init__(self,data,num_steps = 100,batch_size = 128,time_delta = 1,matrix_delta = 20, distance = None):
        super(Loader,self).__init__(dataset = data,batch_size = batch_size)
        self.num_steps = num_steps
        if distance is not None :
            self.distribution = distributions.Distribution_MatrixDistance(len(data), data, distance, 0, matrix_delta)
        elif data.discrete is not None :
            self.distribution = distributions.Distribution_Discrete(len(data), data, time_delta, time_delta)
        else:
            self.distribution = distributions.Distribution(len(data), data, time_delta, time_delta)
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


class SimpleMultiSessionDataset(torch.utils.data.Dataset):
    """A dataset spanning multiple recording sessions.

    Simple Multi session datasets share the same dimensionality across the index 
    as well as between different sessions.

    Multi-session datasets where the number of neurons is constant across sessions
    should utilize the normal ``Dataset`` class with a ``MultisessionLoader`` for
    better efficiency when sampling.

    Attributes:
        offset: The offset determines the shape of the data obtained with the
            ``__getitem__`` and :py:meth:`.base.Dataset.expand_index` methods.
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

        if discrete is not None:
            self.discrete = self._to_tensor(discrete, torch.LongTensor)
        else :
            self.discrete = None

        if continuous is not None : 
            self.continuous = self._to_tensor(continuous, torch.FloatTensor)
        else : 
            self.continuous = None

        self.offset = offset
        self.num_sessions = self.set_num_sessions()

    def _to_tensor(self, array, check_dtype=None):
        if array is None:
            return None
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)
        return array

    def set_num_sessions(self) -> int:
        """The number of sessions in the dataset."""
        return self.neural.shape[0]

    def load_batch(self, index: BatchIndex) -> List[Batch]:
        """Return the data at the specified index location."""
        return Batch(
                reference= torch.stack([self.neural[index[0],index[1]] for index in index.reference]),
                positive= torch.stack([self.neural[index[0],index[1]] for index in index.positive]) ,
                negative= torch.stack([self.neural[index[0],index[1]] for index in index.negative])
            )

    def __getitem__(self, index):
        batch = self.load_batch(index)
        return batch
    
    def get_duration(self):
        return self.neural.shape[1]


class MultiSessionLoader(torch.utils.data.DataLoader):
    """Dataloader for multi-session datasets.

    The loader will enforce a uniform distribution across the sessions.
    Note that if samples within different sessions share the same feature
    dimension, it is better to use a :py:class:`cebra.data.single_session.MixedDataLoader`.
    """

    def __init__(self,data,num_steps = 100,batch_size = 128,time_delta = 1,matrix_delta = 20, distance = None): 
        super(MultiSessionLoader,self).__init__(dataset = data,batch_size = batch_size)
        self.num_steps = num_steps
        if distance is not None and time_delta > 0:
            self.distribution = distributions.MultiSessionDistribution_TimeAndDistanceMatrix(data.get_duration(), data, distance, time_delta, time_delta, matrix_delta)
        elif distance is not None and time_delta == 0:
            self.distribution = distributions.MultiSessionDistribution_DistanceMatrix(data.get_duration(), data, distance, matrix_delta)
        elif data.discrete is not None and time_delta > 0:
            self.distribution = distributions.MultiSessionDistribution_TimeAndDiscrete(data.get_duration(), data, time_delta, time_delta, data.discrete)
        elif data.discrete is not None and time_delta == 0:
            self.distribution = distributions.MultiSessionDistribution_Discrete(data.get_duration(), data, data.discrete)
        else:
            self.distribution = distributions.MultiSessionDistribution_Time(data.get_duration(), data, time_delta, time_delta)
        #    discrete=self.dindex,
        #    continuous=self.cindex,
        #    time_delta=self.time_offset)

    def get_indices(self, num_samples: int) -> List[BatchIndex]:
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

        ref_idx = self.distribution.sample_prior(self.batch_size)
        neg_idx = self.distribution.sample_negative(self.batch_size)
        pos_idx = self.distribution.sample_conditional(ref_idx)

        return BatchIndex(
            reference=ref_idx,
            positive=pos_idx,
            negative=neg_idx
        )

    def __len__(self):
        """The number of batches returned when calling as an iterator."""
        return self.num_steps

    def __iter__(self) -> Batch:
        for i in range(self.num_steps):
            index = self.get_indices(num_samples=self.batch_size)
            yield i,self.dataset.load_batch(index)


class MultiSessionDataset2():
    """A dataset spanning multiple recording sessions.

    Multi session datasets share the same dimensionality across the index,
    but can have differing feature dimensions (e.g. number of neurons) between
    different sessions.

    Multi-session datasets where the number of neurons is constant across sessions
    should utilize the normal ``Dataset`` class with a ``MultisessionLoader`` for
    better efficiency when sampling.

    Attributes:
        offset: The offset determines the shape of the data obtained with the
            ``__getitem__`` and :py:meth:`.base.Dataset.expand_index` methods.
    """

    def __init__(
        self,
        *datasets: TensorDataset,
    ):
        super().__init__()
        self._datasets: List[
            TensorDataset] = self._unpack_dataset_arguments(
                datasets)

    @property
    def num_sessions(self) -> int:
        """The number of sessions in the dataset."""
        return len(self._datasets)

    @property
    def input_dimension(self):
        return super().input_dimension

    def get_input_dimension(self, session_id: int) -> int:
        """Get the feature dimension of the required session.

        Args:
            session_id: The session ID, an integer between 0 and
                :py:attr:`num_sessions`.

        Returns:
            A single session input dimension for the requested session id.
        """
        return self.get_session(session_id).input_dimension

    def get_session(self, session_id: int) -> TensorDataset:
        """Get the dataset for the specified session.

        Args:
            session_id: The session ID, an integer between 0 and
                :py:attr:`num_sessions`.

        Returns:
            A single session dataset for the requested session
            id.
        """
        return self._datasets[session_id]

    def _apply(self, func):
        return (func(data) for data in self.iter_sessions())

    @property
    def session_lengths(self) -> List[int]:
        return [len(session) for session in self.iter_sessions()]

    def iter_sessions(self):
        for i in range(self.num_sessions):
            yield self.get_session(i)

    def __getitem__(self, args) -> List[Batch]:
        """Return a set of samples from all sessions."""
        session_id, index = args
        return self.get_session(session_id).__getitem__(index)

    def load_batch(self, index: BatchIndex) -> List[Batch]:
        """Return the data at the specified index location."""
        return Batch(
                reference= torch.cat([self.get_session(index.session_ref[i])[index.reference[i]] for i in range(len(index.reference))]),
                positive= torch.cat([self.get_session(index.session_pos[i])[index.positive[i]] for i in range(len(index.positive))]) ,
                negative= torch.cat([self.get_session(index.session_neg[i])[index.negative[i]] for i in range(len(index.negative))])
            )
        

class MultiSessionLoader2(torch.utils.data.DataLoader):
    """Dataloader for multi-session datasets.

    The loader will enforce a uniform distribution across the sessions.
    Note that if samples within different sessions share the same feature
    dimension, it is better to use a :py:class:`cebra.data.single_session.MixedDataLoader`.
    """

    def __init__(self,data,num_steps,batch_size):
        super(Loader,self).__init__(dataset = data,batch_size = batch_size)
        self.num_steps = num_steps
        self.distribution = distributions.MultiSessionDistribution(len(data), data, 15)

    def get_indices(self, num_samples: int) -> List[BatchIndex]:
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

        ref_idx = self.distribution.sample_prior(self.batch_size)
        neg_idx = self.distribution.sample_prior(self.batch_size)
        pos_idx, idx = self.distribution.sample_conditional(ref_idx)

        ref_idx = torch.from_numpy(ref_idx[:,0])
        ref_session = torch.from_numpy(ref_idx[:,1])
        neg_idx = torch.from_numpy(neg_idx[:,0])
        neg_session = torch.from_numpy(neg_idx[:,1])
        pos_idx = torch.from_numpy(pos_idx[:,0])
        pos_session = torch.from_numpy(pos_idx[:,1])

        return BatchIndex(
            reference=ref_idx,
            positive=pos_idx,
            negative=neg_idx,
            ref_session = ref_session,
            pos_session = pos_session,
            neg_session = neg_session
        )

    def __len__(self):
        """The number of batches returned when calling as an iterator."""
        return self.num_steps

    def __iter__(self) -> Batch:
        for i in range(self.num_steps):
            index = self.get_indices(num_samples=self.batch_size)
            yield i,self.dataset.load_batch(index)
