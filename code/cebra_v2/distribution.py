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
    """Distribution across a single session which moves closer point close in time"""

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

class Distribution_Discrete:
    """Distribution across a single session which moves closer point close in time and points with same label"""

    def __init__(self, T, data, offset, time_delta: int = 1): #discrete, continuous,
        self.T = T
        self.data = data
        self.discrete = data.discrete
        self.dict_class = self.class_index()
        self.time_delta = time_delta #biggest time difference for positives
        self.offset = offset

    def class_index(self):
        labels = np.unique(self.discrete)
        dict = {}
        for label in labels:
            dict[label] = torch.argwhere(self.discrete == label).reshape((-1))
        return dict

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
    
    def sample_discrete(self,reference_idx):
        labels = self.discrete[reference_idx]
        res = torch.zeros(len(reference_idx),dtype = torch.int)
        for i in range(len(labels)) : 
            id = np.random.randint(len(self.dict_class[labels[i].item()]))
            res[i] = self.dict_class[labels[i].item()][id]
        return res

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
        
        diff_idx = torch.randint(self.time_delta, (num_samples*3//5,))
        class_idx = self.sample_discrete(reference_idx[num_samples*3//5:])
        
        return torch.cat([reference_idx[:num_samples*3//5] + diff_idx , class_idx])
    
class Distribution_MatrixDistance:
    """Distribution across a single session which moves closer points close in the matrix space
    according to the matrix_delta"""

    def __init__(self, T, data, distance, offset, matrix_delta): 
        self.T = T
        self.data = data
        self.distance = distance
        self.graph = self.generate_graph(matrix_delta)
        self.matrix_delta = matrix_delta #biggest matrix distance for positives
        self.offset = offset

    def generate_graph(self,matrix_delta):
        graph = {}
        for t in range(self.T):
            graph[t] = torch.argwhere(self.distance[t,:] < matrix_delta).reshape((-1))
        return graph

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
    
    def sample_graph(self,reference_idx):
        res = torch.zeros(len(reference_idx),dtype = torch.int)
        for i in range(len(reference_idx)) : 
            t = reference_idx[i]
            id = np.random.randint(len(self.graph[t.item()]))
            res[i] = self.graph[t.item()][id]
        return res

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
        
        positive_idx = self.sample_graph(reference_idx)
        
        return positive_idx


class MultiSessionDistribution :
    def f():
        return NotImplementedError

class MultiSessionDistribution_DistanceMatrix:
    """Implements a distribution across multiple sessions where we move closer points with similar labels"""

    def __init__(self, T, data, distance, offset, matrix_delta): 
        self.T = T
        self.data = data
        self.distance = distance
        self.graph = self.generate_graph(matrix_delta)
        self.matrix_delta = matrix_delta #biggest matrix distance for positives
        self.offset = offset

    def generate_graph(self,matrix_delta):
        graph = {}
        for session in range(self.data.num_sessions):
            for t in range(self.T):
                graph[(session,t)] = torch.argwhere(self.distance[session,t,:] < matrix_delta)
        return graph
        
    def sample_graph(self,reference_idx):
        pos_idx = torch.zeros(reference_idx.shape,dtype = torch.int)
        for i in range(len(reference_idx)) : 
            session,t = reference_idx[i].numpy()
            id = np.random.randint(len(self.graph[(session,t)]))
            pos_idx[i] = self.graph[(session,t)][id]
        return pos_idx

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
        idx = torch.randint(self.T - self.offset,size = (num_samples,1))
        session = torch.randint(self.data.num_sessions,size = (num_samples,1))
        return torch.cat([session,idx],dim=1)

    def sample_conditional(self, reference_idx: torch.Tensor) -> torch.Tensor:
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
        
        positive_idx = self.sample_graph(reference_idx)
        
        return positive_idx
    
class MultiSessionDistribution_TimeAndDistanceMatrix:
    """Implements a distribution across multiple sessions where we move closer points with similar labels"""

    def __init__(self, T, data, distance, offset, time_delta, matrix_delta): 
        self.T = T
        self.data = data
        self.distance = distance
        self.graph,self.list_nodes = self.generate_graph(matrix_delta)
        self.matrix_delta = matrix_delta #biggest matrix distance for positives
        self.offset = offset
        self.time_delta = time_delta

    def generate_graph(self,matrix_delta):
        graph = {}
        list_nodes = []
        for session in range(self.data.num_sessions):
            for t in range(self.T):
                accu = torch.argwhere(self.distance[session,t,:] < matrix_delta)
                indices_to_keep = torch.flatten(torch.nonzero(torch.where(accu[:,0] != session,1,0) + torch.where(torch.abs(accu[:,1]-t) > 200,1,0)))
                graph[(session,t)] = torch.index_select(accu,0,indices_to_keep)
                if len(graph[(session,t)]) > 0:
                    list_nodes.append([session,t])
        return graph,torch.tensor(list_nodes,dtype=int)
        
    def sample_graph(self,reference_idx):
        pos_idx = torch.zeros(reference_idx.shape,dtype = torch.int)
        for i in range(len(reference_idx)) : 
            session,t = reference_idx[i].numpy()
            id = np.random.randint(len(self.graph[(session,t)]))
            pos_idx[i] = self.graph[(session,t)][id]
        return pos_idx

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
        idx = torch.randint(self.T - self.offset,size = (num_samples//2,1))
        session = torch.randint(self.data.num_sessions,size = (num_samples//2,1))
        a = torch.cat([session,idx],dim=1)

        idy = torch.randint(len(self.list_nodes),size = (num_samples//2,))
        b = torch.index_select(self.list_nodes,0,idy)

        return torch.cat([a,b],dim=0)

    def sample_negative(self, num_samples: int) -> torch.Tensor:
        """Return indices from a uniform prior distribution.
        Prior distributions return a batch of indices. Indexing
        the dataset with these indices will return samples from
        the prior distribution.
        Args:
            num_samples: The number of samples
        Returns:
            The reference indices of shape ``(num_samples, )``.
        """
        idx = torch.randint(self.T - self.offset,size = (num_samples,1))
        session = torch.randint(self.data.num_sessions,size = (num_samples,1))
        return torch.cat([session,idx],dim=1)

    def sample_conditional(self, reference_idx: torch.Tensor) -> torch.Tensor:
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
        num_samples = len(reference_idx)
        diff_time = torch.randint(self.time_delta, (num_samples//2,))
        a = reference_idx[:num_samples//2]
        a[:,1] += diff_time

        b = self.sample_graph(reference_idx[num_samples//2:])
        
        return torch.cat([a,b])
    
class MultiSessionDistribution_Discrete:
    """Implements a distribution across multiple sessions where we move closer points with similar correlation matrices"""

    def __init__(self, T, data, offset, time_delta, distance_matrix): #discrete, continuous,
        self.T = T
        self.data = data
        #self.discrete = discrete
        #self.continuous = continuous 
        self.time_delta = time_delta #biggest time difference for positives
        self.offset = offset
        self.distance_matrix = distance_matrix


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
        idx = torch.randint(self.T - self.offset,size = (num_samples,1))
        session = torch.randint(self.data.num_sessions(),size = (num_samples,1))
        return torch.cat([session,idx],dim = 1)

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