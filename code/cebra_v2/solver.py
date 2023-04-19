'''Implémentation du solver'''

import cebra_v2.dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable

class SingleSessionSolver():
    """Single session training with a symmetric encoder.
    This solver assumes that reference, positive and negative samples
    are processed by the same features encoder.
    """

    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.log = {
        "pos": [],
        "neg": [],
        "total": [],
        "temperature": []
        }
        self.history = []

    def _inference(self, batch: cebra_v2.dataset.Batch) -> cebra_v2.dataset.Batch:
        """Given a batch of input examples, computes the feature representation/embedding.
        Args:
            batch: The input data, not necessarily aligned across the batch
                dimension. This means that ``batch.index`` specifies the map
                between reference/positive samples, if not equal ``None``.
        Returns:
            Processed batch of data. While the input data might not be aligned
            across the sample dimensions, the output data should be aligned and
            ``batch.index`` should be set to ``None``.
        """
        ref = self.model(batch.reference)
        pos = self.model(batch.positive)
        neg = self.model(batch.negative)
        return cebra_v2.dataset.Batch(ref, pos, neg)
    
    def state_dict(self) -> dict:
        """Return a dictionary fully describing the current solver state.
        Returns:
            State dictionary, including the state dictionary of the models and
            optimizer. Also contains the training history and the CEBRA version
            the model was trained with.
        """

        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss": torch.tensor(self.history),
            "decode": self.decode_history,
            "criterion": self.criterion.state_dict(),
            "log": self.log,
        }
    
    def step(self, batch: cebra_v2.dataset.Batch) -> dict:
        """Perform a single gradient update.
        Args:
            batch: The input samples
        Returns:
            Dictionary containing training metrics.
        """
        self.optimizer.zero_grad()
        prediction = self._inference(batch)
        loss, align, uniform = self.criterion(prediction.reference,
                                              prediction.positive,
                                              prediction.negative)

        loss.backward()
        self.optimizer.step()
        self.history.append(loss.item())
        stats = dict(
            pos=align.item(),
            neg=uniform.item(),
            total=loss.item(),
            temperature=self.criterion.temperature,
        )
        for key, value in stats.items():
            self.log[key].append(value)
                
        return stats
    

    def fit(
        self,
        loader: cebra_v2.dataset.Loader,
        valid_loader: cebra_v2.dataset.Loader = None,
        save_frequency: int = None,
        valid_frequency: int = None,
        decode: bool = False,
        logdir: str = None,
    ):
        """Train model for the specified number of steps.
        Args:
            loader: Data loader, which is an iterator over `cebra.data.Batch` instances.
                Each batch contains reference, positive and negative input samples.
            valid_loader: Data loader used for validation of the model.
            save_frequency: If not `None`, the frequency for automatically saving model checkpoints
                to `logdir`.
            valid_frequency: The frequency for running validation on the ``valid_loader`` instance.
            logdir:  The logging directory for writing model checkpoints. The checkpoints
                can be read again using the `solver.load` function, or manually via loading the
                state dict.
        TODO:
            * Refine the API here. Drop the validation entirely, and implement this via a hook?
        """

        iterator = loader
        self.model.train()
        for id_step, batch in iterator:
            stats = self.step(batch)

            if id_step % 100 == 0 : 
                print('Epoch {}'.format(id_step))
                print('Train loss {:.4f}, Train accuracy {:.2f}%'.format(stats["total"],0))

class MultiSessionSolver():
    """Multi session training, contrasting pairs of neural data."""

    _variant_name = "multi-session"

    def _mix(self, array: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        shape = array.shape
        n, m = shape[:2]
        mixed = array.reshape(n * m, -1)[idx]
        return mixed.reshape(shape)

    def _single_model_inference(self, batch: cebra_v2.dataset.Batch,
                                model: torch.nn.Module) -> cebra_v2.dataset.Batch:
        """Given a single batch of input examples, computes the feature representation/embedding.

        Args:
            batch: The input data, not necessarily aligned across the batch
                dimension. This means that ``batch.index`` specifies the map
                between reference/positive samples, if not equal ``None``.
            model: The model to use for inference.

        Returns:
            Processed batch of data. While the input data might not be aligned
            across the sample dimensions, the output data should be aligned and
            ``batch.index`` should be set to ``None``.
        """
        batch.to(self.device)
        ref = torch.stack([model(batch.reference)], dim=0)
        pos = torch.stack([model(batch.positive)], dim=0)
        neg = torch.stack([model(batch.negative)], dim=0)

        pos = self._mix(pos, batch.index_reversed)

        num_features = neg.shape[2]

        return cebra_v2.dataset.Batch(
            reference=ref.view(-1, num_features),
            positive=pos.view(-1, num_features),
            negative=neg.view(-1, num_features),
        )

    def _inference(self, batches: List[cebra_v2.dataset.Batch]) -> cebra_v2.dataset.Batch:
        """Given batches of input examples, computes the feature representations/embeddings.

        Args:
            batches: A list of input data, not necessarily aligned across the batch
                dimension. This means that ``batch.index`` specifies the map
                between reference/positive samples, if not equal ``None``.

        Returns:
            Processed batch of data. While the input data might not be aligned
            across the sample dimensions, the output data should be aligned and
            ``batch.index`` should be set to ``None``.

        """
        refs = []
        poss = []
        negs = []

        for batch, model in zip(batches, self.model):
            batch.to(self.device)
            refs.append(model(batch.reference))
            poss.append(model(batch.positive))
            negs.append(model(batch.negative))
        ref = torch.stack(refs, dim=0)
        pos = torch.stack(poss, dim=0)
        neg = torch.stack(negs, dim=0)

        pos = self._mix(pos, batches[0].index_reversed)

        num_features = neg.shape[2]

        return cebra_v2.dataset.Batch(
            reference=ref.view(-1, num_features),
            positive=pos.view(-1, num_features),
            negative=neg.view(-1, num_features),
        )
    
    def step(self, batch: cebra_v2.dataset.Batch) -> dict:
        """Perform a single gradient update.

        Args:
            batch: The input samples

        Returns:
            Dictionary containing training metrics.
        """
        self.optimizer.zero_grad()
        prediction = self._inference(batch)
        loss, align, uniform = self.criterion(prediction.reference,
                                              prediction.positive,
                                              prediction.negative)

        loss.backward()
        self.optimizer.step()
        self.history.append(loss.item())
        stats = dict(
            pos=align.item(),
            neg=uniform.item(),
            total=loss.item(),
            temperature=self.criterion.temperature,
        )
        for key, value in stats.items():
            self.log[key].append(value)
                
        return stats
    

    def fit(
        self,
        loader: cebra_v2.dataset.Loader,
        valid_loader: cebra_v2.dataset.Loader = None,
        save_frequency: int = None,
        valid_frequency: int = None,
        decode: bool = False,
        logdir: str = None,
    ):
        """Train model for the specified number of steps.

        Args:
            loader: Data loader, which is an iterator over `cebra.data.Batch` instances.
                Each batch contains reference, positive and negative input samples.
            valid_loader: Data loader used for validation of the model.
            save_frequency: If not `None`, the frequency for automatically saving model checkpoints
                to `logdir`.
            valid_frequency: The frequency for running validation on the ``valid_loader`` instance.
            logdir:  The logging directory for writing model checkpoints. The checkpoints
                can be read again using the `solver.load` function, or manually via loading the
                state dict.

        TODO:
            * Refine the API here. Drop the validation entirely, and implement this via a hook?
        """

        iterator = loader
        self.model.train()
        for id_step, batch in iterator:
            stats = self.step(batch)

            if id_step % 1000 == 0 : 
                print('Epoch {}'.format(id_step))
                print('Train loss {:.4f}, Train accuracy {:.2f}%'.format(
                stats["total"],0))