from abc import ABC

import numpy as np
from torch.utils.data import Dataset

CausalGraph = np.ndarray


class CausalDataset(Dataset, ABC):
    """
    A Causal Dataset is a PyTorch Dataset that posssibly returns the
    original causal graph for evaluation purposes.

    Attributes
    ----------
    n_nodes: int
        Number of nodes in the causal graph.
    causal_graph: CausalGraph
        The causal graph of the dataset.
    observations: np.ndarray
        The observations of the dataset.
        The array has shape [n_samples, n_nodes].
    """
    n_nodes: int
    causal_graph: CausalGraph
    observations: np.ndarray

    def __init__(self, normalize: bool = True):
        self.is_normalized = False
        if normalize:
            self._normalize()

    def __len__(self) -> int:
        """
        Returns the number of observations in the dataset.
        """
        return self.observations.shape[0]

    def __getitem__(self, index) -> np.ndarray:
        """
        Returns the d-dimensional sample at the provided index.
        """
        return self.observations[index]

    def _normalize(self) -> None:
        """
        Normalizes the observations by substracting the mean.
        """
        # Subtract mean
        self.observations -= self.observations.mean(axis=0, keepdims=True)
        # TODO: Divide by standard deviation
        # self.observations /= self.observations.std(axis=0, keepdims=True)
        # Set the corresponding flag
        self.is_normalized = True
