import torch
from torch import nn
from torch.nn.parameter import Parameter

from cosmolib.models import AbstractLinearSCM


class HodgeLinearSCM(AbstractLinearSCM):
    def __init__(self, n_nodes: int):
        """
        Defines a linear structural causal model parameterized by
        a priority vectors on the variables. The model is defined
        as the element-wise product of the adjacency matrix and
        the smooth orientation matrix, parameterized by the
        priorities and the shifted tempered-sigmoid function.

        Parameters
        ----------
        n_nodes: int
            Number of nodes in the model.
        """
        # Model Parameters
        super().__init__(n_nodes)

        # Priority Vectors
        self.nodes_priority = Parameter(torch.zeros((self.n_nodes, 1)))

        # Adjacency Matrix
        self.indices = torch.tril_indices(self.n_nodes, self.n_nodes, -1)
        self.values = Parameter(torch.rand(self.indices[0].size(0)))

        # Init parameters
        self.reset_parameters()

    def _reset_weight(self):
        """
        Initialization of the undirected weights.
        """
        nn.init.normal_(self.values)
        nn.init.normal_(self.nodes_priority)

    def freeze_priorities(self):
        """
        Freezes the priorities.
        """
        self.nodes_priority.requires_grad = False
        self.nodes_priority.grad = None

    def unfreeze_priorities(self):
        """
        Unfreezes the priorities.
        """
        self.nodes_priority.requires_grad = True
        self.nodes_priority.grad = torch.zeros_like(self.nodes_priority)

    def freeze_adjacency(self):
        """
        Freezes the adjacency matrix.
        """
        self.values.requires_grad = False
        self.values.grad = None

    def unfreeze_adjacency(self):
        """
        Unfreezes the adjacency matrix.
        """
        self.values.requires_grad = True
        self.values.grad = torch.zeros_like(self.values)

    @property
    def weight(self) -> torch.Tensor:
        """
        Computes an explicit representation of the weight matrix
        given the undirected adjacency matrix and the orientation.
        """
        # Build skew-symmetric matrix
        skew = torch.zeros((self.n_nodes, self.n_nodes))
        row, col = self.indices
        skew[row, col] = self.values
        skew = skew - skew.T

        # Compute the difference matrix
        grad_p = self.nodes_priority.T - self.nodes_priority

        # Compose adjacency matrix
        return skew * torch.relu(grad_p)
