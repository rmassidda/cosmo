from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn.parameter import Parameter

from cosmolib.models import StructuralCausalModel


class AbstractLinearSCM(StructuralCausalModel):
    weight: torch.Tensor

    def __init__(self, n_nodes: int):
        super().__init__(n_nodes=n_nodes)

    @property
    def weighted_adj(self) -> torch.Tensor:
        return self.weight

    def reset_parameters(self):
        self._reset_weight()

    def _reset_weight(self):
        raise NotImplementedError

    @torch.no_grad()
    def store_parameters(self):
        """
        Store the weights from the linear model
        into a local variable.
        """
        self.old_parameters = self.weight.detach().clone()

    @torch.no_grad()
    def restore_parameters(self):
        """
        Restores the local copy into the linear
        model weights.
        """
        self.weight[:] = self.old_parameters[:]

    def forward(self, batch: torch.Tensor):
        return batch @ self.weight


class LinearSCM(AbstractLinearSCM):
    def __init__(self, n_nodes: int, zero_init: bool = True):
        super().__init__(n_nodes=n_nodes)

        # Weight matrix
        self.weight = Parameter(torch.empty((self.n_nodes, self.n_nodes)))

        # Initialize parameters
        self.zero_init = zero_init
        self.reset_parameters()

    def _reset_weight(self):
        if self.zero_init:
            nn.init.zeros_(self.weight)
        else:
            nn.init.normal_(self.weight, std=1e-3)


class LowRankLinearSCM(AbstractLinearSCM):
    def __init__(self, n_nodes: int, rank: int):
        # Model Parameters
        super().__init__(n_nodes=n_nodes)

        # Low-rank weights
        self.rank = rank
        self.U = Parameter(torch.empty((self.n_nodes, self.rank)))
        self.V = Parameter(torch.empty((self.rank, self.n_nodes)))

        # Initialize parameters
        self.reset_parameters()

    def _reset_weight(self):
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(x, self.U), self.V)

    @property
    def weight(self) -> torch.Tensor:
        weight = torch.matmul(self.U, self.V)
        return weight
