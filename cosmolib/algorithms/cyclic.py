"""
Dummy algorithm learning a linear model.
"""
import torch

from ..datasets import CausalDataset
from ..models import LinearSCM
from ..models import StructuralCausalModel
from ..models.trainable import Trainable
from .generic import ContinuousCausalDiscoveryAlgorithm


class Cyclic_Trainable(Trainable):
    def __init__(self,
                 module: StructuralCausalModel,
                 learning_rate: float,
                 lambda_reg: float
                 ):
        # Init super model
        super().__init__(
            module=module,
            learning_rate=learning_rate
        )

        # Optimization Parameters
        self.loss = torch.nn.MSELoss()
        self.lambda_reg = lambda_reg

    def step(self, batch: torch.Tensor, batch_idx: int, split: str = 'Train') \
            -> torch.Tensor:
        # MSE loss
        loss = self.loss(self.forward(batch), batch)

        # Regularization
        l1_reg = self.lambda_reg * torch.norm(self.module.weighted_adj,
                                              p=2)  # type: ignore

        # Overall objective function
        full_step = loss + l1_reg

        # Log values
        self.log(f'{split}_MSE', loss)
        self.log(f'{split}_L1', l1_reg)
        self.log(f'{split}_Step', full_step)

        # Return Augmented Lagrangian
        return full_step


class Cyclic(ContinuousCausalDiscoveryAlgorithm):
    def __init__(self,
                 learning_rate: float,
                 lambda_reg: float,
                 *generic_args, **generic_kwargs):
        # Model specific parameters
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

        # Init generic model
        super().__init__(*generic_args, **generic_kwargs)

    def _build_model(self, dataset: CausalDataset) -> LinearSCM:
        return LinearSCM(n_nodes=dataset.n_nodes)

    def _wrap_model(self) -> Trainable:
        return Cyclic_Trainable(
            module=self.causal_model,
            learning_rate=self.learning_rate,
            lambda_reg=self.lambda_reg
        )
