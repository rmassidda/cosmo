"""
GOLEM Algorithm from Ng et al.
"""
import torch

from ..datasets import CausalDataset
from ..models import LinearSCM
from ..models import StructuralCausalModel
from ..models.trainable import Trainable
from ..utils import dag_constraint
from .generic import ContinuousCausalDiscoveryAlgorithm


class GOLEM_Trainable(Trainable):
    """
    GOLEM learning procedure from in Ng et al.

    In the equal variance case, the model minimizes the least squares loss
    regularized by the L1 norm of the weights and the continuous acyclicity
    constraint as introduced by Zheng et al. (2018).
    """
    def __init__(self, module: StructuralCausalModel,
                 learning_rate: float, lambda_reg: float,
                 dag_reg: float, equal_variance: bool):
        # Init super model
        super().__init__(module=module,
                         learning_rate=learning_rate)

        # Optimization Parameters
        self.loss = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.dag_reg = dag_reg
        self.equal_variance = equal_variance

    def _likelihood(self, batch: torch.Tensor):
        """
        Data likelihood of a linear Gaussian model.
        """
        if self.equal_variance:
            return 0.5 * self.module.n_nodes * torch.log(
                self.loss(self.forward(batch), batch)
                # torch.sum(
                #     torch.square(batch - self.forward(batch))
                # )
            ) - torch.slogdet(torch.eye(self.module.n_nodes)
                              - self.module.weighted_adj)[1]
        else:
            return 0.5 * torch.sum(torch.log(
                torch.sum(
                    torch.square(batch - self.forward(batch)),
                    dim=0)),
            ) - torch.slogdet(torch.eye(self.module.n_nodes)
                              - self.module.weighted_adj)[1]

    def step(self, batch: torch.Tensor, batch_idx: int, split: str = 'Train') \
            -> torch.Tensor:
        # Negative log-likelihood
        nl_likelihood = self._likelihood(batch)

        # Regularization
        l1_reg = self.lambda_reg * torch.norm(self.module.weighted_adj,
                                              p=1)  # type: ignore

        # Constraint violation
        dagness = self.dag_reg * dag_constraint(self.module.weighted_adj)

        # Overall objective function
        full_step = nl_likelihood + l1_reg + dagness

        # Log values
        self.log(f'{split}_NLL', nl_likelihood)
        self.log(f'{split}_L1', l1_reg)
        self.log(f'{split}_Dagness', dagness)
        self.log(f'{split}_Step', full_step)

        # Store for convergence
        self.curr_obj_fun = full_step

        # Return Augmented Lagrangian
        return full_step


class GOLEM(ContinuousCausalDiscoveryAlgorithm):
    """
    GOLEM Causal Discovery Algorithm from Ng et al.

    The corresponding model is implemented in the :class:`_GOLEM` class.

    Parameters
    ----------
    learning_rate
        Learning rate for the Adam optimizer.
    lambda_reg
        Regularization strength for the L1 norm of the weights.
    dag_reg
        Regularization strength for the continuous acyclicity constraint.
    equal_variance
        Whether to assume equal variance for the noise terms.
    convergence_interval
        Interval in epochs to check for convergence.
    tolerance
        Tolerance for convergence.
    *generic_args
        Positional arguments for :class:`ContinuousCausalDiscoveryAlgorithm`.
    **generic_kwargs
        Keyword arguments for :class:`ContinuousCausalDiscoveryAlgorithm`.
    """
    def __init__(self, learning_rate: float, lambda_reg: float,
                 dag_reg: float, equal_variance: bool,
                 *generic_args, **generic_kwargs):
        # Model specific parameters
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.dag_reg = dag_reg
        self.equal_variance = equal_variance

        # Init generic model
        super().__init__(*generic_args, **generic_kwargs)

    def _build_model(self, dataset: CausalDataset) -> LinearSCM:
        return LinearSCM(
            n_nodes=dataset.n_nodes,
            zero_init=False,
        )

    def _wrap_model(self) -> GOLEM_Trainable:
        return GOLEM_Trainable(
            module=self.causal_model,
            learning_rate=self.learning_rate,
            lambda_reg=self.lambda_reg,
            dag_reg=self.dag_reg,
            equal_variance=self.equal_variance,
            )
