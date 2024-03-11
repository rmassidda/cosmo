"""
COSMO Algorithm
"""
from typing import Optional

import numpy as np
import torch

from ..models import PriorityLinearSCM
from ..models.priority_linear_scm import PriorityNonLinearSCM
from ..models.trainable import Trainable
from ..datasets import CausalDataset, CausalGraph
from .generic import ContinuousCausalDiscoveryAlgorithm


class COSMO_Trainable(Trainable):
    """
    Causal Ordering discovery with SMooth Orientations.
    COSMO

    The class implements a PyTorch Lightning trainable
    Linear SCM model via smooth orientations.
    """
    module: PriorityLinearSCM

    def __init__(self, module: PriorityLinearSCM,
                 learning_rate: float,
                 l1_adj_reg: float,
                 l2_adj_reg: float,
                 priority_reg: float,
                 init_temperature: float, temperature: float,
                 anneal: Optional[str]):
        # Init super model
        super().__init__(module=module,
                         learning_rate=learning_rate)

        # Loss function
        self.loss = torch.nn.MSELoss()

        # Optimization Parameters
        self.learning_rate = learning_rate
        self.l1_adj_reg = l1_adj_reg
        self.l2_adj_reg = l2_adj_reg
        self.priority_reg = priority_reg
        self.init_temperature = init_temperature
        self.min_temperature = temperature
        self.anneal = anneal

        # TODO: enable monitoring as a parameter
        self.monitor = False

    def step(self, batch: torch.Tensor, _: int, split: str = 'Train') \
            -> torch.Tensor:
        # MSE Loss
        loss = self.loss(self.forward(batch), batch)

        # L2 Regularization on the Undirected Matrix
        # adj = self.module.adj.view(
        #     self.module.n_nodes, self.module.n_nodes, -1).sum(dim=2)
        # adj_reg = self.adj_reg * torch.norm(adj, p=2)  # type: ignore
        # L1 & L2 Regularization on the Undirected Matrix
        l1_rorm = torch.norm(self.module.adj, p=1)
        l2_norm = torch.norm(self.module.adj, p=2)
        adj_reg = self.l1_adj_reg * l1_rorm + self.l2_adj_reg * l2_norm

        # L2 Regularization on the Priorities
        priority_reg = self.priority_reg * torch.norm(
            self.module.nodes_priority, p=2)  # type: ignore

        # Full loss
        full_step = loss + adj_reg + priority_reg

        # Log values
        self.log(f'{split}_MSE', loss)
        self.log(f'{split}_Adj_L2', adj_reg)
        self.log(f'{split}_Priority_L2', priority_reg)
        self.log(f'{split}_Step', full_step)

        return full_step

    def on_train_epoch_end(self):
        if self.anneal is not None:
            # Runtime assertions
            assert self.trainer is not None
            assert self.trainer.max_epochs is not None
            n_epochs = self.trainer.current_epoch + 1
            if self.anneal == 'cosine':
                # Cosine Annealing
                self.module.temperature = self.min_temperature + \
                    0.5 * (self.init_temperature - self.min_temperature) * \
                    (1 + np.cos(np.pi * n_epochs /
                                self.trainer.max_epochs))
            elif self.anneal == 'linear':
                # Linear Decay
                self.module.temperature = self.min_temperature + \
                    (1-(n_epochs / self.trainer.max_epochs)) *\
                    (self.init_temperature - self.min_temperature)
            elif self.anneal == 'constant':
                self.module.temperature = self.init_temperature
            else:
                raise ValueError(f'Unknown decay style {self.anneal}')

        # Log temperature
        self.log('Temperature', self.module.temperature)

        # if self.monitor:
        #     # Log of the weight gradient norm
        #     if hasattr(self, '_weight') and \
        #             self.module._weight.grad is not None:
        #         self.log('weight_grad_norm', torch.norm(
        #             self.module._weight.grad.cpu(), p=2))  # type: ignore
        #     # Log of the adjacency norm
        #     if self.module.adj.grad is not None:
        #         self.log('adj_grad_norm', torch.norm(
        #             self.module.adj.grad.cpu(), p=2))  # type: ignore
        #     # Log of the priority norm
        #     if self.module.nodes_priority.grad is not None:
        #         self.log('priority_grad_norm', torch.norm(
        #             self.module.nodes_priority.grad.cpu(),
        #             p=2))  # type: ignore


class COSMO(ContinuousCausalDiscoveryAlgorithm):
    """
    Causal Ordering discovery with SMooth Orientation.

    When the temperature reaches the minimum, the resulting weight
    matrix describes a DAG. Nonetheless, the gradient also vanishes
    and the model stops learning. To overcome this issue, we decay
    the temperature with several strategies during training.

    Further, we regularize the adjacency matrix to contain small
    values and the priorities to do not diverge further than required.

    Parameters
    ----------
    learning_rate: float
        Learning rate for the optimizer.
    adj_reg: float
        Regularization coefficient for the adjacency matrix.
    priority_reg: float
        Regularization coefficient for the priorities.
    init_temperature: float
        Initial temperature for the smooth orientations.
    temperature: float
        Temperature for the smooth orientations.
    shift: float
        Minimum distance between different priorities.
    hard_threshold: bool
        Whether to use a hard threshold for the priorities.
    decay_style: str
        Style of the temperature decay, one of 'cosine', 'linear',
        'exponential', 'step', 'none'.
    decay_steps: Optional[int]
        Number of steps for the step decay.
    """
    causal_model: PriorityLinearSCM

    def __init__(self, learning_rate: float,
                 l1_adj_reg: float,
                 l2_adj_reg: float,
                 priority_reg: float,
                 hidden_size: Optional[list[int]],
                 init_temperature: float,
                 temperature: float, shift: float,
                 hard_threshold: bool, anneal: Optional[str],
                 symmetric: bool,
                 *generic_args, **generic_kwargs):

        # Model specific parameters
        self.learning_rate = learning_rate
        self.l1_adj_reg = l1_adj_reg
        self.l2_adj_reg = l2_adj_reg
        self.priority_reg = priority_reg
        self.hidden_size = hidden_size
        self.init_temperature = init_temperature
        self.temperature = temperature
        self.shift = shift
        self.hard_threshold = hard_threshold
        self.anneal = anneal
        self.symmetric = symmetric

        # Init generic algorithm
        super().__init__(*generic_args, **generic_kwargs)

    def _build_model(self, dataset: CausalDataset) -> PriorityLinearSCM:
        if self.hidden_size is None:
            return PriorityLinearSCM(
                dataset.n_nodes,
                temperature=self.init_temperature,
                shift=self.shift,
                hard_threshold=self.hard_threshold,
                symmetric=self.symmetric,
            )
        else:
            return PriorityNonLinearSCM(
                dataset.n_nodes,
                hidden_size=self.hidden_size,
                temperature=self.init_temperature,
                shift=self.shift,
                hard_threshold=self.hard_threshold,
                symmetric=self.symmetric,
            )

    def _wrap_model(self) -> COSMO_Trainable:
        return COSMO_Trainable(
            self.causal_model,
            learning_rate=self.learning_rate,
            l1_adj_reg=self.l1_adj_reg,
            l2_adj_reg=self.l2_adj_reg,
            priority_reg=self.priority_reg,
            init_temperature=self.init_temperature,
            temperature=self.temperature,
            anneal=self.anneal,
        )

    def evaluate(self, gt_graph: CausalGraph) -> dict:
        # Evaluate using super metrics
        super_evaluation = super().evaluate(gt_graph)

        # Retrive priorities
        priorities = self.causal_model.nodes_priority.clone(
        ).detach().cpu().numpy()  # type: ignore
        # Priority differences
        priorities_diff = np.abs(priorities[:, None] - priorities[None, :])
        # remove last dimension
        priorities_diff = priorities_diff.squeeze()
        # assert square shape
        assert priorities_diff.shape[0] == priorities_diff.shape[1]
        # assert diagonal is zero
        assert np.allclose(priorities_diff.diagonal(), 0)
        # remove upper triangular
        priorities_diff = priorities_diff[np.tril_indices_from(
            priorities_diff, k=-1)]
        # absolute difference
        priorities_diff = np.abs(priorities_diff)
        # compute the mean of the differences
        priorities_diff_mean = np.mean(priorities_diff)
        # compute the std of the differences
        priorities_diff_std = np.std(priorities_diff)

        # L2 norm of the priorities
        l2_norm = np.linalg.norm(priorities, ord=2)

        # Aggregate evaluation
        evaluation = {**super_evaluation,
                      'priority_l2_norm': l2_norm,
                      'priority_diff_mean': priorities_diff_mean,
                      'priority_diff_std': priorities_diff_std}

        return evaluation

    def _fit(self, dataset: CausalDataset) -> CausalGraph:
        # Check if the anneal epoch is set
        if self.anneal is not None:
            return super()._fit(dataset)

        # Build model
        self.causal_model = self._build_model(dataset)
        trainable_wrapper = self._wrap_model()

        # Load dataset
        loader = self._get_loader(dataset)

        # Initial values
        trainable_wrapper.module.temperature = self.init_temperature

        # Max number of optimization problems
        n_repeat = 0
        # Check if we should stop based on penalty value
        while trainable_wrapper.module.temperature > self.temperature:
            # Increment number of repeats
            n_repeat += 1

            # Build trainer
            trainer = self._get_trainer(dataset,
                                        f'Temperature_{n_repeat}')

            # Fit model
            trainer.fit(trainable_wrapper, loader)

            # Update temperature
            trainable_wrapper.module.temperature = \
                max(
                    self.temperature,
                    trainable_wrapper.module.temperature * 0.5
                )

        # Last fit at the final temperature
        trainer = self._get_trainer(dataset, f'Temperature_{n_repeat + 1}')
        trainer.fit(trainable_wrapper, loader)

        # Return graph
        return self._mask_weights()
