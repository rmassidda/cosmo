"""
Implementation of the NOTEARS algorithm.

Zheng, Xun, Bryon Aragam, Pradeep K. Ravikumar, and Eric P. Xing.
"Dags with no tears: Continuous optimization for structure learning."
Advances in neural information processing systems 31 (2018).

The reference implementation is at:
https://github.com/xunzheng/notears/
"""
from typing import Optional

import torch

from ..datasets import CausalDataset
from ..datasets import CausalGraph
from ..models import LinearSCM
from ..models import LocallyConnectedMLP
from ..models import StructuralCausalModel
from ..models import CausalModel
from ..models.trainable import Trainable
from ..utils import dag_constraint
from .generic import ContinuousCausalDiscoveryAlgorithm


class NOTEARS_Trainable(Trainable):
    """
    The class defines the learning procedure for a
    structural causal model as described in
    the NOTEARS algorithm by Zheng et al.
    """
    def __init__(self, module: StructuralCausalModel,
                 learning_rate: float, lambda_reg: float,
                 penalty: float, multiplier: float,
                 ):
        # Init super model
        super().__init__(
            module=module,
            learning_rate=learning_rate)

        # Loss function
        self.loss = torch.nn.MSELoss()

        # Primal Optimization Parameters
        self.lambda_reg = lambda_reg
        self.penalty = penalty
        self.multiplier = multiplier

    def step(self, batch: torch.Tensor, _: int, split: str = 'Train') \
            -> torch.Tensor:
        """
        Computes the Augmented Lagrangian on the current batch.
        """
        # Loss
        loss = self.loss(self.forward(batch), batch)

        # Constraint violation
        dagness = dag_constraint(self.module.weighted_adj)

        # Augmented lagrangian
        aug_lag = loss \
            + 0.5 * self.penalty * (dagness**2) \
            + self.multiplier * dagness \

        l1_reg = self.lambda_reg * \
            torch.norm(self.module.weighted_adj, p=1)  # type: ignore

        # Objective function
        full_step = aug_lag + l1_reg

        # Log values
        self.log(f'{split}_Augmented_Lagrangian', aug_lag)
        self.log(f'{split}_Dagness', dagness)
        self.log(f'{split}_L1', l1_reg)
        self.log(f'{split}_MSE', loss)
        self.log(f'{split}_Multiplier', self.multiplier)
        self.log(f'{split}_Penalty', self.penalty)
        self.log(f'{split}_Step', full_step)

        # Return Regularized Augmented Lagrangian
        return full_step


class NOTEARS(ContinuousCausalDiscoveryAlgorithm):
    """
    NOTEARS Algorithm from Zheng et al.

    The model learns an acyclic Linear SCM by using an acyclicity constraint
    and the Augmented Lagriangian constrained optimization technique.

    Overload the default optimizer to employ L-BFGS as described in
    the paper from Zheng et al. Notably, the author's implementation
    does not use L-BFGS but L-BFGS-B, which is a variant of L-BFGS
    that allows for bounds on the parameters. Further, they parameterize
    the weight matrix as the difference of two positive matrices,
        w = w_pos - w_neg
    and employ bounds to ensure their positivity.

    Further, there is a notable difference between the scipy and pytorch
    implementations of L-BFGS. For this reason, we employ Adam as the
    optimizer and we assume that the model has converged after a fixed
    number of epochs.

    For reference, see the original code and the discussion at:
    https://github.com/xunzheng/notears/issues/5
    """
    def __init__(self, learning_rate: float, lambda_reg: float,
                 hidden_size: Optional[list[int]],
                 init_penalty: float, penalty_fact: float, max_penalty: float,
                 multiplier: float, dagness_tol: float,
                 progress_rate: float, max_iter: int,
                 *generic_args, **generic_kwargs):

        # Model specific parameters
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.hidden_size = hidden_size
        self.init_penalty = init_penalty
        self.penalty_fact = penalty_fact
        self.max_penalty = max_penalty
        self.dagness_tol = dagness_tol
        self.progress_rate = progress_rate
        self.max_iter = max_iter
        self.multiplier = multiplier

        # Init generic model
        super().__init__(*generic_args, **generic_kwargs)

    def _build_model(self, dataset: CausalDataset) -> CausalModel:
        if self.hidden_size is None:
            return LinearSCM(
                n_nodes=dataset.n_nodes,
                )
        else:
            return LocallyConnectedMLP(
                n_nodes=dataset.n_nodes,
                hidden_size=self.hidden_size,
            )

    def _wrap_model(self) -> NOTEARS_Trainable:
        return NOTEARS_Trainable(
                module=self.causal_model,
                learning_rate=self.learning_rate,
                lambda_reg=self.lambda_reg,
                penalty=self.init_penalty,
                multiplier=self.multiplier,
        )

    def _fit(self, dataset: CausalDataset) -> CausalGraph:
        """
        The algorithm solves multiple unconstrained optimization problems
        regularized according to the DAGness of the weight matrix.
        At the end of the primal optimization problem, it evaluates the
        improvement in terms of the DAGness constraint. If the progress is
        sufficient, we perform the dual step; otherwise, we try again the
        primal problem by restoring the weights and increasing the penalty
        times the penalty factor.
        """
        # Build model
        self.causal_model = self._build_model(dataset)
        trainable_wrapper = self._wrap_model()

        # Load dataset
        loader = self._get_loader(dataset)

        # Initial values
        old_dagness = torch.inf
        dagness = torch.tensor(0.0)
        self.causal_model.store_parameters()

        # Max number of optimization problems
        for iter_num in range(self.max_iter):
            n_repeat = 0
            # Check if we should stop based on penalty value
            while trainable_wrapper.penalty < self.max_penalty:
                # Increment number of repeats
                n_repeat += 1

                # Build trainer
                trainer = self._get_trainer(dataset,
                                            f'Primal_{iter_num}_{n_repeat}')

                # Fit model
                trainer.fit(trainable_wrapper, loader)

                # Check DAGness improvement
                dagness = dag_constraint(self.causal_model.weighted_adj)
                if dagness > self.progress_rate * old_dagness:
                    # Restore old weights
                    self.causal_model.restore_parameters()
                    # Increase penalty
                    trainable_wrapper.penalty *= self.penalty_fact
                else:
                    break

            # Update values
            old_dagness = dagness
            self.causal_model.store_parameters()

            # Dual step
            trainable_wrapper.multiplier += \
                trainable_wrapper.penalty * dagness.item()

            # Check if we should stop
            if dagness < self.dagness_tol or \
                    trainable_wrapper.penalty >= self.max_penalty:
                break

        # Return graph
        return self._mask_weights()
