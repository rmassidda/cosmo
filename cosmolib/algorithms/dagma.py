"""
Implementation of the DAGMA algorithm.

Bello, Kevin, Bryon Aragam, and Pradeep Kumar Ravikumar.
"DAGMA: Learning DAGs via M-matrices and a Log-Determinant
Acyclicity Characterization."
Advances in Neural Information Processing Systems (2022).

The reference implementation is at:
https://github.com/kevinsbello/dagma
"""
from typing import Optional

import torch
import numpy as np

from ..models import StructuralCausalModel
from ..datasets import CausalDataset
from ..models.trainable import Trainable
from ..models import LinearSCM, LocallyConnectedMLP
from .generic import ContinuousCausalDiscoveryAlgorithm


class M_MatrixError(Exception):
    """
    Exception raised when the weighted adjacency matrix is not an M-matrix.
    """
    pass


class DAGMA_Trainable(Trainable):
    """
    DAGMA learning procedure from Bello et al.
    """
    def __init__(self, module: StructuralCausalModel,
                 learning_rate: float, lambda_reg: float,
                 path_coefficient: float,
                 log_det_parameter: float,
                 ):
        # Init super model with zero weights
        super().__init__(
            module=module,
            learning_rate=learning_rate)

        # Loss function
        self.loss = torch.nn.MSELoss()

        # Optimization Parameters
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.path_coefficient = path_coefficient
        self.log_det_parameter = log_det_parameter

    @property
    def m_mat(self):
        return self.log_det_parameter * \
              torch.eye(self.module.n_nodes, self.module.n_nodes) \
              - self.module.weighted_adj * self.module.weighted_adj

    def check_m_matrix(self):
        return torch.all((torch.inverse(self.m_mat) + 1e-5) >= 0)

    def dag_constraint(self):
        return - torch.logdet(self.m_mat) + self.module.n_nodes \
            * np.log(self.log_det_parameter)

    def step(self, batch: torch.Tensor, _: int, split: str = 'Train') \
            -> torch.Tensor:
        """
        Computes a step to minimize the augmented lagrangian.
        """
        # Loss
        loss = self.loss(self.forward(batch), batch)

        # L1 Regularization
        l1_reg = self.lambda_reg * torch.norm(self.module.weighted_adj,
                                              p=1)  # type: ignore

        # Weighted loss
        path_loss = self.path_coefficient * (loss + l1_reg)

        # Constraint violation
        dagness = self.dag_constraint()

        # Log values
        self.log(f'{split}_MSE', loss)
        self.log(f'{split}_L1', l1_reg)
        self.log(f'{split}_Step', path_loss + dagness)

        return path_loss + dagness

    def on_train_epoch_start(self):
        """
        Stores the weights of the model at the beginning
        of each epoch.
        """
        self.epoch_start_params = self.module.get_parameters()

    def on_train_epoch_end(self):
        """
        DAGMA implements a series of controls to ensure that the algorithm
        converges to a solution.

        Firstly, they divide the training procedure into T steps.
        The first T-1 steps are called the warm-up phase,
        and the last step is called the last step. The maximum
        number of epochs is larger for the last step.

        A step can be terminated early if:
          1. The algorithm detects that the solution is not an M-matrix,
          2. If the improvement is less than a tolerance value, or
          3. If the number of epochs exceeds the maximum number of epochs.

        In the first case, the algorithm resets the weights and restarts the
        step with a larger value of s and a smaller learning rate. If the
        value of s is too large, instead of resetting at the beginning of the
        step, the algorithm resets at the beginning of the epoch.

        In the remaining cases (2, 3), the algorithm proceeds to the next step
        by decreasing the value of mu and resetting the learning rate.
        """
        with torch.no_grad():
            # Check if the solution is not an M-matrix
            if not self.check_m_matrix():
                # Starting point is not an M-matrix
                if self.current_epoch == 0 or self.log_det_parameter <= 0.9:
                    raise M_MatrixError(
                        'The starting point is not an M-matrix.')

                # Revert to previous weights
                self.module.set_parameters(self.epoch_start_params)

                # Decrease learning rate
                self.learning_rate *= 0.5

                # Learning rate too small, we pass to the next step
                if self.learning_rate < 1e-6:
                    self.trainer.should_stop = True


class DAGMA(ContinuousCausalDiscoveryAlgorithm):
    """
    The model learns an acyclic Linear SCM.

    Parameters
    ----------
    learning_rate
        Learning rate for the ADAM optimizer.
    lambda_reg
        Regularization parameter for the L1 norm.
    initial_path_coefficient
        Initial value of the path coefficient.
    path_decay_factor
        Factor to decrease the path coefficient at the end of each step.
    log_det_parameter
        Value of the log-det parameter for each step.
    tolerance
        Tolerance for the convergence.
    convergence_interval
        Number of steps to check for convergence.
    max_steps
        Maximum number of steps, also called T.
    """
    def __init__(self, learning_rate: float, lambda_reg: float,
                 hidden_size: Optional[list[int]],
                 initial_path_coefficient: float, path_decay_factor: float,
                 log_det_parameter: list[float] | float,
                 max_steps: int,
                 *generic_args, **generic_kwargs):

        # Model specific parameters
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.hidden_size = hidden_size
        self.initial_path_coefficient = initial_path_coefficient
        self.path_decay_factor = path_decay_factor
        self.max_steps = max_steps

        # Adapt log-det parameters
        if isinstance(log_det_parameter, float):
            self.log_det_parameter = [log_det_parameter] * max_steps
        elif isinstance(log_det_parameter, list):
            if len(log_det_parameter) != max_steps:
                raise ValueError('The length of log_det_parameter must be'
                                 'equal to max_steps.')
            self.log_det_parameter = log_det_parameter
        else:
            raise ValueError('log_det_parameter must be a float or a list of'
                             'floats.')

        # Init generic algorithm
        super().__init__(*generic_args, **generic_kwargs)

    def _build_model(self, dataset: CausalDataset) -> StructuralCausalModel:
        if self.hidden_size is None:
            return LinearSCM(
                n_nodes=dataset.n_nodes,
                zero_init=True
                )
        else:
            return LocallyConnectedMLP(
                n_nodes=dataset.n_nodes,
                hidden_size=self.hidden_size,
                zero_init=True
            )

    def _wrap_model(self) -> DAGMA_Trainable:
        return DAGMA_Trainable(
            module=self.causal_model,
            learning_rate=self.learning_rate,
            lambda_reg=self.lambda_reg,
            path_coefficient=self.initial_path_coefficient,
            log_det_parameter=self.log_det_parameter[0],
        )

    def _fit(self, dataset: CausalDataset):
        """
        The DAGMA algorithm solves multiple unconstrained optimization
        problems by iteratively increasing the constraint on the path
        coefficients. In each step, the resulting matrix should be an
        M-matrix. If the algorithm detects that the solution is not an
        M-matrix, it restarts the step with a larger value of s and a
        smaller learning rate.
        """
        # Build model
        self.causal_model = self._build_model(dataset)
        trainable_wrapper = self._wrap_model()

        # Load dataset
        loader = self._get_loader(dataset)

        # Initial model values
        self.causal_model.store_parameters()

        # Iterate over steps
        for step in range(self.max_steps):
            if self.verbose:
                print(f'Step {step + 1}/{self.max_steps}...')
            # Initialize learning rate
            trainable_wrapper.learning_rate = self.learning_rate

            # Try to fit the model
            success = False
            while not success:
                # Build trainer
                trainer = self._get_trainer(dataset, f'step_{step}')

                # Train model
                try:
                    trainer.fit(trainable_wrapper, loader)
                    success = True
                except M_MatrixError:
                    if self.verbose:
                        print('The solution is not an M-matrix. Restarting.')
                    # Decrease learning rate
                    trainable_wrapper.learning_rate *= 0.5
                    # Increase s
                    trainable_wrapper.log_det_parameter += 0.1
                    # Restore weights
                    self.causal_model.restore_parameters()

            # Update path coefficient
            trainable_wrapper.path_coefficient *= self.path_decay_factor
            # Update log-det parameter
            trainable_wrapper.log_det_parameter = self.log_det_parameter[step]
            # Store weights
            self.causal_model.store_parameters()

        # Sqrt of the weights if non-linear model
        if isinstance(self.causal_model, LocallyConnectedMLP):
            self.causal_model.sqrt_weight = True

        # Return graph
        return self._mask_weights()
