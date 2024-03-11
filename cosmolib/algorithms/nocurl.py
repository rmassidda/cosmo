"""
Implementation of the NO-CURL algorithm.

Yu, Yue, Tian Gao, Naiyu Yin, and Qiang Ji.
"Dags with no curl: An efficient dag structure learning approach."
International Conference on Machine Learning, pp. 12156-12166. PMLR, 2021.

The reference implementation is at:
https://github.com/fishmoon1234/DAG-NoCurl
"""
from typing import Optional
from typing import Union

import torch

from ..datasets import CausalDataset
from ..datasets import CausalGraph
from ..models import HodgeLinearSCM
from ..models import StructuralCausalModel
from ..models import LinearSCM
from ..models.trainable import Trainable
from .generic import ContinuousCausalDiscoveryAlgorithm
from .notears import NOTEARS_Trainable


def compute_ordering(pre_weights: torch.Tensor,
                     n_nodes: int) -> torch.Tensor:
    # Decompose the matrix
    connection_matrix = torch.sign(torch.abs(pre_weights))
    connection_matrix = torch.sign(
        torch.linalg.matrix_exp(connection_matrix) - torch.eye(n_nodes)
    )
    connection_matrix = connection_matrix - connection_matrix.T
    connection_matrix = connection_matrix / 2

    # Create the Laplacian matrix
    laplacian = torch.ones(n_nodes - 1, n_nodes - 1)
    for node in range(n_nodes - 1):
        laplacian[node, node] = -(n_nodes - 1)

    # Compute the ordering
    pre_ordering = torch.linalg.solve(
        laplacian, torch.sum(connection_matrix, dim=1)[:-1]
    )

    # Add zero at the end
    pre_ordering = torch.cat([pre_ordering, torch.zeros(1)])

    return pre_ordering


class NOCURL_Trainable(Trainable):
    def __init__(self, module: StructuralCausalModel,
                 learning_rate: float, lambda_reg: float,
                 ):
        # Init super model
        super().__init__(
            module=module,
            learning_rate=learning_rate)

        # Loss function
        self.loss = torch.nn.MSELoss()

        # Primal Optimization Parameters
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

    def step(self, batch: torch.Tensor, _: int, split: str = 'Train') \
            -> torch.Tensor:
        # Loss
        loss = self.loss(self.forward(batch), batch)

        l1_reg = self.lambda_reg * \
            torch.norm(self.module.weighted_adj, p=1)  # type: ignore

        # Objective function
        full_step = loss + l1_reg

        # Log values
        self.log(f'{split}_L1', l1_reg)
        self.log(f'{split}_MSE', loss)
        self.log(f'{split}_Step', full_step)

        # Return Regularized Augmented Lagrangian
        return full_step


class NOCURL(ContinuousCausalDiscoveryAlgorithm):
    """
    NOCURL Algorithm from Yue et al.

    The algorithm learns a directed acyclic graph (DAG) structure
    from observational data by solving a fixed number of unconstrained
    optimization problems.

    Firstly, the algorithm learns the weights of a possibly cyclic
    linear model by regularizing the MSE loss with an acyclicity
    constraint. After training, the model is optionally retrained
    with an higher regularization strength to enforce a DAG structure.
    In the code, we regularize the first training procedure with the
    "constraint_multiplier" parameter. The second training procedure is
    regularized with the "constraint_multiplier_second" parameter; if the
    latter is None, the second training procedure is skipped.

    Then, the algorithm thresholds the learned weights against the
    "prethreshold" parameter and computes an approximation of the
    topological ordering of the graph.

    Similarly, the algorithm computes the undirected weights of the graph.
    If the parameter "compute_weights" is set to False, the weights are
    fitted by freezing the ordering and minimizing a non-constrained
    MSE loss. Otherwise, the weights are computed via a closed form.

    Consequently, if the "joint_learn" parameter is set to True, the algorithm
    jointly learns the weights and the ordering of the graph by minimizing
    the non-constrained MSE loss. Otherwise, the step is skipped.

    Finally, the algorithm returns a thresholded version of the learned
    weights and the topological ordering of the graph.
    """
    def __init__(self,
                 learning_rate: float,
                 lambda_reg: float,
                 multiplier: Union[float, list[float]],
                 prethreshold: Optional[float],
                 compute_weights: bool,
                 joint_learn: bool,
                 progress_rate: float,
                 *generic_args, **generic_kwargs):

        # Model specific parameters
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.multiplier = multiplier if isinstance(multiplier, list) \
            else [multiplier]
        self.prethreshold = prethreshold
        self.compute_weights = compute_weights
        self.joint_learn = joint_learn
        self.progress_rate = progress_rate

        # Init generic model
        super().__init__(*generic_args, **generic_kwargs)

    def _build_model(self, dataset: CausalDataset) -> HodgeLinearSCM:
        return HodgeLinearSCM(n_nodes=dataset.n_nodes)

    def _wrap_model(self) -> NOCURL_Trainable:
        return NOCURL_Trainable(
            module=self.causal_model,
            learning_rate=self.learning_rate,
            lambda_reg=self.lambda_reg,
        )

    def _compute_weights(self) -> None:
        raise NotImplementedError

    def _fit(self, dataset: CausalDataset) -> CausalGraph:
        # Jointly learn weights and ordering without preliminary step
        if self.joint_learn:
            return super()._fit(dataset)

        # Load dataset
        loader = self._get_loader(dataset)

        # Build preliminary model
        self.causal_model = LinearSCM(dataset.n_nodes)
        trainable_wrapper = NOTEARS_Trainable(
            self.causal_model,
            learning_rate=self.learning_rate,
            lambda_reg=self.lambda_reg,
            penalty=0.0,
            multiplier=0.0,
        )

        # Train multiple times with higher multiplier
        for n_step, multiplier in enumerate(self.multiplier):
            trainable_wrapper.multiplier = multiplier
            trainer = self._get_trainer(dataset,
                                        f'Presolution_{n_step + 1}')
            trainer.fit(trainable_wrapper, loader)

        # Mask the learned weights
        pre_weights = self.causal_model.weighted_adj.detach().clone()
        pre_weights[torch.abs(pre_weights) < self.prethreshold] = 0.

        # Compute the ordering
        pre_ordering = compute_ordering(pre_weights, dataset.n_nodes)

        # Build the final model
        self.causal_model = self._build_model(dataset)
        trainable_wrapper = NOCURL_Trainable(
            self.causal_model,
            learning_rate=self.learning_rate,
            lambda_reg=self.lambda_reg,
        )

        # Set the ordering
        with torch.no_grad():
            self.causal_model.nodes_priority[:, 0] = pre_ordering

        # Compute or learn the weights
        if self.compute_weights:
            raise NotImplementedError
        else:
            # Freeze the ordering
            self.causal_model.freeze_priorities()
            # Learn the weights
            trainer = self._get_trainer(dataset, 'WeightLearning')
            trainer.fit(trainable_wrapper, loader)
            # Unfreeze the ordering
            self.causal_model.unfreeze_priorities()

        # Jointly learn the ordering and the weights
        # if self.joint_learn:
        #     trainer = self._get_trainer(dataset, 'JointLearning')
        #     trainer.fit(trainable_wrapper, loader)

        # Return graph
        return self._mask_weights()
