import numpy as np
import torch

from cosmolib.algorithms import NOTEARS
from cosmolib.datasets import CausalDataset
from cosmolib.models import DagGnn_VAE
from cosmolib.models.trainable import Trainable


class DagGnnTrainable(Trainable):
    """
    TODO: Also here there are similarities with the NOTEARS
    trainable class. Maybe we can merge them into one class.
    See the discussion below in the DagGnn algorithm class.
    """
    module: DagGnn_VAE

    def __init__(self, module: DagGnn_VAE, learning_rate: float,
                 lambda_reg: float, penalty: float,
                 multiplier: float):
        # Init super model
        super().__init__(module=module, learning_rate=learning_rate)

        # Optimization parameters
        self.lambda_reg = lambda_reg
        self.penalty = penalty
        self.multiplier = multiplier

    def configure_optimizers(self):
        # Optimizer
        optim = torch.optim.Adam(self.parameters(),
                                 lr=self.learning_rate)
        # Scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=200, gamma=0.5
        )

        return [optim], [scheduler]

    def kl_divergence(self, mu_z: torch.Tensor) -> torch.Tensor:
        """
        Computes the KL divergence betweeen the approximated posterior
        q(Z|X) with mean mu_z and the normally distributed prior p(Z).
        Equation (8) from the paper where S_z = 0.
        """
        kl_div = mu_z * mu_z
        kl_sum = kl_div.sum()
        return (kl_sum / (mu_z.size(0))) * 0.5

    def reconstruction_acc(self, x_pred: torch.Tensor,
                           x_target: torch.Tensor,
                           variance: float = 0.,
                           # add_const: bool = False
                           ) -> torch.Tensor:
        """
        Computes the negative log-likelihood between the
        predicted and target samples.
        """
        mean1 = x_pred
        mean2 = x_target
        neg_log_p = variance + \
            torch.div(torch.pow(mean1 - mean2, 2),
                      2.*np.exp(2. * variance))
        # if add_const:
        #     const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        #     neg_log_p += const
        return neg_log_p.sum() / (x_target.size(0))

    def constraint(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Computes the acyclicity constraint on the adjacency matrix
        as defined in Equation (13) from the paper. The value of
        alpha is implicitly set to 1/n_nodes as in the author's
        implementation.
        """
        n_nodes = self.module.n_nodes
        adj = adj * adj
        # alpha = 1. / n_nodes
        eye = torch.eye(n_nodes, n_nodes).to(adj)
        adj = eye + torch.div(adj, n_nodes)
        adj = torch.matrix_power(adj, n_nodes)
        h_val = torch.trace(adj) - n_nodes
        return h_val

    def step(self, batch: torch.Tensor, _: int, split: str = 'Train') \
            -> torch.Tensor:
        """
        If the input has shape (batch_size, n_nodes) it is
        reshaped to (batch_size, n_nodes, 1).
        """
        # Reshape input
        if batch.dim() == 2:
            batch = batch.unsqueeze(2)

        # Get latent and reconstructed batch
        z_batch, x_batch = self.module.forward(batch)

        # Compute DAGness
        dagness = self.constraint(self.module.weighted_adj)

        # Reconstruction accuracy
        loss_nll = self.reconstruction_acc(x_batch, batch)
        # KL divergence
        loss_kl = self.kl_divergence(z_batch)
        # Loss
        loss = loss_nll + loss_kl
        # Augmented lagrangian
        aug_lag = loss + \
            + 0.5 * self.penalty * (dagness**2) \
            + self.multiplier * dagness
        # L1 regularization
        l1_reg = self.lambda_reg * \
            torch.norm(self.module.weighted_adj, p=1)  # type: ignore
        # Objective function
        full_step = aug_lag + l1_reg

        # Log values
        self.log(f'{split}_Loss', loss)
        self.log(f'{split}_NLL', loss_nll)
        self.log(f'{split}_KL', loss_kl)
        self.log(f'{split}_DAGness', dagness)
        self.log(f'{split}_AugLag', aug_lag)
        self.log(f'{split}_L1Reg', l1_reg)
        self.log(f'{split}_Step', full_step)

        return full_step


class DAGGNN(NOTEARS):
    causal_model: DagGnn_VAE
    """
    DAG-GNN Causal Discovery Method from Yu et al. (2019)

    TODO: We extend the NOTEARS class since the DAG-GNN method
    also uses the augmented lagrangian method to enforce DAGness.
    A cleaner solution would be to create a new class for the
    augmented lagrangian method and let both NOTEARS and DAG-GNN
    inherit from it.
    """
    def __init__(self, learning_rate: float, lambda_reg: float,
                 hidden_size: int, latent_size: int,
                 init_penalty: float, penalty_fact: float, max_penalty: float,
                 multiplier: float, dagness_tol: float,
                 progress_rate: float, max_iter: int,
                 *generic_args, **generic_kwargs):
        # Init NOTEARS super class
        super().__init__(
            learning_rate=learning_rate, lambda_reg=lambda_reg,
            hidden_size=None,
            init_penalty=init_penalty, penalty_fact=penalty_fact,
            max_penalty=max_penalty, multiplier=multiplier,
            dagness_tol=dagness_tol, progress_rate=progress_rate,
            max_iter=max_iter,
            *generic_args, **generic_kwargs)

        # Specific VAE model parameters
        self.hidden_size = hidden_size
        self.latent_size = latent_size

    def _build_model(self, dataset: CausalDataset) -> DagGnn_VAE:
        return DagGnn_VAE(
            n_nodes=dataset.n_nodes,
            n_features=1,
            n_hidden=self.hidden_size,
            n_latent=self.latent_size,
        )

    def _wrap_model(self) -> DagGnnTrainable:
        return DagGnnTrainable(
            module=self.causal_model,
            learning_rate=self.learning_rate,
            lambda_reg=self.lambda_reg,
            penalty=self.init_penalty,
            multiplier=self.multiplier,
        )
