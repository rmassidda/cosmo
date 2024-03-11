import torch

from cosmolib.models import StructuralCausalModel


class DagGnn_VAE(StructuralCausalModel):
    """
    Non-linear DAG GNN model with single feature nodes.
    """

    def __init__(self, n_nodes: int, n_features: int,
                 n_hidden: int, n_latent: int):
        super().__init__(n_nodes=n_nodes)

        # Adjacency matrix
        self.weighted_adj = torch.nn.Parameter(torch.zeros((n_nodes, n_nodes)))

        # Encoder
        self.fc1 = torch.nn.Linear(n_features, n_hidden, bias=True)
        self.fc2 = torch.nn.Linear(n_hidden, n_latent, bias=True)
        # Decoder
        self.fc3 = torch.nn.Linear(n_latent, n_hidden, bias=True)
        self.fc4 = torch.nn.Linear(n_hidden, n_features, bias=True)

        # Init weights
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        torch.nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x_batch: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        """
        The forward method expects a batch of samples
        with shape (batch_size, n_nodes, n_features) and
        returns two batch of samples with the same shape.
        The first batch is the latent representation of the
        input batch and the second batch is the reconstructed
        input batch.
        If the input has shape (batch_size, n_nodes) it is
        reshaped to (batch_size, n_nodes, 1).
        """
        # Reshape input
        if x_batch.dim() == 2:
            x_batch = x_batch.unsqueeze(2)

        # Transformed Adjacency matrix
        identity = torch.eye(self.n_nodes).to(x_batch)
        trans_adj = identity - self.weighted_adj.T

        # Encoder
        z_batch = self.fc2(torch.relu(self.fc1(x_batch)))
        z_batch = trans_adj @ z_batch

        # Decoder
        x_batch = torch.inverse(trans_adj) @ z_batch
        x_batch = self.fc4(torch.relu(self.fc3(x_batch)))

        return z_batch, x_batch
