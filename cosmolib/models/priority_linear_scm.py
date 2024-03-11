from typing import Optional
import math

import torch
from torch import nn
from torch.nn.parameter import Parameter

from cosmolib.models import AbstractLinearSCM
from cosmolib.models.locally_connected import LocallyConnected


class PriorityLinearSCM(AbstractLinearSCM):
    def __init__(self, n_nodes: int, temperature: float,
                 shift: float, hard_threshold: bool,
                 symmetric: bool,
                 adjacency_var: float = 0.0,
                 priority_var: Optional[float] = None,
                 monitor: bool = False):
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
        temperature: float
            Temperature for the smooth orientations.
        shift: float
            Center of the sigmoid function.
        hard_threshold: bool
            Whether to use a hard threshold in forward mode
            for the smooth orientation matrix.
        symmetric: bool
            Whether to use a symmetric adjacency matrix.
        adjacency_var: float
            Variance of the prior on the adjacency matrix.
        priority_var: float
            Variance of the prior on the priorities.
        monitor: bool
            Whether to monitor the model parameters by retaining
            gradients.
        """
        # Model Parameters
        super().__init__(n_nodes)

        # Check epsilon
        assert shift > 0, 'Shift must be strictly positive'

        # Priority Vectors
        self.nodes_priority = Parameter(torch.zeros((self.n_nodes, 1)))
        self.priority_var = priority_var if priority_var is not None \
            else shift / math.sqrt(2)

        # Smooth Orientation Parameters
        self.temperature = temperature
        self.shift = shift
        self.hard_threshold = hard_threshold

        # Adjacency Matrix
        self.adj = Parameter(torch.empty((self.n_nodes, self.n_nodes)))
        self.adjacency_var = adjacency_var
        self.symmetric = symmetric

        # Monitor
        self.monitor = monitor

        # Init parameters
        self.reset_parameters()

    def _reset_weight(self):
        """
        Initialization of the undirected weights.
        """
        # nn.init.normal_(self.adj, std=self.adjacency_var)
        nn.init.kaiming_uniform_(self.adj, nonlinearity='linear')
        nn.init.normal_(self.nodes_priority, std=self.priority_var)

    @property
    def orientation(self) -> torch.Tensor:
        """
        Computes the orientation matrix given the priority vectors.
        If the hard_threshold flag is set to True, the orientation
        if thresholded against the shift parameter.

        The matrix containing the priority differences is computed
        as diff_mat[i, j] = priority[j] - priority[i]. We want an arc
        whenever p[i] < p[j], therefore, whenever
            dif_mat[i, j] > self.shift
        """
        # Difference Matrix
        dif_mat = self.nodes_priority.T - self.nodes_priority

        # Apply the shifted-tempered sigmoid
        orient_mat = torch.sigmoid((dif_mat - self.shift) / self.temperature)

        # Remove the diagonal
        orient_mat = orient_mat * (1 - torch.eye(self.n_nodes))

        # Hard Thresholding
        if self.hard_threshold:
            # Compute the hard orientation
            hard_orient_mat = dif_mat > self.shift
            hard_orient_mat = hard_orient_mat.float()

            # Apply soft detaching trick
            orient_mat = orient_mat + \
                (hard_orient_mat - orient_mat).detach()

        return orient_mat

    @property
    def weight(self) -> torch.Tensor:
        """
        Computes an explicit representation of the weight matrix
        given the undirected adjacency matrix and the orientation.
        """
        # Compute the adjacency matrix
        if self.symmetric:
            adj = self.adj + self.adj.T
        else:
            adj = self.adj

        if self.monitor:
            # Compute the weight matrix
            self._weight = adj * self.orientation
            # Retain the gradient
            self._weight.retain_grad()
            # Return the weight matrix
            return self._weight

        return adj * self.orientation


class PriorityNonLinearSCM(PriorityLinearSCM):
    def __init__(self,
                 n_nodes: int,
                 hidden_size: list[int],
                 temperature: float,
                 shift: float, hard_threshold: bool,
                 symmetric: bool,
                 adjacency_var: float = 1e-1,
                 priority_var: Optional[float] = None,
                 monitor: bool = False,
                 bias: bool = True):
        """
        Defines a non-linear structural causal model parameterized by
        a priority vector on the variables.
        """
        super().__init__(n_nodes, temperature, shift, hard_threshold,
                         symmetric, adjacency_var, priority_var, monitor)

        # Dimensionality of the layers
        self.n_nodes = n_nodes
        self.hidden_size = hidden_size + [1]

        # Adjacency Matrix
        self.adj = Parameter(torch.empty((
            self.n_nodes, self.n_nodes * self.hidden_size[0])))

        # Bias term
        if bias:
            self.bias = Parameter(torch.empty(
                self.n_nodes * self.hidden_size[0]))
        else:
            self.register_parameter('bias', None)

        # Fully connected layers
        self.fully_connected = torch.nn.ModuleList(
            [LocallyConnected(
                self.n_nodes,
                self.hidden_size[i],
                self.hidden_size[i + 1], bias=bias)
             for i in range(len(self.hidden_size) - 1)])

        # Init parameters
        self.init_adj()

    def init_adj(self):
        """
        Initialize the adjacency matrix
        and the bias term, if any.
        """
        # Adjacency Matrix
        nn.init.kaiming_uniform_(self.adj, a=math.sqrt(5))
        # Bias
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.adj)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_batch: torch.Tensor):
        """
        Forward pass of the multilayer perceptron.
        Both the input and the output tensors have
        shape [B, d].
        """
        # View weights as [d, d * h0]
        weight = self.weight.view(self.n_nodes,
                                  self.n_nodes * self.hidden_size[0])

        # Apply linear transformation [B, d] -> [B, d * h0]
        x_batch = torch.nn.functional.linear(x_batch, weight.T, self.bias)

        # Reshape the batch to have shape [B, d, h0]
        x_batch = x_batch.view(-1, self.n_nodes, self.hidden_size[0])

        # Locally connected layers
        for fc in self.fully_connected:
            # [B, d, h_l] -> [B, d, h_l]
            x_batch = torch.sigmoid(x_batch)
            # [B, d, h_l] -> [B, d, h_{l+1}]
            x_batch = fc(x_batch)

        # [B, d, 1] -> [B, d]
        x_batch = x_batch.squeeze(dim=2)

        return x_batch

    @property
    def weight(self) -> torch.Tensor:
        """
        Applies the orientation to the weights of the SCM.
        """
        # View weights as [d, d, h0]
        weight = self.adj.view(self.n_nodes, self.n_nodes, self.hidden_size[0])

        # Apply the orientation to each hidden dimension
        weight = weight * self.orientation.unsqueeze(-1)

        if self.monitor:
            # Store the weight matrix
            self._weight = weight
            # Retain the gradient
            self._weight.retain_grad()

        return weight

    @property
    def weighted_adj(self) -> torch.Tensor:
        """
        The weighted adjacency matrix of the SCM.
        """
        return self.weight.sum(dim=2)
