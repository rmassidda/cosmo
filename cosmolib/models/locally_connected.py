import torch
import torch.nn as nn
import math

from cosmolib.models import StructuralCausalModel


class LocallyConnected(nn.Module):
    """
    A locally connected layer is a convolutional layer with
    unitary filter size and without shared weights. Therefore,
    each output channel is a different linear combination of
    its inputs.

    Params:
    -------
    num_channels: int
        Number of output channels.
    input_features: int
        Number of input features per channel.
    output_features: int
        Number of output features per channel.
    bias: int
        Whether to use bias in the linear layers.
    """

    def __init__(self,
                 num_channels: int,
                 input_features: int,
                 output_features: int,
                 bias: bool = True):
        super(LocallyConnected, self).__init__()

        self.num_channels = num_channels
        self.input_features = input_features
        self.output_features = output_features

        # Weight matrix [d, m1, m2]
        self.weight = nn.Parameter(torch.Tensor(num_channels,
                                                input_features,
                                                output_features))
        # Bias [d, m2]
        if bias:
            self.bias = nn.Parameter(torch.Tensor(
                num_channels, output_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """
        Initialize the parameters of the linear layers.
        """
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        """
        Applies the locally connected layer to the input.

        The input is a tensor of shape [bs, d, m1] where
        bs is the batch size, d is the number of channels
        and m1 is the number of input features per channel.
        """
        # [bs, d, 1, m2] = [bs, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(input.unsqueeze(dim=2),
                           self.weight.unsqueeze(dim=0))
        # [bs, d, m2]
        out = out.squeeze(dim=2)
        # [bs, d, m2] += [:, d, m2]
        if self.bias is not None:
            out += self.bias
        return out


class LocallyConnectedMLP(StructuralCausalModel):
    """
    Multilayer perceptron.
    """

    def __init__(self,
                 n_nodes: int,
                 hidden_size: list[int],
                 bias: bool = True,
                 zero_init: bool = False):
        super(LocallyConnectedMLP, self).__init__(n_nodes=n_nodes)

        # Dimensionality of the layers
        self.hidden_size = hidden_size + [1]

        # First layer maps to hidden_size[1] for each node
        self.fc1 = torch.nn.Linear(self.n_nodes,
                                   self.n_nodes * hidden_size[0],
                                   bias=bias)

        # Eventually initialize the first layer to zero
        if zero_init:
            nn.init.zeros_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)

        # Remaining fully connected layers map to other hidden sizes
        self.fc2 = torch.nn.ModuleList(
            [LocallyConnected(
                self.n_nodes,
                self.hidden_size[i],
                self.hidden_size[i + 1], bias=bias)
             for i in range(len(self.hidden_size) - 1)])

        self.sqrt_weight = False

    def forward(self, x_batch: torch.Tensor):
        """
        Forward pass of the multilayer perceptron.
        Both the input and the output tensors have
        shape [bs, n_nodes].
        """
        # First fully connected layer
        # [bs, n_nodes] -> [bs, n_nodes * mi]
        x_batch = self.fc1(x_batch)
        # [bs, n_nodes * m1] -> [bs, n_nodes, m1]
        x_batch = x_batch.view(-1, self.n_nodes, self.hidden_size[0])

        # Locally connected layers
        for fc in self.fc2:
            # [bs, n_nodes, m1] -> [bs, n_nodes, m1]
            x_batch = torch.sigmoid(x_batch)
            # [bs, n_nodes, m1] -> [bs, n_nodes, m2]
            x_batch = fc(x_batch)

        # [bs, n_nodes, 1] -> [bs, n_nodes]
        x_batch = x_batch.squeeze(dim=2)

        return x_batch

    @property
    def weighted_adj(self) -> torch.Tensor:
        fc1_weight = self.fc1.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(self.n_nodes, -1, self.n_nodes)
        fc1_weight = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        # Square root of the weight matrix
        if self.sqrt_weight:
            fc1_weight = torch.sqrt(fc1_weight)
        return fc1_weight
