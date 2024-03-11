"""
Generic interface for a Causal Model.
"""
import torch


class CausalModel(torch.nn.Module):
    """
    Abstract Structural Causal Model defined as a
    differentiable PyTorch Module.
    """
    weighted_adj: torch.Tensor

    def __init__(self, n_nodes: int):
        """
        Abstract class to implement a causal model.

        Parameters
        ----------
        n_nodes: int
            The number of nodes of the model.
        """
        super().__init__()
        self.n_nodes = n_nodes

    @torch.no_grad()
    def get_parameters(self):
        return [p.detach().clone() for p in self.parameters()]

    @torch.no_grad()
    def set_parameters(self, parameters: list):
        for p, new_p in zip(self.parameters(), parameters):
            p[:] = new_p[:]

    @torch.no_grad()
    def store_parameters(self):
        """
        The function store all parameters of the model
        in a local property called old_parameters.
        """
        self.old_parameters = self.get_parameters()

    @torch.no_grad()
    def restore_parameters(self):
        """
        The function restore all parameters of the model
        from the local property called old_parameters.
        """
        self.set_parameters(self.old_parameters)


StructuralCausalModel = CausalModel
