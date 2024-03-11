from abc import abstractmethod
import pytorch_lightning as pl
import torch

from .causalmodel import CausalModel


class Trainable(pl.LightningModule):
    """
    An object of a Trainable class is a PyTorch Lightning Module
    that wraps a CausalModel, i.e. a differentiable PyTorch Module.
    """
    learning_rate: float
    module: CausalModel

    def __init__(self,
                 module: CausalModel,
                 learning_rate: float):
        """
        Abstract class to implement a trainable model.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use.
        """
        super().__init__()
        self.module = module
        self.learning_rate = learning_rate

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        return self.module(x_batch)

    def configure_optimizers(self):
        """
        Configures Adam as the default optimizer.
        """
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate)

    def reset_optimizers(self):
        assert self.trainer is not None
        self.trainer.optimizers = [self.configure_optimizers()]

    @abstractmethod
    def step(self, batch: torch.Tensor, _: int, split: str = 'Train') \
            -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch: torch.Tensor, batch_idx: int) \
            -> torch.Tensor:
        return self.step(batch, batch_idx, split='Train')
