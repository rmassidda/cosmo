"""
This module contains the generic classes to implement both
combinatorial and continuous causal discovery algorithms.
"""
from abc import ABC, abstractmethod
from typing import Optional
from typing import Union
import math
import time

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl

from ..datasets import CausalGraph, CausalDataset
from ..metrics import get_metric
from ..utils import dag_constraint, mask_weights
from ..models import CausalModel
from ..models.trainable import Trainable


class CausalDiscoveryAlgorithm(ABC):
    """
    The abstract class describes the skeleton of a Causal Discovery Algorithm.

    Attributes
    ----------
    causal_graph: np.ndarray
        The causal graph estimated by the algorithm.
    execution_time: float
        The execution time of the algorithm.

    Methods
    -------
    _fit(dataset: CausalDataset)
        Executes the algorithm on the given dataset and returns
        the estimated causal graph.
    fit(dataset: CausalDataset)
        Wraps the method _fit() and stores the estimated causal
        graph in the attribute `causal_graph`.
    evaluate(gt_graph: CausalGraph)
        Evaluates the algorithm on the ground truth graph.
    """
    causal_graph: CausalGraph
    execution_time: float

    def __init__(self):
        # Execution time
        self.execution_time = 0.0

    @abstractmethod
    def _fit(self, dataset: CausalDataset) -> CausalGraph:
        """
        The method executes the algorithm on the given dataset
        and returns the estimated causal graph.
        """
        raise NotImplementedError

    def fit(self, dataset: CausalDataset) -> None:
        """
        The method executes the algorithm on the given dataset.
        The resulting causal graph is stored in the attribute
        `causal_graph`. The execution time is stored in the
        attribute `execution_time`. The internal method `_fit`
        is called to execute the algorithm. It is implemented
        by the subclasses.

        Parameters
        ----------
        dataset: CausalDataset
            The dataset on which the algorithm is executed.
        """
        # Start timer
        start_time = time.process_time()
        # Execute algorithm
        self.causal_graph = self._fit(dataset)
        # Stop timer
        end_time = time.process_time()
        # Compute execution time
        self.execution_time = end_time - start_time

    def evaluate(self, gt_graph: CausalGraph) -> dict:
        """
        Evaluates the algorithm on the ground truth graph.
        """
        # Flatten arrays
        mask_true = gt_graph.flatten()
        mask_pred = self.causal_graph.flatten()

        _metrics = ['NHD', 'SHD', 'TPR', 'FDR', 'PPV', 'F1', 'ACC', 'FPR']

        # Compute metrics on predicted mask
        run_metrics = {met_name: get_metric(met_name)(mask_true, mask_pred)
                       for met_name in _metrics}

        return run_metrics


class EvaluationCallback(pl.Callback):
    def __init__(self, algorithm: CausalDiscoveryAlgorithm,
                 ground_truth: CausalGraph,
                 logging_interval: int = 1):
        """
        Callback to evaluate the algorithm on the ground truth graph
        at the end of each epoch. The evaluation is performed only
        every logging_interval epochs.

        Parameters
        ----------
        algorithm: CausalDiscoveryAlgorithm
            The algorithm to evaluate.
        ground_truth: CausalGraph
            The ground truth graph.
        logging_interval: int
            The interval between two evaluations.
        """
        self.algorithm = algorithm
        self.ground_truth = ground_truth
        self.logging_interval = logging_interval

        # Call super constructor
        super().__init__()

    def on_train_epoch_end(self,
                           trainer: pl.Trainer,
                           _: Trainable):
        # Check if we need to log
        if trainer.current_epoch % self.logging_interval == 0:
            # Compute the causal graph
            self.algorithm._mask_weights()  # type: ignore
        # with torch.no_grad():
            assert trainer.logger is not None
            # Get metrics dictionary
            metrics = self.algorithm.evaluate(self.ground_truth)
            # Remove keys 'fpr' and 'tpr'
            metrics.pop('fpr')
            metrics.pop('tpr')
            # Add prefix 'test_' to the keys
            metrics = {f'test_{k}': v for k, v in metrics.items()}
            # Log metrics
            trainer.logger.log_metrics(metrics, step=trainer.global_step)


class ConvergenceStop(pl.Callback):
    def __init__(self,
                 tolerance: float = 1e-6,
                 interval: int = 100,
                 metric: str = 'Train_Step'
                 ):
        """
        Callback to evaluate whether the algorithm has converged
        at the end of a fixed number of training epochs. Convergence
        is estimated by comparing the absolute difference between
        the current and previous values of the loss function,
        normalized by the latter.

        Parameters
        ----------
        tolerance: float
            The tolerance to use to check convergence.
        interval: int
            The interval between two evaluations.
        metric: str
            The metric to use to check convergence.
        """
        self.tolerance = tolerance
        self.interval = interval
        self.metric = metric
        self.previous_loss = 1e6

        # Call super constructor
        super().__init__()

    def on_train_epoch_end(self,
                           trainer: pl.Trainer,
                           model: Trainable):
        # Eventually return
        if trainer.current_epoch % self.interval != 0 or \
                trainer.current_epoch == 0:
            return

        # Get current loss
        logs = trainer.callback_metrics
        current = logs[self.metric].squeeze()

        # Absolute ratio of change
        convergence_val = self.previous_loss - current
        convergence_val = convergence_val / self.previous_loss
        convergence_val = abs(convergence_val)

        # Log convergence ratio
        model.log('Convergence', convergence_val)

        # Update objective function
        self.previous_loss = current

        # Check convergence
        if convergence_val < self.tolerance:
            trainer.should_stop = True


class ContinuousCausalDiscoveryAlgorithm(CausalDiscoveryAlgorithm):
    """
    The class implements the skeleton of a generic continuous causal
    discovery algorithm.

    To implement a new algorithm, you need to implement the method
    _build_model() that returns a PyTorch Lightning model which
    exposes the method get_weighted_adjacency_matrix() returning
    the torch.Tensor of the estimated weighted adjacency matrix.
    Further, if the method requires further aspects to be implemented,
    you can override the method _fit().

    Attributes
    ----------
    causal_graph: np.ndarray
        The causal graph estimated by the algorithm.
    causal_model: Any
        The causal model estimated by the algorithm.
    threshold: float
        The threshold used to binarize the estimated causal graph.
    force_dag: bool
        Whether to force the estimated causal graph to be a DAG.
    n_workers: int
        The number of workers used to load the data.
    batch_size: int
        The batch size used to train the model.
    max_epochs: int
        The maximum number of epochs to train the model.
    exp_name: str
        The name of the experiment, used for logging.
    version: Optional[str]
        The version of the experiment, used for logging.
    monitor: bool
        Whether to enable monitoring and logging of the training.
    verbose: bool
        Whether to enable verbose printing to the console.

    Methods
    -------
    fit(dataset: CausalDataset)
        Executes the algorithm on the given dataset.
    evaluate(gt_graph: CausalGraph)
        Evaluates the algorithm on the given dataset.
    """
    causal_graph: CausalGraph
    causal_model: CausalModel

    def __init__(self,
                 threshold: float = 0.3,
                 force_dag: bool = False,
                 n_workers: int = 0,
                 batch_size: int = 32,
                 max_epochs: int = 20,
                 early_stop: bool = True,
                 exp_name: str = 'default',
                 version: Optional[str] = None,
                 monitor: bool = False,
                 verbose: bool = False,
                 gpus: Optional[Union[int, list[int]]] = None,
                 ):

        # Super constructor
        super().__init__()

        # Thresholding parameters
        self.threshold = threshold
        self.force_dag = force_dag

        # Training parameters
        self.num_workers = n_workers
        self.batch_size = batch_size
        self.exp_name = exp_name
        self.version = version
        self.max_epochs = max_epochs
        self.early_stop = early_stop

        # Monitoring parameters
        self.verbose = verbose
        self.monitor = monitor

        # GPU parameters
        if gpus is not None and not isinstance(gpus, list):
            gpus = [gpus]
        self.gpus = gpus

    @abstractmethod
    def _build_model(self, dataset: CausalDataset) -> CausalModel:
        raise NotImplementedError

    @abstractmethod
    def _wrap_model(self) -> Trainable:
        raise NotImplementedError

    def _mask_weights(self) -> CausalGraph:
        # Retrieve weighted tensor
        self.weighted_adj = self.causal_model.weighted_adj
        # Compute the causal graph
        self.causal_graph = mask_weights(
            self.weighted_adj.detach().clone().cpu().numpy(),
            self.threshold, self.force_dag)
        return self.causal_graph

    def _get_loader(self, dataset: CausalDataset) -> DataLoader:
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def _get_logger(self,
                    version_suffix: Optional[str] = None) -> TensorBoardLogger:
        # Set version
        if version_suffix:
            version = f"{self.version}_{version_suffix}"
        else:
            version = self.version

        return TensorBoardLogger("logs/", name=self.exp_name,
                                 default_hp_metric=False, version=version)

    def _get_trainer(self,
                     dataset: CausalDataset,
                     version_suffix: Optional[str] = None) -> pl.Trainer:
        """
        Returns a PyTorch Lightning trainer configured with the
        parameters of the algorithm. If the parameter `monitor`
        is set to True, the trainer is configured to log the
        training to TensorBoard. A custom version suffix can
        be passed as argument to distinguish different phases
        of the algorithm execution.

        Parameters
        ----------
        dataset: CausalDataset
            The dataset on which the algorithm is executed.
        version_suffix: Optional[str]
            A custom version suffix to distinguish different phases.
        """
        # Eventually set logger
        if self.monitor:
            logger = self._get_logger(version_suffix)
            log_interval = min(math.floor(len(dataset) / self.batch_size), 50)
        else:
            logger = False
            log_interval = 1

        # Set callbacks
        callbacks = []
        if self.early_stop:
            # callbacks.append(EarlyStopping('Train_Step'))
            callbacks.append(ConvergenceStop())
        if self.monitor:
            callbacks.append(EvaluationCallback(self, dataset.causal_graph))

        # training
        trainer = pl.Trainer(
                             logger=logger,
                             max_epochs=self.max_epochs,
                             callbacks=callbacks,
                             enable_progress_bar=self.verbose,
                             enable_model_summary=self.verbose,
                             log_every_n_steps=log_interval,
                             enable_checkpointing=False,
                             accelerator='gpu' if self.gpus else None,
                             devices=self.gpus,
                             )

        return trainer

    def _fit(self, dataset: CausalDataset) -> CausalGraph:
        # Build model
        self.causal_model = self._build_model(dataset)
        # Wrap model
        trainable_wrapper = self._wrap_model()
        # Load data
        loader = self._get_loader(dataset)
        # Get trainer
        trainer = self._get_trainer(dataset)
        # Fit model
        trainer.fit(trainable_wrapper, loader)

        return self._mask_weights()

    def evaluate(self, gt_graph: CausalGraph) -> dict:
        # Evaluate using super metrics
        super_evaluation = super().evaluate(gt_graph)

        # Compute DAGness
        dagness = dag_constraint(self.weighted_adj).item()

        # Flatten arrays
        mask_true = gt_graph.flatten()
        weight_pred = self.weighted_adj.detach().cpu().numpy().flatten()

        # Compute ROC curve
        fpr, tpr, _ = get_metric('ROC')(mask_true, np.abs(weight_pred))
        # Interpolate in the (0, 1) range
        base_fpr = np.linspace(0, 1, 101)
        tpr = np.interp(base_fpr, fpr, tpr)
        # Force TPR to zero with the higher threshold
        tpr[0] = 0.0

        # Compute roc_auc
        roc_auc_score = get_metric('ROCAUC')(mask_true, np.abs(weight_pred))

        # Aggregate evaluation
        evaluation = {**super_evaluation, 'dagness': dagness,
                      'roc_auc': roc_auc_score,
                      'fpr': base_fpr.tolist(), 'tpr': tpr.tolist(),
                      'time': self.execution_time}
        return evaluation
