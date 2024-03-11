from typing import Callable

import numpy as np
import sklearn.metrics as skm


_metrics: dict[str, Callable] = {
    'FDR': lambda y_true, y_pred: 1 - skm.precision_score(y_true, y_pred),
    'HD': skm.hamming_loss,
    'PPV': skm.precision_score,
    'SHD': lambda y_true, y_pred:
        skm.hamming_loss(y_true, y_pred) * len(y_true),
    'NHD': lambda y_true, y_pred:
        skm.hamming_loss(y_true, y_pred) * np.sqrt(len(y_true)),
    'TPR': skm.recall_score,
    'TNR': lambda y_true, y_pred:
        skm.recall_score(y_true, y_pred, pos_label=0),
    'FPR': lambda y_true, y_pred:
        1 - skm.recall_score(y_true, y_pred, pos_label=0),
    'ROCAUC': skm.roc_auc_score,
    'F1': skm.f1_score,
    'ACC': skm.accuracy_score,
    'ROC': skm.roc_curve
}


def get_metric(metric_name: str) -> Callable:
    """
    Wrapper for various metrics on NumPy arrays from Scikit-learn.
    """
    return _metrics[metric_name]


# class EvaluationCallback(pl.Callback):
#     def __init__(self, algorithm: CausalDiscoveryAlgorithm,
#                  ground_truth: CausalGraph,
#                  logging_interval: int = 1):
#         """
#         Callback to evaluate the algorithm on the ground truth graph
#         at the end of each epoch. The evaluation is performed only
#         every logging_interval epochs.

#         Parameters
#         ----------
#         algorithm: CausalDiscoveryAlgorithm
#             The algorithm to evaluate.
#         ground_truth: CausalGraph
#             The ground truth graph.
#         logging_interval: int
#             The interval between two evaluations.
#         """
#         self.algorithm = algorithm
#         self.ground_truth = ground_truth
#         self.logging_interval = logging_interval

#         # Call super constructor
#         super().__init__()

#     def on_train_epoch_end(self,
#                            trainer: pl.Trainer,
#                            _: CausalModel):
#         # Check if we need to log
#         if trainer.current_epoch % self.logging_interval == 0:
#             # Compute the causal graph
#             self.algorithm._mask_weights()  # type: ignore
#         # with torch.no_grad():
#             assert trainer.logger is not None
#             # Get metrics dictionary
#             metrics = self.algorithm.evaluate(self.ground_truth)
#             # Remove keys 'fpr' and 'tpr'
#             metrics.pop('fpr')
#             metrics.pop('tpr')
#             # Add prefix 'test_' to the keys
#             metrics = {f'test_{k}': v for k, v in metrics.items()}
#             # Log metrics
#             trainer.logger.log_metrics(metrics, step=trainer.global_step)
