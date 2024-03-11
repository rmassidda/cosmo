"""
The module contains the main function to run the experiments
on causal discovery algorithms.
"""
from datetime import datetime
import logging
import warnings

import pandas as pd
from pytorch_lightning.utilities.warnings \
    import PossibleUserWarning  # type:ignore
import ray
from sklearn.exceptions import UndefinedMetricWarning

from cosmolib.algorithms import get_algorithm, CausalDiscoveryAlgorithm
from cosmolib.datasets import get_dataset, CausalDataset


# Enable current module logging up to INFO level
logging.basicConfig(level=logging.INFO)


def _iteration_run(algorithm_name: str, dataset_name: str,
                   flavor_name: str, config: dict,
                   run_iter: int, n_repetitions: int,
                   verbose: bool, dry_run: bool) -> dict:
    """
    Runs a causal discovery algorithm, identified by its name and flavor,
    on a specific observational dataset.
    """

    # Build experiment name
    exp_name = f'{algorithm_name}_{flavor_name}_{dataset_name}'

    # Add experiment name to config
    # FIXME: this won't work for non-continuous causal discovery algorithms
    config['exp_name'] = exp_name
    config['version'] = f'{datetime.now().strftime("%y%m%d%H%M")}_{run_iter}'

    # Disable PyTorch Lightning logging up to WARNING level
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=PossibleUserWarning)

    # Disable sklearn warnings
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    # Print experiment name
    logging.info(
        f'Experiment {exp_name} - '
        f'Run {run_iter+1}/{n_repetitions} (start)')

    # Load dataset
    dataset: CausalDataset = get_dataset(dataset_name)

    # Load model
    algorithm: CausalDiscoveryAlgorithm = get_algorithm(
        algorithm_name, **config)

    # If dry run, return empty evaluation
    if dry_run:
        return {
            'exp/name': exp_name,
            'exp/iter': run_iter,
            'exp/datetime': pd.Timestamp.now(),
        }

    # Fit algorithm
    algorithm.fit(dataset)

    # Evaluate algorithm
    evaluation = algorithm.evaluate(dataset.causal_graph)

    # Append "eval/" prefix to evaluation keys
    evaluation = {f'eval/{key}': value for key, value
                  in evaluation.items()}

    # Add experiment information
    evaluation['exp/name'] = exp_name
    evaluation['exp/iter'] = run_iter
    evaluation['exp/datetime'] = pd.Timestamp.now()

    # Add dataset information
    evaluation['dset/name'] = dataset_name
    evaluation['dset/nodes'] = dataset.n_nodes
    evaluation['dset/samples'] = len(dataset)

    # Add algorithm information
    evaluation['algo/name'] = algorithm_name
    evaluation['algo/flavor'] = flavor_name

    # Add hyperparameters information
    for key, value in config.items():
        evaluation[f'hparam/{key}'] = value

    logging.info(
        f'Experiment {exp_name} - '
        f'Run {run_iter+1}/{n_repetitions} (end)')

    # Return evaluation
    return evaluation


@ray.remote  # type: ignore
def iteration_run(*args, **kwargs):
    return _iteration_run(*args, **kwargs)


@ray.remote  # type: ignore
def aggregate_experiment(*evaluations):
    # Return list of evaluations
    return list(evaluations)
