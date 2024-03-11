"""
Grid search across hyperparameters for different CCD Models.
"""
from typing import Optional
import logging
import os
import sys

from pytorch_lightning import seed_everything
from ray import tune
import pandas as pd
import ray
import yaml

from benchmark.run import _iteration_run
from benchmark.run import iteration_run
from benchmark.run import aggregate_experiment


def yaml_to_raytune(yaml_config: dict) -> dict:
    """
    Given a YAML configuration file for a grid search,
    returns a dictionary compatible with Ray Tune.
    Each entry in the dictionary is a hyperparameter name,
    and the value is either dictionary with one of the
    following keys:
    - `grid_search`: list of values to search over
    - `loguniform`: tuple of (min, max) values to search over
    or a single value.
    """
    search_space = {}
    for key, value in yaml_config.items():
        if isinstance(value, dict):
            if 'grid_search' in value:
                search_space[key] = tune.grid_search(value['grid_search'])
            elif 'loguniform' in value:
                search_space[key] = tune.loguniform(*value['loguniform'])
            else:
                raise ValueError(f'Invalid configuration {value}')
        else:
            search_space[key] = value
    return search_space


def grid_search(algorithm_name: str, dataset_name: str, config_path: str,
                n_repetitions: int = 5, seed: Optional[int] = None,
                grid_metric: str = 'eval/roc_auc',
                grid_metric_mode: str = "max",
                n_grid_samples: int = 40, num_cpus: int = 8,
                validation_directory: str = 'validation',
                verbose: bool = False,
                **alg_kwargs):
    """
    The function repeatedly applies a continuous causal discovery method
    to a dataset to evaluate the best hyperparameters.
    """
    # Load configuration
    try:
        with open(config_path, 'r') as fp:
            yaml_grid = yaml.safe_load(fp)
    except FileNotFoundError:
        logging.error(f'Configuration file {config_path} not found.')
        sys.exit(1)

    # Get config file name, without extension
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    # Build the search space
    search_space = yaml_to_raytune(yaml_grid)

    # Eventually seed everything
    if seed is not None:
        seed_everything(seed)

    # Initialize ray
    if not ray.is_initialized():  # type: ignore
        ray.init(num_cpus=num_cpus)  # type: ignore

    # Wrapper function
    def wrapper(config):
        # Add search space to algorithm kwargs
        configured_alg_kwargs = {**alg_kwargs, **config}

        # Build experiments
        experiments = [
            [algorithm_name, dataset_name, 'default',
             configured_alg_kwargs, run_iter,
             n_repetitions, verbose,
             False]
            for run_iter in range(n_repetitions)]

        # Run experiments in parallel
        # results = [iteration_run.remote(*experiment)
        #            for experiment in experiments]
        # results = aggregate_experiment.remote(*results)  # type: ignore
        # results = ray.get(results)  # type: ignore

        # Run experiments sequentially
        results = [_iteration_run(*experiment) for experiment in experiments]

        # Build dataframe
        dataframe = pd.DataFrame(results)

        # Return averaged metrics
        return dataframe.mean(numeric_only=True).to_dict()

    # Configure analysis
    analysis = tune.run(
        tune.with_parameters(wrapper),
        # resources_per_trial=tune.PlacementGroupFactory(
        #     [{'CPU': 1.0}] + [{'CPU': 1.0}] * n_repetitions
        # ),
        resources_per_trial={'cpu': 1, 'gpu': 0},
        metric=grid_metric,
        mode=grid_metric_mode,
        config=search_space,
        num_samples=n_grid_samples,
        name=f"tune_{dataset_name}_{config_name}"
    )

    # Print best configuration
    print(analysis.best_config)
    print(analysis.best_result)

    # Store results
    results: pd.DataFrame = analysis.results_df
    results_fname = f"grids/{dataset_name}_{config_name}.csv"
    results.to_csv(results_fname)
    print(f"Results saved to {results_fname}")

    # Compose best configuration
    best_config = {**alg_kwargs, **analysis.best_config}

    # Compose validation experiments
    validation_experiments = [
        [algorithm_name, dataset_name, 'default',
         best_config, run_iter,
         n_repetitions, verbose, False]
        for run_iter in range(n_repetitions)]

    # Run validation experiments
    remote_results = [iteration_run.remote(*experiment)
                      for experiment in validation_experiments]
    remote_results = aggregate_experiment.remote(*remote_results)
    validation_results = ray.get(remote_results)  # type: ignore

    # Build dataframe
    validation_dframe = pd.DataFrame(validation_results)
    # Create folder if it does not exist
    if not os.path.exists(validation_directory):
        os.makedirs(validation_directory)
    # Build filename
    filename = f'{validation_directory}/{dataset_name}_{config_name}.csv'
    # Save dataframe
    validation_dframe.to_csv(filename, index=False, mode='w')
