import pandas as pd
from cosmolib.algorithms.notears import NOTEARS_Trainable
from cosmolib.algorithms.dagma import DAGMA_Trainable
from cosmolib.models import LinearSCM
from cosmolib.datasets import get_dataset
from cosmolib.algorithms.cosmo import COSMO_Trainable
from cosmolib.models import PriorityLinearSCM
import pytorch_lightning as pl
import os
import time
from typing import List
from torch.utils.data import DataLoader
import logging
import sys

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

ALL_NODES = [10, 20,  50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 24000]
SMALL_NODES = [10, 20,  50, 100, 250, 500, 1000, 2000, 4000]


def timeperstep(model: pl.LightningModule,
                loader: DataLoader,
                max_epochs: int = 5,
                n_repetitions: int = 5) -> List[float]:
    """
    The following function measures the time
    per optimization step for a given model.

    The function returns the average and the variance
    of the time per step.
    """

    # Measure time per step
    times = []
    for n_run in range(n_repetitions):
        # Build trainer
        trainer = pl.Trainer(max_epochs=max_epochs,
                             logger=False,
                             # enable_progress_bar=False,
                             enable_model_summary=False,
                             enable_checkpointing=False)
        # Measure time
        start = time.time()
        trainer.fit(model, loader)
        times.append((time.time() - start)/max_epochs)
        print(
            f'Run {n_run + 1}/{n_repetitions} finished. '
            f'Time per step: {times[-1]}')

    # Return average and variance
    return times


results = []
model_name = sys.argv[1]

if model_name == "cosmo":
    # COSMO
    for nodes in ALL_NODES:
        # Create dataset
        dataset = get_dataset(f'n1000_d{nodes}_ER4_gauss')
        # Create loader
        loader = DataLoader(dataset, batch_size=64,
                            shuffle=True, num_workers=2)

        causal_model = PriorityLinearSCM(
            n_nodes=nodes, temperature=0.001, shift=0.1, hard_threshold=False,
            symmetric=False)
        trainable = COSMO_Trainable(
            causal_model,
            learning_rate=0.001,
            l1_adj_reg=0.1,
            l2_adj_reg=0.1,
            priority_reg=0.0001,
            init_temperature=0.8,
            temperature=0.001,
            anneal='cosine',
        )

        cosmo_results = timeperstep(trainable, loader)
        for run in cosmo_results:
            results = results + [{
                'model': 'cosmo',
                'nodes': nodes,
                'time': run,
            }]
        print(f'Finished COSMO for {nodes} nodes.')
elif model_name == "nocurl":
    # NoCurl
    from cosmolib.models import HodgeLinearSCM
    from cosmolib.algorithms.nocurl import NOCURL_Trainable
    from cosmolib.datasets import get_dataset

    for nodes in ALL_NODES:
        # Create dataset
        dataset = get_dataset(f'n1000_d{nodes}_ER4_gauss')
        # Create loader
        loader = DataLoader(dataset, batch_size=64,
                            shuffle=True, num_workers=2)

        causal_model = HodgeLinearSCM(n_nodes=nodes)
        trainable = NOCURL_Trainable(
            causal_model,
            learning_rate=0.001,
            lambda_reg=0.1
        )

        nocurl_results = timeperstep(trainable, loader)
        for run in nocurl_results:
            results = results + [{
                'model': 'nocurl',
                'nodes': nodes,
                'time': run,
            }]
        print(f'Finished NOCURL for {nodes} nodes.')
elif model_name == "dagma":
    # DAGMA
    for nodes in ALL_NODES:
        # Create dataset
        dataset = get_dataset(f'n1000_d{nodes}_ER4_gauss')
        # Create loader
        loader = DataLoader(dataset, batch_size=64,
                            shuffle=True, num_workers=2)

        causal_model = LinearSCM(n_nodes=nodes, zero_init=True)
        trainable = DAGMA_Trainable(
            causal_model,
            learning_rate=0.0001,
            lambda_reg=0.001,
            path_coefficient=0.1,
            log_det_parameter=0.9
        )

        dagma_results = timeperstep(trainable, loader)
        for run in dagma_results:
            results = results + [{
                'model': 'dagma',
                'nodes': nodes,
                'time': run,
            }]
        print(f'Finished DAGMA for {nodes} nodes.')
elif model_name == "notears":
    # NOTEARS
    for nodes in SMALL_NODES:
        # Create dataset
        dataset = get_dataset(f'n1000_d{nodes}_ER4_gauss')
        # Create loader
        loader = DataLoader(dataset, batch_size=64,
                            shuffle=True, num_workers=2)

        causal_model = LinearSCM(n_nodes=nodes)
        trainable = NOTEARS_Trainable(
            causal_model,
            learning_rate=0.01,
            lambda_reg=0.001,
            penalty=0.1,
            multiplier=0.1
        )

        notears_results = timeperstep(trainable, loader)
        for run in notears_results:
            results = results + [{
                'model': 'notears',
                'nodes': nodes,
                'time': run,
            }]
        print(f'Finished NOTEARS for {nodes} nodes.')

# Convert in dataframe
dataframe = pd.DataFrame(results)
# Create steptime folder if not exists
if not os.path.exists('steptime'):
    os.makedirs('steptime')
# Store dataframe to csv
dataframe.to_csv(f'steptime/{model_name}.csv', index=False)
