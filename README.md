# Supplementary Material of "Constraint-Free Structure Learning with Smooth Acyclic Orientations" @ICLR 2024

## Environment

The conda environment is specified in `envs/environment.yml`.
Therefore,
to install the environment,
run the following command:

```bash
conda env create -f envs/environment.yml
```

## Synthetic Dataset Notation

We use the notation `n{samples}_d{nodes}_{ER\|SF}{edge_factor}_{noise}`
to denote
syntethic datasets.
Therefore,
the dataset `n1000_d100_SF6_gauss` has 1000 samples,
on a scale-free graph with 100 nodes and edge factor 6,
and the noise is gaussian.
We admit the following noise terms:
`gauss`, `exp`, and `gumbel` for the linear case
and
`mlp` for the non-linear case.

## Experimental run

For each experiment,
run the following command:

```bash
python -m benchmark grid {model} {dataset} config/grids/{grid_name} --n_grid_samples={NSAMPLES} --num_cpus={NCPUS} --n_repetitions=5
```

The command performs a grid search, whose results are store in the `grids` folder, and validates the best configuration (according to the ROCAUC metric), whose results are stored in the `validation` folder. Both files are named `{dataset}_{grid_name}.csv`. We include the results of our runs in the grids and validation directories.

The available model names are:

- `cosmo`
- `dagma`
- `nocurl`
- `notears`

The hyperparameter ranges for the grid search in the `config/grids/` directory are:

- `cosmo.yaml`; Linear COSMO
- `cosmo_nl.yaml`; Non-Linear COSMO
- `dagma.yaml`; Linear DAGMA
- `dagma_nl.yaml`; Non-Linear DAGMA
- `nocurl.yaml`; Linear NOCURL
- `nocurl_joint.yaml`; Linear NOCURL-U, i.e., the unconstrained variant.
- `notears.yaml`; Linear NOTEARS

For instance, to run non-linear `cosmo` on `n1000_d20_ER4_mlp`, the command would be:
    
```bash
python -m benchmark grid cosmo n1000_d20_ER4_mlp config/grids/cosmo_nl.yaml --n_grid_samples=200 --num_cpus=50 --n_repetitions=5
```

The `table.ipynb` notebook can be used
to generate
the tables in the paper
by selecting
the number of nodes,
the edge factor
and the noise type.

## Step time

The step time simulation experiment can be replicated for each model by running the following command:

```bash
python steptime.py cosmo
```

The model names are, as before, `cosmo`, `dagma`, `nocurl`, and `notears`.

Each run produces a `{modelname}.csv` file in the `steptime` directory.
Then, the `steptime.ipynb` notebook can be used to concatenate and plot the results across all models.
