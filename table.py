"""
The scrtip builds the necessary LaTeX tables
for each dataset and each model.
"""

import os
import pandas as pd
import sys

# Constants
COLUMNS = ['eval/NHD', 'eval/TPR', 'eval/roc_auc', 'eval/time']
COLUMNS_HEAD = [r'\nhd', r'\tpr', r'\auc', r'Time (s)']
COLUMNS_ASC = [False, True, True, False]
COLUMNS_NAMES = [col.split('/')[1] for col in COLUMNS]


def main(NODES, MODELS, MODELS_TEX, NOISE, KEEP_BEST):
    # Create the directory for the tables
    if not os.path.exists('tables'):
        os.makedirs('tables')

    # Iterate on each dataset
    for edges in ['ER4', 'ER6', 'SF4', 'SF6']:
        for noise in NOISE:
            # Create the file
            filename = f'tables/{edges}_{noise}.tex'
            filepointer = open(filename, 'w')
            filepointer.write(r'\begin{tabular}{clrrrr}' + '\n')
            filepointer.write(r'  \toprule' + '\n')
            filepointer.write(
                r'  $d$ & Algorithm & \nhd & \tpr & \auc & Time (s)\\' + '\n')
            filepointer.write(r'  \midrule' + '\n')
            for n_nodes in NODES:
                # Create the dictionary for the results
                results = {col: {} for col in COLUMNS}
                # Compute the best models
                best = {}
                second_best = {}

                # Iterate over the models
                for model in MODELS:
                    run_id = f'n1000_d{n_nodes}_{edges}_{noise}_{model}'
                    val_fname = f'validation/{run_id}.csv'
                    try:
                        # Load results
                        dataframe = pd.read_csv(val_fname, index_col=0)
                        # Normalize the SHD
                        dataframe['eval/NHD'] = dataframe['eval/SHD'] / n_nodes
                        # Iterate over metrics
                        for col in COLUMNS:
                            # Populate the results dictionary
                            results[col][model] = \
                                (dataframe[col].mean(), dataframe[col].std())
                    except FileNotFoundError:
                        continue

                # Compute the best model for each metric
                for col in COLUMNS:
                    try:
                        # Select the models with results
                        models = list(results[col].keys())
                        # Sort the models by the metric
                        sorted_models = sorted(
                            models,
                            key=lambda x: results[col][x][0],
                            reverse=COLUMNS_ASC[COLUMNS.index(col)]
                        )
                        # Get the best and second best models
                        best[col] = sorted_models[0]
                        if len(sorted_models) > 1:
                            second_best[col] = sorted_models[1]
                    except KeyError:
                        continue

                # Get the list of available models (test on the first metric)
                models = list(results[COLUMNS[0]].keys())
                # Count models
                n_models = len(models)
                # Iterate on models
                first = True
                for model in models:
                    if first:
                        first = False
                        filepointer.write(r'  \multirow{' + str(n_models) +
                                          r'}{*}{' + str(n_nodes) + r'} & ' +
                                          MODELS_TEX[MODELS.index(model)] +
                                          ' &\n')
                    else:
                        filepointer.write(r'  & ' +
                                          MODELS_TEX[MODELS.index(model)] +
                                          ' &\n')
                    for col in COLUMNS:
                        # Check if last column
                        if col == COLUMNS[-1]:
                            end = r'\\'
                        else:
                            end = r'&'

                        # Comment
                        comment = f'{model}_n1000_d{n_nodes}_{edges}_{noise}' \
                                  + f'_{model}' \
                                  + f'_{COLUMNS_NAMES[COLUMNS.index(col)]}'
                        # Get the mean and std
                        mean, std = results[col][model]
                        # Keep only 3 decimals for the mean
                        # and 2 for the std. If the column
                        # is the time, keep no decimals.
                        if col == 'eval/time':
                            mean = f'{mean:.0f}'
                            std = f'{std:.0f}'
                        else:
                            mean = f'{mean:.3f}'
                            std = f'{std:.2f}'

                        # Check if the model is the best
                        if model == best[col] and KEEP_BEST:
                            # Write the mean and std
                            filepointer.write(
                                            r'  \best{'
                                            + f'{mean}'
                                            + r' $\pm$ '
                                            + f'{std}'
                                            + r'}'
                                            + f' {end} %{comment}\n')
                        elif model == second_best[col] and KEEP_BEST:
                            # Write the mean and std
                            filepointer.write(
                                            r'  \rest{'
                                            + f'{mean}'
                                            + r' $\pm$ '
                                            + f'{std}'
                                            + r'}'
                                            + f' {end} %{comment}\n')
                        else:
                            # Write the mean and std
                            filepointer.write(
                                            f'  {mean}'
                                            + r' $\pm$ '
                                            + f'{std}'
                                            + f' {end} %{comment}\n')
                if n_nodes != NODES[-1]:
                    filepointer.write(r'  \midrule' + '\n')
            filepointer.write(r'  \bottomrule' + '\n')
            filepointer.write(r'\end{tabular}')


if __name__ == '__main__':
    if sys.argv[1] == 'linear':
        nodes = [30, 100, 500]
        models = ['cosmo', 'dagma', 'nocurl', 'nocurl_joint', 'notears']
        models_tex = [r'\underline{\cosmo}', r'\dagma', r'\nocurl',
                      r'\nocurljoint', r'\notears']
        noise = ['gauss', 'exp', 'gumbel']
        keep_best = True
    elif sys.argv[1] == 'nonlinear':
        nodes = [20, 40, 100]
        models = ['cosmo_nl', 'dagma_nl']
        models_tex = [r'\underline{\cosmo}', r'\dagma']
        noise = ['mlp']
        keep_best = False
    else:
        print('Usage: python3 tables.py [linear|nonlinear]', file=sys.stderr)
        sys.exit(1)

    main(nodes, models, models_tex, noise, keep_best)
