from .simulated import SimulatedDataset
from .real import DreamDataset, SachsDataset
from .generic import CausalDataset, CausalGraph


def get_dataset(
        name: str = 'n2000_d10_ER2_gauss',
        normalize: bool = True) -> CausalDataset:
    """
    Parses a dataset name and returns the corresponding dataset.

    Synthetically generated datasets are generated using the syntax
    'n<#samples>_d<#dimensions>_<#graph_type><#edge_factor>_<#noise_type>'
    where samples and dimensions are integers, graph_type is one of
    ER and SF, edge_factor is integer, and noise_type is one of
    'gauss', 'exp', 'uniform', and 'gumbel' for linear datasets
    and 'mlp', 'mim', 'gp' and 'gp-add' for non-linear datasets.

    Real datasets are generated using the syntax 'dream<#num>', where
    num is the serial number, or 'sachs'.

    Parameters
    ----------
    name: class, default='sachs'
        Dataset name.

    Return
    ------
    out: Dataset
        standard training dataset.
    """
    if name == 'sachs':
        return SachsDataset(normalize)
    elif name[:5] == 'dream':
        serial_num = int(name[5])
        return DreamDataset(serial_num, normalize)
    else:
        # Split name
        samples, dimensions, graph, noise_type = name.split('_')
        # Parse Values
        samples = int(samples[1:])
        dimensions = int(dimensions[1:])
        graph_type = graph[:2]
        edge_factor = int(graph[2:])
        # Simulated Dataset
        return SimulatedDataset(
            n_nodes=dimensions, n_edges=dimensions*edge_factor,
            graph_type=graph_type, noise_type=noise_type, n_samples=samples,
            normalize=normalize
        )

    # TODO: catch parsing errors and raise this exception
    # raise ValueError('Unknown dataset name. '
    #                  'The dataset {} has not been registered.'.format(name))


__all__ = [
    'SimulatedDataset',
    'DreamDataset',
    'SachsDataset',
    'get_dataset',
    'CausalDataset',
    'CausalGraph',
]
