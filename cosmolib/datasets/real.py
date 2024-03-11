import io
import os
import zipfile

import networkx as nx
import numpy as np
import pandas as pd
import requests

from .generic import CausalDataset


class DreamDataset(CausalDataset):
    """
    Dream Series Dataset: Simulated and in-vivo gene regulation networks.
    Data generated with GeneNetWeaver 2.0, 5 graphs of 100 variables x 100
    samples.

    Darbach D, Prill RJ, Schaffter T, Mattiussi C, Floreano D,
    and Stolovitzky G.
    Revealing strengths and weaknesses of methods for gene network inference.
    PNAS, 107(14):6286-6291, 2010.
    """

    def __init__(self, num: int = 1, normalize: bool = True):
        # TODO: let the user choose the resources location
        # Choose graph
        self.serial_number = num

        # Read samples
        self.observations = self._load_data()

        # Read edges
        dirname = os.path.dirname(os.path.realpath(__file__))
        self.edges = pd.read_csv(
            '{}/resources/dream_edges_{}.csv'.format(dirname, num))

        # Build ground truth
        self.causal_graph = self.get_ground_truth()
        self.n_nodes = self.causal_graph.shape[0]

        # Super constructor
        super().__init__(normalize)

    def _load_data(self) -> np.ndarray:
        """
        Downloads the remote dataset and loads it in memory.
        """
        # Retrieve remote dataset
        data = requests.get('https://www.synapse.org/Portal/filehandle?'
                            'ownerId=syn3049712&ownerType=ENTITY&fileName'
                            '=DREAM4_InSilico_Size100_Multifactorial.zip&'
                            'preview=false&wikiId=74630')
        # Unzip and load in memory
        # TODO: cache locally
        with zipfile.ZipFile(io.BytesIO(data.content)) as f:
            for name in f.namelist():
                if name == 'insilico_size100_{}_multifactorial.tsv' \
                            .format(self.serial_number):
                    data = pd.read_csv(f.open(name), sep='\t')
                    return np.array(data.values, dtype=np.float32)
        raise ValueError

    def get_ground_truth(self):
        """
        Builds the ground truth matrix from the edge connnections.
        """
        graph = nx.DiGraph()
        for idx, row in self.edges.iterrows():
            graph.add_edge(row['Cause'], row['Effect'])
        return nx.to_numpy_array(graph).astype(int)


class SachsDataset(CausalDataset):
    """
    Sachs Dataset: proteins and phospholipids in human cells.
    11 variables x 7466 samples;
    Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A.,
    & Nolan, G. P. (2005).
    Causal protein-signaling networks derived from multiparameter
    single-cell data.
    Science, 308(5721), 523-529.
    """

    def __init__(self, normalize: bool = True):
        # Loads observations
        dirname = os.path.dirname(os.path.realpath(__file__))
        data = pd.read_csv('{}/resources/sachs.csv'.format(dirname))
        self.observations = np.array(data.values, dtype=np.float32)

        # Loads edges
        self.edges = pd.read_csv(
            '{}/resources/sachs_edges.csv'.format(dirname))

        # Build ground truth
        self.causal_graph = self.get_ground_truth()
        self.n_nodes = self.causal_graph.shape[0]

        # Super constructor
        super().__init__(normalize)

    def get_ground_truth(self, return_dag=False):
        graph = nx.DiGraph()
        for idx, row in self.edges.iterrows():
            graph.add_edge(row['Cause'], row['Effect'])
        return nx.to_numpy_array(graph).astype(int)
