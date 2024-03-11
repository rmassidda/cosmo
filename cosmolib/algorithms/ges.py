"""
GES Algorithm
The idea of GES is to greedily search through the space of CPDAGs.
It can be outlined in two steps:
i) it starts with a CPDAG and then adds edges maximising the increase in score
ii) edges are greedily removed until an optimum of the score is reached.

References
----------
https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf
"""

from typing import Tuple, Union

import numpy as np

from .generic import CausalDiscoveryAlgorithm
from ..datasets import CausalDataset
from ..utils import apply_insert, insert_validity, apply_delete, delete_validity, get_cpdag, get_parents, get_NAyx, \
    get_adjacent, get_nodes, generate_subsets


class GES(CausalDiscoveryAlgorithm):

    def __init__(self):
        self.n, self.d = None, None
        self.scatter = None

    def fit(self, dataset: CausalDataset):
        """
        Given the provided dataset, the algorithm learns the causal graph
        and sets the proper attribute.
        """
        self.n, self.d = len(dataset), dataset.observations.shape[1]
        self.scatter = np.cov(dataset.observations, rowvar=False, ddof=0)
        empty_graph = np.zeros((self.d, self.d), dtype=int)  # empty graph
        graph = self.FES(empty_graph)  # Perform the forward search phase
        graph = self.BES(graph)  # Perform the backward search phase
        self.causal_graph = graph

    def FES(self, graph: np.ndarray) -> np.ndarray:
        """
        Forward Equivalence Search

        Parameters
        ----------
        graph: np.ndarray
            The initial graph. Defaults to the empty graph, namely a matrix of zeros

        Returns
        -------
        graph: np.ndarray
            the adjacency matrix of the estimated CPDAG
        """
        while True:
            x, y, T = self.FS(graph)  # Get the best edge to insert
            if x is None or y is None:  # If no edge is found, break the loop
                break
            graph = apply_insert(graph, x, y, T)  # Insert the edge and update the graph
            graph = get_cpdag(graph)  # Convert the pdag into cpdag
        return graph

    def FS(self, graph: np.ndarray) -> Tuple[Union[int, None], Union[int, None], set]:
        """
        Forward Search
        """
        edge, subset = (None, None), set()
        max_score = 0
        for x in range(graph.shape[0]):
            nodes = get_nodes(graph, x)
            for y in nodes:
                T0 = generate_subsets(graph, x, y)
                for T in T0:
                    if not insert_validity(graph, x, y, T):
                        continue
                    parents = get_parents(graph, y)
                    na_yx = get_NAyx(graph, x, y)
                    pa_with_x = parents.union({x}, na_yx, T)
                    pa_without_x = parents.union(na_yx, T)
                    score = self.local_score(pa_with_x, y) - self.local_score(pa_without_x, y)
                    if score > max_score:
                        max_score = score
                        edge = (x, y)
                        subset = T

        return edge[0], edge[1], subset

    def BES(self, graph: np.ndarray) -> np.ndarray:
        """
        Backward Equivalence Search
        """
        while True:
            x, y, H = self.BS(graph)
            if y is None or y is None:
                break
            graph = apply_delete(graph, x, y, H)
            graph = get_cpdag(graph)

        return graph

    def BS(self, graph: np.ndarray) -> Tuple[Union[int, None], Union[int, None], set]:
        """
        Backward Search
        """
        max_score = 0
        edge, subset = (None, None), set()
        for x in range(self.d):
            adjacent = get_adjacent(graph, x)
            for y in adjacent:
                H0 = generate_subsets(graph, x, y)
                for H in H0:
                    if not delete_validity(graph, x, y, H):
                        continue
                    parents = get_parents(graph, y)
                    na_yx = get_NAyx(graph, x, y)
                    na_yx_without_H = na_yx - H
                    pa_with_x = parents.union((na_yx_without_H - {x}))
                    pa_without_x = parents.union(na_yx_without_H)
                    score = (self.local_score(pa_with_x, y) - self.local_score(pa_without_x, y))
                    if score > max_score:
                        max_score = score
                        edge = (x, y)
                        subset = H
        return edge[0], edge[1], subset

    def local_score(self, pa: set, y: int) -> float:
        """
        The function returns the local score of the vertex y given the parent set pa
        """
        sigma = self.scatter[y, y]
        pa = list(pa)
        k = len(pa)
        if k > 0:
            pa_cov = self.scatter[pa, :][:, pa]
            y_cov = self.scatter[y, pa]
            coef = np.linalg.solve(pa_cov, y_cov)
            sigma = sigma - y_cov @ coef
        bic_score = - (self.n * (1 + np.log(sigma)) + (k + 1) * np.log(self.n))
        return bic_score
