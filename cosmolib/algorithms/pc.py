"""
PC Algorithm
A combinatorial causal discovery algorithm based on conditional independence
tests.  Starting from a complete undirected graph, PC consists of two phases:
(i)  it removes edges recursively according to the outcome of the conditional
     independence tests;
(ii) it applies various rules to orient dependencies.

References
----------
https://www.jmlr.org/papers/volume8/kalisch07a/kalisch07a.pdf

"""
from copy import deepcopy
from itertools import combinations, permutations

import numpy as np

from ..utils import fisher_z_test, find_adjacent_vertices
from ..datasets import CausalDataset
from .generic import CausalDiscoveryAlgorithm


class PC(CausalDiscoveryAlgorithm):
    """
    PC Algorithm
    """

    def __init__(self,
                 ci_test: str = 'fisherz',
                 alpha: float = 0.05):

        # Input dimensionality
        self.alpha = alpha
        self.ci_test = ci_test
        self._columns = None

    def learn(self, data):
        # phase 1
        causal_skeleton, separation_sets = self._find_skeleton(data)
        # phase 2
        cpdag = self._orient_undirected_graph(causal_skeleton, separation_sets)

        return cpdag

    def fit(self, dataset: CausalDataset):
        """
        Given the provided dataset, the algorithm learns the causal graph
        and sets the proper attribute.
        """
        self.causal_graph = self.learn(dataset.observations)

    def _find_skeleton(self, data):
        """
        Parameters
        ----------
        data: np.ndarray
            training data

        Returns
        -------
        Causal skeleton: array
          The undirected graph
        Separation sets: dict
          Such as key is (x, y), then value is a set of other variables
          not contains x and y.
        """
        n_feature = data.shape[1]
        skeleton = np.ones((n_feature, n_feature)) - np.eye(n_feature)
        separation_sets = dict()
        depth = -1
        # maximum number of edges connected to a node in the adjacency matrix
        max_degree = max(np.sum(skeleton != 0, axis=1))
        while max_degree - 1 > depth:
            depth += 1
            adjacent_vertices = find_adjacent_vertices(skeleton)
            for x, y in adjacent_vertices:
                # if there is no edges,continue
                if skeleton[x, y] == 0:
                    continue
                neighbors_x = set(np.argwhere(skeleton[x] == 1).reshape(-1, ))
                # check adjacent vertices excluding y
                neighbors_x_without_y = neighbors_x - {y}
                if len(neighbors_x_without_y) >= depth:
                    for sub in combinations(neighbors_x_without_y, depth):
                        sub = list(sub)
                        if self.ci_test != 'fisherz':
                            raise ValueError('Unknown conditional independence test. '
                                             '{} test has not been implemented.'.format(self.ci_test))
                        else:
                            _, _, p_value = fisher_z_test(data, x, y, sub)
                        if p_value >= self.alpha:
                            # remove edges
                            skeleton[x, y] = skeleton[y, x] = 0
                            separation_sets[(x, y)] = sub
                            break

        return skeleton, separation_sets

    def _orient_undirected_graph(self, skeleton, separation_sets):
        """
        It orients the undirected edges to form an equivalence class of DAGs.

        Parameters
        ----------
        skeleton : np.ndarray
            The causal skeleton of the true causal graph.
        separation_sets : dict
            The separation sets.

        Returns
        -------
            Completed Partially Directed Acyclic Graph (CPDAG) of the true causal graph

        """
        self._columns = list(range(skeleton.shape[1]))
        cpdag = deepcopy(abs(skeleton))
        for x, y in separation_sets.keys():
            all_k = [col for col in self._columns if col not in (x, y)]
            for k in all_k:
                if cpdag[x, k] + cpdag[k, x] != 0 \
                        and cpdag[k, y] + cpdag[y, k] != 0:
                    if k not in separation_sets[(x, y)]:
                        if cpdag[x, k] + cpdag[k, x] == 2:
                            cpdag[k, x] = 0
                        if cpdag[y, k] + cpdag[k, y] == 2:
                            cpdag[k, y] = 0

        old_dag = np.zeros((cpdag.shape[1], cpdag.shape[1]))

        while not np.array_equal(old_dag, cpdag):
            old_dag = deepcopy(cpdag)
            pairs = list(combinations(self._columns, 2))
            for i, j in pairs:
                # check if it has both edges
                if cpdag[i, j] == 1 and cpdag[j, i] == 1:
                    # rules implementation is based on gcastle
                    cpdag = self._rule1(cpdag, i, j)
                    cpdag = self._rule2(cpdag, i, j)
                    cpdag = self._rule3(cpdag, i, j, separation_sets)
                    cpdag = self._rule4(cpdag, i, j, separation_sets)

        return cpdag

    def _rule1(self, cpdag, i, j):
        """
        Orient i-j into i->j whenever there is an arrow k->i
        such that k and j are nonadjacent.
        """
        for i, j in permutations((i, j), 2):
            all_k = [x for x in self._columns if x not in (i, j)]
            for k in all_k:
                # if there is an arrow i -> k
                if cpdag[k, i] == 1 and cpdag[i, k] == 0 \
                        and cpdag[k, j] + cpdag[j, k] == 0:
                    cpdag[j, i] = 0
        return cpdag

    def _rule2(self, cpdag, i, j):
        """
        Orient i-j into i->j whenever there is a chain i->k->j.
        """
        for i, j in permutations((i, j), 2):
            all_k = [x for x in self._columns if x not in (i, j)]
            for k in all_k:
                if cpdag[k, i] == 0 and cpdag[i, k] == 1 \
                        and (cpdag[k, j] == 1 and cpdag[j, k] == 0):
                    cpdag[j, i] = 0
        return cpdag

    def _rule3(self, cpdag, i, j, separation_sets):
        """
        Orient i-j into i->j whenever there are two chains i-k->j and i-l->j
        such that k and l are nonadjacent.
        """
        for i, j in permutations((i, j), 2):
            for kl in separation_sets.keys():  # k and l are nonadjacent.
                k, l = kl
                if cpdag[i, k] == 1 \
                        and cpdag[k, i] == 1 \
                        and cpdag[k, j] == 1 \
                        and cpdag[j, k] == 0 \
                        and cpdag[i, l] == 1 \
                        and cpdag[l, i] == 1 \
                        and cpdag[l, j] == 1 \
                        and cpdag[j, l] == 0:
                    cpdag[j, i] = 0
        return cpdag

    def _rule4(self, cpdag, i, j, separation_sets):
        """
        Orient i-j into i->j whenever there are two chains i-k->l and k->l->j
        such that k and j are nonadjacent.
        """

        for i, j in permutations((i, j), 2):
            for kj in separation_sets.keys():  # k and l are nonadjacent.
                if j not in kj:
                    continue
                else:
                    kj = list(kj)
                    kj.remove(j)
                    k = kj[0]
                    ls = [x for x in self._columns if x not in [i, j, k]]
                    for l in ls:
                        if cpdag[k, l] == 1 \
                                and cpdag[l, k] == 0 \
                                and cpdag[i, k] == 1 \
                                and cpdag[k, i] == 1 \
                                and cpdag[l, j] == 1 \
                                and cpdag[j, l] == 0:
                            cpdag[j, i] = 0
        return cpdag
