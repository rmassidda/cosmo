"""
Utility functions.
"""
from itertools import combinations
import math

from scipy import stats
import networkx as nx
import numpy as np
import torch
# from castle.algorithms.ges.functional.graph import pdag_to_cpdag


def is_dag(adj_mat: np.ndarray) -> bool:
    """Check whether B corresponds to a DAG.
    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(adj_mat))


def threshold_till_dag(adj_mat: np.ndarray) -> np.ndarray:
    """Remove the edges with smallest absolute weight until a DAG is obtained.
    Args:
        B (numpy.ndarray): [d, d] weighted matrix.
    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
        float: Minimum threshold to obtain DAG.
    """
    if is_dag(adj_mat):
        return adj_mat

    # Avoid working on original matrix
    adj_mat = np.copy(adj_mat)

    # Get the indices with non-zero weight
    nonzero_indices = np.where(adj_mat != 0)

    # Each element in the list is a tuple (weight, j, i)
    weight_indices_ls = list(zip(adj_mat[nonzero_indices],
                                 nonzero_indices[0],
                                 nonzero_indices[1]))

    # Sort based on absolute weight
    sorted_weight_indices_ls = sorted(weight_indices_ls,
                                      key=lambda tup: abs(tup[0]))

    # Iterate on sorted indices
    for _, j, i in sorted_weight_indices_ls:
        if is_dag(adj_mat):
            # A DAG is found
            break

        # Remove edge with smallest absolute weight
        adj_mat[j, i] = 0.

    return adj_mat


def mask_weights(weights: np.ndarray,
                 threshold: float = 0.1,
                 relative: bool = False,
                 force_dag: bool = False) -> np.ndarray:
    """
    It returns a copy of the original array.
    """

    # Work on a copy
    weights = weights.copy()

    # Relative comparison
    if relative:
        weights = weights / np.max(weights)

    # Threshold weights
    weights[np.abs(weights) <= threshold] = 0.

    # Eventually force DAGness
    if force_dag:
        weights = threshold_till_dag(weights)

    # Mask weights
    weight_mask = np.abs(weights) > 0
    weight_mask = weight_mask.astype(int)

    return weight_mask


def dag_constraint(weight_adj: torch.Tensor) -> torch.Tensor:
    """
    DAGness measure
    """
    return torch.trace(torch.matrix_exp(
        weight_adj * weight_adj)) - weight_adj.shape[0]


def fisher_z_test(data, x, y, z):
    """
    Fisher's z-transform for conditional independence test from castle library
    """

    n = data.shape[0]
    k = len(z)
    if k == 0:
        r = np.corrcoef(data[:, [x, y]].T)[0][1]
    else:
        sub_index = [x, y]
        sub_index.extend(z)
        sub_corr = np.corrcoef(data[:, sub_index].T)
        # inverse matrix
        try:
            PM = np.linalg.inv(sub_corr)
        except np.linalg.LinAlgError:
            PM = np.linalg.pinv(sub_corr)
        r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))
    cut_at = 0.99999
    r = min(cut_at, max(-1 * cut_at, r))  # make r between -1 and 1

    # Fisher’s z-transform
    res = math.sqrt(n - k - 3) * .5 * math.log1p((2 * r) / (1 - r))
    p_value = 2 * (1 - stats.norm.cdf(abs(res)))

    return None, None, p_value


def find_adjacent_vertices(skeleton):
    return [(x, y) for x, y in combinations(set(range(skeleton.shape[0])), 2)]


def require_nonleaf_grad(torch_variable):
    """
    Records the gradient of a nonleaf variable in the
    grad_nonleaf attribute.
    Solution taken from
    https://discuss.pytorch.org/t/how-do-i-calculate-the-gradients-of-a-non-leaf-variable-w-r-t-to-a-loss-function/5112/2
    """

    def hook(gradient):
        torch_variable.grad_nonleaf = gradient

    torch_variable.register_hook(hook)


def get_NAyx(graph: np.ndarray, x: int, y: int) -> set:
    """
    It returns the set of all the nodes neighbors to y and that are adjacent to x
    """
    return get_neighbors(graph, y) & get_adjacent(graph, x)


def generate_subsets(graph: np.ndarray, x: int, y: int) -> set:
    """
    The function generates all the subsets of the neighbours of the vertex y that are not adjacent to x
    """
    s = get_neighbors(graph, y) - get_adjacent(graph, x)
    for i in range(len(s) + 1):
        for subset in combinations(s, i):
            yield set(subset)


def get_neighbors(graph: np.ndarray, i: int) -> set:
    """
    Find the neighbours of the node at index i in the graph.

    Parameters
    __________
        graph: np.ndarray
        i: int
            index of the node whose neighbours need to be found

    Returns
    _______
        Set of indices representing the neighbours of the node at index i
    """
    from_i = graph[i, :] != 0
    to_i = graph[:, i] != 0
    idx = np.where(np.logical_and(from_i, to_i))[0]
    return set(idx) - {i}


def get_adjacent(graph: np.ndarray, i: int) -> set:
    """
    Find the adjacent of the node at index i in the graph.

    Parameters
    __________
        graph: np.ndarray
        i: int
            index of the node whose adjacent need to be found

    Returns
    _______
        Set of indices representing the adjacent of the node at index i
    """
    from_i = graph[i, :] != 0
    to_i = graph[:, i] != 0
    idx = np.where(np.logical_or(from_i, to_i))[0]
    return set(idx) - {i}


def get_parents(graph: np.ndarray, i: int) -> set:
    """
    Find the parents of the node at index i in the graph.

    Parameters
    __________
        graph: np.ndarray
        i: int
            index of the node whose parents need to be found

    Returns
    _______
        Set of indices representing the parents of the node at index i
    """
    from_i = graph[i, :] != 0
    to_i = graph[:, i] != 0
    idx = np.where(np.logical_and(~from_i, to_i))[0]
    return set(idx) - {i}


def get_nodes(graph: np.ndarray, i: int) -> set:
    """
    Find the indices in the graph where there is no connection from and to node i

    Parameters
    __________
        graph: np.ndarray
        i: int
            index of the node

    Returns
    _______
        Set of indices representing where there is no connection from and to node i
    """
    from_i = graph[i, :] != 0
    to_i = graph[:, i] != 0
    idx = np.where(np.logical_and(~from_i, ~to_i))[0]
    return set(idx) - {i}


def get_child(graph: np.ndarray, i: int) -> set:
    """
    Find the child of the node at index i in the graph.

    Parameters
    __________
        graph: np.ndarray
        i: int
            index of the node

    Returns
    _______
        Set of indices representing the child of the node at index i
    """
    from_i = graph[i, :] != 0
    to_i = graph[:, i] != 0
    idx = np.where(np.logical_and(from_i, ~to_i))[0]
    return set(idx) - {i}


def insert_validity(graph: np.ndarray, x: int, y: int, T: set) -> bool:
    """
    Check whether an insert operator is valid.
    First, the function checks if the set of nodes NAyx U T is a clique in the graph,
    and then checks if every semi-directed path from node x
    to node y contains a node in the set NAyx U T.
    """
    na_yx = get_NAyx(graph, x, y)
    na_yx_t = na_yx.union(T)
    return is_clique(graph, na_yx_t) and is_path(graph, x, y, na_yx_t)


def is_path(graph: np.ndarray, x: int, y: int, na_yx_t: set) -> bool:
    """
    This function checks if every semi-directed path
    from node x to node y contains at least one node in the set na_yx_t.
    """
    semi_paths = semi_directed_path(graph, y, x)
    check = True
    for path in semi_paths:
        if len(set(path) & na_yx_t) == 0:
            check = False
            break
    return check


def semi_directed_path(graph, x, y):
    """
    Implementation based on castle library
    """
    semi_paths = []
    visitable = {i: get_child(graph, i) | get_neighbors(graph, i)
                 for i in range(graph.shape[0])}
    cache = [[x]]
    while len(cache) > 0:
        current_path = cache.pop(0)
        next = list(visitable[current_path[-1]] - set(current_path))
        for next_node in next:
            new_path = current_path.copy()
            new_path.append(next_node)
            if next_node == y:
                semi_paths.append(new_path)
            else:
                cache.append(new_path)

    return semi_paths


def is_clique(graph: np.ndarray, sub_nodes: set) -> bool:
    """
    Checks whether the nodes are a clique or not
    """
    sub_graph = graph[np.ix_(list(sub_nodes), list(sub_nodes))]
    skeleton = (sub_graph + sub_graph.T != 0).astype(int)
    return skeleton.sum() == len(sub_nodes) * (len(sub_nodes) - 1)


def delete_validity(graph: np.ndarray, x: int, y: int, H: set) -> bool:
    """
    Checks whether the delete operator is valid
    """
    na_yx = get_NAyx(graph, x, y)
    na_yx_h = na_yx - H
    return is_clique(graph, na_yx_h)


def apply_insert(graph: np.ndarray, x: int, y: int, T: set) -> np.ndarray:
    """
    This function applies an insert operator to the graph
    by adding an edge x -> y and orienting edges t - y to t -> y, for t in T.
    """
    graph[x, y] = 1
    if len(T) != 0:
        T = list(T)
        graph[T, y] = 1
        graph[y, T] = 0
    return graph


def apply_delete(graph: np.ndarray, x: int, y: int, H: set) -> np.ndarray:
    """
    This function applies a delete operator to the graph by
    deleting the edge between x and y, orienting the undirected
    edges between y and H towards H and orienting any
    undirected edges between x and H towards H.
    """
    graph[y, x], graph[x, y] = 0, 0
    graph[list(H), y] = 0
    neighbor_x = get_neighbors(graph, x)
    graph[list(H & neighbor_x), y] = 0
    return graph


def get_cpdag(graph):
    """
    Implementation based on castle library
    """
    raise NotImplementedError
    # return pdag_to_cpdag(graph)
