"""
Test the graph projection implementation used by the NOCURL algorithm.
"""
import unittest

import torch

from cosmolib.algorithms.nocurl import compute_ordering


def graph_proj_tester(pre_matrix: torch.Tensor, target: torch.Tensor) -> bool:
    n_nodes = pre_matrix.shape[0]
    proj = compute_ordering(pre_matrix, n_nodes)
    return torch.allclose(proj, target, atol=1e-7)


class TestGraphProjection(unittest.TestCase):

    def test_example_1(self):
        """
        Test Example 1 from Appendix D.
        """
        self.assertTrue(graph_proj_tester(
            torch.tensor(
                [[0., -1., 0., 0.],
                 [0., 0., 2., 0.],
                 [0., 0., 0., 5.],
                 [0., 0., 0., 0.]]
            ),
            torch.tensor([-0.75, -0.5, -0.25, 0.])
        ))

    def test_example_2(self):
        """
        Test Example 2 from Appendix D.
        """
        self.assertTrue(graph_proj_tester(
            torch.tensor(
                [[0., -1., 0., 0.],
                 [0., 0., 0., 0.],
                 [0., 0., 0., 5.],
                 [0., 0., 0., 0.]]
            ),
            torch.tensor([-0.25, 0., -0.25, 0.])
        ))

    def test_example_3(self):
        """
        Test Example 3 from Appendix D.
        """
        self.assertTrue(graph_proj_tester(
            torch.tensor(
                [[0., -1., 0., 0.],
                 [2., 0., 0., 0.],
                 [0., 0., 0., 5.],
                 [-2., 0., 0., 0.]]
            ),
            torch.tensor([0.375, 0.375, -0.25, 0.])
        ))

    def test_example_4(self):
        """
        Test Example 4 from Appendix D.
        """
        self.assertTrue(graph_proj_tester(
            torch.tensor(
                [[0., -1., 0., 0.],
                 [0., 0., 2., 0.],
                 [0., 0., 0., 5.],
                 [-2., 0., 0., 0.]]
            ),
            torch.tensor([0., 0., 0., 0.])
        ))
