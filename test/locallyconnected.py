"""
Test the locally connected module.
"""
import unittest

import numpy as np
import torch

from cosmolib.models.locally_connected import LocallyConnected


class TestLocallyConnected(unittest.TestCase):

    def test_shape(self):
        """
        Test the generation of a simulated dataset.
        """
        n, d, m1, m2 = 2, 3, 5, 7

        # numpy
        input_numpy = np.random.randn(n, d, m1)
        weight = np.random.randn(d, m1, m2)
        output_numpy = np.zeros([n, d, m2])
        for j in range(d):
            # [n, m2] = [n, m1] @ [m1, m2]
            output_numpy[:, j, :] = input_numpy[:, j, :] @ weight[j, :, :]

        # torch
        torch.set_default_dtype(torch.double)
        input_torch = torch.from_numpy(input_numpy)
        locally_connected = LocallyConnected(d, m1, m2, bias=False)
        locally_connected.weight.data[:] = torch.from_numpy(weight)
        output_torch = locally_connected(input_torch)

        # compare
        self.assertTrue(torch.allclose(output_torch,
                                       torch.from_numpy(output_numpy)))