"""
Test SimulatedDataset class.
"""
import unittest

import torch

from cosmolib.utils import decode_const


class Testconst(unittest.TestCase):
    """
    Test const decoding.
    """

    def test_decode_const(self):
        # Sorted vector
        x = torch.tensor([1, 2, 3, 4])
        y = decode_const(x, eps=1)
        self.assertTrue(torch.all(y == torch.tensor([1, 3, 5, 7])))

        # Equal elements vector
        x = torch.tensor([3, 3, 3, 3])
        y = decode_const(x, eps=1)
        self.assertTrue(torch.all(y == torch.tensor([3, 4, 5, 6])))

        # Reverse sorted elements vector
        x = torch.tensor([4, 3, 2, 1])
        y = decode_const(x)
        self.assertTrue(torch.all(y == torch.tensor([4.15, 3.1, 2.05, 1])))

        # Unsorted elements vector
        x = torch.tensor([3, 4, 1, 2])
        y = decode_const(x, eps=2)
        self.assertTrue(torch.all(y == torch.tensor([7, 10, 1, 4])))

    def test_decode_const_bidimensional(self):
        # Sorted vector
        x = torch.tensor([1, 2, 3, 4])
        x = x.reshape((4, 1))
        y = decode_const(x)
        y_gt = torch.tensor([1, 2.05, 3.1, 4.15]).reshape((4, 1))
        self.assertTrue(torch.all(y == y_gt))

        # Equal elements vector
        x = torch.tensor([3, 3, 3, 3])
        x = x.reshape((4, 1))
        y = decode_const(x)
        y_gt = torch.tensor([3, 3.05, 3.1, 3.15]).reshape((4, 1))
        self.assertTrue(torch.all(y == y_gt))

        # Reverse sorted elements vector
        x = torch.tensor([4, 3, 2, 1])
        x = x.reshape((4, 1))
        y = decode_const(x)
        y_gt = torch.tensor([4.15, 3.1, 2.05, 1]).reshape((4, 1))
        self.assertTrue(torch.all(y == y_gt))

        # Unsorted elements vector
        x = torch.tensor([3, 4, 1, 2])
        x = x.reshape((4, 1))
        y = decode_const(x)
        y_gt = torch.tensor([3.10, 4.15, 1, 2.05]).reshape((4, 1))
        self.assertTrue(torch.all(y == y_gt))

if __name__ == '__main__':
    unittest.main()
