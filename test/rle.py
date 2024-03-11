"""
Test SimulatedDataset class.
"""
import unittest

import torch

from cosmolib.utils import decode_rle


class TestRLE(unittest.TestCase):
    """
    Test RLE decoding.
    """

    def test_decode_rle(self):
        # Sorted vector
        x = torch.tensor([1, 2, 3, 4])
        y = decode_rle(x)
        self.assertTrue(torch.all(y == torch.tensor([1, 3, 6, 10])))

        # Equal elements vector
        x = torch.tensor([3, 3, 3, 3])
        y = decode_rle(x)
        self.assertTrue(torch.all(y == torch.tensor([3, 6, 9, 12])))

        # Reverse sorted elements vector
        x = torch.tensor([4, 3, 2, 1])
        y = decode_rle(x)
        self.assertTrue(torch.all(y == torch.tensor([10, 6, 3, 1])))

        # Unsorted elements vector
        x = torch.tensor([3, 4, 1, 2])
        y = decode_rle(x)
        self.assertTrue(torch.all(y == torch.tensor([6, 10, 1, 3])))

    def test_decode_rle_bidimensional(self):
        # Sorted vector
        x = torch.tensor([1, 2, 3, 4])
        x = x.reshape((4, 1))
        y = decode_rle(x)
        y_gt = torch.tensor([1, 3, 6, 10]).reshape((4, 1))
        self.assertTrue(torch.all(y == y_gt))

        # Equal elements vector
        x = torch.tensor([3, 3, 3, 3])
        x = x.reshape((4, 1))
        y = decode_rle(x)
        y_gt = torch.tensor([3, 6, 9, 12]).reshape((4, 1))
        self.assertTrue(torch.all(y == y_gt))

        # Reverse sorted elements vector
        x = torch.tensor([4, 3, 2, 1])
        x = x.reshape((4, 1))
        y = decode_rle(x)
        y_gt = torch.tensor([10, 6, 3, 1]).reshape((4, 1))
        self.assertTrue(torch.all(y == y_gt))

        # Unsorted elements vector
        x = torch.tensor([3, 4, 1, 2])
        x = x.reshape((4, 1))
        y = decode_rle(x)
        y_gt = torch.tensor([6, 10, 1, 3]).reshape((4, 1))
        self.assertTrue(torch.all(y == y_gt))

if __name__ == '__main__':
    unittest.main()
