"""
Test SimulatedDataset class.
"""
import unittest

import numpy as np

from cosmolib.utils import mask_weights


class TestMask(unittest.TestCase):

    def test_mask_weights(self):
        print('Test .1')
        weight_adj = np.array([[0.1, 0.4],[0.1, 0.2]])

        print(mask_weights(weight_adj))
        print(mask_weights(weight_adj, relative=True))
        print(mask_weights(weight_adj, threshold=0., relative=True))
        print(mask_weights(weight_adj, force_dag=True))
        # TODO: real test cases

    def test_mask_weights_2(self):
        print('Test .2')
        weight_adj = np.array([[1., 1.],[1., 1.]])

        print(mask_weights(weight_adj))
        print(mask_weights(weight_adj, relative=True))
        print(mask_weights(weight_adj, threshold=0., relative=True))
        print(mask_weights(weight_adj, force_dag=True))
        # TODO: real test cases


if __name__ == '__main__':
    unittest.main()
