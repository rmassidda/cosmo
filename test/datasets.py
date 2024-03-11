"""
Test the datasets module.
"""
import unittest

import numpy as np

from cosmolib.datasets.simulated import SimulatedDataset
from cosmolib.datasets import get_dataset


class TestCausalDataset(unittest.TestCase):

    def test_generation(self):
        """
        Test the generation of a simulated dataset.
        """
        dset = SimulatedDataset(n_edges=5, n_nodes=10, n_samples=10)

        self.assertEqual(len(dset), 10)
        self.assertEqual(dset.causal_graph.shape[0],
                         dset.causal_graph.shape[1])
        self.assertEqual(dset.causal_graph.shape[0], 10)

    def test_parsing(self):
        """
        Test the name parsing of the dataset.
        """
        dset = get_dataset('n2000_d10_ER2_gauss')
        self.assertEqual(len(dset), 2000)
        self.assertEqual(dset.n_nodes, 10)
        self.assertEqual(dset.causal_graph.shape, (10, 10))

        dset = get_dataset('n20_d25_ER2_gauss')
        self.assertEqual(len(dset), 20)
        self.assertEqual(dset.n_nodes, 25)
        self.assertEqual(dset.causal_graph.shape, (25, 25))

    def test_nonlinear(self):
        """
        Test the creation of nonlinear datasets.
        """
        for noise_type in ['mlp', 'mim', 'gp', 'gp-add']:
            dset = get_dataset(f'n20_d25_ER2_{noise_type}')
            self.assertEqual(len(dset), 20)
            self.assertEqual(dset.n_nodes, 25)
            self.assertEqual(dset.causal_graph.shape, (25, 25))

        for noise_type in ['mlp', 'mim', 'gp', 'gp-add']:
            dset = get_dataset(f'n2000_d10_ER2_{noise_type}')
            self.assertEqual(len(dset), 2000)
            self.assertEqual(dset.n_nodes, 10)
            self.assertEqual(dset.causal_graph.shape, (10, 10))

    def test_sachs(self):
        dset = get_dataset('sachs')
        self.assertEqual(len(dset), 7466)
        self.assertEqual(dset.n_nodes, 11)
        self.assertEqual(dset.causal_graph.shape, (11, 11))

    def test_dream(self):
        self.assertRaises(ValueError, get_dataset, 'dream0')
        self.assertRaises(ValueError, get_dataset, 'dream6')

        for i in range(1, 6):
            dset = get_dataset(f'dream{i}')
            self.assertEqual(len(dset), 100)
            self.assertEqual(dset.n_nodes, 100)
            self.assertEqual(dset.causal_graph.shape, (100, 100))

    def test_normalization(self):
        dset = get_dataset('n2000_d10_ER2_gauss', normalize=False)
        self.assertTrue(np.any(np.abs(dset.observations.mean(axis=0)) > 1e-4))
        dset = get_dataset('n2000_d10_ER2_gauss', normalize=True)
        self.assertTrue(np.all(np.abs(dset.observations.mean(axis=0)) < 1e-4))

        dset = get_dataset('sachs', normalize=False)
        self.assertTrue(np.any(np.abs(dset.observations.mean(axis=0)) > 1e-4))
        dset = get_dataset('sachs', normalize=True)
        self.assertTrue(np.all(np.abs(dset.observations.mean(axis=0)) < 1e-4))


if __name__ == '__main__':
    unittest.main()
