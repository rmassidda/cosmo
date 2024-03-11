"""
Test Orientation matrix generation.
"""
import unittest

import torch

from cosmolib.models import PriorityLinearSCM


class TestPriorityToOrientation(unittest.TestCase):
    """
    Test conversion from priority vector to orientations.
    """

    @torch.no_grad()
    def test_sigmoid(self):
        # Create model
        model = PriorityLinearSCM(n_nodes=4, temperature=1e-2, shift=0.20,
                                  precision=1e-6, hard_threshold=True)

        # Hard v1
        model.nodes_priority[:] = torch.Tensor([1., 2., 3., 4.]).unsqueeze(1)
        print(model.orientation)

        # Hard v2
        model.nodes_priority[:] = torch.Tensor([4., 3., 2., 1.]).unsqueeze(1)
        print(model.orientation)

        # Hard v3
        model.nodes_priority[:] = torch.Tensor([4., 2., 1., 3.]).unsqueeze(1)
        print(model.orientation)

        # Hard v4
        model.nodes_priority[:] = torch.Tensor([4., 2., 1., 4.01]).unsqueeze(1)
        print(model.orientation)

        # Hard v5
        model.nodes_priority[:] = torch.Tensor([4., 2., 1., 4.1]).unsqueeze(1)
        print(model.orientation)

        # Soft
        model.hard_threshold = False

        # Soft v1
        model.nodes_priority[:] = torch.Tensor([1., 2., 3., 4.]).unsqueeze(1)
        print(model.orientation)

        # Soft v2
        model.nodes_priority[:] = torch.Tensor([4., 3., 2., 1.]).unsqueeze(1)
        print(model.orientation)

        # Soft v3
        model.nodes_priority[:] = torch.Tensor([4., 2., 1., 3.]).unsqueeze(1)
        print(model.orientation)

        # Soft v4
        model.nodes_priority[:] = torch.Tensor([4., 2., 1., 4.01]).unsqueeze(1)
        print(model.orientation)

        # Soft v5
        model.nodes_priority[:] = torch.Tensor([4., 2., 1., 4.1]).unsqueeze(1)
        print(model.orientation)

        # TODO: actually test stuff :)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
