import unittest
import numpy as np
from planning.navigation_reference import navigation_reference


class RouteReferenceTests(unittest.TestCase):
    def test_straight_nominal_and_corrective_lateral_sign(self):
        ego = np.tile([.7, 0.], (2, 4, 1))
        route = np.zeros((2, 8, 2)); route[:, :, 0] = np.arange(8)*5.
        valid = np.ones((2, 8), bool)
        vehicle = np.tile([4.5, 1.9, 2.47, .7], (2, 1))
        route[1, :, 1] = -1.
        prior = navigation_reference(ego, route, valid, vehicle)
        np.testing.assert_allclose(prior[0, :, 0], np.arange(1, 11)*1.4, atol=1e-5)
        np.testing.assert_allclose(prior[0, :, 1], 0, atol=1e-5)
        self.assertLess(prior[1, -1, 1], -.5)
        self.assertTrue((prior[:, :, 2] >= 0).all())
        start = np.zeros((1, 4, 2))
        self.assertGreater(navigation_reference(start, route[:1], valid[:1], vehicle[:1])[0, -1, 2], 0)


if __name__ == '__main__':
    unittest.main()
