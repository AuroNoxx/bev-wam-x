import unittest
import numpy as np
from dataset.navigation import eligible_navigation_anchors


class RecoveryTargetTests(unittest.TestCase):
    def test_noisy_future_is_not_imitated_but_recovery_after_it_is_valid(self):
        valid = np.ones((60, 10), bool); routes = np.ones((60, 8), bool)
        events = np.zeros((60, 7), bool); noise = np.zeros(60, bool)
        noise[20:26] = True
        keep = eligible_navigation_anchors(valid, routes, events, noise)
        self.assertTrue(keep[0])  # state20 is before transition20 begins.
        self.assertFalse(keep[1])
        self.assertFalse(keep[25])
        self.assertTrue(keep[26])  # observed noisy history is allowed, future is clean.
        events[50, 4] = True
        keep = eligible_navigation_anchors(valid, routes, events, noise)
        self.assertTrue(keep[30]); self.assertFalse(keep[31]); self.assertFalse(keep[50])
        valid[58, -1] = False
        self.assertFalse(eligible_navigation_anchors(valid, routes, events, noise)[58])


if __name__ == '__main__':
    unittest.main()
