import unittest
import torch
from models.navigation_head import NavigationTrajectoryHead


class NavigationHeadTests(unittest.TestCase):
    def test_masked_navigation_cannot_change_output_and_gradients_train(self):
        torch.manual_seed(4)
        head = NavigationTrajectoryHead(tokens=4, token_dim=8, hidden=32)
        history = torch.randn(2, 4, 4, 8)
        ego = torch.zeros(2, 4, 2); ego[..., 0] = .7
        route = torch.randn(2, 8, 2)
        valid = torch.ones(2, 8, dtype=torch.bool); valid[:, -2:] = False
        vehicle = torch.tensor([[4.5, 1.9, 2.6, .7]]).expand(2, -1)
        optimizer = torch.optim.Adam(head.parameters(), lr=.01)
        target = torch.zeros(2, 10, 4)
        for _ in range(3):
            loss = (head(history, ego, route, valid, vehicle)-target).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        prediction = head(history, ego, route, valid, vehicle)
        changed = route.clone(); changed[:, -2:] = float('nan')
        torch.testing.assert_close(prediction, head(history, ego, changed, valid, vehicle))
        changed = route.clone(); changed[:, :6] += 10
        self.assertGreater(float((prediction-head(history, ego, changed, valid, vehicle)).detach().abs().max()), 1e-5)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertIsNotNone(head.route[0].weight.grad)
        self.assertGreater(float(head.route[0].weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
