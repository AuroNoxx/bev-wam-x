import unittest
import torch
from models.flow_matcher_transformer import kinematic_bicycle_rollout, get_physical_candidate_actions
from models.integrated_physics import batched_vehicle_priors


class VehiclePhysicsTests(unittest.TestCase):
    def test_batch_matches_serial_fixed_vehicle_and_uses_actual_wheelbase(self):
        ego = torch.tensor([[[.7, .1]]*4, [[.4, -.05]]*4])
        vehicle = torch.tensor([[4.5, 1.9, 2.47, .7], [4.5, 1.9, 2.47, .7]])
        actions = torch.tensor(get_physical_candidate_actions(7.))
        result = batched_vehicle_priors(ego, actions, vehicle)
        serial = torch.stack([kinematic_bicycle_rollout(ego, a=a, delta=d) for a, d in actions], dim=1)
        torch.testing.assert_close(result, serial, rtol=0, atol=0)
        vehicle[1, 2] = 3.5
        changed = batched_vehicle_priors(ego, actions, vehicle)
        torch.testing.assert_close(changed[0], result[0])
        self.assertGreater(float((changed[1, 3]-result[1, 3]).abs().sum()), .01)


if __name__ == '__main__':
    unittest.main()
