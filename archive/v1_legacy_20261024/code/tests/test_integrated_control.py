import unittest
from types import SimpleNamespace as NS
import numpy as np
from env.integrated import checked_step
from metadrive.policy.env_input_policy import EnvInputPolicy


def simulator(policy, applied):
    car = NS(id='car', current_action=applied)
    def step(action):
        return None, 0., False, False, dict(raw_action=applied)
    return NS(agent=car, engine=NS(get_policy=lambda _:policy), step=step)


class AppliedControlTests(unittest.TestCase):
    def test_rejects_idm_before_claiming_external_control(self):
        with self.assertRaisesRegex(RuntimeError, 'not EnvInputPolicy'):
            checked_step(simulator(object(), [0, -1]), [0, -1])

    def test_checks_actual_policy_and_vehicle_action(self):
        policy = object.__new__(EnvInputPolicy)
        command = np.array([.2, -1.], np.float32)
        self.assertEqual(len(checked_step(simulator(policy, command), command)), 5)
        with self.assertRaisesRegex(RuntimeError, 'differs'):
            checked_step(simulator(policy, [0, 1]), command)


if __name__ == '__main__':
    unittest.main()
