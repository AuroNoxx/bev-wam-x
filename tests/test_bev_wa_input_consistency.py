"""Integration test: verifies that offline dataset extraction and online closed-loop step buffer
produce identical features for:
1. ego_motion (speed / 10.0, yaw_rate)
2. applied_action history (causal execution feedback)
3. action_valid masks
across every single step of a recorded rollout.
"""

import unittest
from pathlib import Path
import numpy as np

from dataset.bev_wa_dataset import extract_causal_actions
from mini_wa_jepa.geometry import wrap_angle


class TestInputConsistency(unittest.TestCase):
    def test_offline_vs_online_consistency(self):
        # Find any existing valid episode NPZ
        sample_path = Path("artifacts/stage7_integrated/targeted_supplement_train_001/seed_4500001/episode.npz")
        if not sample_path.exists():
            candidates = list(Path("artifacts/stage7_integrated").glob("**/*.npz"))
            self.assertTrue(len(candidates) > 0, "No episode NPZ found for testing")
            sample_path = candidates[0]

        data = np.load(sample_path)
        states = data["states"]  # [N+1, 6]
        raw_ego = data["ego"]  # [N, 2]
        raw_applied = data["applied_action"]  # [N, 2]
        n_steps = len(raw_ego)

        # Simulate online step-by-step buffer
        online_ego_history = []
        online_applied_history = []
        prev_heading = None

        for t in range(n_steps):
            # 1. State observation at step t
            state = states[t]
            speed = float(state[3]) / 10.0
            heading = float(state[2])
            if prev_heading is None:
                yaw_rate = 0.0
            else:
                yaw_rate = float(np.arctan2(np.sin(heading - prev_heading), np.cos(heading - prev_heading)) / 0.1)
            prev_heading = heading

            curr_ego = np.array([speed, yaw_rate], dtype=np.float32)
            online_ego_history.append(curr_ego)
            if len(online_ego_history) > 4:
                online_ego_history.pop(0)

            # Pad online ego to 4 frames
            online_h_ego = np.stack(
                [online_ego_history[0]] * (4 - len(online_ego_history)) + online_ego_history, axis=0
            )

            # Online causal applied action extraction
            online_h_app, online_app_valid = extract_causal_actions(online_applied_history, t, num_lags=4)

            # 2. Offline dataset extraction at step t
            offline_indices = [max(0, t - lag) for lag in (3, 2, 1, 0)]
            offline_h_ego = raw_ego[offline_indices]
            offline_h_app, offline_app_valid = extract_causal_actions(raw_applied, t, num_lags=4)

            # Assert identical features at every step
            np.testing.assert_allclose(
                online_h_ego, offline_h_ego, atol=1e-5,
                err_msg=f"Ego motion mismatch at step {t}!"
            )
            np.testing.assert_allclose(
                online_h_app, offline_h_app, atol=1e-5,
                err_msg=f"Applied action mismatch at step {t}!"
            )
            np.testing.assert_array_equal(
                online_app_valid, offline_app_valid,
                err_msg=f"Action valid mask mismatch at step {t}!"
            )

            # 3. Step execution (recording action that took ego from t to t+1)
            online_applied_history.append(raw_applied[t])


if __name__ == "__main__":
    unittest.main()
