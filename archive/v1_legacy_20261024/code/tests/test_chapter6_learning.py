"""核对教学页与正式记录的对应关系，以及缺失/并列结果的诚实展示。"""
import unittest
import numpy as np
from chapters.chapter6_action_conditioned.src import lesson_data as data


class OutcomeMatchingTests(unittest.TestCase):
    def test_missing_and_tied_outcomes_do_not_invent_a_choice(self):
        actual = np.array([0., 1., 2., 3.])[:, None, None]
        self.assertIsNone(data.closest_outcome(np.array([[0.]]), actual, [False]*4)[0])
        self.assertIsNone(data.closest_outcome(np.array([[.5]]), actual, [True]*4)[0])
        self.assertEqual(data.closest_outcome(np.array([[0.]]), actual, [False, True, True, True])[0], 1)


@unittest.skipUnless(data.available(), 'Formal Stage6 recordings unavailable')
class RecordedLessonTests(unittest.TestCase):
    def test_common_context_and_world_positions_match_recorded_labels(self):
        self.assertEqual(len(data.scene_options()), 32)
        for scene in data.scene_options().values():
            context = data.context(scene)
            self.assertEqual(len({r['group'] for r in context['metadata']}), 1)
            for branch in context['branches']:
                xy = data.local_positions(branch['states'])
                np.testing.assert_allclose(xy[0], [0., 0.], atol=1e-6)
                mask = branch['waypoint_valid'].astype(bool)
                np.testing.assert_allclose(xy[2::2][mask], branch['realized'][mask, :2], atol=1e-4)
        with self.assertRaises(IndexError):
            data.context(-1)

    def test_history_only_prediction_does_not_depend_on_candidate(self):
        predictions = data.frozen_predictions('unconditional')['prediction']
        for group in predictions.reshape(-1, 4, *predictions.shape[1:]):
            for candidate in group[1:]:
                np.testing.assert_allclose(candidate, group[0], atol=2e-5, rtol=1e-4)

    def test_wrong_command_output_is_the_supplied_candidate(self):
        # 正式评估将四条命令循环右移，A收到D、B收到A；页面文字须与此一致。
        archive = data.frozen_predictions('conditioned')
        expected = np.roll(archive['prediction'].reshape(-1, 4, *archive['prediction'].shape[1:]), 1, axis=1)
        np.testing.assert_allclose(archive['wrong_prediction'], expected.reshape(archive['prediction'].shape), atol=2e-5, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
