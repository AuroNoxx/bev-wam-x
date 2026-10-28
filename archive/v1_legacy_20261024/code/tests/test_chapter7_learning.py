import unittest
import numpy as np
from chapters.chapter7_candidate_planning import ui


class Chapter7Tests(unittest.TestCase):
    def test_actual_flow_states_change_without_changing_noise_or_truth(self):
        values=ui.make_controls().value;a=ui.compute(dict(values,flow_step=0));b=ui.compute(dict(values,flow_step=10))
        if 'missing' in a:self.skipTest('Measured Flow path not available')
        np.testing.assert_array_equal(a['data']['ground_truth'],b['data']['ground_truth'])
        self.assertFalse(np.allclose(a['data']['path'][a['step']],b['data']['path'][b['step']]))
        self.assertEqual(a['data']['path'].shape,(11,8,10,4))

    def test_actual_decision_and_next_cycle_share_one_recording(self):
        values=ui.make_controls().value
        if values['recording']=='missing':self.skipTest('No recorded episode')
        a=ui.compute(dict(values,section='2 看评分、选择与执行',decision=0))
        b=ui.compute(dict(values,section='2 看评分、选择与执行',decision=1))
        self.assertIs(a['data'],b['data']);self.assertEqual(b['index'],1)
        self.assertEqual(len(a['rows']),len(a['data']['candidates'][0]))
        for row,p in zip(a['rows'],a['data']['probability'][0]):
            if np.isnan(p[0]):self.assertIsNone(row['碰撞概率'])

    def test_render_uses_real_artifacts_for_both_views(self):
        controls=ui.make_controls();values=controls.value
        self.assertIsNotNone(ui.render(values,controls))
        if values['recording']!='missing':
            self.assertIsNotNone(ui.render(dict(values,section='2 看评分、选择与执行'),controls))


if __name__=='__main__':unittest.main()
