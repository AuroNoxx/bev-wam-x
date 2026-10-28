"""阶段5：相同退化、跨episode因果性、记录目标与正式权重接口。"""
import unittest
import numpy as np
from configs.stage5 import CONDITIONS, condition, Stage5Config
from evaluation.degradation import SpatialDegrader, temporal_sources, OnlineDegrader


class StressDegradationTests(unittest.TestCase):
    def test_clean_is_identity_and_never_mutates_input(self):
        bev=np.full((4,4,128,128),255,np.uint8)
        output=SpatialDegrader(condition('clean'))(bev,42)
        np.testing.assert_array_equal(output,bev)
        self.assertIsNot(output,bev)

    def test_occlusion_geometry_visibility_and_nested_severity(self):
        bev=np.full((4,128,128),255,np.uint8)
        masks=[]
        for level in (1,2,3):
            output=SpatialDegrader(condition(f'occlusion_l{level}'))(bev,1)
            mask=output[3]==0;masks.append(mask)
            np.testing.assert_array_equal(output[:,mask],0)
            self.assertEqual(int(mask.sum()),32*(16*level))
            self.assertFalse(mask[64:].any())
        self.assertTrue((masks[0]<=masks[1]).all());self.assertTrue((masks[1]<=masks[2]).all())
        self.assertTrue((bev==255).all())

    def test_boundary_episode_determinism_channel_isolation_and_nested_masks(self):
        bev=np.full((4,4,128,128),255,np.uint8)
        small=SpatialDegrader(condition('boundary_l1'),7)(bev,13)
        repeated=SpatialDegrader(condition('boundary_l1'),7)(bev,13)
        large=SpatialDegrader(condition('boundary_l2'),7)(bev,13)
        np.testing.assert_array_equal(small,repeated)
        for index in (0,2,3):np.testing.assert_array_equal(small[:,index],bev[:,index])
        self.assertTrue(((small[:,1]==0)<=(large[:,1]==0)).all())
        np.testing.assert_array_equal(small[0],small[3])

    def test_three_frame_hold_has_no_future_or_episode_leakage(self):
        episodes=np.repeat([0,1],10);steps=np.tile(np.arange(10),2)
        sources=temporal_sources(episodes,steps,3)
        np.testing.assert_array_equal(sources,np.array([0,1,2,3,4,5,6,6,6,6,10,11,12,13,14,15,16,16,16,16]))
        self.assertTrue((sources<=np.arange(20)).all())
        np.testing.assert_array_equal(episodes[sources],episodes)

    def test_discontinuous_segment_cannot_reach_old_frames(self):
        episodes=np.array([0,0,0,0]);steps=np.array([5,6,8,9])
        sources=temporal_sources(episodes,steps,3)
        np.testing.assert_array_equal(sources,[0,1,2,2])

    def test_online_and_offline_delivery_agree_at_observation_ticks(self):
        episodes=np.zeros(30,int);steps=np.arange(30)
        for level in (0,1,2,3):
            spec=condition('clean' if level==0 else f'dropout_l{level}')
            online=OnlineDegrader(spec,7,0)
            received=[]
            for step in range(150):
                clean=np.full((4,128,128),step,np.uint8)
                output,source=online(clean,step)
                self.assertLessEqual(source,step)
                if step%5==0:received.append(int(output[0,0,0])/5)
            np.testing.assert_array_equal(received,temporal_sources(episodes,steps,level))

    def test_closed_design_has_exact_126_prespecified_episodes(self):
        cfg=Stage5Config()
        self.assertEqual(len(CONDITIONS),10)
        self.assertEqual(len(cfg.closed_conditions)*3*2*cfg.closed_episodes_per_map,126)
        self.assertEqual(cfg.sdpa_backend,'MATH')


if __name__=='__main__':unittest.main()
