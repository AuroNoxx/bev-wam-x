import json
import numpy as np
from evaluation.stage4_diagnostic import causal_ego, classify_outcome, scene_suite, scene_overrides


def test_yaw_wrap_and_initial_motion():
    assert causal_ego(4, 3., None).tolist() == [np.float32(.4), 0.]
    assert np.isclose(causal_ego(0, -np.pi+.01, np.pi-.01)[1], .2)


def row(gap=None, **events):
    return dict(events=events, tracked_lead_clearance_m=gap, state=[0, 0], post_state=[2, 0], route_completion=.3)


def test_missing_gap_not_success():
    assert classify_outcome('following', [row()], 1)['safe_follow_observation'] is None
    assert classify_outcome('following', [row(4)], 2)['safe_follow_observation'] is False
    assert classify_outcome('following', [row(4)], 1)['safe_follow_observation'] is True
    assert classify_outcome('following', [row(4, crash_vehicle=True)], 1)['safe_follow_observation'] is False


def test_navigation_arrival_is_distinct():
    assert classify_outcome('navigation', [row(4)], 1)['navigation_success'] is False
    assert classify_outcome('navigation', [row(arrive_dest=True)], 1)['navigation_success'] is True
    assert classify_outcome('following', [row(4)], 1)['navigation_success'] is None


def test_legacy_suites_stay_distinct():
    assert all(s['map']=='S' and s['horizon']==250 for s in scene_suite('legacy_six_S', None))
    scenes=scene_suite('legacy_four', None, 20261023)
    assert len(scenes)==4 and all(s['seed']==20261023 for s in scenes)
    assert scene_overrides(scenes[0])['map_config']['config']=='SSS'


def test_manifest_no_blind_access(tmp_path):
    import pytest
    p=tmp_path/'manifest.json'
    p.write_text(json.dumps(dict(split='validation', completed=True, scenes=[dict(seed=4400000+i,map='S') for i in range(6)])))
    with pytest.raises(ValueError): scene_suite('manifest_nav', p)
