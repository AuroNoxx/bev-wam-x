from types import SimpleNamespace
import numpy as np
from env.traffic_teacher import quintic_blend, legal_adjacent_lane, lane_clearance


def test_legal_adjacent_lane_requires_both_broken_boundaries_and_same_road():
    from metadrive.component.lane.straight_lane import StraightLane
    from metadrive.constants import PGLineType
    a = StraightLane([0, 0], [200, 0]); a.index = ('a', 'b', 1)
    b = StraightLane([0, 3.5], [200, 3.5]); b.index = ('a', 'b', 0)
    assert legal_adjacent_lane(a, b)
    b.line_types = [PGLineType.BROKEN, PGLineType.CONTINUOUS]
    assert not legal_adjacent_lane(a, b)
    b.line_types = [PGLineType.BROKEN, PGLineType.BROKEN]; b.forbidden = True
    assert not legal_adjacent_lane(a, b)
    b.forbidden = False; b.index = ('x', 'b', 0)
    assert not legal_adjacent_lane(a, b)


def test_lane_clearance_accounts_for_fast_rear_and_both_footprints():
    from metadrive.component.lane.straight_lane import StraightLane
    lane = StraightLane([0, 0], [200, 0])
    def car(x, speed, y=0):
        return SimpleNamespace(position=np.array([x, y]), speed=speed, WIDTH=2., LENGTH=4.)
    ego = car(50., 4.); rear = car(35., 10.); front = car(80., 3.)
    f, r, t = lane_clearance(lane, ego, [ego, rear, front, car(52., 20., 10.)])
    assert f == 26. and r == 11.
    assert abs(t-11./6.) < 1e-8
    x = quintic_blend(np.array([-1., 0., .5, 1., 2.]))
    np.testing.assert_allclose(x, [0., 0., .5, 1., 1.])
