from types import SimpleNamespace
import numpy as np
from planning.integrated import permission_rejections


def test_turning_with_route_is_not_a_lane_change_and_return_uses_current_lane_permissions():
    angle = np.arange(8)*.16
    route = np.column_stack([30*np.sin(angle), 30*(1-np.cos(angle))])
    nav = SimpleNamespace(points=route, valid=np.ones(8, bool), lane_width_m=3.5)
    # Interpolate points exactly on the declared curved polyline.
    samples = np.linspace(0, 5, 10)
    plan = np.zeros((10, 4)); plan[:, :2] = np.column_stack([np.interp(samples, np.arange(8), route[:, j]) for j in range(2)])
    assert plan[-1, 1] > 2.
    assert permission_rejections(plan, nav, [False, False]) == []
    nav.points = np.column_stack([np.arange(8)*5., np.full(8, -3.5)])
    plan[:, 0] = np.arange(1, 11); plan[:, 1] = np.linspace(-.1, -3.5, 10)
    assert permission_rejections(plan, nav, [False, True]) == []
    assert permission_rejections(plan, nav, [True, False]) == ['static_right_lane_permission']
