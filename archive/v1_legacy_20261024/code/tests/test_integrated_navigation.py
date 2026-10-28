import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from configs.integrated import IntegratedConfig
from env.navigation import NavigationProvider, NavigationFrame, vehicle_parameters
from mini_wa_jepa.geometry import ego_to_world
from planning.integrated import curvature_braking, route_progress, strict_score


class Lane:
    def __init__(self, start, end, origin=(0., 0.), angle=0., length=20.):
        self.index = (start, end, 0)
        self.origin, self.angle, self.length = np.array(origin), angle, length
    def position(self, s, lateral):
        return ego_to_world([s, lateral], self.origin, self.angle)
    def local_coordinates(self, p):
        from mini_wa_jepa.geometry import world_to_ego
        return world_to_ego(p, self.origin, self.angle)
    def distance(self, p):
        s, y = self.local_coordinates(p)
        return abs(y)+max(-s, s-self.length, 0)
    def heading_theta_at(self, s):
        return self.angle
    def width_at(self, s):
        return 3.5


def simulator(origin=(0., 0.), angle=0.):
    a = Lane('a', 'b', origin, angle)
    b = Lane('b', 'c', a.position(20, 0), angle)
    nav = NS(checkpoints=['a', 'b', 'c'], current_road=NS(start_node='a', end_node='b'))
    car = NS(position=a.position(2, .5), heading_theta=angle, navigation=nav,
             LENGTH=4.5, WIDTH=1.9, FRONT_WHEELBASE=1.2, REAR_WHEELBASE=1.4, max_steering=40.)
    return NS(agent=car, current_map=NS(road_network=NS(graph={'a': {'b': [a]}, 'b': {'c': [b]}})))


class NavigationTests(unittest.TestCase):
    def test_route_uses_declared_map_and_is_rigid_transform_equivariant(self):
        a = NavigationProvider(simulator()).capture()
        b = NavigationProvider(simulator((100, -30), 1.4)).capture()
        np.testing.assert_allclose(a.points, b.points, atol=1e-5)
        np.testing.assert_allclose(a.points[:, 1], -.5, atol=1e-5)
        self.assertTrue(a.valid.all())
        self.assertEqual(a.source, 'map_assisted_route_lane_centerline')

    def test_navigation_has_no_dependency_on_traffic_or_expert_future(self):
        sim = simulator()
        before = NavigationProvider(sim).capture().points.copy()
        sim.traffic = object()
        sim.teacher_future = np.full((10, 4), np.nan)
        np.testing.assert_array_equal(before, NavigationProvider(sim).capture().points)

    def test_ending_masks_missing_route_points_and_reset_does_not_cache(self):
        sim = simulator()
        sim.agent.position = np.array([38., 0.])
        frame = NavigationProvider(sim).capture()
        self.assertEqual(frame.valid.sum(), 1)
        self.assertTrue(np.all(frame.points[~frame.valid] == 0))
        sim.agent.position = np.array([2., 0.])
        self.assertTrue(NavigationProvider(sim).capture().valid.all())

    def test_declared_vehicle_geometry_is_actual_not_default(self):
        np.testing.assert_allclose(vehicle_parameters(simulator().agent), [4.5, 1.9, 2.6, np.deg2rad(40)])

    def test_navigation_logger_records_route_and_rejects_wrong_mask(self):
        from mini_wa_jepa.visualization import RerunLogger
        frame = NavigationProvider(simulator()).capture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'route.rrd'
            with RerunLogger(spawn=False, save_path=path) as logger:
                logger.set_step(0, 0.)
                logger.log_navigation(frame.points, frame.valid, source=frame.source)
                logger.log_navigation(frame.points, np.zeros(8, bool), source=frame.source)
                with self.assertRaises(ValueError):
                    logger.log_navigation(frame.points, np.zeros(7, bool), source=frame.source)
            self.assertGreater(path.stat().st_size, 1000)

    def test_route_progress_follows_turn_not_ego_forward_x(self):
        route = np.array([[0, 0], [5, 0], [5, 5], [5, 10]])
        plan = np.zeros((10, 4)); plan[-1, :2] = [5, 8]
        self.assertAlmostEqual(route_progress(plan, route, np.ones(4, bool)), 13.)

    def test_all_risk_rejected_does_not_restore_kinematic_candidate(self):
        bev = np.zeros((4, 128, 128), np.uint8); bev[0] = 255; bev[3] = 255
        state = np.zeros(6)
        plan = np.column_stack([np.arange(1, 11)*1.4, np.zeros(10), np.full(10, 7), np.zeros(10)])
        nav = NavigationFrame(np.column_stack([np.arange(8)*5., np.zeros(8)]), np.ones(8, bool),
                              np.zeros((8, 2)), 3.5, ('a', 'b', 0))
        result = strict_score(plan[None], bev, [bev], [state], 4.5, 1.9, nav,
                              np.array([[.9, .9]]), [.5, .5])
        self.assertEqual(result['selected_index'], -1)
        self.assertFalse(result['eligible'].any())

    def test_braking_keeps_turn_and_stops_without_reverse(self):
        straight = curvature_braking(7., 0.)
        curved = curvature_braking(7., .2)
        self.assertTrue(np.all(np.diff(curved[:, 2]) <= 0))
        self.assertEqual(curved[-1, 2], 0.)
        self.assertGreater(curved[-1, 1], 0.)
        self.assertTrue(np.all(straight[:, 1] == 0))
        np.testing.assert_array_equal(curvature_braking(0., 0.), np.zeros((10, 4)))
        with self.assertRaises(ValueError):
            curvature_braking(float('nan'), 0.)


if __name__ == '__main__':
    unittest.main()
