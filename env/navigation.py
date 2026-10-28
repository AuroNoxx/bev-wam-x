"""Declared map-assisted route input, independent of teacher/traffic/future states.

NodeNetworkNavigation.checkpoints, current_road and road_network.graph were
verified against installed MetaDrive source. Exact lane geometry is privileged
map input here, provided equally to the learner baselines, not BEV perception.
"""
from dataclasses import dataclass
import numpy as np
from configs.integrated import IntegratedConfig
from mini_wa_jepa.geometry import world_to_ego, wrap_angle


@dataclass(frozen=True)
class NavigationFrame:
    points: np.ndarray
    valid: np.ndarray
    world_points: np.ndarray
    lane_width_m: float
    lane_index: tuple
    source: str = 'map_assisted_route_lane_centerline'


class NavigationProvider:
    def __init__(self, sim, config=IntegratedConfig()):
        self.sim, self.config = sim, config

    def capture(self):
        car = self.sim.agent
        nav = car.navigation
        checkpoints = list(nav.checkpoints)
        graph = self.sim.current_map.road_network.graph
        edges = list(zip(checkpoints[:-1], checkpoints[1:]))
        current = (nav.current_road.start_node, nav.current_road.end_node)
        if current not in edges:
            raise ValueError('Current navigation road is absent from declared route')
        index = edges.index(current)
        lanes = []
        for start, end in edges[index:]:
            candidates = graph[start][end]
            lanes.append(candidates[min(self.config.preferred_lane, len(candidates)-1)])
        # Localization can transition after the physical vehicle enters next road.
        # Choose among current/next route segments, never arbitrary future roads.
        choices = []
        for i, lane in enumerate(lanes[:2]):
            s, _ = lane.local_coordinates(car.position)
            heading = lane.heading_theta_at(np.clip(s, 0, lane.length))
            error = abs(float(wrap_angle(heading-car.heading_theta)))
            choices.append((float(lane.distance(car.position))+2*error, i))
        selected = min(choices)[1]
        lanes = lanes[selected:]
        longitudinal, _ = lanes[0].local_coordinates(car.position)
        cursor = float(np.clip(longitudinal, 0, lanes[0].length))
        points = np.zeros((self.config.navigation_points, 2), np.float64)
        valid = np.zeros(self.config.navigation_points, bool)
        for k in range(self.config.navigation_points):
            remaining = cursor+k*self.config.navigation_spacing_m
            for lane in lanes:
                if remaining <= lane.length+1e-8:
                    points[k] = lane.position(min(remaining, lane.length), 0)
                    valid[k] = True
                    break
                remaining -= lane.length
        local = np.zeros_like(points)
        local[valid] = world_to_ego(points[valid], car.position, car.heading_theta)
        if not np.isfinite(local).all() or not valid[0]:
            raise ValueError('Invalid navigation observation')
        return NavigationFrame(local.astype(np.float32), valid, points,
                               float(lanes[0].width_at(cursor)), tuple(lanes[0].index))


def vehicle_parameters(car):
    """Declared own-vehicle parameters, not traffic actor ground truth."""
    result = np.array([car.LENGTH, car.WIDTH,
                       car.FRONT_WHEELBASE+car.REAR_WHEELBASE,
                       np.deg2rad(car.max_steering)], np.float32)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError('Invalid vehicle parameters')
    return result

