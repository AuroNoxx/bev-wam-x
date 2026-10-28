"""Conservative map-assisted lane-change permissions; no actor/future access."""
import numpy as np
from mini_wa_jepa.geometry import wrap_angle


def legal_adjacent_lane(source, target):
    from metadrive.constants import PGLineType
    from metadrive.component.lane.straight_lane import StraightLane
    if not isinstance(source, StraightLane) or not isinstance(target, StraightLane):
        return False
    delta = target.index[-1]-source.index[-1]
    if source.index[:2] != target.index[:2] or abs(delta) != 1 or target.forbidden:
        return False
    side = 0 if delta < 0 else 1
    return (source.line_types[side] == PGLineType.BROKEN and
            target.line_types[1-side] == PGLineType.BROKEN and
            abs(float(wrap_angle(target.heading_theta_at(0)-source.heading_theta_at(0)))) < .1)


def maneuver_permissions(sim):
    """[toward smaller lane index, toward larger] at own map localization.

    Curved lanes/junctions are conservatively disallowed in this first profile.
    These are static permissions, not clearance or a recommendation to change.
    """
    lane = sim.agent.lane
    lanes = sim.current_map.road_network.graph[lane.index[0]][lane.index[1]]
    result = np.zeros(2, bool)
    for column, index in enumerate((lane.index[-1]-1, lane.index[-1]+1)):
        if 0 <= index < len(lanes):
            result[column] = legal_adjacent_lane(lane, lanes[index])
    return result
