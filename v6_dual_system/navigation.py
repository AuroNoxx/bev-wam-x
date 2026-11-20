"""Navigation comes from the selected route, projected at the sensor timestamp."""
import math

import numpy as np


def model_navigation_state(speed, metric):
    nav=np.asarray(metric,dtype=np.float32)
    if nav.shape not in ((4,),(12,)):
        raise ValueError("Expected 2 or 6 FLU navigation anchors")
    return np.r_[np.float32(speed),nav[:4]/60.,nav[4:]/50.]


class RouteEnd(ValueError):
    """The remaining route cannot supply both requested anchors."""


FILM_STATIONS = (30., 60., 10., 20., 35., 50.)


def anchors(reference, pose, distances=(30., 60.)):
    arc, _, _ = reference.project(pose)
    stations = arc + np.asarray(distances)
    if stations.max() > reference.arc[-1]:
        raise RouteEnd("Selected route ends within the navigation horizon")
    points = np.column_stack([np.interp(stations, reference.arc, c) for c in reference.xyz.T[:2]])
    delta = points - pose[:2, 3]
    yaw = math.atan2(pose[1, 0], pose[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    return (delta @ np.array([[c, s], [s, -c]])).astype(np.float32).ravel()
