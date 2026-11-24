"""Training-only targets built from synchronized CARLA state."""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .geometry import FUTURE_TICKS, H, W, X_MAX, X_MIN, Y_MAX, Y_MIN, Z, Z_EDGES, local, pixels, wrap

LANE_CHANNELS = ("drivable_area", "dashed_line", "solid_line", "road_edge")
MAP_CLASSES = ("dashed_divider", "solid_divider", "road_edge")
MAP_SLOTS = 24
MAP_POINTS = 20
VECTOR_NAMES = ("ego_center", "left_line", "right_line", "left_edge", "right_edge",
                "left_neighbor", "right_neighbor")
VECTOR_X = np.linspace(-20., 55., 24, dtype=np.float32)
OBJECT_CLASSES = ("car", "suv", "truck", "two_wheeler", "pedestrian")
OBJECT_SLOTS = 32
OBJECT_FIELDS = ("x", "y", "z", "l", "w", "h", "yaw", "vx", "vy", "cls_id", "valid")
STATE_FIELDS = ("v", "a_lon", "a_lat", "yaw_rate", "steer", "throttle", "brake")
EVENT_FIELDS = ("collision", "solid_crossing", "offroute")
MIN_BOX = np.array([1.2, .6, .8])
MAX_BOX = np.array([16., 3.8, 4.8])
SOLID = {"Solid", "SolidSolid", "SolidBroken"}
BROKEN = {"Broken", "BrokenBroken", "BrokenSolid"}


def xyz(value):
    return np.array([value.x, value.y, value.z], dtype=np.float64)


def name(value):
    return str(value).rsplit(".", 1)[-1]


def mark_type(marking):
    return name(marking.type) if marking is not None else ""


def driving(waypoint):
    return waypoint is not None and name(waypoint.lane_type) == "Driving"


def neighbor(wp, side):
    other = (wp.get_left_lane if side < 0 else wp.get_right_lane)()
    if driving(other):
        return other
    return None


def same_direction(wp, other):
    return other is not None and wp.transform.get_forward_vector().dot(other.transform.get_forward_vector()) > .5


def side_point(wp, side):
    tr = wp.transform
    return xyz(tr.location) + side * wp.lane_width / 2 * xyz(tr.get_right_vector())


def polygon(mask, xy):
    # OpenCV fixed-point polygon rasterization shares geometry.pixels' cell centers.
    cv2.fillPoly(mask, [np.rint(pixels(xy) * 16).astype(np.int32)], 1, shift=4)


def line(mask, xy, width=1):
    cv2.polylines(mask, [np.rint(pixels(xy) * 16).astype(np.int32)],
                  False, 1, width, shift=4)


class RoadSurfaces:
    """Spatially indexed OpenDRIVE lane-strip geometry. Loaded once per map."""

    def __init__(self, road_map):
        strips, marks, edges = [], [], []
        for wp in road_map.generate_waypoints(1.0):
            if not driving(wp):
                continue
            for nxt in wp.next(1.0):
                if not driving(nxt) or (nxt.road_id, nxt.lane_id) != (wp.road_id, wp.lane_id):
                    continue
                strip = [side_point(wp, -1), side_point(wp, 1),
                         side_point(nxt, 1), side_point(nxt, -1)]
                strips.append(strip)
                sides = []
                hard = []
                for side, marking in ((-1, wp.left_lane_marking), (1, wp.right_lane_marking)):
                    kind = mark_type(marking)
                    sides.append(2 if kind in SOLID else 1 if kind in BROKEN and wp.s % 9 < 3 else 0)
                    hard.append(neighbor(wp, side) is None or kind == "Curb")
                marks.append(sides)
                edges.append(hard)
        if not strips:
            raise ValueError("Map has no driving strips")
        self.strips = np.asarray(strips)
        self.marks = np.asarray(marks)
        self.edges = np.asarray(edges)
        self.index = cKDTree(self.strips[:, :, :2].mean(axis=1))

    def masks(self, pose):
        ids = self.index.query_ball_point(pose[:2, 3], np.hypot(80, 30) + 4)
        strips = local(self.strips[ids], pose)
        bounds = np.array([X_MIN, Y_MIN]), np.array([X_MAX, Y_MAX])
        near = ((strips[..., :2].max(axis=1) >= bounds[0]) &
                (strips[..., :2].min(axis=1) <= bounds[1])).all(axis=1)
        # Reject overhead levels. Keep sloped road strips connected to the footprint.
        near &= np.max(np.abs(strips[..., 2]), axis=1) < 3.5
        masks = np.zeros((4, H, W), np.uint8)
        for strip, markings, edges in zip(strips[near], self.marks[np.asarray(ids, dtype=int)[near]],
                                          self.edges[np.asarray(ids, dtype=int)[near]]):
            xy = strip[:, :2]
            polygon(masks[0], xy)
            for segment, marking, edge in zip((xy[[0, 3]], xy[[1, 2]]), markings, edges):
                if marking:
                    line(masks[marking], segment)
                if edge:
                    line(masks[3], segment)
        return exclusive_markings(masks)


def exclusive_markings(masks):
    masks[1, masks[2].astype(bool)] = 0
    return masks


def lane_vectors(anchor, pose):
    points = [[] for _ in VECTOR_NAMES]
    if anchor is not None:
        chain = [anchor]
        for earlier, count in ((True, 24), (False, 62)):
            cursor = anchor
            for _ in range(count):
                options = cursor.previous(1.) if earlier else cursor.next(1.)
                if not options:
                    break
                nxt = max(options, key=lambda w: w.transform.get_forward_vector().dot(
                    cursor.transform.get_forward_vector()))
                chain.insert(0, nxt) if earlier else chain.append(nxt)
                cursor = nxt
        for wp in chain:
            left, right = (neighbor(wp, s) for s in (-1, 1))
            outer = []
            for side in (-1, 1):
                edge = wp
                for _ in range(12):
                    other = neighbor(edge, side)
                    if other is None:
                        outer.append(side_point(edge, side))
                        break
                    if not same_direction(edge, other):
                        outer.append(None)
                        break
                    edge = other
                else:
                    outer.append(None)
            sample = (xyz(wp.transform.location), side_point(wp, -1), side_point(wp, 1), *outer,
                      xyz(left.transform.location) if same_direction(wp, left) else None,
                      xyz(right.transform.location) if same_direction(wp, right) else None)
            for path, point in zip(points, sample):
                path.append(point)
    vectors = np.zeros((7, len(VECTOR_X), 2), dtype=np.float32)
    valid = np.zeros(vectors.shape[:2], dtype=bool)
    for index, path in enumerate(points):
        # Preserve gaps and path order, including lanes that end at a merge.
        for a, b in zip(path, path[1:]):
            if a is None or b is None:
                continue
            p, q = local(np.stack((a, b)), pose)[:, :2]
            if q[0] - p[0] <= 1e-5:
                continue
            mask = (VECTOR_X >= p[0]) & (VECTOR_X <= q[0])
            x = VECTOR_X[mask]
            vectors[index, mask] = np.column_stack((x, p[1] + (x - p[0]) * (q[1] - p[1]) / (q[0] - p[0])))
            valid[index] |= mask
    return vectors, valid, valid.any(axis=1)


@dataclass
class ActorState:
    center: np.ndarray
    heading: np.ndarray
    velocity: np.ndarray
    size: np.ndarray
    cls_id: int
    actor_id: int


def local_objects(actors, pose):
    objects, ids = [], []
    for actor in actors:
        center = local(actor.center, pose)
        if not (X_MIN <= center[0] <= X_MAX and Y_MIN <= center[1] <= Y_MAX):
            continue
        head = (actor.heading @ pose[:3, :3]) * [1, -1, 1]
        vel = (actor.velocity @ pose[:3, :3]) * [1, -1, 1]
        yaw = wrap(np.arctan2(head[1], head[0]))
        objects.append([*center, *actor.size, yaw, *vel[:2], actor.cls_id, 1.])
        ids.append(actor.actor_id)
    return np.asarray(objects, dtype=np.float32).reshape(-1, len(OBJECT_FIELDS)), np.asarray(ids, dtype=np.int64)


def object_slots(objects, actor_ids=None):
    good = ((objects[:, 3:6] >= MIN_BOX) & (objects[:, 3:6] <= MAX_BOX)).all(axis=1)
    # Pedestrians contribute to occupancy even when outside the locked vehicle dimension range.
    eligible = objects[good]
    order = np.argsort(np.linalg.norm(eligible[:, :3], axis=1), kind="stable")
    packed = np.zeros((OBJECT_SLOTS, len(OBJECT_FIELDS)), np.float32)
    count = min(len(order), OBJECT_SLOTS)
    packed[:count] = eligible[order[:count]]
    if actor_ids is None:
        return packed
    ids = np.full(OBJECT_SLOTS, -1, dtype=np.int64)
    ids[:count] = actor_ids[good][order[:count]]
    return packed, ids


def encode_boxes(boxes):
    """Reversible detector target components plus the original float32 yaw."""
    yaw = boxes[..., 6].copy()
    return dict(center_size=boxes[..., :6].copy(), yaw=yaw, velocity=boxes[..., 7:9].copy(),
                cls_id=boxes[..., 9].astype(np.int64), valid=boxes[..., 10].astype(bool),
                sin_yaw=np.sin(yaw), cos_yaw=np.cos(yaw))


def decode_boxes(encoded):
    # Stored yaw is authoritative for exact round trips. Predictions use atan2(sin, cos).
    geometry = encoded["center_size"]
    return np.concatenate((geometry, encoded["yaw"][..., None], encoded["velocity"],
                           encoded["cls_id"][..., None].astype(geometry.dtype),
                           encoded["valid"][..., None].astype(geometry.dtype)), axis=-1)


def footprint(box):
    x, y, _, length, width, _, yaw = box[:7]
    corners = np.array([[.5, .5], [.5, -.5], [-.5, -.5], [-.5, .5]]) * (length, width)
    c, s = np.cos(yaw), np.sin(yaw)
    return corners @ np.array([[c, s], [-s, c]]) + (x, y)


def occupancy(scan, objects):
    out = np.zeros((Z, H, W), np.uint8)
    col, row = np.floor(pixels(scan[:, :2]) + .5).astype(int).T
    height = np.searchsorted(Z_EDGES, scan[:, 2], side="right") - 1
    keep = (col >= 0) & (col < W) & (row >= 0) & (row < H) & (height >= 0) & (height < Z)
    out[height[keep], row[keep], col[keep]] = 1
    for box in objects:
        layers = (Z_EDGES[:-1] < box[2] + box[5] / 2) & (Z_EDGES[1:] > box[2] - box[5] / 2)
        corners = footprint(box)
        for layer in np.flatnonzero(layers):
            polygon(out[layer], corners)
    return out


def future_targets(pose, future_poses, future_speeds, future_actors):
    if len(future_poses) != FUTURE_TICKS[-1]:
        raise ValueError("Full 25-tick future is required")
    xy = local(np.asarray(future_poses)[:, :3, 3], pose)[:, :2]
    trajectory = np.column_stack((xy, future_speeds)).astype(np.float32)
    dynamic = np.zeros((len(FUTURE_TICKS), H, W), np.uint8)
    for out, tick in zip(dynamic, FUTURE_TICKS):
        boxes, _ = local_objects(future_actors[tick - 1], pose)
        for box in boxes:
            polygon(out, footprint(box))
    return trajectory, dynamic
