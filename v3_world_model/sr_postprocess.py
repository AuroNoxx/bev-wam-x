"""Car-OS post-processing for BEV object detections: gating, NMS, heading snap, and tracking."""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import H, W, X_MAX, X_MIN, Y_MAX, Y_MIN, pixels
from .labels import OBJECT_SLOTS
from .sr_geometry import clean_drivable_mask, estimate_road_tangent

# [L_min, L_max, W_min, W_max, H_min, H_max] per coarse class: car, suv, truck, two_wheeler, pedestrian.
_DIM_BOUNDS = np.array([
    [3.20, 5.40, 1.60, 2.20, 1.30, 1.85],
    [3.80, 5.80, 1.75, 2.35, 1.50, 2.25],
    [5.50, 14.0, 2.10, 2.80, 2.20, 4.00],
    [1.40, 2.60, 0.55, 1.10, 1.10, 1.80],
    [0.45, 1.10, 0.45, 1.10, 1.40, 2.00],
], dtype=np.float32)

_GRID_OFFSETS = np.array([(u, v) for u in (-0.38, 0.0, 0.38) for v in (-0.38, 0.0, 0.38)], dtype=np.float32)


def _wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _dilate_corridor(drivable_mask, dilation_m=2.0):
    clean = (clean_drivable_mask(drivable_mask) >= 0.5).astype(np.uint8)
    if clean.max() == 0:
        return clean
    k = max(3, 2 * int(round(dilation_m / 0.6)) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(clean, kernel)


def _sample_lidar(boxes, lidar_bev):
    n = len(boxes)
    if lidar_bev is None or n == 0:
        empty = np.zeros(n, dtype=bool)
        return empty, empty, empty
    lidar = np.asarray(lidar_bev, dtype=np.float32)
    cy = np.cos(boxes[:, 6])
    sy = np.sin(boxes[:, 6])
    local_xy = _GRID_OFFSETS[None, :, :] * boxes[:, None, 3:5]
    wx = boxes[:, None, 0] + local_xy[:, :, 0] * cy[:, None] - local_xy[:, :, 1] * sy[:, None]
    wy = boxes[:, None, 1] + local_xy[:, :, 0] * sy[:, None] + local_xy[:, :, 1] * cy[:, None]
    uv = np.floor(pixels(np.stack((wx, wy), axis=-1)) + 0.5).astype(np.int64)
    cols = np.clip(uv[:, :, 0], 0, W - 1)
    rows = np.clip(uv[:, :, 1], 0, H - 1)
    hi_z = lidar[1, rows, cols].max(axis=1)
    elev_density = (lidar[2, rows, cols] + lidar[3, rows, cols]).max(axis=1)
    log_cnt = lidar[5, rows, cols].max(axis=1)
    in_bev = ((boxes[:, 0] >= X_MIN) & (boxes[:, 0] <= X_MAX) &
              (np.abs(boxes[:, 1]) <= Y_MAX))
    elevated = in_bev & (hi_z > 0.35) & ((elev_density > 0.5) | (log_cnt > 0.69))
    strong_lidar = in_bev & (hi_z > 0.50) & (elev_density >= 2.0)
    empty_ground = in_bev & (log_cnt > 0.69) & (hi_z <= 0.25) & (elev_density < 0.25)
    return elevated, strong_lidar, empty_ground


def _rotated_iou(box_a, box_b):
    dx = float(box_a[0] - box_b[0])
    dy = float(box_a[1] - box_b[1])
    ra = 0.5 * float(np.hypot(box_a[3], box_a[4]))
    rb = 0.5 * float(np.hypot(box_b[3], box_b[4]))
    if np.hypot(dx, dy) > ra + rb:
        return 0.0
    rect_a = ((float(box_a[0]), float(box_a[1])), (float(box_a[3]), float(box_a[4])), float(np.degrees(box_a[6])))
    rect_b = ((float(box_b[0]), float(box_b[1])), (float(box_b[3]), float(box_b[4])), float(np.degrees(box_b[6])))
    ret, pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if ret == cv2.INTERSECT_NONE or pts is None or len(pts) < 3:
        return 0.0
    inter = float(cv2.contourArea(cv2.convexHull(pts.reshape(-1, 2).astype(np.float32))))
    union = max(float(box_a[3] * box_a[4] + box_b[3] * box_b[4] - inter), 1e-6)
    return inter / union


def _nms_indices(boxes, scores, iou_thresh=0.12):
    order = np.argsort(-scores, kind="stable")
    kept = []
    for idx in order:
        cand = boxes[idx]
        cls_c = int(cand[9])
        suppress = False
        for k in kept:
            ref = boxes[k]
            cls_r = int(ref[9])
            if _rotated_iou(cand, ref) > iou_thresh:
                suppress = True
                break
            if cls_c < 4 and cls_r < 4:
                dx = float(cand[0] - ref[0])
                dy = float(cand[1] - ref[1])
                cos_r, sin_r = float(np.cos(ref[6])), float(np.sin(ref[6]))
                along = dx * cos_r + dy * sin_r
                across = -dx * sin_r + dy * cos_r
                ang = abs(float(_wrap(cand[6] - ref[6])))
                aligned = min(ang, abs(np.pi - ang)) < np.radians(50.0)
                if (cls_c == 3) != (cls_r == 3):
                    v_l = float(ref[3] if cls_r < 3 else cand[3])
                    v_w = float(ref[4] if cls_r < 3 else cand[4])
                    if abs(along) < 0.65 * v_l + 0.8 and abs(across) < 0.65 * v_w + 0.7:
                        suppress = True
                        break
                elif aligned:
                    if abs(along) < 0.55 * float(ref[3] + cand[3]) and abs(across) < 0.65 * float(ref[4] + cand[4]):
                        suppress = True
                        break
                    if cls_c < 3 and cls_r < 3 and np.hypot(dx, dy) < 2.8:
                        suppress = True
                        break
        if not suppress:
            kept.append(int(idx))
    return np.asarray(kept, dtype=np.int64)


def _align_headings(boxes, lane_paths):
    if len(boxes) == 0 or not lane_paths:
        return boxes
    tangent, dist = estimate_road_tangent(boxes[:, :2], lane_paths)
    is_vehicle = boxes[:, 9] < 3.5  # vehicles & two-wheelers align; pedestrians excluded
    yaw = boxes[:, 6].astype(np.float32)
    diff = _wrap(tangent - yaw)
    flipped = _wrap(tangent + np.pi - yaw)
    target = np.where(np.abs(flipped) < np.abs(diff), tangent + np.pi, tangent)
    residual = np.abs(_wrap(target - yaw))
    snap = is_vehicle & (dist < 8.5) & (residual < np.radians(50.0))
    if snap.any():
        boxes = boxes.copy()
        boxes[snap, 6] = _wrap(yaw[snap] + 0.80 * _wrap(target[snap] - yaw[snap]))
    return boxes


@dataclass
class _Track:
    track_id: int
    box: np.ndarray
    score: float
    hits: int
    missed: int
    confirmed: bool
    prev_vx: float
    decelerating: bool = False


class SRPostProcessor:
    """Stateless Car-OS cleanup plus stateful ego-compensated BEV vehicle tracking."""

    def __init__(self, *, max_missed=1, min_confirm_hits=2, road_dilation_m=2.2, iou_thresh=0.12):
        self.max_missed = int(max_missed)
        self.min_confirm_hits = int(min_confirm_hits)
        self.road_dilation_m = float(road_dilation_m)
        self.iou_thresh = float(iou_thresh)
        self._tracks: list[_Track] = []
        self._next_id = 1

    def reset(self):
        self._tracks.clear()
        self._next_id = 1

    @staticmethod
    def clean_frame(boxes, scores=None, drivable_mask=None, lane_paths=None, lidar_bev=None, *,
                    road_dilation_m=2.2, iou_thresh=0.12, return_lidar_mask=False):
        """Filter ego overlap, off-road ghosts, rotated duplicates, and align lane headings."""
        boxes = np.asarray(boxes, dtype=np.float32)
        if len(boxes) == 0:
            empty = np.empty((0, 11), dtype=np.float32)
            empty_s = np.empty((0,), dtype=np.float32)
            return (empty, empty_s, np.empty((0,), dtype=bool)) if return_lidar_mask else (empty, empty_s)
        scores = np.asarray(boxes[:, -1] if scores is None else scores, dtype=np.float32).reshape(-1)
        finite = np.isfinite(boxes[:, :10]).all(axis=1) & np.isfinite(scores)
        not_ego = ~((np.abs(boxes[:, 0]) < 2.2) & (np.abs(boxes[:, 1]) < 1.1))
        in_roi = (boxes[:, 0] >= X_MIN - 2.0) & (boxes[:, 0] <= X_MAX + 2.0) & (np.abs(boxes[:, 1]) <= Y_MAX + 2.0)
        base_thresh = 0.45 if lidar_bev is not None else 0.50
        keep = finite & not_ego & in_roi & (scores >= base_thresh)
        if not keep.any():
            empty = np.empty((0, 11), dtype=np.float32)
            empty_s = np.empty((0,), dtype=np.float32)
            return (empty, empty_s, np.empty((0,), dtype=bool)) if return_lidar_mask else (empty, empty_s)

        boxes = boxes[keep].copy()
        scores = scores[keep].copy()
        cls_id = np.clip(np.rint(boxes[:, 9]).astype(np.int64), 0, len(_DIM_BOUNDS) - 1)
        boxes[:, 9] = cls_id.astype(np.float32)
        bounds = _DIM_BOUNDS[cls_id]
        boxes[:, 3] = np.clip(boxes[:, 3], bounds[:, 0], bounds[:, 1])
        boxes[:, 4] = np.clip(boxes[:, 4], bounds[:, 2], bounds[:, 3])
        boxes[:, 5] = np.clip(boxes[:, 5], bounds[:, 4], bounds[:, 5])
        boxes[:, 2] = np.clip(boxes[:, 2], -0.5, 1.0)
        boxes[:, 6] = _wrap(boxes[:, 6])

        elevated, strong_lidar, empty_ground = _sample_lidar(boxes, lidar_bev)
        if lidar_bev is not None:
            pass_score = np.where(strong_lidar, 0.45, np.where(elevated, 0.48, 0.52))
            unconfirmed_ground = empty_ground & (scores < np.where(cls_id < 3, 0.65, 0.56))
            lidar_keep = (scores >= pass_score) & (~unconfirmed_ground)
            if not lidar_keep.any():
                empty = np.empty((0, 11), dtype=np.float32)
                empty_s = np.empty((0,), dtype=np.float32)
                return (empty, empty_s, np.empty((0,), dtype=bool)) if return_lidar_mask else (empty, empty_s)
            boxes, scores = boxes[lidar_keep], scores[lidar_keep]
            elevated, strong_lidar = elevated[lidar_keep], strong_lidar[lidar_keep]
            cls_id = cls_id[lidar_keep]

        if drivable_mask is not None:
            corridor = _dilate_corridor(drivable_mask, dilation_m=road_dilation_m)
            if corridor.max() > 0:
                uv = np.rint(pixels(boxes[:, :2])).astype(np.int64)
                in_grid = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                cols = np.clip(uv[:, 0], 0, W - 1)
                rows = np.clip(uv[:, 1], 0, H - 1)
                on_road = in_grid & (corridor[rows, cols] > 0)
                ped_sidewalk = (cls_id == 4) & (scores >= 0.62)
                strong_shoulder = (scores >= 0.62) & strong_lidar
                road_keep = on_road | ped_sidewalk | strong_shoulder
                if not road_keep.any():
                    empty = np.empty((0, 11), dtype=np.float32)
                    empty_s = np.empty((0,), dtype=np.float32)
                    return (empty, empty_s, np.empty((0,), dtype=bool)) if return_lidar_mask else (empty, empty_s)
                boxes, scores, elevated = boxes[road_keep], scores[road_keep], elevated[road_keep]

        # Align headings before NMS so parallel duplicate suppression sees clean headings.
        boxes = _align_headings(boxes, lane_paths)
        idx = _nms_indices(boxes, scores, iou_thresh=iou_thresh)
        boxes = boxes[idx]
        scores = scores[idx]
        elevated = elevated[idx]
        boxes[:, 10] = 1.0
        if return_lidar_mask:
            return boxes, scores, elevated
        return boxes, scores

    def _warp_tracks(self, ego_delta, dt=0.1):
        if not self._tracks:
            return
        for tr in self._tracks:
            tr.box[:2] = tr.box[:2] + np.clip(tr.box[7:9], -35.0, 35.0) * float(dt)
        if ego_delta is None:
            return
        dx, dy, angle = float(ego_delta[0]), float(ego_delta[1]), float(ego_delta[2])
        if np.hypot(dx, dy) > 8.0 or abs(angle) > 0.35:
            self._tracks.clear()
            return
        c, s = float(np.cos(angle)), float(np.sin(angle))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        shift = np.array([dx, dy], dtype=np.float32)
        for tr in self._tracks:
            tr.box[:2] = tr.box[:2] @ rot + shift
            tr.box[6] = float(_wrap(tr.box[6] - angle))
            tr.box[7:9] = tr.box[7:9] @ rot

    def update(self, boxes, scores=None, drivable_mask=None, lane_paths=None,
               lidar_bev=None, ego_delta=None, dt=0.1):
        """Run single-frame cleanup and temporal association, returning padded OBJECT_SLOTS outputs."""
        slots = max(OBJECT_SLOTS, len(boxes) if boxes is not None else 0)
        self._warp_tracks(ego_delta, dt=dt)
        det_boxes, det_scores, _ = self.clean_frame(
            boxes, scores, drivable_mask, lane_paths, lidar_bev,
            road_dilation_m=self.road_dilation_m, iou_thresh=self.iou_thresh,
            return_lidar_mask=True,
        )

        matched_tracks = set()
        matched_dets = set()
        if self._tracks and len(det_boxes):
            nt, nd = len(self._tracks), len(det_boxes)
            cost = np.full((nt, nd), 1e6, dtype=np.float32)
            t_xy = np.array([t.box[:2] for t in self._tracks], dtype=np.float32)
            t_cls = np.array([int(t.box[9]) for t in self._tracks], dtype=np.int64)
            d_xy = det_boxes[:, :2]
            d_cls = det_boxes[:, 9].astype(np.int64)
            dists = np.linalg.norm(t_xy[:, None, :] - d_xy[None, :, :], axis=-1)
            # Disallow pedestrian-to-heavy-vehicle matching while allowing minor car/suv relabeling.
            t_group = np.where(t_cls < 3, 0, t_cls)
            d_group = np.where(d_cls < 3, 0, d_cls)
            compat = t_group[:, None] == d_group[None, :]
            cls_pen = np.where(t_cls[:, None] == d_cls[None, :], 0.0, 1.0)
            valid = compat & (dists <= 3.8)
            cost[valid] = dists[valid] + cls_pen[valid]
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] > 10.0:
                    continue
                tr = self._tracks[r]
                det = det_boxes[c]
                sc = float(det_scores[c])
                matched_tracks.add(r)
                matched_dets.add(c)
                tr.missed = 0
                tr.hits += 1
                tr.decelerating = bool((tr.prev_vx - float(det[7])) > 0.38)
                tr.prev_vx = float(det[7])
                tr.box[:2] = 0.70 * det[:2] + 0.30 * tr.box[:2]
                tr.box[2] = float(det[2])
                tr.box[3:6] = 0.35 * det[3:6] + 0.65 * tr.box[3:6]
                tr.box[6] = float(_wrap(tr.box[6] + 0.65 * _wrap(det[6] - tr.box[6])))
                tr.box[7:10] = det[7:10]
                tr.box[10] = 1.0
                tr.score = 0.70 * sc + 0.30 * tr.score
                if tr.hits >= self.min_confirm_hits or tr.score >= 0.78:
                    tr.confirmed = True

        survivors = []
        for idx, tr in enumerate(self._tracks):
            if idx in matched_tracks:
                survivors.append(tr)
            elif tr.confirmed and tr.missed + 1 <= self.max_missed:
                tr.missed += 1
                tr.score *= 0.85
                tr.decelerating = False
                survivors.append(tr)

        for d_idx in range(len(det_boxes)):
            if d_idx in matched_dets:
                continue
            det = det_boxes[d_idx].copy()
            sc = float(det_scores[d_idx])
            confirmed = (self.min_confirm_hits <= 1) or (sc >= 0.78)
            survivors.append(_Track(
                track_id=self._next_id, box=det, score=sc, hits=1, missed=0,
                confirmed=confirmed, prev_vx=float(det[7]),
            ))
            self._next_id += 1

        self._tracks = survivors
        confirmed_tracks = [t for t in self._tracks if t.confirmed]
        if confirmed_tracks:
            t_boxes = np.stack([t.box for t in confirmed_tracks], axis=0)
            eff_scores = np.array(
                [t.score + (1.0 if t.missed == 0 else 0.0) for t in confirmed_tracks], dtype=np.float32
            )
            keep_idx = _nms_indices(t_boxes, eff_scores, iou_thresh=self.iou_thresh)
            confirmed_tracks = [confirmed_tracks[int(i)] for i in keep_idx]
        active = sorted(confirmed_tracks, key=lambda t: -t.score)[:slots]

        out_boxes = np.zeros((slots, 11), dtype=np.float32)
        out_scores = np.zeros((slots,), dtype=np.float32)
        out_ids = np.full((slots,), -1, dtype=np.int32)
        out_decel = np.zeros((slots,), dtype=bool)
        for i, tr in enumerate(active):
            out_boxes[i] = tr.box
            out_boxes[i, 10] = 1.0
            out_scores[i] = tr.score
            out_ids[i] = tr.track_id
            out_decel[i] = tr.decelerating
        return out_boxes, out_scores, out_ids, out_decel
