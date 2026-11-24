"""Metric road meshes from raster observations; FLU geometry throughout."""
import cv2
import mapbox_earcut
import numpy as np
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize

from .geometry import DX, DY, H, W, X_MAX, X_MIN, Y_MAX, Y_MIN, grid, metric, pixels


def smooth(points, closed=False):
    # Simplify sub-cell staircase oscillations before corner cutting.
    line = cv2.approxPolyDP(np.ascontiguousarray(points, np.float32), .43, closed).reshape(-1, 2)
    if len(line) < (3 if closed else 2):
        return np.empty((0, 2), np.float32)
    if len(line) == 2:
        return line
    for _ in range(2):
        a, b = (line, np.roll(line, -1, axis=0)) if closed else (line[:-1], line[1:])
        delta = b - a
        # Bound corner cuts in metres on long straight edges (especially at ROI borders).
        fraction = np.minimum(.25, .5 / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-6))
        corners = np.stack((a + fraction*delta, b - fraction*delta), axis=1).reshape(-1, 2)
        line = corners if closed else np.vstack((line[0], corners, line[-1]))
    return line


def fit_curve(points, closed=False):
    """Fit an observed metric polyline, keeping branch endpoints and sharp corners."""
    if len(points) < 3:
        return points
    path = np.vstack((points, points[0])) if closed else points
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    length = distance[-1]
    if length < (12. if closed else 2.):
        return points
    count = max(5, int(np.ceil(length / .45)))
    t = np.linspace(0., length, count, endpoint=not closed)
    samples = np.column_stack([np.interp(t, distance, axis) for axis in path.T])
    window = min(25, (len(samples) - 1) // 2 * 2 + 1)
    fitted = savgol_filter(samples, window, 2, axis=0, mode="wrap" if closed else "interp")
    if not closed:
        chord = samples[-1] - samples[0]
        tangent = chord / max(np.linalg.norm(chord), 1e-6)
        straight = samples[0] + ((samples - samples[0]) @ tangent)[:, None] * tangent
        error = np.linalg.norm(straight - samples, axis=1)
        if np.linalg.norm(chord) > 2 and np.quantile(error, .9) < .48 and error.max() < .85:
            fitted = straight
    prev = np.roll(samples, 4, axis=0) if closed else samples[np.maximum(0, np.arange(count) - 4)]
    nxt = np.roll(samples, -4, axis=0) if closed else samples[np.minimum(count - 1, np.arange(count) + 4)]
    a, b = samples - prev, nxt - samples
    cosine = (a * b).sum(axis=1) / np.maximum(1e-8, np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    delta = fitted - samples
    delta *= np.minimum(1., .85 / np.maximum(1e-8, np.linalg.norm(delta, axis=1)))[:, None]
    delta *= np.clip((cosine - .62) / .25, 0., 1.)[:, None]
    if not closed:
        delta[[0, -1]] = 0.
    result = (samples + delta).astype(np.float32)
    if closed:
        old_area = cv2.contourArea(np.asarray(points, np.float32), oriented=True)
        new_area = cv2.contourArea(result, oriented=True)
        if not .9 < new_area / (old_area or 1e-8) < 1.1:
            return points
    return result


def _dilate_road(drivable_mask, radius_m=1.0):
    binary = (np.asarray(drivable_mask, np.float32) >= .5).astype(np.uint8)
    if binary.max() == 0:
        return binary
    k = max(3, 2 * int(round(radius_m / 0.6)) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(binary, kernel)


def clean_drivable_mask(prob):
    """Remove isolated asphalt islands and fill tiny pinholes while keeping real medians/holes."""
    binary = (np.asarray(prob, np.float32) >= .5).astype(np.uint8)
    if binary.max() == 0:
        return np.zeros(binary.shape, np.float32)
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k7, borderType=cv2.BORDER_REPLICATE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, borderType=cv2.BORDER_REPLICATE)
    h, w = mask.shape
    num_bg, bg_labels, bg_stats, _ = cv2.connectedComponentsWithStats(1 - mask, connectivity=4)
    if num_bg > 1:
        border_labels = set(np.unique(np.concatenate((
            bg_labels[0, :], bg_labels[-1, :], bg_labels[:, 0], bg_labels[:, -1]
        ))).tolist())
        for k in range(1, num_bg):
            if k not in border_labels and bg_stats[k, cv2.CC_STAT_AREA] < 28:
                mask[bg_labels == k] = 1
    num_fg, fg_labels, fg_stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_fg <= 1:
        return np.zeros(binary.shape, np.float32)
    ego_uv = np.rint(pixels(np.array([[0., 0.]], np.float32))[0]).astype(int)
    ego_col = int(np.clip(ego_uv[0], 0, w - 1))
    ego_row = int(np.clip(ego_uv[1], 0, h - 1))
    ego_label = int(fg_labels[ego_row, ego_col])
    if ego_label == 0:
        patch = fg_labels[max(0, ego_row - 4):min(h, ego_row + 5),
                          max(0, ego_col - 4):min(w, ego_col + 5)]
        nz = patch[patch > 0]
        if len(nz):
            vals, counts = np.unique(nz, return_counts=True)
            ego_label = int(vals[counts.argmax()])
    keep = np.zeros(binary.shape, np.float32)
    chosen = ego_label if ego_label > 0 else int(np.argmax(fg_stats[1:, cv2.CC_STAT_AREA])) + 1
    keep[fg_labels == chosen] = 1.0
    return keep


def _surface_contours(prob, *, fit=False):
    scale = 4
    fine = cv2.resize(np.asarray(prob, np.float32), None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    binary = (cv2.GaussianBlur(fine, (0, 0), 1.6) >= .5).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], None, scale
    rings = [smooth(metric((poly.reshape(-1, 2) + .5)/scale - .5), True) for poly in contours]
    if fit:
        rings = [fit_curve(ring, closed=True) if len(ring) >= 4 else ring for ring in rings]
    return rings, hierarchy[0], scale


def _triangulate_contours(rings, hier):
    if hier is None:
        return np.empty((0, 2), np.float32)
    triangles = []
    for i, (_, _, child, parent) in enumerate(hier):
        if parent >= 0 or len(rings[i]) < 3:
            continue
        group = [rings[i]]
        while child >= 0:
            if len(rings[child]) >= 3:
                group.append(rings[child])
            child = hier[child, 0]
        vertices = np.concatenate(group)
        ends = np.cumsum([len(r) for r in group], dtype=np.uint32)
        triangles.append(vertices[mapbox_earcut.triangulate_float32(vertices, ends)])
    return np.concatenate(triangles) if triangles else np.empty((0, 2), np.float32)


def surface_triangles(prob, *, fit=False):
    # Clean noisy far islands/pinholes in smooth Car-OS mode while preserving large holes.
    if fit:
        prob = clean_drivable_mask(prob)
    rings, hier, _ = _surface_contours(prob, fit=fit)
    return _triangulate_contours(rings, hier)


def centerlines(prob):
    binary = np.asarray(prob) >= .5
    skeleton = skeletonize(binary)
    pixels_rc = set(map(tuple, np.argwhere(skeleton).tolist()))
    # Suppress diagonal shortcuts across an existing orthogonal link.
    neighbors = {}
    for r, c in pixels_rc:
        neighbors[r, c] = {(r+dr, c+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                           if (dr or dc) and (r+dr, c+dc) in pixels_rc and
                           not (dr and dc and ((r+dr,c) in pixels_rc or (r,c+dc) in pixels_rc))}
    used = set()
    paths = []
    starts = [p for p in neighbors if len(neighbors[p]) != 2]
    for start in starts + list(neighbors):
        for nxt in neighbors[start]:
            if (start,nxt) in used:
                continue
            path = [start]
            a,b = start,nxt
            while (a,b) not in used:
                used.update(((a,b),(b,a)))
                path.append(b)
                if len(neighbors[b]) != 2 or b == start:
                    break
                a,b = b,next(iter(neighbors[b] - {a}))
            closed = path[-1] == start
            points = metric(np.asarray(path[:-1] if closed else path)[:, ::-1])
            points = smooth(points, closed)
            if len(points):
                paths.append(np.vstack((points, points[0])) if closed else points)
    return paths


def temporal_probabilities(current, previous, delta=None, *, current_weight=0.65):
    if previous is None:
        return current
    if delta is not None:
        dx, dy, angle = delta
        if np.hypot(dx,dy) > 8 or abs(angle) > .35:
            return current
        c,s = np.cos(angle),np.sin(angle)
        uv = pixels((grid() - [dx,dy]) @ np.array([[c,s],[-s,c]])).astype(np.float32)
        warped = cv2.remap(previous.transpose(1,2,0), uv, None, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,borderValue=(-1,)*4).transpose(2,0,1)
        previous = np.where(warped >= 0,warped,current)
    w = np.float32(current_weight)
    return w * current + (np.float32(1.0) - w) * previous


def _pack_paths(paths):
    paths = [np.asarray(path, np.float32) for path in paths if len(path) >= 2]
    if not paths:
        return None
    lengths = np.array([len(path) for path in paths])
    points = np.concatenate(paths)
    start = np.r_[0, lengths.cumsum()[:-1]]
    end = lengths.cumsum() - 1
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangent[start] = points[start + 1] - points[start]
    tangent[end] = points[end] - points[end - 1]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    segments = np.ones(len(points), bool)
    segments[end] = False
    return points, normal, np.flatnonzero(segments)


def _arc_length(pts):
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) >= 2 else 0.0


def _ring_area(pts):
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _resample_open(pts, step=0.5):
    pts = np.asarray(pts, np.float32)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.r_[True, seg > 1e-4]]
    if len(pts) < 2:
        return pts
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    total = float(s[-1])
    if total <= step:
        return pts
    n = max(int(np.ceil(total / step)) + 1, 2)
    s_new = np.linspace(0.0, total, n, dtype=np.float32)
    return np.column_stack((np.interp(s_new, s, pts[:, 0]), np.interp(s_new, s, pts[:, 1]))).astype(np.float32)


def _end_dir(pts, at_end=True):
    k = min(len(pts) - 1, 4)
    vec = (pts[-1] - pts[-1 - k]) if at_end else (pts[k] - pts[0])
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        vec = pts[-1] - pts[0]
        norm = float(np.linalg.norm(vec))
    return (vec / max(norm, 1e-6)).astype(np.float32)


def _merge_collinear_paths(paths, *, max_gap=10.0, cos_thresh=0.94, max_lat=0.65):
    chains = [np.asarray(p, np.float32) for p in paths if _arc_length(p) >= 1.0]
    while len(chains) >= 2:
        m = len(chains)
        s_pt = np.array([c[0] for c in chains], dtype=np.float32)
        e_pt = np.array([c[-1] for c in chains], dtype=np.float32)
        ds = np.array([_end_dir(c, at_end=False) for c in chains], dtype=np.float32)
        de = np.array([_end_dir(c, at_end=True) for c in chains], dtype=np.float32)
        out_pt = np.vstack((e_pt, s_pt))
        out_dir = np.vstack((de, -ds))
        in_pt = np.vstack((s_pt, e_pt))
        in_dir = np.vstack((ds, -de))
        cid = np.tile(np.arange(m), 2)

        delta = in_pt[None, :, :] - out_pt[:, None, :]
        dist = np.hypot(delta[:, :, 0], delta[:, :, 1])
        valid = (cid[:, None] != cid[None, :]) & (dist <= max_gap)
        valid &= (out_dir @ in_dir.T) >= cos_thresh
        far = dist > 0.25
        if np.any(far & valid):
            inv_d = 1.0 / np.maximum(dist, 1e-6)
            chord = delta * inv_d[:, :, None]
            dot_oc = np.sum(out_dir[:, None, :] * chord, axis=-1) >= cos_thresh
            dot_ic = np.sum(in_dir[None, :, :] * chord, axis=-1) >= cos_thresh
            lat = np.abs(out_dir[:, None, 0] * delta[:, :, 1] - out_dir[:, None, 1] * delta[:, :, 0]) <= max_lat
            valid &= (~far) | (dot_oc & dot_ic & lat)
        if not np.any(valid):
            break
        u, v = np.unravel_index(int(np.argmin(np.where(valid, dist, np.inf))), dist.shape)
        i, rev_i = int(u % m), bool(u >= m)
        j, rev_j = int(v % m), bool(v >= m)
        ci = chains[i][::-1] if rev_i else chains[i]
        cj = chains[j][::-1] if rev_j else chains[j]
        merged = np.vstack((ci, cj[1:] if np.linalg.norm(cj[0] - ci[-1]) < 1e-3 else cj))
        hi, lo = max(i, j), min(i, j)
        chains.pop(hi)
        chains.pop(lo)
        chains.append(merged.astype(np.float32))
    return chains


def _extract_clean_paths(mask, drivable_mask=None, exclude_mask=None, *,
                         min_len=2.0, max_gap=10.0, fit=True):
    binary = (np.asarray(mask, np.float32) >= .5)
    if drivable_mask is not None:
        binary = binary & (_dilate_road(drivable_mask, 0.9) > 0)
    if exclude_mask is not None:
        excl = cv2.dilate((np.asarray(exclude_mask, np.float32) >= .5).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        binary = binary & (excl == 0)
    if not binary.any():
        return []
    raw_paths = centerlines(binary.astype(np.float32))
    closed_paths, open_paths = [], []
    for path in raw_paths:
        if len(path) >= 3 and np.array_equal(path[0], path[-1]):
            closed_paths.append(np.asarray(path, np.float32))
        else:
            open_paths.append(np.asarray(path, np.float32))
    merged = _merge_collinear_paths(open_paths, max_gap=max_gap)
    out = []
    for path in merged:
        if _arc_length(path) < min_len:
            continue
        sampled = _resample_open(path, step=0.5)
        if fit and len(sampled) >= 4:
            sampled = fit_curve(sampled, closed=False)
        if _arc_length(sampled) >= min_len:
            out.append(sampled.astype(np.float32))
    for path in closed_paths:
        if _arc_length(path) < min_len:
            continue
        if fit and len(path) > 4:
            curve = fit_curve(path[:-1], closed=True)
            path = np.vstack((curve, curve[0])).astype(np.float32)
        out.append(path)
    return out


def _slice_dashes(paths, *, dash_len=3.0, gap_len=5.5, min_dash_len=0.8):
    period = float(dash_len + gap_len)
    dashes = []
    for path in paths:
        pts = _resample_open(path, step=0.35)
        if len(pts) < 2:
            continue
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        if s[-1] < min_dash_len:
            continue
        span = pts[-1] - pts[0]
        mean_dir = span / max(float(np.linalg.norm(span)), 1e-6)
        s_global = s + float(np.dot(pts[0], mean_dir))
        k0 = int(np.floor((s_global[0] - dash_len) / period))
        k1 = int(np.ceil(s_global[-1] / period))
        for k in range(k0, k1 + 1):
            a = max(float(s_global[0]), k * period)
            b = min(float(s_global[-1]), k * period + float(dash_len))
            if b - a < min_dash_len:
                continue
            mid_mask = (s_global > a + 1e-4) & (s_global < b - 1e-4)
            s_sample = np.r_[a, s_global[mid_mask], b]
            dash = np.column_stack((np.interp(s_sample, s_global, pts[:, 0]),
                                    np.interp(s_sample, s_global, pts[:, 1]))).astype(np.float32)
            dashes.append(dash)
    return dashes


def dashed_lane_paths(mask, drivable_mask=None, *, fit=True):
    """Cleaned continuous polylines tracing dashed lane dividers."""
    return _extract_clean_paths(mask, drivable_mask, min_len=2.0, max_gap=10.0, fit=fit)


def solid_lane_paths(mask, drivable_mask=None, dashed_mask=None, *, fit=True):
    """Cleaned continuous polylines tracing solid lane/shoulder markings."""
    return _extract_clean_paths(mask, drivable_mask, exclude_mask=dashed_mask,
                                min_len=3.0, max_gap=3.5, fit=fit)


def pack_paths(paths):
    """Pack continuous polylines into `(points, normal, segment_starts)` for ribbon/curb sweeping."""
    return _pack_paths(paths)


def pack_dashed_paths(paths, *, dash_len=3.0, gap_len=5.5):
    """Procedurally slice smooth lane polylines into uniform Car-OS dashes."""
    return _pack_paths(_slice_dashes(paths, dash_len=dash_len, gap_len=gap_len))


def dashed_segments(mask, drivable_mask=None, *, dash_len=3.0, gap_len=5.5, fit=False):
    """Procedural uniform dashed lane strips snapped to smoothed dashed centerlines."""
    paths = dashed_lane_paths(mask, drivable_mask, fit=fit)
    return pack_dashed_paths(paths, dash_len=dash_len, gap_len=gap_len)


def solid_segments(mask, drivable_mask=None, dashed_mask=None, *, fit=False):
    """Cleaned continuous solid lane markings inside the drivable road."""
    return _pack_paths(solid_lane_paths(mask, drivable_mask, dashed_mask=dashed_mask, fit=fit))


def _split_ring_on_roi(ring, *, min_arc_len=4.0, fit=False):
    on_roi = ((ring[:, 0] >= X_MAX - 1.6) | (ring[:, 0] <= X_MIN + 1.6) |
              (np.abs(ring[:, 1]) >= Y_MAX - 1.4))
    inside = ~on_roi
    if inside.all():
        closed = np.vstack((ring, ring[0])).astype(np.float32)
        return [closed] if _arc_length(closed) >= min_arc_len else []
    if not inside.any():
        return []
    cut = int(np.flatnonzero(on_roi)[0])
    rolled_ring = np.roll(ring, -cut, axis=0)
    rolled_in = np.roll(inside, -cut)
    edges = np.flatnonzero(np.diff(np.r_[False, rolled_in, False].astype(np.int8)))
    arcs = []
    for start, end in edges.reshape(-1, 2):
        if end - start >= 2:
            arc = rolled_ring[start:end].astype(np.float32)
            if _arc_length(arc) >= min_arc_len:
                if fit:
                    arc = _resample_open(arc, step=1.0)
                    if len(arc) >= 6:
                        arc = fit_curve(arc, closed=False).astype(np.float32)
                arcs.append(arc)
    return arcs


def _curb_paths_from_contours(rings, hier, *, fit=False):
    curb_paths = []
    min_hole_area = 40.0 * DX * DY
    if hier is not None:
        for i, (_, _, _, parent) in enumerate(hier):
            ring = rings[i]
            if len(ring) < 3:
                continue
            if parent >= 0 and _ring_area(ring) < min_hole_area:
                continue
            curb_paths.extend(_split_ring_on_roi(ring, min_arc_len=4.0, fit=fit))
    return curb_paths


def road_surface_and_curbs(clean_road, *, fit=True):
    """Build triangulated road surface and continuous open border curbs from one contour pass."""
    rings, hier, _ = _surface_contours(clean_road, fit=fit)
    triangles = _triangulate_contours(rings, hier)
    curbs = _pack_paths(_curb_paths_from_contours(rings, hier, fit=fit))
    return triangles, curbs


def road_curb_paths(drivable_prob, edge_prob=None, *, fit=False):
    """Continuous open curb barriers extracted from cleaned road outer/median contours."""
    del edge_prob  # CARLA junction internal edge masks create cobweb artifacts across intersections.
    clean = clean_drivable_mask(drivable_prob)
    rings, hier, _ = _surface_contours(clean, fit=fit)
    return _pack_paths(_curb_paths_from_contours(rings, hier, fit=fit))


def estimate_road_tangent(xy, lane_paths):
    """Nearest smooth polyline tangent angle (rad) and distance (m) for BEV query points."""
    pts = np.atleast_2d(np.asarray(xy, np.float32))
    n = len(pts)
    if not lane_paths or n == 0:
        return np.zeros(n, np.float32), np.full(n, np.inf, np.float32)
    starts, ends = [], []
    for path in lane_paths:
        arr = np.asarray(path, np.float32)
        if len(arr) >= 2:
            starts.append(arr[:-1])
            ends.append(arr[1:])
    if not starts:
        return np.zeros(n, np.float32), np.full(n, np.inf, np.float32)
    a = np.concatenate(starts, axis=0)
    b = np.concatenate(ends, axis=0)
    ab = b - a
    len2 = np.sum(ab * ab, axis=1)
    keep = len2 > 1e-6
    if not keep.any():
        return np.zeros(n, np.float32), np.full(n, np.inf, np.float32)
    a, ab, len2 = a[keep], ab[keep], len2[keep]
    angles = np.arctan2(ab[:, 1], ab[:, 0]).astype(np.float32)
    ap = pts[:, None, :] - a[None, :, :]
    t = np.clip(np.sum(ap * ab[None, :, :], axis=-1) / len2[None, :], 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    dist2 = np.sum((pts[:, None, :] - closest) ** 2, axis=-1)
    best = dist2.argmin(axis=1)
    return angles[best], np.sqrt(dist2[np.arange(n), best]).astype(np.float32)


def sweep_segments(mask, *, fit=False):
    paths = centerlines(mask)
    if fit:
        fitted = []
        for path in paths:
            closed = np.array_equal(path[0], path[-1])
            curve = fit_curve(path[:-1] if closed else path, closed)
            fitted.append(np.vstack((curve, curve[0])) if closed else curve)
        paths = fitted
    return _pack_paths(paths)
