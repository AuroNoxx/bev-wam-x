"""Metric road meshes from raster observations; FLU geometry throughout."""
import cv2
import mapbox_earcut
import numpy as np
from skimage.morphology import skeletonize

from .geometry import grid, metric, pixels


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


def surface_triangles(prob):
    # Oversample probabilities before thresholding to retain sub-cell boundaries.
    scale = 4
    fine = cv2.resize(np.asarray(prob, np.float32), None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    binary = (cv2.GaussianBlur(fine, (0, 0), 1.6) >= .5).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return np.empty((0, 2), np.float32)
    rings = [smooth(metric((poly.reshape(-1, 2) + .5)/scale - .5), True) for poly in contours]
    triangles = []
    for i, (_, _, child, parent) in enumerate(hierarchy[0]):
        if parent >= 0 or len(rings[i]) < 3:
            continue
        group = [rings[i]]
        while child >= 0:
            if len(rings[child]) >= 3:
                group.append(rings[child])
            child = hierarchy[0, child, 0]
        vertices = np.concatenate(group)
        ends = np.cumsum([len(r) for r in group], dtype=np.uint32)
        triangles.append(vertices[mapbox_earcut.triangulate_float32(vertices, ends)])
    return np.concatenate(triangles) if triangles else np.empty((0, 2), np.float32)


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


def temporal_probabilities(current, previous, delta=None):
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
    return .65*current + .35*previous


def sweep_segments(mask):
    paths = centerlines(mask)
    if not paths:
        return None
    lengths = np.array([len(path) for path in paths])
    points = np.concatenate(paths)
    start = np.r_[0, lengths.cumsum()[:-1]]
    end = lengths.cumsum() - 1
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangent[start] = points[start+1] - points[start]
    tangent[end] = points[end] - points[end-1]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    segments = np.ones(len(points), bool)
    segments[end] = False
    return points, normal, np.flatnonzero(segments)
