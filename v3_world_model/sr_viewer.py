"""Tesla FSD style 3D Surround Reconstruction viewer with glassmorphic telemetry HUD."""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import H, W, metric
from .sr_geometry import sweep_segments, surface_triangles, temporal_probabilities
from .labels import OBJECT_SLOTS
from .qc_v3 import load_npz

HORIZON_RGB = (.068, .098, .148)
THW_RED = (.86, .19, .20, 1.)
DRIVABLE_RGB = (.115, .138, .172, 1.)
DASHED_RGB = (.91, .94, .99, .95)
SOLID_RGB = (.96, .78, .28, .98)
CURB_RGB = (.36, .43, .52, 1.)
FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=np.int64)
QUAD_TRI = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
PRESETS = (
    ("CHASE", -12., 13., 28., (17., 0., .65), 62.),
    ("ELEVATED", 0., 52., 43., (17., 0., .20), 67.),
    ("TOP", 0., 76., 38., (12., 0., 0.), 40.),
)
INTENT_LABELS = {
    "Follow": ("Follow Lane", "车道保持巡航"),
    "跟车": ("Follow Lane", "车道保持巡航"),
    "Left Change": ("Change Left", "向左变道"),
    "Overtake left": ("Overtake Left", "左变道超车"),
    "Right Change": ("Change Right", "向右变道"),
    "Replay": ("Drive Replay", "实采轨迹回放"),
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-8)


def _select_font(paths, size):
    for candidate in paths:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, max(9, int(round(size))))
    return ImageFont.load_default()


class FontPack:
    def __init__(self, scale):
        sans_bold = (r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\segoeuib.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        sans_reg = (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\segoeui.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        num_font = (r"C:\Windows\Fonts\bahnschrift.ttf", r"C:\Windows\Fonts\segoeuib.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        self.speed = _select_font(num_font, 54 * scale)
        self.h1 = _select_font(sans_bold, 17 * scale)
        self.h2 = _select_font(sans_bold, 14 * scale)
        self.kpi = _select_font(num_font, 19 * scale)
        self.body = _select_font(sans_reg, 12.5 * scale)
        self.small = _select_font(sans_reg, 11 * scale)


def prediction_scene(pred, *, speed=0., cruise=0., intent="Follow", alternatives=(),
                     previous_scene=None, ego_motion=None, steer=0., throttle=0., brake=0.,
                     accel_lon=0., thw=None, infer_ms=None, limit=80.,
                     nav_hint="", mode="LIVE"):
    """Convert one batched model result into the unbatched renderer contract."""
    def as_numpy(x):
        return x.detach().float().cpu().numpy()[0]

    geom = as_numpy(pred["boxes"])
    yaw = np.arctan2(geom[:, 6], geom[:, 7])
    cls = as_numpy(pred["cls_logits"]).argmax(axis=-1)
    scores = as_numpy(pred["obj_logits"].float().sigmoid())
    visible = scores > .5
    boxes = np.column_stack((geom[:, :6], yaw, geom[:, 8:10], cls, visible)).astype(np.float32)
    decelerating = np.zeros(OBJECT_SLOTS, dtype=bool)
    if previous_scene is not None:
        previous = np.asarray(previous_scene["boxes"])
        for i in np.flatnonzero(visible):
            candidates = previous[(previous[:, -1] > .5) & (previous[:, 9] == cls[i])]
            if len(candidates):
                distances = np.linalg.norm(candidates[:, :2] - boxes[i, :2], axis=1)
                match = candidates[distances.argmin()]
                decelerating[i] = distances.min() < 3. and match[7] - boxes[i, 7] > .38
    probs = as_numpy(pred["lane_logits"].float().sigmoid())
    if mode == "LIVE" and previous_scene is not None:
        probs = temporal_probabilities(probs, previous_scene.get("lane_probs"), ego_motion)
    return dict(lanes=probs > .5, lane_probs=probs, boxes=boxes,
                future_occupancy=as_numpy(pred["future_occupancy_logits"]) > 0.,
                trajectory=as_numpy(pred["trajectory"]), speed=speed, cruise=cruise,
                intent=intent, alternatives=alternatives, decelerating=decelerating,
                steer=steer, throttle=throttle, brake=brake, accel_lon=accel_lon,
                thw=thw, infer_ms=infer_ms, limit=limit, nav_hint=nav_hint, mode=mode,
                source="model", object_scores=scores, fit_geometry=mode == "LIVE")


def from_capture(frame):
    labels = load_npz(frame / "labels.npz")
    state = labels["ego_state"]
    intent = "Replay"
    obs_path = frame / "obs.npz"
    obs = load_npz(obs_path) if obs_path.is_file() else {}
    if "cmd" in obs:
        intent = ("Follow", "Left Change", "Right Change")[int(np.argmax(obs["cmd"]))]
    anchors = obs.get("nav_anchors", labels.get("nav_anchors"))
    nav_hint = ""
    if anchors is not None and np.isfinite(anchors).all():
        far = anchors[-1]
        turn = "主线直行" if abs(far[1]) < 2.2 else ("前方左转/匝道" if far[1] > 0 else "前方右转/匝道")
        nav_hint = f"SD Nav: {max(far[0], 0.):.0f}m {turn}"
    return dict(lanes=labels["lanes"], boxes=labels["boxes"], trajectory=labels["trajectory"],
                future_occupancy=labels["future_occupancy"], speed=float(state[0]),
                accel_lon=float(state[1]), steer=float(state[4]),
                throttle=float(state[5]), brake=float(state[6]),
                thw=float(labels["thw"]) if "thw" in labels else -1.,
                cruise=0., limit=80., intent=intent, nav_hint=nav_hint, mode="REPLAY")


def ribbon(path, width=1.9):
    """Triangle strip ordered left/right, with monotone alpha fade by arc length."""
    xy = np.asarray(path, dtype=np.float32)[:, :2]
    xy = np.vstack((np.zeros((1, 2), np.float32), xy))
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-3]]
    if len(xy) < 2:
        return np.empty((0, 7), np.float32)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    steps = np.linspace(0, arc[-1], max(48, len(xy) * 2))
    p = np.column_stack([np.interp(steps, arc, xy[:, j]) for j in range(2)])
    p[1:-1] = (.20 * p[:-2] + .60 * p[1:-1] + .20 * p[2:])
    tangent = np.gradient(p, axis=0)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    rails = np.stack((p + width / 2 * normal, p - width / 2 * normal), axis=1)
    linear = steps / max(arc[-1], 1e-6)
    colors = np.column_stack((.14 - .08 * linear, .65 - .16 * linear,
                              np.ones_like(steps), .91 - .82 * linear))
    return np.column_stack((rails.reshape(-1, 2), np.full(rails.size // 2, .050),
                             np.repeat(colors, 2, axis=0))).astype(np.float32)


def ribbon_triangles(strip):
    if len(strip) < 3:
        return strip
    ids = np.arange(len(strip) - 2)
    return strip[np.column_stack((ids, ids + 1, ids + 2)).ravel()]


def glowing_ribbon_mesh(path, width=2.85, z=.052):
    """Build smooth C1 ribbon triangles with (pos 3f, uv_lat_s 2f, rgba 4f)."""
    xy = np.asarray(path, dtype=np.float32)[:, :2]
    xy = np.vstack((np.zeros((1, 2), np.float32), xy))
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-3]]
    if len(xy) < 2:
        return np.empty((0, 9), np.float32)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    steps = np.linspace(0., arc[-1], max(72, len(xy) * 3))
    p = np.column_stack([np.interp(steps, arc, xy[:, j]) for j in range(2)])
    for _ in range(3):
        p[1:-1] = .22 * p[:-2] + .56 * p[1:-1] + .22 * p[2:]
    tangent = np.gradient(p, axis=0)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    rails = np.stack((p + width / 2 * normal, p - width / 2 * normal), axis=1)
    s_norm = steps / max(arc[-1], 1e-6)
    uv = np.stack((np.column_stack((np.full_like(s_norm, 1.), s_norm)),
                   np.column_stack((np.full_like(s_norm, -1.), s_norm))), axis=1)
    rgba = np.column_stack((.14 - .05 * s_norm, .64 - .12 * s_norm,
                            np.ones_like(s_norm), .95 - .78 * s_norm))
    verts = np.concatenate((rails.reshape(-1, 2), np.full((rails.size // 2, 1), z),
                            uv.reshape(-1, 2), np.repeat(rgba, 2, axis=0)), axis=-1).astype(np.float32)
    ids = np.arange(len(verts) - 2)
    return verts[np.column_stack((ids, ids + 1, ids + 2)).ravel()]


def _tri_normals(tris, outward_center=None):
    """Compute unit normals for (M, 3, 3) triangles in one vectorized call."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nl = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(nl > 1e-8, n / np.maximum(nl, 1e-8), np.array([0., 0., 1.], np.float32))
    if outward_center is not None:
        tri_c = tris.mean(axis=1)
        flip = (np.sum(n * (tri_c - outward_center), axis=-1) < 0.)[:, None]
        n = np.where(flip, -n, n)
    return n.astype(np.float32)


def _quads_to_tris_batch(quads, rgba, mat, outward_center=None):
    """Convert (Q, 4, 3) quads into (6*Q, 11) vertex rows with exact normals."""
    tris = quads[:, QUAD_TRI].reshape(-1, 3, 3).astype(np.float32)
    n = _tri_normals(tris, outward_center)
    norm_v = np.repeat(n, 3, axis=0)
    q_count = len(quads)
    if np.asarray(rgba).ndim == 1:
        rgba_v = np.broadcast_to(np.asarray(rgba, np.float32), (6 * q_count, 4))
    else:
        rgba_v = np.repeat(np.asarray(rgba, np.float32), 6, axis=0)
    if np.ndim(mat) == 0:
        mat_v = np.full((6 * q_count, 1), mat, np.float32)
    else:
        mat_v = np.repeat(np.asarray(mat, np.float32), 6, axis=0).reshape(-1, 1)
    return np.concatenate((tris.reshape(-1, 3), norm_v, rgba_v, mat_v), axis=-1)


class UnitVehicleBuilder:
    """Build normalized (L=1, W=1, H=1, z in [0, 1]) vehicle mesh once per class."""

    def __init__(self):
        self.parts = []  # tuples: (verts (V,3), rgba (V,4), mat (V,1), body_mask (V,))

    def add_quads(self, quads, rgba, mat, body_mask, outward_center=None):
        tris = np.asarray(quads, np.float32)[:, QUAD_TRI].reshape(-1, 3, 3)
        n = _tri_normals(tris, outward_center)
        norm_v = np.repeat(n, 3, axis=0)
        v_count = len(tris) * 3
        rgba_v = (np.broadcast_to(np.asarray(rgba, np.float32), (v_count, 4))
                  if np.asarray(rgba).ndim == 1 else np.repeat(np.asarray(rgba, np.float32), 6, axis=0))
        mat_v = (np.full((v_count, 1), mat, np.float32)
                 if np.ndim(mat) == 0 else np.repeat(np.asarray(mat, np.float32), 6, axis=0).reshape(-1, 1))
        bm = (np.full(v_count, bool(body_mask), dtype=bool)
              if np.ndim(body_mask) == 0 else np.repeat(np.asarray(body_mask, bool), 6))
        self.parts.append((tris.reshape(-1, 3), norm_v, rgba_v, mat_v, bm))

    def add_loft(self, rings, colors, mats, body_flags, cap_ends=True):
        rings_a = [np.asarray(r, np.float32) for r in rings]
        center = np.mean(np.vstack(rings_a), axis=0)
        m = len(rings_a[0])
        quads, q_col, q_mat, q_bm = [], [], [], []
        for s in range(len(rings_a) - 1):
            a, b = rings_a[s], rings_a[s + 1]
            for j in range(m):
                jn = (j + 1) % m
                quads.append((a[j], a[jn], b[jn], b[j]))
                q_col.append(colors[j])
                q_mat.append(mats[j])
                q_bm.append(body_flags[j])
        self.add_quads(quads, q_col, q_mat, q_bm, outward_center=center)
        if cap_ends and m >= 3:
            t_pos, t_col, t_mat, t_bm = [], [], [], []
            for end_ring in (rings_a[0], rings_a[-1]):
                rc = end_ring.mean(axis=0)
                for j in range(m):
                    jn = (j + 1) % m
                    t_pos.append((rc, end_ring[j], end_ring[jn]))
                    t_col.append(colors[0])
                    t_mat.append(mats[0])
                    t_bm.append(body_flags[0])
            tris = np.asarray(t_pos, np.float32)
            norm_v = np.repeat(_tri_normals(tris, center), 3, axis=0)
            self.parts.append((tris.reshape(-1, 3), norm_v,
                               np.repeat(np.asarray(t_col, np.float32), 3, axis=0),
                               np.repeat(np.asarray(t_mat, np.float32), 3, axis=0).reshape(-1, 1),
                               np.repeat(np.asarray(t_bm, bool), 3)))

    def add_wheel(self, lx, ly, rad_x, rad_z, width_y):
        ang = np.linspace(0., 2 * math.pi, 11)[:-1]
        y_out = ly + (width_y * .5 if ly > 0 else -width_y * .5)
        y_in = ly - (width_y * .5 if ly > 0 else -width_y * .5)
        ring_out = np.column_stack((lx + rad_x * np.cos(ang), np.full_like(ang, y_out), rad_z * np.sin(ang) + rad_z))
        ring_in = np.column_stack((lx + rad_x * np.cos(ang), np.full_like(ang, y_in), rad_z * np.sin(ang) + rad_z))
        c_mid = (ring_out.mean(axis=0) + ring_in.mean(axis=0)) * .5
        n = len(ang)
        quads = [(ring_out[j], ring_out[(j + 1) % n], ring_in[(j + 1) % n], ring_in[j]) for j in range(n)]
        self.add_quads(quads, (.10, .11, .14, 1.), .12, False, outward_center=c_mid)
        hub_y = ly + (width_y * .53 if ly > 0 else -width_y * .53)
        hub = np.column_stack((lx + rad_x * .54 * np.cos(ang), np.full_like(ang, hub_y), rad_z * .54 * np.sin(ang) + rad_z))
        rc = hub.mean(axis=0)
        tris = np.asarray([(rc, hub[j], hub[(j + 1) % n]) for j in range(n)], np.float32)
        nv = np.repeat(_tri_normals(tris, c_mid), 3, axis=0)
        self.parts.append((tris.reshape(-1, 3), nv,
                           np.broadcast_to(np.array([.62, .68, .76, 1.], np.float32), (3 * n, 4)),
                           np.full((3 * n, 1), .75, np.float32), np.zeros(3 * n, dtype=bool)))

    def add_prism(self, bot, top, rgba, mat, body_mask, shade_factors=None):
        pts = np.vstack((np.asarray(bot, np.float32), np.asarray(top, np.float32)))
        center = pts.mean(axis=0)
        quads = pts[FACES]
        if shade_factors is None:
            cols = rgba
        else:
            cols = [(rgba[0] * s, rgba[1] * s, rgba[2] * s, rgba[3]) for s in shade_factors]
        self.add_quads(quads, cols, mat, body_mask, outward_center=center)

    def compile(self):
        return (np.concatenate([p[0] for p in self.parts]).astype(np.float32),
                np.concatenate([p[1] for p in self.parts]).astype(np.float32),
                np.concatenate([p[2] for p in self.parts]).astype(np.float32),
                np.concatenate([p[3] for p in self.parts]).astype(np.float32),
                np.concatenate([p[4] for p in self.parts]))


def _build_vehicle_templates():
    templates = {}
    glass = (.07, .10, .15, 1.)
    dark_trim = (.11, .13, .17, 1.)
    drl = (.90, .96, 1., 1.)

    # Kind 0 (sedan), Kind 1 (suv), Kind 5 (ego with glass roof)
    for key in (0, 1, 5):
        b = UnitVehicleBuilder()
        suv = (key == 1)
        fuse_spec = (
            (.50, .80, .14, .39 if not suv else .44),
            (.43, .96, .11, .50 if not suv else .55),
            (.18, 1.00, .11, .57 if not suv else .61),
            (-.10, 1.02, .11, .59 if not suv else .62),
            (-.38, .97, .12, .62 if not suv else .64),
            (-.50, .88, .15, .57 if not suv else .60),
        )
        fuse_rings = []
        for x, fw, zb, zt in fuse_spec:
            wb, wm, wt = .5 * fw * .90, .5 * fw, .5 * fw * .91
            zm = zb + .56 * (zt - zb)
            fuse_rings.append([[x, -wb, zb], [x, -wm, zm], [x, -wt, zt],
                               [x, wt, zt], [x, wm, zm], [x, wb, zb]])
        b.add_loft(fuse_rings,
                   [(1., 1., 1., 1.)] * 5 + [dark_trim],
                   (.46, .52, .58, .52, .46, .10),
                   (True, True, True, True, True, False))
        if suv:
            cab_spec = ((.17, .88, .58, .62), (.00, .76, .62, .99),
                        (-.35, .75, .64, 1.00), (-.45, .83, .61, .65))
        else:
            cab_spec = ((.20, .88, .55, .58), (.01, .75, .59, .98),
                        (-.19, .75, .61, 1.00), (-.41, .84, .58, .63))
        cab_rings = []
        for x, fw, zb, zt in cab_spec:
            w0, wt = .5 * fw, .5 * fw * .89
            cab_rings.append([[x, -w0, zb], [x, -wt, zt], [x, wt, zt], [x, w0, zb]])
        b.add_loft(cab_rings,
                   ((1., 1., 1., 1.) if key != 5 else (.92, .95, .99, 1.),
                    (1., 1., 1., 1.), glass, dark_trim),
                   (.65, .62, .95, .10),
                   (True, True, False, False), cap_ends=False)
        # Flush panoramic glass roof inset (dark reflective glass nested inside painted roof rails).
        if suv:
            pan = np.array([[[.01, -.29, .998], [-.34, -.28, 1.005],
                             [-.34, .28, 1.005], [.01, .29, .998]]], np.float32)
        else:
            pan = np.array([[[.01, -.29, .988], [-.28, -.28, 1.005],
                             [-.28, .28, 1.005], [.01, .29, .988]]], np.float32)
        b.add_quads(pan, glass, .96, False)
        # Sleek aerodynamic side-view mirrors.
        for side in (-1., 1.):
            mx0, mx1, mz0, mz1 = .16, .11, .56, .63
            y_in, y_out = side * .47, side * .56
            b.add_prism([[mx0, y_in, mz0], [mx1, y_in, mz0], [mx1, y_out, mz0], [mx0, y_out, mz0]],
                        [[mx0, y_in, mz1], [mx1, y_in, mz1], [mx1, y_out, mz1], [mx0, y_out, mz1]],
                        (1., 1., 1., 1.), .52, True)
        # Rear lower aero diffuser trim.
        b.add_quads([[[-.502, -.34, .16], [-.502, .34, .16], [-.502, .31, .27], [-.502, -.31, .27]]],
                    dark_trim, .18, False)
        for wx in (.31, -.31):
            for side in (-1., 1.):
                b.add_wheel(wx, side * .46, .078, .21, .125)
        for side in (-1., 1.):
            b.add_quads([[[.47, side * .26, .38], [.43, side * .43, .41],
                          [.43, side * .43, .47], [.47, side * .26, .44]]], drl, 0., False)
        templates[key] = b.compile()

    # Kind 2 (truck)
    bt = UnitVehicleBuilder()
    cab_spec = ((.50, .84, .16, .52), (.46, .96, .14, .64), (.35, .96, .14, .92), (.22, .96, .14, .92))
    cab_rings = []
    for x, fw, zb, zt in cab_spec:
        w0, wt = .5 * fw, .5 * fw * .90
        cab_rings.append([[x, -w0, zb], [x, -wt, zb + .42 * (zt - zb)], [x, -wt * .94, zt],
                          [x, wt * .94, zt], [x, wt, zb + .42 * (zt - zb)], [x, w0, zb]])
    bt.add_loft(cab_rings,
                [(1., 1., 1., 1.), glass, (1., 1., 1., 1.), glass, (1., 1., 1., 1.), dark_trim],
                (.48, .92, .55, .92, .48, .10),
                (True, False, True, False, True, False))
    bt.add_prism([[.18, -.49, .26], [-.50, -.49, .26], [-.50, .49, .26], [.18, .49, .26]],
                 [[.18, -.48, 1.00], [-.50, -.48, 1.00], [-.50, .48, 1.00], [.18, .48, 1.00]],
                 (.78, .82, .88, 1.), .38, True, shade_factors=(.65, 1.04, .92, .85, .92, .85))
    for wx in (.37, -.22, -.39):
        for side in (-1., 1.):
            bt.add_wheel(wx, side * .44, .058, .19, .12)
    templates[2] = bt.compile()

    # Kind 3 (two_wheeler)
    bm = UnitVehicleBuilder()
    bm.add_prism([[.45, -.18, .14], [-.45, -.18, .14], [-.45, .18, .14], [.45, .18, .14]],
                 [[.32, -.12, .62], [-.35, -.12, .60], [-.35, .12, .60], [.32, .12, .62]],
                 (1., 1., 1., 1.), .55, True)
    bm.add_prism([[.10, -.16, .56], [-.18, -.16, .56], [-.18, .16, .56], [.10, .16, .56]],
                 [[.06, -.10, 1.00], [-.14, -.10, 1.00], [-.14, .10, 1.00], [.06, .10, 1.00]],
                 (.18, .22, .28, 1.), .35, False)
    for wx in (.34, -.34):
        bm.add_wheel(wx, 0., .15, .24, .10)
    templates[3] = bm.compile()

    # Kind 4 (pedestrian)
    bp = UnitVehicleBuilder()
    bp.add_prism([[.28, -.28, .02], [-.28, -.28, .02], [-.28, .28, .02], [.28, .28, .02]],
                 [[.22, -.22, .78], [-.22, -.22, .78], [-.22, .22, .78], [.22, .22, .78]],
                 (.78, .68, .54, 1.), .25, False)
    bp.add_prism([[.18, -.18, .79], [-.18, -.18, .79], [-.18, .18, .79], [.18, .18, .79]],
                 [[.15, -.15, 1.0], [-.15, -.15, 1.0], [-.15, .15, 1.0], [.15, .15, 1.0]],
                 (.92, .82, .70, 1.), .30, False)
    templates[4] = bp.compile()
    return templates


VEHICLE_TEMPLATES = _build_vehicle_templates()
_ANG_16 = np.linspace(0., 2 * math.pi, 17)[:-1]
_SHADOW_OUTER = np.column_stack((.61 * np.cos(_ANG_16), .64 * np.sin(_ANG_16), np.zeros_like(_ANG_16))).astype(np.float32)
_ANG_24 = np.linspace(0., 2 * math.pi, 25)
_RING_IN = np.column_stack((.58 * np.cos(_ANG_24), .68 * np.sin(_ANG_24), np.zeros_like(_ANG_24))).astype(np.float32)
_RING_OUT = np.column_stack((.67 * np.cos(_ANG_24), .78 * np.sin(_ANG_24), np.zeros_like(_ANG_24))).astype(np.float32)


def _static_range_rings():
    parts = []
    ang = np.linspace(-math.pi * .42, math.pi * .42, 49)
    for r_m in (15., 30., 50.):
        r_in = np.column_stack((r_m * np.cos(ang), r_m * np.sin(ang)))
        r_out = np.column_stack(((r_m + .08) * np.cos(ang), (r_m + .08) * np.sin(ang)))
        q = np.stack((r_in[:-1], r_out[:-1], r_out[1:], r_in[1:]), axis=1)
        keep = (np.abs(q[..., 1]).max(axis=1) <= 29.8) & (q[..., 0].max(axis=1) <= 79.8)
        if keep.any():
            q3 = np.concatenate((q[keep], np.full((keep.sum(), 4, 1), -.015, np.float32)), axis=-1)
            parts.append(_quads_to_tris_batch(q3, (.24, .34, .48, .13), 0.))
    return np.concatenate(parts).astype(np.float32) if parts else np.empty((0, 11), np.float32)


RANGE_RINGS_11 = _static_range_rings()


class SceneMesh:
    """Collect lit (pos, norm, rgba, mat) triangles using vectorized NumPy batches."""

    def __init__(self):
        self.opaque = []
        self.blend = [RANGE_RINGS_11] if len(RANGE_RINGS_11) else []

    def mask_surface(self, mask, z, rgba, mat=.05, *, fit=False):
        xy = surface_triangles(mask, fit=fit)
        if not len(xy):
            return
        verts = np.column_stack((xy, np.full(len(xy), z, np.float32)))
        n = np.broadcast_to(np.array([0., 0., 1.], np.float32), (len(verts), 3))
        col = np.broadcast_to(np.asarray(rgba, np.float32), (len(verts), 4))
        self.opaque.append(np.concatenate((verts, n, col, np.full((len(verts), 1), mat, np.float32)), axis=-1))

    def lane_strips(self, mask, z, rgba, width, *, fit=False):
        sweep = sweep_segments(mask, fit=fit)
        if sweep is None:
            return
        xy, norm, segments = sweep
        n_up = np.array([0., 0., 1.], np.float32)
        left = np.column_stack((xy + (width * .5) * norm, np.full(len(xy), z, np.float32)))
        right = np.column_stack((xy - (width * .5) * norm, np.full(len(xy), z, np.float32)))
        quads = np.stack((left[segments], right[segments], right[segments+1], left[segments+1]), axis=1)
        verts = quads[:, QUAD_TRI].reshape(-1, 3)
        self.opaque.append(np.concatenate((
            verts,
            np.broadcast_to(n_up, (len(verts), 3)),
            np.broadcast_to(np.asarray(rgba, np.float32), (len(verts), 4)),
            np.zeros((len(verts), 1), np.float32),
        ), axis=-1))

    def curb_barriers(self, mask, z_base=-.01, height=.20, base_half=.25, top_half=.13, *, fit=False):
        sweep = sweep_segments(mask, fit=fit)
        if sweep is None:
            return
        xy, norm, segments = sweep
        top_col = (CURB_RGB[0] * 1.08, CURB_RGB[1] * 1.08, CURB_RGB[2] * 1.08, 1.)
        side_col = (CURB_RGB[0] * .86, CURB_RGB[1] * .86, CURB_RGB[2] * .86, 1.)
        bot_col = (CURB_RGB[0] * .65, CURB_RGB[1] * .65, CURB_RGB[2] * .65, 1.)
        bl = np.column_stack((xy + base_half * norm, np.full(len(xy), z_base, np.float32)))
        br = np.column_stack((xy - base_half * norm, np.full(len(xy), z_base, np.float32)))
        tl = np.column_stack((xy + top_half * norm, np.full(len(xy), z_base + height, np.float32)))
        tr = np.column_stack((xy - top_half * norm, np.full(len(xy), z_base + height, np.float32)))
        rings = np.stack((br, tr, tl, bl), axis=1)  # (K, 4, 3)
        a, b = rings[segments], rings[segments+1]
        q_parts = []
        for j in range(4):
            jn = (j + 1) % 4
            q_parts.append(np.stack((a[:, j], a[:, jn], b[:, jn], b[:, j]), axis=1))
        quads = np.concatenate(q_parts, axis=0)
        seg_n = len(a)
        cols = np.concatenate([np.broadcast_to(c, (seg_n, 4)) for c in (side_col, top_col, side_col, bot_col)])
        mats = np.concatenate([np.full(seg_n, m, np.float32) for m in (.30, .42, .30, .20)])
        centers = np.repeat(np.tile((a.mean(axis=1) + b.mean(axis=1)) * .5, (4, 1)), 2, axis=0)
        self.opaque.append(_quads_to_tris_batch(quads, cols, mats, outward_center=centers))

    def vehicle(self, box, color, ego=False, urgent=False, decelerating=False):
        cx, cy, cz, L, W_v, H_v, yaw = [float(v) for v in box[:7]]
        kind = int(box[9])
        t_key = 5 if ego else (kind if kind in VEHICLE_TEMPLATES else 0)
        v_loc, n_loc, c_loc, m_loc, body_mask = VEHICLE_TEMPLATES[t_key]

        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]], dtype=np.float32)
        g_z = cz - H_v * .5
        scale = np.array([L, W_v, H_v], np.float32)
        pos = (v_loc * scale) @ rot + np.array([cx, cy, g_z], np.float32)
        inv_scale = np.array([1. / max(L, 1e-3), 1. / max(W_v, 1e-3), 1. / max(H_v, 1e-3)], np.float32)
        nrm = (n_loc * inv_scale) @ rot
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
        cols = c_loc.copy()
        cols[body_mask] = np.array([color[0], color[1], color[2], 1.], np.float32)
        self.opaque.append(np.concatenate((pos, nrm, cols, m_loc), axis=-1))

        # Rear LED taillight signature (mounted cleanly proud of the rear deck; glows brighter under braking/THW)
        if kind in (0, 1, 2):
            tail_bright = decelerating or urgent
            tail_col = (1., .16, .15, 1.) if tail_bright else (.88, .16, .16, 1.)
            zt0 = (.58 if kind == 2 else .46) * H_v
            zt1 = (.65 if kind == 2 else .535) * H_v
            x_rear = -.503 * L
            tl_quads = []
            for side in (-1., 1.):
                tl_quads.append([[x_rear, side * W_v * .22, zt0], [x_rear, side * W_v * .42, zt0],
                                 [x_rear, side * W_v * .39, zt1], [x_rear, side * W_v * .20, zt1]])
            if ego or tail_bright:
                # Subtle continuous center LED lightbar connecting left and right clusters.
                tl_quads.append([[x_rear, -W_v * .22, zt0 + .018 * H_v], [x_rear, W_v * .22, zt0 + .018 * H_v],
                                 [x_rear, W_v * .22, zt1 - .015 * H_v], [x_rear, -W_v * .22, zt1 - .015 * H_v]])
            tl_w = np.asarray(tl_quads, np.float32) @ rot + np.array([cx, cy, g_z], np.float32)
            self.opaque.append(_quads_to_tris_batch(tl_w, tail_col, 0.))

        # Vectorized soft radial contact shadow
        outer = (_SHADOW_OUTER * np.array([L, W_v, 1.], np.float32)) @ rot + np.array([cx, cy, .006], np.float32)
        center = np.array([cx, cy, .006], np.float32)
        idx_next = np.roll(np.arange(len(outer)), -1)
        sh_tris = np.stack((np.broadcast_to(center, outer.shape), outer, outer[idx_next]), axis=1)
        sh_rgba = np.broadcast_to(np.array([[.015, .020, .030, .56], [.015, .020, .030, 0.], [.015, .020, .030, 0.]], np.float32),
                                  (len(outer), 3, 4))
        sh_n = np.broadcast_to(np.array([0., 0., 1.], np.float32), (len(outer) * 3, 3))
        self.blend.append(np.concatenate((sh_tris.reshape(-1, 3), sh_n, sh_rgba.reshape(-1, 4),
                                          np.zeros((len(outer) * 3, 1), np.float32)), axis=-1))

        if urgent:
            p_in = (_RING_IN * np.array([L, W_v, 1.], np.float32)) @ rot + np.array([cx, cy, .009], np.float32)
            p_out = (_RING_OUT * np.array([L, W_v, 1.], np.float32)) @ rot + np.array([cx, cy, .009], np.float32)
            rq = np.stack((p_in[:-1], p_out[:-1], p_out[1:], p_in[1:]), axis=1)
            self.blend.append(_quads_to_tris_batch(rq, (.96, .20, .20, .48), 0.))

        if ego:
            cone = np.array([[[.50 * L, -W_v * .32, .06], [10.5, -W_v * 1.05, .02],
                              [10.5, W_v * 1.05, .02], [.50 * L, W_v * .32, .06]]], np.float32) @ rot + np.array([cx, cy, 0.], np.float32)
            c_tris = cone[:, QUAD_TRI].reshape(-1, 3)
            c_rgba = np.array([[.65, .86, 1., .15], [.40, .65, 1., 0.], [.40, .65, 1., 0.],
                               [.65, .86, 1., .15], [.40, .65, 1., 0.], [.40, .65, 1., 0.]], np.float32)
            self.blend.append(np.concatenate((c_tris, np.broadcast_to(np.array([0., 0., 1.], np.float32), (6, 3)),
                                              c_rgba, np.zeros((6, 1), np.float32)), axis=-1))


def build_scene(scene, show_occupancy=False):
    """Build GPU lit meshes, glowing ribbon triangles, and backward-compatible 7-float meshes."""
    masks = np.asarray(scene.get("lane_probs", scene["lanes"]))
    boxes = np.asarray(scene["boxes"])
    trajectory = np.asarray(scene["trajectory"])
    if masks.shape != (4, H, W) or boxes.shape != (32, 11) or trajectory.shape != (25, 3):
        raise ValueError("Expected lane masks 4x160x120, 32 boxes, and 25 trajectory steps")
    if not all(np.isfinite(a).all() for a in (masks, boxes, trajectory)):
        raise ValueError("Non-finite viewer input")

    mesh = SceneMesh()
    ground_q = np.array([[[115., 75., -.035], [115., -75., -.035], [-45., -75., -.035], [-45., 75., -.035]]], np.float32)
    mesh.opaque.append(_quads_to_tris_batch(ground_q, (.055, .072, .098, 1.), 0.))
    fit = scene.get("fit_geometry", False)
    mesh.mask_surface(masks[0], 0., DRIVABLE_RGB, mat=.05, fit=fit)
    mesh.lane_strips(masks[1], .014, DASHED_RGB, width=.20, fit=fit)
    mesh.lane_strips(masks[2], .018, SOLID_RGB, width=.24, fit=fit)
    # CARLA junction labels can contain boundaries internal to overlapping lane strips.
    # Extrude curbs only near an actual drivable/non-drivable interface.
    interface = cv2.morphologyEx((masks[0] >= .5).astype(np.uint8), cv2.MORPH_GRADIENT,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mesh.curb_barriers(masks[3] * interface, fit=fit)

    decel = scene.get("decelerating", np.zeros(OBJECT_SLOTS, dtype=bool))
    speed = float(scene.get("speed", 0.))
    for i, box in enumerate(boxes):
        if box[-1] < .5 or min(box[3:6]) <= 0:
            continue
        ahead = 0 < box[0] < 45 and abs(box[1]) < 2.2 and abs(box[6]) < 1.2
        thw = max(0., box[0] - box[3] / 2 - 2.3) / max(speed, .1)
        urgent = ahead and (thw < 1.5 or bool(decel[i]))
        mesh.vehicle(box, THW_RED if urgent else (.62, .69, .77, 1.),
                     urgent=urgent, decelerating=bool(decel[i]))

    ego = np.array([0., 0., .74, 4.65, 1.9, 1.48, 0., 0., 0., 0., 1.], dtype=np.float32)
    mesh.vehicle(ego, (.89, .93, .97, 1.), ego=True)

    if show_occupancy and "future_occupancy" in scene:
        future = np.asarray(scene["future_occupancy"])
        if future.shape != (5, H, W):
            raise ValueError("Future occupancy must be 5x160x120")
        for i, mask in reversed(list(enumerate(future))):
            rows, bounds = np.nonzero(np.diff(np.pad(mask.astype(bool), ((0, 0), (1, 1))), axis=1))
            if not len(rows):
                continue
            a, b = bounds.reshape(-1, 2).T - .5
            row = rows[::2]
            u = np.stack((a, b, b, a), axis=1)
            v = row[:, None] + np.array([-.5, -.5, .5, .5])
            xy = metric(np.stack((u, v), axis=-1))
            z0 = .028 + i * .036
            z1 = z0 + .16
            bot = np.concatenate((xy, np.full((*xy.shape[:2], 1), z0, np.float32)), axis=-1)
            top = np.concatenate((xy, np.full((*xy.shape[:2], 1), z1, np.float32)), axis=-1)
            pts = np.concatenate((bot, top), axis=1)
            quads = pts[:, FACES].reshape(-1, 4, 3)
            centers = np.repeat(pts.mean(axis=1), 12, axis=0)
            col = (.20 + .04 * i, .58 + .05 * (4 - i), .99, .26 - .038 * i)
            mesh.blend.append(_quads_to_tris_batch(quads, col, 0., outward_center=centers))

    for other in scene.get("alternatives", ()):
        alt = ribbon(np.asarray(other), .22)
        alt[:, 2] += .004
        alt[:, 3:7] = (.42, .84, 1., .42)
        alt_tris = ribbon_triangles(alt)
        if len(alt_tris):
            n_up = np.broadcast_to(np.array([0., 0., 1.], np.float32), (len(alt_tris), 3))
            mesh.blend.append(np.concatenate((alt_tris[:, :3], n_up, alt_tris[:, 3:7],
                                              np.zeros((len(alt_tris), 1), np.float32)), axis=-1))

    opaque_11 = np.concatenate(mesh.opaque).astype(np.float32) if mesh.opaque else np.empty((0, 11), np.float32)
    blend_11 = np.concatenate(mesh.blend).astype(np.float32) if mesh.blend else np.empty((0, 11), np.float32)
    ribbon_9 = glowing_ribbon_mesh(trajectory, width=2.85, z=.050)
    return opaque_11, blend_11, ribbon_9


def geometry(scene, show_occupancy=False):
    """Public compatibility helper returning (opaque_7, blend_7) vertex arrays."""
    opaque_11, blend_11, _ = build_scene(scene, show_occupancy)
    opaque_7 = np.concatenate((opaque_11[:, :3], opaque_11[:, 6:10]), axis=-1) if len(opaque_11) else np.empty((0, 7), np.float32)
    parts = []
    if len(blend_11):
        parts.append(np.concatenate((blend_11[:, :3], blend_11[:, 6:10]), axis=-1))
    trajectory = np.asarray(scene["trajectory"])
    glow = ribbon(trajectory, width=2.35)
    glow[:, 2] -= .001
    glow[:, 6] *= .22
    core = ribbon(trajectory, width=1.9)
    for r in (glow, core):
        rt = ribbon_triangles(r)
        if len(rt):
            parts.append(rt)
    blend_7 = np.concatenate(parts).astype(np.float32) if parts else np.empty((0, 7), np.float32)
    return opaque_7, blend_7


def camera_matrix(size, azimuth, elevation, distance, target=(6.5, 0., .42), fov=48.):
    w, h = size
    target = np.asarray(target, dtype=np.float64)
    angle, tilt = np.deg2rad([azimuth, elevation])
    eye = target + distance * np.array([-np.cos(tilt) * np.cos(angle),
                                        np.cos(tilt) * np.sin(angle), np.sin(tilt)])
    fwd = unit(target - eye)
    side = unit(np.cross(fwd, [0., 0., 1.]))
    up = np.cross(side, fwd)
    view = np.eye(4)
    view[:3, :3] = np.stack((side, up, -fwd))
    view[:3, 3] = -view[:3, :3] @ eye
    f = 1. / math.tan(math.radians(fov) / 2.)
    near, far = .15, 220.
    perspective = np.array([[f * h / w, 0, 0, 0], [0, f, 0, 0],
                            [0, 0, (far + near) / (near - far), 2 * near * far / (near - far)],
                            [0, 0, -1, 0]])
    return (perspective @ view).astype(np.float32), eye.astype(np.float32)


def _scene_thw_lead(scene):
    speed = float(scene.get("speed", 0.))
    boxes = np.asarray(scene["boxes"])
    active = boxes[boxes[:, -1] > .5]
    ahead = active[(active[:, 0] > 0) & (abs(active[:, 1]) < 2.2) & (abs(active[:, 6]) < 1.2)]
    lead_gap = float((ahead[:, 0] - ahead[:, 3] * .5 - 2.3).min()) if len(ahead) else -1.
    thw = scene.get("thw")
    if thw is None or (isinstance(thw, (int, float)) and thw < 0):
        thw_val = max(0., lead_gap) / max(speed, .5) if lead_gap >= 0 and speed >= .5 else -1.
    else:
        thw_val = float(thw)
    return thw_val, lead_gap, int(len(active))


def perception_summary(scene):
    """Missing detections are not evidence of a clear road."""
    thw, gap, count = _scene_thw_lead(scene)
    scores = np.asarray(scene.get("object_scores", []), dtype=np.float32)
    confidence = float(scores.max()) if scores.size else None
    lead_text = f"{gap:.1f}m" if gap >= 0 else "未检出前车"
    object_text = f"检出{count}辆"
    if confidence is not None:
        object_text += f" / 最高{confidence:.0%}"
    return thw, gap, count, confidence, lead_text, object_text


def hud(scene, fps, *, size=(1280, 720), gpu_ms=None, show_occupancy=False,
        show_telemetry=True, preset_name="CHASE", fonts=None):
    """Render an anti-aliased Tesla FSD glassmorphic RGBA HUD overlay."""
    w, h = size
    s = max(.48, min(w / 1280., h / 720.))
    fonts = fonts or FontPack(s)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    def rr(xy, radius, fill, outline=(255, 255, 255, 36), width=1):
        d.rounded_rectangle([int(v) for v in xy], radius=max(4, int(radius)),
                            fill=fill, outline=outline, width=max(1, int(width)))

    speed_ms = float(scene.get("speed", 0.))
    kmh = speed_ms * 3.6
    cruise_kmh = float(scene.get("cruise", 0.)) * 3.6
    limit_kmh = float(scene.get("limit", 80.))
    steer = float(scene.get("steer", 0.))
    throttle = float(np.clip(scene.get("throttle", 0.), 0., 1.))
    brake = float(np.clip(scene.get("brake", 0.), 0., 1.))
    accel = float(scene.get("accel_lon", 0.))
    intent_raw = str(scene.get("intent", "Follow"))
    intent_en, intent_zh = INTENT_LABELS.get(intent_raw, (intent_raw, intent_raw))
    nav_hint = str(scene.get("nav_hint", "")) or "SD Route: Mainline Cruise"
    mode = str(scene.get("mode", "SYNTH"))
    infer_ms = scene.get("infer_ms")
    thw_val, lead_gap, count_objs, _, lead_text, object_text = perception_summary(scene)
    thw_warn = 0. <= thw_val < 1.5

    # 1. Top-Left Main Driving Cluster
    pad = 20 * s
    cw, ch = 264 * s, 192 * s
    rr((pad, pad, pad + cw, pad + ch), 18 * s, (10, 15, 24, 210), (140, 180, 235, 50), 1.5 * s)
    bar_x0, bar_x1, bar_y = pad + 18 * s, pad + cw - 18 * s, pad + 12 * s
    mid_x = .5 * (bar_x0 + bar_x1)
    d.rounded_rectangle([int(bar_x0), int(bar_y), int(bar_x1), int(bar_y + 4 * s)],
                        radius=2, fill=(45, 58, 78, 180))
    drive_lvl = throttle if throttle > .01 else max(0., accel / 3.5)
    regen_lvl = brake if brake > .01 else max(0., -accel / 4.0)
    if drive_lvl > .01:
        fill_x = mid_x + min(1., drive_lvl) * .5 * (bar_x1 - bar_x0)
        d.rounded_rectangle([int(mid_x), int(bar_y), int(fill_x), int(bar_y + 4 * s)],
                            radius=2, fill=(46, 224, 148, 245))
    elif regen_lvl > .01:
        fill_x = mid_x - min(1., regen_lvl) * .5 * (bar_x1 - bar_x0)
        d.rounded_rectangle([int(fill_x), int(bar_y), int(mid_x), int(bar_y + 4 * s)],
                            radius=2, fill=(255, 162, 54, 245))

    gx, gy = pad + 20 * s, pad + 26 * s
    for idx, g in enumerate(("P", "R", "N", "D")):
        col = (242, 247, 255, 255) if g == "D" else (105, 120, 142, 200)
        d.text((gx + idx * 24 * s, gy), g, font=fonts.h2, fill=col)

    wx, wy, wr = pad + cw - 34 * s, gy + 9 * s, 12 * s
    d.ellipse([wx - wr, wy - wr, wx + wr, wy + wr], outline=(48, 152, 255, 255), width=max(2, int(2.4 * s)))
    d.line([wx - wr * .72, wy - wr * .15, wx + wr * .72, wy - wr * .15], fill=(48, 152, 255, 255), width=max(2, int(2 * s)))
    d.line([wx, wy - wr * .15, wx, wy + wr * .75], fill=(48, 152, 255, 255), width=max(2, int(2 * s)))
    d.text((wx - 28 * s, wy), "FSD", font=fonts.small, fill=(84, 178, 255, 245), anchor="rm")

    spd_str = f"{kmh:.0f}"
    d.text((pad + 20 * s, pad + 52 * s), spd_str, font=fonts.speed, fill=(246, 250, 255, 255))
    spd_w = d.textlength(spd_str, font=fonts.speed)
    d.text((pad + 28 * s + spd_w, pad + 90 * s), "km/h", font=fonts.body, fill=(148, 166, 192, 235))

    py = pad + 130 * s
    max_x0, max_x1 = pad + 18 * s, pad + 132 * s
    rr((max_x0, py, max_x1, py + 44 * s), 12 * s, (18, 30, 48, 225), (48, 152, 255, 220), 1.6 * s)
    d.text((max_x0 + 12 * s, py + 22 * s), "MAX", font=fonts.small, fill=(84, 178, 255, 255), anchor="lm")
    cruise_str = f"{cruise_kmh:.0f}" if cruise_kmh > 1 else "AUTO"
    d.text((max_x1 - 14 * s, py + 22 * s), cruise_str, font=fonts.h1, fill=(242, 248, 255, 255), anchor="rm")

    lim_x0, lim_x1 = pad + 142 * s, pad + cw - 18 * s
    rr((lim_x0, py, lim_x1, py + 44 * s), 10 * s, (238, 243, 250, 245), (35, 42, 54, 240), 1.6 * s)
    d.text((.5 * (lim_x0 + lim_x1), py + 11 * s), "LIMIT", font=fonts.small, fill=(65, 76, 94, 255), anchor="mm")
    d.text((.5 * (lim_x0 + lim_x1), py + 30 * s), f"{limit_kmh:.0f}", font=fonts.h1, fill=(18, 24, 34, 255), anchor="mm")

    # 2. Top-Center Decision & Navigation Capsule
    cap_w, cap_h = min(430 * s, w - 560 * s if w > 820 else w * .45), 68 * s
    if cap_w > 180 * s:
        cap_x = .5 * (w - cap_w)
        rr((cap_x, pad, cap_x + cap_w, pad + cap_h), 18 * s, (12, 18, 29, 210), (96, 170, 255, 85), 1.5 * s)
        ix, iy, ir = cap_x + 34 * s, pad + cap_h * .5, 19 * s
        d.ellipse([ix - ir, iy - ir, ix + ir, iy + ir], fill=(32, 118, 232, 235), outline=(120, 200, 255, 220), width=1)
        if "Left" in intent_en or "左" in intent_zh:
            d.line([ix + 6 * s, iy + 8 * s, ix + 6 * s, iy - 1 * s, ix - 6 * s, iy - 6 * s],
                   fill=(255, 255, 255, 255), width=max(2, int(2.6 * s)))
            d.polygon([(ix - 9 * s, iy - 8 * s), (ix - 1 * s, iy - 10 * s), (ix - 5 * s, iy - 1 * s)],
                      fill=(255, 255, 255, 255))
        elif "Right" in intent_en or "右" in intent_zh:
            d.line([ix - 6 * s, iy + 8 * s, ix - 6 * s, iy - 1 * s, ix + 6 * s, iy - 6 * s],
                   fill=(255, 255, 255, 255), width=max(2, int(2.6 * s)))
            d.polygon([(ix + 9 * s, iy - 8 * s), (ix + 5 * s, iy - 1 * s), (ix + 1 * s, iy - 10 * s)],
                      fill=(255, 255, 255, 255))
        else:
            d.line([ix, iy + 9 * s, ix, iy - 7 * s], fill=(255, 255, 255, 255), width=max(2, int(2.8 * s)))
            d.polygon([(ix, iy - 10 * s), (ix - 6 * s, iy - 2 * s), (ix + 6 * s, iy - 2 * s)],
                      fill=(255, 255, 255, 255))
        d.text((cap_x + 66 * s, pad + 11 * s), f"{intent_zh} · {intent_en}", font=fonts.h1, fill=(242, 248, 255, 255))
        d.text((cap_x + 66 * s, pad + 37 * s), nav_hint, font=fonts.body, fill=(148, 174, 208, 235))

    # 3. Top-Right Safety & Telemetry Card
    if show_telemetry:
        tw, th = 272 * s, 192 * s
        tx = w - pad - tw
        rr((tx, pad, tx + tw, pad + th), 18 * s, (10, 15, 24, 210), (140, 180, 235, 50), 1.5 * s)
        d.text((tx + 16 * s, pad + 11 * s), "TELEMETRY", font=fonts.h2, fill=(176, 196, 224, 240))
        badge_col = (34, 156, 108, 220) if mode == "LIVE" else (42, 118, 216, 220)
        bx0, bx1, by0, by1 = tx + tw - 76 * s, tx + tw - 14 * s, pad + 10 * s, pad + 30 * s
        rr((bx0, by0, bx1, by1), 7 * s, badge_col, outline=None)
        d.text((.5 * (bx0 + bx1), .5 * (by0 + by1)), mode, font=fonts.small, fill=(246, 252, 255, 255), anchor="mm")

        thw_txt = f"{thw_val:.2f}s" if thw_val >= 0 else "--"
        gap_txt = lead_text
        inf_txt = f"{infer_ms:.1f}ms" if infer_ms is not None else "Replay"
        fps_txt = f"{fps:.0f} FPS" if fps > 1 else "GPU"
        gpu_txt = f"{gpu_ms:.2f}ms" if gpu_ms is not None else "4xMSAA"
        tiles = (
            (0, 0, "THW 时距", thw_txt, (255, 92, 98, 255) if thw_warn else (82, 228, 162, 255)),
            (1, 0, object_text, gap_txt, (238, 244, 252, 255)),
            (0, 1, "INFER / CTRL", f"{inf_txt} / 100ms", (152, 206, 255, 255)),
            (1, 1, "SR 渲染", fps_txt, (152, 206, 255, 255)),
        )
        tile_w, tile_h = 116 * s, 52 * s
        for col_i, row_i, k_label, k_val, val_col in tiles:
            bx = tx + 14 * s + col_i * 126 * s
            by = pad + 40 * s + row_i * 60 * s
            fill_tile = (68, 22, 28, 190) if (col_i == 0 and row_i == 0 and thw_warn) else (20, 29, 44, 195)
            out_tile = (255, 92, 98, 180) if (col_i == 0 and row_i == 0 and thw_warn) else (120, 155, 200, 35)
            rr((bx, by, bx + tile_w, by + tile_h), 10 * s, fill_tile, out_tile)
            d.text((bx + 8 * s, by + 6 * s), k_label, font=fonts.small, fill=(142, 162, 190, 230))
            val_font = fonts.body if len(k_val) > 8 or not k_val.isascii() else fonts.kpi
            d.text((bx + 8 * s, by + 24 * s), k_val, font=val_font, fill=val_col)

        occ_fill = (32, 128, 232, 225) if show_occupancy else (32, 42, 58, 200)
        ox0, ox1, oy0, oy1 = tx + 14 * s, tx + 128 * s, pad + 161 * s, pad + 183 * s
        rr((ox0, oy0, ox1, oy1), 7 * s, occ_fill, (110, 180, 255, 80))
        d.text((.5 * (ox0 + ox1), .5 * (oy0 + oy1)), f"3D Occ: {'ON' if show_occupancy else 'OFF'}",
               font=fonts.small, fill=(242, 248, 255, 255), anchor="mm")
        vx0, vx1 = tx + 136 * s, tx + tw - 14 * s
        rr((vx0, oy0, vx1, oy1), 7 * s, (24, 34, 50, 210), (140, 175, 220, 60))
        d.text((.5 * (vx0 + vx1), .5 * (oy0 + oy1)), f"{preset_name} · {gpu_txt}",
               font=fonts.small, fill=(190, 214, 244, 240), anchor="mm")
    else:
        pill_w = 104 * s
        rr((w - pad - pill_w, pad, w - pad, pad + 32 * s), 12 * s, (10, 15, 24, 195), (140, 180, 235, 50))
        d.text((w - pad - pill_w * .5, pad + 16 * s), "[H] 遥测", font=fonts.body, fill=(200, 220, 245, 240), anchor="mm")

    # 4. Bottom-Left Control Actuator Mini Card
    bw, bh = 264 * s, 74 * s
    by = h - pad - bh
    rr((pad, by, pad + bw, by + bh), 16 * s, (10, 15, 24, 205), (140, 180, 235, 45), 1.4 * s)
    sx, sy, sr = pad + 40 * s, by + 29 * s, 16 * s
    ang_rad = steer * math.radians(150.)
    d.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], outline=(190, 214, 242, 235), width=max(2, int(2.6 * s)))
    c_a, s_a = math.cos(ang_rad), math.sin(ang_rad)
    d.line([sx - sr * .82 * c_a, sy - sr * .82 * s_a, sx + sr * .82 * c_a, sy + sr * .82 * s_a],
           fill=(88, 178, 255, 245), width=max(2, int(2.4 * s)))
    d.line([sx, sy, sx - sr * .78 * s_a, sy + sr * .78 * c_a],
           fill=(88, 178, 255, 245), width=max(2, int(2.2 * s)))
    d.text((sx, by + 51 * s), f"STR {steer * 450:+.0f}°", font=fonts.small, fill=(158, 180, 208, 235), anchor="mt")

    for row_idx, (lbl, val, bar_col) in enumerate((("THR", throttle, (48, 224, 144, 245)),
                                                    ("BRK", brake, (255, 86, 94, 245)))):
        rx, ry = pad + 80 * s, by + 16 * s + row_idx * 26 * s
        d.text((rx, ry), lbl, font=fonts.small, fill=(176, 192, 214, 240))
        bx0, bx1 = rx + 34 * s, pad + bw - 48 * s
        d.rounded_rectangle([int(bx0), int(ry + 3 * s), int(bx1), int(ry + 11 * s)],
                            radius=4, fill=(32, 42, 58, 210))
        if val > .005:
            d.rounded_rectangle([int(bx0), int(ry + 3 * s), int(bx0 + (bx1 - bx0) * val), int(ry + 11 * s)],
                                radius=4, fill=bar_col)
        d.text((bx1 + 8 * s, ry), f"{val * 100:3.0f}%", font=fonts.small, fill=(228, 236, 248, 240))

    # 5. Bottom-Center Slim Keybind Strip
    if w >= 720:
        hint = "[C] 视角   [O] 未来占用   [H] 遥测   [左键拖拽/滚轮] 环绕   [Esc] 退出"
        hw_pill = 470 * s
        hx, hy = .5 * (w - hw_pill), h - pad - 28 * s
        rr((hx, hy, hx + hw_pill, hy + 28 * s), 12 * s, (10, 15, 24, 165), (140, 180, 235, 32))
        d.text((.5 * w, hy + 14 * s), hint, font=fonts.small, fill=(166, 184, 210, 220), anchor="mm")

    return np.asarray(img, dtype=np.uint8)


class SRViewer:
    def __init__(self, size=(1280, 720), backend="auto", window=False):
        self.size = tuple(size)
        self.window = None
        self.ctx = None
        self.preset_index = 0
        _, self.azimuth, self.elevation, self.distance, self.target, self.fov = PRESETS[0]
        self.show_occupancy = False
        self.show_telemetry = True
        self.last_frame_time, self.gpu_ms, self.fps = None, None, 0.
        self.start_time = perf_counter()
        self.hud_key = None
        self.drag = None
        scale = max(.48, min(self.size[0] / 1280., self.size[1] / 720.))
        self.fonts = FontPack(scale)
        if backend == "software":
            if window:
                raise ValueError("Window mode requires an OpenGL context")
            return
        try:
            self.open_gl(window)
        except (RuntimeError, ValueError, OSError, ImportError):
            self.close()
            if backend == "gpu" or window:
                raise
            self.ctx = None

    def apply_preset(self, idx):
        self.preset_index = idx % len(PRESETS)
        _, self.azimuth, self.elevation, self.distance, self.target, self.fov = PRESETS[self.preset_index]

    def open_gl(self, visible):
        import moderngl

        if not visible:
            try:
                self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
            except (ValueError, OSError, RuntimeError, ImportError):
                self.ctx = None
        if self.ctx is None:
            import glfw

            if not glfw.init():
                raise RuntimeError("Could not initialize GLFW")
            self.glfw = glfw
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.SAMPLES, 4)
            glfw.window_hint(glfw.VISIBLE, visible)
            glfw.window_hint(glfw.RESIZABLE, False)
            self.window = glfw.create_window(*self.size, "BEV-WAM-X 3D SR", None, None)
            if not self.window:
                raise RuntimeError("Could not create GLFW OpenGL context")
            glfw.make_context_current(self.window)
            glfw.swap_interval(0)
            self.ctx = moderngl.create_context(require=330)
            glfw.set_key_callback(self.window, self.on_key)
            glfw.set_cursor_pos_callback(self.window, self.on_mouse)
            glfw.set_scroll_callback(self.window, lambda _w, _dx, dy: self.zoom(dy))
        self.gl = moderngl
        ctx = self.ctx
        if ctx.max_samples < 4:
            raise RuntimeError("4x MSAA is unavailable")
        self.msaa = ctx.framebuffer(ctx.renderbuffer(self.size, components=4, samples=4),
                                   ctx.depth_renderbuffer(self.size, samples=4))
        self.resolved = ctx.simple_framebuffer(self.size, components=4)

        self.bg_program = ctx.program(vertex_shader="""
            #version 330
            out vec2 uv;
            void main() {
                uv = vec2(gl_VertexID % 2, gl_VertexID / 2);
                gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
            }
            """, fragment_shader="""
            #version 330
            in vec2 uv;
            out vec4 pixel;
            void main() {
                vec3 zenith = vec3(0.024, 0.036, 0.062);
                vec3 horizon = vec3(0.068, 0.098, 0.148);
                vec3 floor_col = vec3(0.042, 0.056, 0.084);
                float h_band = exp(-pow((uv.y - 0.68) * 3.8, 2.0));
                vec3 col = mix(floor_col, zenith, smoothstep(0.15, 1.0, uv.y));
                col = mix(col, horizon, 0.85 * h_band);
                pixel = vec4(col, 1.0);
            }
            """)
        self.bg_vao = ctx.vertex_array(self.bg_program, [])

        self.program = ctx.program(vertex_shader="""
            #version 330
            uniform mat4 camera;
            in vec3 pos;
            in vec3 norm;
            in vec4 rgba;
            in float mat;
            out vec3 v_pos;
            out vec3 v_norm;
            out vec4 v_rgba;
            out float v_mat;
            void main() {
                gl_Position = camera * vec4(pos, 1.0);
                v_pos = pos;
                v_norm = norm;
                v_rgba = rgba;
                v_mat = mat;
            }
            """, fragment_shader="""
            #version 330
            uniform vec3 eye_pos;
            in vec3 v_pos;
            in vec3 v_norm;
            in vec4 v_rgba;
            in float v_mat;
            out vec4 pixel;
            void main() {
                vec3 base = v_rgba.rgb;
                float alpha = v_rgba.a;
                if (v_mat > 0.01) {
                    vec3 n = normalize(v_norm);
                    vec3 v = normalize(eye_pos - v_pos);
                    if (dot(n, v) < 0.0) n = -n;
                    vec3 l1 = normalize(vec3(0.45, 0.32, 0.84));
                    vec3 l2 = normalize(vec3(-0.65, -0.50, 0.42));
                    float ndl1 = max(dot(n, l1), 0.0);
                    float ndl2 = max(dot(n, l2), 0.0);
                    float hemi = 0.5 + 0.5 * n.z;
                    vec3 amb = mix(vec3(0.18, 0.21, 0.27), vec3(0.48, 0.54, 0.64), hemi);
                    vec3 h = normalize(l1 + v);
                    float shine = mix(18.0, 96.0, clamp(v_mat, 0.0, 1.0));
                    float spec = pow(max(dot(n, h), 0.0), shine) * mix(0.08, 0.50, clamp(v_mat, 0.0, 1.0));
                    float fresnel = pow(1.0 - max(dot(n, v), 0.0), 3.0) * (0.14 + 0.22 * v_mat);
                    vec3 rim_col = mix(vec3(0.55, 0.72, 0.96), base, 0.35);
                    base = base * (amb + 0.65 * ndl1 + 0.22 * ndl2) + vec3(spec) + rim_col * fresnel;
                }
                float d = length(vec2(max(v_pos.x, 0.0) * 0.95, abs(v_pos.y) * 1.15));
                float fog = smoothstep(38.0, 79.0, d);
                base = mix(base, vec3(0.068, 0.098, 0.148), fog);
                alpha *= (1.0 - smoothstep(66.0, 81.0, d));
                pixel = vec4(clamp(base, 0.0, 1.0), alpha);
            }
            """)
        self.stream = ctx.buffer(reserve=2**21, dynamic=True)
        self.vao = ctx.vertex_array(self.program, [(self.stream, "3f 3f 4f 1f", "pos", "norm", "rgba", "mat")])

        self.ribbon_program = ctx.program(vertex_shader="""
            #version 330
            uniform mat4 camera;
            in vec3 pos;
            in vec2 uv_ls;
            in vec4 rgba;
            out vec3 v_pos;
            out vec2 v_ls;
            out vec4 v_rgba;
            void main() {
                gl_Position = camera * vec4(pos, 1.0);
                v_pos = pos;
                v_ls = uv_ls;
                v_rgba = rgba;
            }
            """, fragment_shader="""
            #version 330
            uniform float u_time;
            in vec3 v_pos;
            in vec2 v_ls;
            in vec4 v_rgba;
            out vec4 pixel;
            void main() {
                float u = abs(v_ls.x);
                float s = v_ls.y;
                float core = 1.0 - smoothstep(0.32, 0.58, u);
                float edge_rail = exp(-pow((u - 0.54) * 12.0, 2.0));
                float halo = exp(-u * u * 3.8);
                float flow = 0.90 + 0.10 * sin(s * 18.0 - u_time * 5.0);
                float intensity = (0.78 * core + 0.32 * halo + 0.24 * edge_rail) * flow;
                vec3 col = mix(vec3(0.10, 0.45, 0.96), vec3(0.32, 0.80, 1.0), clamp(core + 0.5 * edge_rail, 0.0, 1.0));
                float entry = smoothstep(0.0, 0.035, s);
                float d = length(vec2(max(v_pos.x, 0.0) * 0.95, abs(v_pos.y) * 1.15));
                float far_fade = 1.0 - smoothstep(55.0, 79.0, d);
                float alpha = v_rgba.a * intensity * entry * far_fade;
                pixel = vec4(col, clamp(alpha, 0.0, 0.94));
            }
            """)
        self.ribbon_buffer = ctx.buffer(reserve=2**16, dynamic=True)
        self.ribbon_vao = ctx.vertex_array(self.ribbon_program,
                                           [(self.ribbon_buffer, "3f 2f 4f", "pos", "uv_ls", "rgba")])

        self.timer = ctx.query(time=True)
        self.hud_texture = ctx.texture(self.size, 4)
        self.hud_program = ctx.program(vertex_shader="""
            #version 330
            out vec2 uv;
            void main() {
                uv = vec2(gl_VertexID % 2, 1.0 - float(gl_VertexID / 2));
                gl_Position = vec4(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 0.0, 1.0);
            }
            """, fragment_shader="""
            #version 330
            uniform sampler2D tex;
            in vec2 uv;
            out vec4 pixel;
            void main() { pixel = texture(tex, uv); }
            """)
        self.hud_vao = ctx.vertex_array(self.hud_program, [])

    def on_key(self, window, key, scancode, action, modifiers):
        glfw = self.glfw
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_O and action == glfw.PRESS:
            self.show_occupancy = not self.show_occupancy
        elif key == glfw.KEY_H and action == glfw.PRESS:
            self.show_telemetry = not self.show_telemetry
        elif key == glfw.KEY_C and action == glfw.PRESS:
            self.apply_preset(self.preset_index + 1)
        elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
            self.azimuth += 3 if key == glfw.KEY_RIGHT else -3
        elif key in (glfw.KEY_UP, glfw.KEY_DOWN):
            self.elevation = float(np.clip(self.elevation + (3 if key == glfw.KEY_UP else -3), 12., 84.))

    def on_mouse(self, window, x, y):
        if self.glfw.get_mouse_button(window, self.glfw.MOUSE_BUTTON_LEFT) == self.glfw.PRESS:
            if self.drag is not None:
                self.azimuth += (x - self.drag[0]) * .26
                self.elevation = float(np.clip(self.elevation + (y - self.drag[1]) * .25, 12., 84.))
            self.drag = (x, y)
        else:
            self.drag = None

    def zoom(self, delta):
        self.distance = float(np.clip(self.distance - 2 * delta, 10., 75.))

    def close(self):
        if self.ctx is not None:
            self.ctx.release()
            self.ctx = None
        if hasattr(self, "glfw"):
            if self.window:
                self.glfw.destroy_window(self.window)
                self.window = None
            self.glfw.terminate()
            del self.glfw

    def draw(self, vertices, offset):
        if len(vertices):
            self.vao.render(self.gl.TRIANGLES, vertices=len(vertices), first=offset)

    def render(self, pred_dict, readback=True):
        """Render a prediction-format dictionary and optionally return a BGR uint8 image."""
        now = perf_counter()
        if self.last_frame_time is not None:
            sample = 1. / max(now - self.last_frame_time, .0001)
            self.fps = sample if not self.fps else .95 * self.fps + .05 * sample
        self.last_frame_time = now
        if pred_dict is not getattr(self, "_scene", None) or self.show_occupancy != getattr(self, "_occ", None):
            self._scene, self._occ = pred_dict, self.show_occupancy
            self._gpu_mesh = build_scene(pred_dict, self.show_occupancy)
            if self.ctx:
                opaque_11, blend_11, ribbon_9 = self._gpu_mesh
                verts = np.concatenate((opaque_11, blend_11))
                if self.stream.size < verts.nbytes:
                    self.stream.orphan(max(verts.nbytes * 2, 2**21))
                if len(verts):
                    self.stream.write(verts.tobytes())
                if self.ribbon_buffer.size < ribbon_9.nbytes:
                    self.ribbon_buffer.orphan(max(ribbon_9.nbytes * 2, 2**16))
                if len(ribbon_9):
                    self.ribbon_buffer.write(ribbon_9.tobytes())
        opaque_11, blend_11, ribbon_9 = self._gpu_mesh
        transform, eye = camera_matrix(self.size, self.azimuth, self.elevation,
                                       self.distance, self.target, self.fov)
        preset_name = PRESETS[self.preset_index][0]
        key = (round(float(pred_dict.get("speed", 0.)), 1), round(float(pred_dict.get("cruise", 0.)), 1),
               pred_dict.get("intent", "Follow"), round(float(pred_dict.get("steer", 0.)), 2),
               round(float(pred_dict.get("throttle", 0.)), 2), round(float(pred_dict.get("brake", 0.)), 2),
               round(self.fps / 4), self.show_occupancy, self.show_telemetry, preset_name,
               perception_summary(pred_dict), pred_dict.get("infer_ms"),
               pred_dict.get("nav_hint"), pred_dict.get("mode"))
        if key != self.hud_key:
            self.hud_rgba = hud(pred_dict, self.fps, size=self.size, gpu_ms=self.gpu_ms,
                                show_occupancy=self.show_occupancy, show_telemetry=self.show_telemetry,
                                preset_name=preset_name, fonts=self.fonts)
            self.hud_key = key
            if self.ctx:
                self.hud_texture.write(np.ascontiguousarray(self.hud_rgba).tobytes())
        if not self.ctx:
            opaque_7, blend_7 = geometry(pred_dict, self.show_occupancy)
            return self.software(opaque_7, blend_7, transform, self.hud_rgba)
        ctx, gl = self.ctx, self.gl
        self.msaa.use()
        self.program["camera"].write(transform.T.copy().tobytes())
        self.program["eye_pos"].write(eye.tobytes())
        self.ribbon_program["camera"].write(transform.T.copy().tobytes())
        self.ribbon_program["u_time"].value = float(now - self.start_time)
        with self.timer:
            self.msaa.clear(.024, .036, .062, 1., depth=1.)
            ctx.disable(gl.DEPTH_TEST | gl.BLEND)
            self.bg_vao.render(gl.TRIANGLE_STRIP, vertices=4)
            ctx.enable_only(gl.DEPTH_TEST)
            self.msaa.depth_mask = True
            self.draw(opaque_11, 0)
            ctx.enable(gl.BLEND)
            ctx.blend_func = gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA
            self.msaa.depth_mask = False
            self.draw(blend_11, len(opaque_11))
            if len(ribbon_9):
                self.ribbon_vao.render(gl.TRIANGLES, vertices=len(ribbon_9))
            self.msaa.depth_mask = True
            ctx.disable(gl.DEPTH_TEST)
            self.hud_texture.use()
            self.hud_vao.render(gl.TRIANGLE_STRIP, vertices=4)
            ctx.copy_framebuffer(self.resolved, self.msaa)
        self.gpu_ms = self.timer.elapsed / 1e6
        if self.window and self.glfw.get_window_attrib(self.window, self.glfw.VISIBLE):
            ctx.copy_framebuffer(ctx.screen, self.resolved)
            self.glfw.swap_buffers(self.window)
            self.glfw.poll_events()
        if readback:
            w, h = self.size
            pix = np.frombuffer(self.resolved.read(components=3, alignment=1), np.uint8)
            return pix.reshape(h, w, 3)[::-1, :, ::-1].copy()
        return None

    def software(self, opaque, blended, transform, hud_rgba):
        w, h = self.size
        y_grid = np.linspace(0., 1., h, dtype=np.float32)[:, None, None]
        bg = np.clip((1. - y_grid) * np.array([15, 25, 38]) + y_grid * np.array([8, 12, 18]), 0, 255).astype(np.uint8)
        canvas = np.broadcast_to(bg, (h, w, 3)).copy()
        verts = np.concatenate((opaque, blended)).reshape(-1, 3, 7)
        clips = np.concatenate((verts[..., :3], np.ones((*verts.shape[:2], 1))), axis=-1) @ transform.T
        # Clip triangles crossing the eye plane. Smooth contours often produce
        # long triangles spanning both sides of the camera.
        front = clips[..., 3] >= .15
        clipped_pos, clipped_attr = [], []
        for i in np.flatnonzero(front.any(axis=1) & ~front.all(axis=1)):
            polygon = []
            for j in range(3):
                k = (j + 1) % 3
                if front[i,j]:
                    polygon.append((clips[i,j], verts[i,j]))
                if front[i,j] != front[i,k]:
                    t = (.15 - clips[i,j,3]) / (clips[i,k,3] - clips[i,j,3])
                    polygon.append((clips[i,j] + t*(clips[i,k]-clips[i,j]),
                                    verts[i,j] + t*(verts[i,k]-verts[i,j])))
            for j in range(1,len(polygon)-1):
                tri = (polygon[0],polygon[j],polygon[j+1])
                clipped_pos.append([v[0] for v in tri])
                clipped_attr.append([v[1] for v in tri])
        verts, clips = verts[front.all(axis=1)], clips[front.all(axis=1)]
        if clipped_pos:
            verts = np.concatenate((verts,clipped_attr))
            clips = np.concatenate((clips,clipped_pos))
        xyz = clips[..., :3] / clips[..., 3:4]
        screen = (xyz[..., :2] * [1., -1.] + 1.) * [w / 2, h / 2]
        colors = verts[..., 3:].mean(axis=1)
        zbuffer = np.full((h,w), np.inf)
        order = np.lexsort((-xyz[...,2].mean(axis=1), colors[:,3] < .999))
        for i in order:
            poly, color = screen[i], colors[i]
            x0,y0 = np.floor(poly.min(axis=0)).astype(int).clip(0,(w-1,h-1))
            x1,y1 = np.ceil(poly.max(axis=0)).astype(int).clip(0,(w-1,h-1))
            if x0 > x1 or y0 > y1:
                continue
            u,v = poly[1:] - poly[0]
            det = u[0]*v[1]-u[1]*v[0]
            if abs(det) < 1e-8:
                continue
            yy,xx = np.ogrid[y0:y1+1,x0:x1+1]
            x,y = xx + .5-poly[0,0], yy + .5-poly[0,1]
            a,b = (x*v[1]-y*v[0])/det,(y*u[0]-x*u[1])/det
            z = xyz[i,0,2] + a*(xyz[i,1,2]-xyz[i,0,2]) + b*(xyz[i,2,2]-xyz[i,0,2])
            depth = zbuffer[y0:y1+1,x0:x1+1]
            visible = (a>=0)&(b>=0)&(a+b<=1)&(z<=depth)
            roi = canvas[y0:y1+1,x0:x1+1]
            rgb = color[2::-1]*255
            if color[3]>=.999:
                roi[visible] = rgb
                depth[visible] = z[visible]
            else:
                roi[visible] = roi[visible]*(1-color[3])+rgb*color[3]
        alpha = hud_rgba[..., 3:4].astype(np.float32) / 255.
        hud_bgr = hud_rgba[..., 2::-1].astype(np.float32)
        return np.rint(canvas.astype(np.float32) * (1. - alpha) + hud_bgr * alpha).astype(np.uint8)


def synthetic_scene():
    y = metric(np.column_stack((np.arange(W), np.zeros(W))))[:, 1]
    mask = np.zeros((4, H, W), np.uint8)
    mask[0, :, abs(y) < 7.] = 1
    center = int(W / 2)
    for row in range(H):
        if row % 10 < 5:
            mask[1, row, [center - 7, center + 7]] = 1
        mask[2, row, [center - 12, center + 12]] = 1
        mask[3, row, [center - 15, center + 15]] = 1
    boxes = np.zeros((32, 11), dtype=np.float32)
    boxes[:4] = [[17, 0, .77, 4.6, 1.9, 1.55, 0, 11, 0, 0, 1],
                 [24, 3.5, .95, 4.9, 1.9, 1.90, .10, 16, .15, 1, 1],
                 [29, -3.5, 1.7, 9.5, 2.65, 3.4, 0, 17, 0, 2, 1],
                 [-8, 3.6, .82, 4.5, 1.8, 1.65, .2, 14, 0, 0, 1]]
    x = np.linspace(.9, 32, 25)
    path = np.column_stack((x, 3.5 / (1 + np.exp(-(x - 13.) / 3.0)), np.full_like(x, 16.5)))
    alt = np.column_stack((x, -path[:, 1] * .45))
    occ = np.zeros((5, H, W), np.uint8)
    for i in range(5):
        r0 = max(0, 72 - 5 * i)
        occ[i, r0:r0 + 5, center - 2:center + 2] = 1
    return dict(lanes=mask, boxes=boxes, trajectory=path, speed=17.8, cruise=22.2,
                limit=80., intent="Overtake left", alternatives=(alt,),
                future_occupancy=occ, steer=-.04, throttle=.42, brake=0.,
                accel_lon=.8, infer_ms=6.7, mode="SYNTH",
                nav_hint="SD Nav: 180m Mainline Straight")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frame", type=Path, help="Offline capture frame converted to the prediction schema")
    p.add_argument("--out", type=Path, default=Path("sr_preview.png"))
    p.add_argument("--window", action="store_true")
    p.add_argument("--backend", choices=("gpu", "software", "auto"), default="auto")
    p.add_argument("--occupancy", action="store_true")
    p.add_argument("--benchmark-frames", type=int, default=0)
    args = p.parse_args()
    scene = from_capture(args.frame) if args.frame else synthetic_scene()
    viewer = SRViewer(backend=args.backend, window=args.window)
    viewer.show_occupancy = args.occupancy
    try:
        image = viewer.render(scene)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(cv2.imencode(".png", image)[1].tobytes())
        if args.benchmark_frames:
            if viewer.ctx is None:
                raise RuntimeError("GPU timing requires an OpenGL context")
            timings = []
            started = perf_counter()
            for _ in range(args.benchmark_frames):
                viewer.render(dict(scene), readback=False)
                timings.append(viewer.gpu_ms)
            print(json.dumps(dict(frames=len(timings), gpu_median_ms=float(np.median(timings)),
                                  gpu_p95_ms=float(np.quantile(timings, .95)),
                                  fps=len(timings) / (perf_counter() - started),
                                  resolution=viewer.size, renderer=viewer.ctx.info["GL_RENDERER"])))
        if args.window:
            while not viewer.glfw.window_should_close(viewer.window):
                viewer.render(scene, readback=False)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
