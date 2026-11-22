"""Minimal surround-reconstruction viewer. Render inputs use ego FLU prediction geometry."""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .geometry import DX, DY, H, W, metric
from .labels import OBJECT_SLOTS, footprint
from .qc_v3 import load_npz

BG = (.035, .045, .062, 1.)
PAINT = {0: (.18, .21, .25, 1.), 1: (.89, .94, .99, 1.),
         2: (1., .82, .38, 1.), 3: (.43, .57, .65, 1.)}
FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
         (2, 3, 7, 6), (3, 0, 4, 7))
THW_RED = (.95, .20, .19, 1.)


def unit(x):
    return x / max(np.linalg.norm(x), 1e-8)


def prediction_scene(pred, *, speed=0., cruise=0., intent="跟车", alternatives=(), previous_scene=None):
    """Convert one batched model result into the unbatched renderer contract."""
    def as_numpy(x):
        return x.detach().float().cpu().numpy()[0]

    geom = as_numpy(pred["boxes"])
    yaw = np.arctan2(geom[:, 6], geom[:, 7])
    cls = as_numpy(pred["cls_logits"]).argmax(axis=-1)
    visible = as_numpy(pred["obj_logits"]) > 0.
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
    return dict(lanes=as_numpy(pred["lane_logits"]) > 0., boxes=boxes,
                future_occupancy=as_numpy(pred["future_occupancy_logits"]) > 0.,
                trajectory=as_numpy(pred["trajectory"]), speed=speed, cruise=cruise, intent=intent,
                alternatives=alternatives, decelerating=decelerating)


def from_capture(frame):
    labels = load_npz(frame / "labels.npz")
    return dict(lanes=labels["lanes"], boxes=labels["boxes"], trajectory=labels["trajectory"],
                future_occupancy=labels["future_occupancy"], speed=float(labels["ego_state"][0]),
                cruise=0., intent="Replay")


def ribbon(path, width=1.9):
    """Triangle strip ordered left/right, with monotone alpha fade by arc length."""
    xy = np.asarray(path, dtype=np.float32)[:, :2]
    xy = np.vstack((np.zeros((1, 2), np.float32), xy))
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-3]]
    if len(xy) < 2:
        return np.empty((0, 7), np.float32)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    # Interpolate then use local chord direction for a continuous, centered ribbon.
    steps = np.linspace(0, arc[-1], max(48, len(xy) * 2))
    p = np.column_stack([np.interp(steps, arc, xy[:, j]) for j in range(2)])
    p[1:-1] = (.20 * p[:-2] + .60 * p[1:-1] + .20 * p[2:])
    tangent = np.gradient(p, axis=0)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    rails = np.stack((p + width / 2 * normal, p - width / 2 * normal), axis=1)
    linear = steps / max(arc[-1], 1e-6)
    colors = np.column_stack((.12 - .07 * linear, .64 - .18 * linear,
                              np.ones_like(steps), .91 - .82 * linear))
    return np.column_stack((rails.reshape(-1, 2), np.full(rails.size // 2, .047),
                             np.repeat(colors, 2, axis=0))).astype(np.float32)


def ribbon_triangles(strip):
    if len(strip) < 3:
        return strip
    ids = np.arange(len(strip) - 2)
    return strip[np.column_stack((ids, ids + 1, ids + 2)).ravel()]


class Mesh:
    def __init__(self):
        self.opaque, self.blend = [], []

    def quad(self, corners, color, blend=False):
        p = np.asarray(corners)[[0, 1, 2, 0, 2, 3]]
        vertices = np.column_stack((p, np.broadcast_to(color, (6, 4))))
        (self.blend if blend else self.opaque).append(vertices)

    def flat(self, xy, z, color, blend=False):
        self.quad(np.column_stack((xy, np.full(4, z))), color, blend)

    def cuboid(self, box, color, top_scale=1.):
        base = footprint(box)
        top = (base - box[:2]) * top_scale + box[:2]
        h = box[5]
        points = np.vstack((np.column_stack((base, np.full(4, box[2] - h / 2))),
                            np.column_stack((top, np.full(4, box[2] + h / 2)))))
        for indices, shade in zip(FACES, (.63, 1., .84, .72, .58, .78)):
            rgb = np.asarray(color[:3]) * shade
            self.quad(points[list(indices)], (*rgb, color[3]))

    def mask(self, mask, z, color, blend=False, curb=False):
        rows, bounds = np.nonzero(np.diff(np.pad(mask.astype(bool), ((0, 0), (1, 1))), axis=1))
        if not len(rows):
            return
        a, b = bounds.reshape(-1, 2).T - .5
        row = rows[::2]
        u = np.stack((a, b, b, a), axis=1)
        v = row[:, None] + np.array([-.5, -.5, .5, .5])
        xy = metric(np.stack((u, v), axis=-1))

        def emit(quads, rgba):
            vertices = quads[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3)
            (self.blend if blend else self.opaque).append(
                np.column_stack((vertices, np.broadcast_to(rgba, (len(vertices), 4)))))

        if not curb:
            emit(np.concatenate((xy, np.full((*xy.shape[:2], 1), z)), axis=-1), color)
            return
        center = xy.mean(axis=1, keepdims=True)
        inset = center + (xy - center) * [.54, .66]
        base = np.concatenate((inset, np.full((*xy.shape[:2], 1), z - .025)), axis=-1)
        top = base + [0., 0., .16]
        corners = np.concatenate((base, top), axis=1)
        for ids, shade in zip(FACES[1:], (1., .84, .72, .58, .78)):
            emit(corners[:, ids], (*[c * shade for c in color[:3]], color[3]))

    def vehicle(self, box, color, ego=False):
        length, height, yaw, kind = box[3], box[5], box[6], int(box[9])
        ground = box[2] - height / 2
        body = np.array(box, dtype=float)
        base_fraction = .78 if kind == 2 else .55
        body[2], body[5] = ground + height * base_fraction / 2, height * base_fraction
        self.cuboid(body, color, .94)
        cabin = body.copy()
        shift = (-.14 if kind == 2 else -.10) * length
        cabin[:2] += [shift * math.cos(yaw), shift * math.sin(yaw)]
        cabin[2] = ground + height * (1. + base_fraction) / 2
        cabin[3] *= .67 if kind == 2 else .50 if kind == 0 else .62
        cabin[4] *= .80
        cabin[5] = height * (1. - base_fraction)
        cabin_color = tuple(np.asarray(color[:3]) * (.72 if ego else .67)) + (1.,)
        self.cuboid(cabin, cabin_color, .82 if kind != 2 else .98)
        # Two light details at the local forward end make orientation visible.
        for side in (-1, 1):
            light = body.copy()
            dx, dy = length * .48, side * box[4] * .31
            light[:2] += [dx * math.cos(yaw) - dy * math.sin(yaw),
                          dx * math.sin(yaw) + dy * math.cos(yaw)]
            light[3:6] = (.09, min(.28, box[4] * .21), .09)
            self.cuboid(light, (.84, .94, 1., 1.))


def geometry(scene, show_occupancy=False):
    masks = np.asarray(scene["lanes"])
    boxes = np.asarray(scene["boxes"])
    trajectory = np.asarray(scene["trajectory"])
    if masks.shape != (4, H, W) or boxes.shape != (32, 11) or trajectory.shape != (25, 3):
        raise ValueError("Expected lane masks 4x160x120, 32 boxes, and 25 trajectory steps")
    if not all(np.isfinite(a).all() for a in (masks, boxes, trajectory)):
        raise ValueError("Non-finite viewer input")
    mesh = Mesh()
    mesh.flat(np.array([[80, 30], [80, -30], [-40, -30], [-40, 30]]), -.045, (.087, .106, .133, 1.))
    for i, mask in enumerate(masks):
        mesh.mask(mask, i * .012, PAINT[i], curb=i == 3)
    decel = scene.get("decelerating", np.zeros(OBJECT_SLOTS, dtype=bool))
    speed = float(scene.get("speed", 0))
    for i, box in enumerate(boxes):
        if box[-1] < .5 or min(box[3:6]) <= 0:
            continue
        ahead = 0 < box[0] < 40 and abs(box[1]) < 2 and abs(box[6]) < 1.2
        thw = max(0., box[0] - box[3] / 2 - 2.3) / max(speed, .1)
        urgent = ahead and (thw < 1.5 or decel[i])
        mesh.vehicle(box, THW_RED if urgent else (.58, .67, .76, 1.))
    # Layered opacity beneath a light gray ego model.
    ego = np.array([0., 0., .74, 4.65, 1.9, 1.48, 0., 0., 0., 0., 1.])
    for j, factor in enumerate((1.46, 1.30, 1.17)):
        # Layer gray-blended ground colors for a soft contact shadow.
        mesh.flat(footprint(ego) * factor, .002 + .001 * j, (.075 - .015 * j, .090 - .017 * j, .112 - .02 * j, 1.))
    mesh.vehicle(ego, (.81, .86, .92, 1.), ego=True)
    if show_occupancy and "future_occupancy" in scene:
        future = np.asarray(scene["future_occupancy"])
        if future.shape != (5, H, W):
            raise ValueError("Future occupancy must be 5x160x120")
        for i, mask in reversed(list(enumerate(future))):
            mesh.mask(mask, .042 - i * .002, (.24, .55, .93, .24 - .033 * i), blend=True)
    for other in scene.get("alternatives", ()):
        alternative = ribbon(np.asarray(other), .20)
        alternative[:, 2] += .005
        alternative[:, 3:7] = (.40, .80, 1., .38)
        mesh.blend.append(ribbon_triangles(alternative))
    glow = ribbon(trajectory, width=2.35)
    glow[:, 2] -= .001
    glow[:, 6] *= .21
    mesh.blend.append(ribbon_triangles(glow))
    mesh.blend.append(ribbon_triangles(ribbon(trajectory)))
    opaque, blend = (np.concatenate(v).astype(np.float32) if len(v) else np.empty((0, 7), np.float32)
                     for v in (mesh.opaque, mesh.blend))
    return opaque, blend


def camera_matrix(size, azimuth, elevation, distance):
    w, h = size
    target = np.array([7., 0., 0.])
    angle, tilt = np.deg2rad([azimuth, elevation])
    eye = target + distance * np.array([-np.cos(tilt) * np.cos(angle),
                                       np.cos(tilt) * np.sin(angle), np.sin(tilt)])
    fwd = unit(target - eye)
    side = unit(np.cross(fwd, [0, 0, 1.]))
    up = np.cross(side, fwd)
    view = np.eye(4)
    view[:3, :3] = np.stack((side, up, -fwd))
    view[:3, 3] = -view[:3, :3] @ eye
    f = 1. / math.tan(math.radians(42.) / 2)
    near, far = .15, 200.
    perspective = np.array([[f * h / w, 0, 0, 0], [0, f, 0, 0],
                            [0, 0, (far + near) / (near - far), 2 * near * far / (near - far)],
                            [0, 0, -1, 0]])
    return (perspective @ view).astype(np.float32)


def hud(scene, fps):
    text = f"{scene.get('speed', 0) * 3.6:5.1f} km/h   Cruise {scene.get('cruise', 0) * 3.6:4.0f}   {scene.get('intent', 'Follow')}   {fps:4.0f} FPS"
    img = np.full((38, 605, 3), (26, 20, 15), dtype=np.uint8)
    cv2.putText(img, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .54, (225, 231, 241), 1, cv2.LINE_AA)
    return img


class SRViewer:
    def __init__(self, size=(960, 600), backend="auto", window=False):
        self.size = tuple(size)
        self.window = None
        self.ctx = None
        self.azimuth, self.elevation, self.distance = 10., 46., 28.
        self.show_occupancy = False
        self.last_frame_time, self.gpu_ms, self.fps = None, None, 0.
        self.hud_key = None
        self.drag = None
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
            self.window = glfw.create_window(*self.size, "Stage 3 Surround View", None, None)
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
        self.program = ctx.program(vertex_shader="""
            #version 330
            uniform mat4 camera;
            in vec3 pos;
            in vec4 rgba;
            out vec4 color;
            void main() { gl_Position = camera * vec4(pos, 1); color = rgba; }
            """, fragment_shader="""
            #version 330
            in vec4 color;
            out vec4 pixel;
            void main() { pixel = color; }
            """)
        self.stream = ctx.buffer(reserve=2**20, dynamic=True)
        self.vao = ctx.vertex_array(self.program, [(self.stream, "3f 4f", "pos", "rgba")])
        self.timer = ctx.query(time=True)
        self.hud_texture = ctx.texture((605, 38), 3)
        self.hud_program = ctx.program(vertex_shader="""
            #version 330
            uniform vec2 scale;
            out vec2 uv;
            void main() {
                uv = vec2(gl_VertexID % 2, gl_VertexID / 2);
                gl_Position = vec4(vec2(-.975, .96) + uv * vec2(scale.x, -scale.y), 0, 1);
            }
            """, fragment_shader="""
            #version 330
            uniform sampler2D tex;
            in vec2 uv;
            out vec4 pixel;
            void main() { pixel = vec4(texture(tex, uv).rgb, 1); }
            """)
        self.hud_program["scale"].value = (min(1.94, 1210 / self.size[0]), 76 / self.size[1])
        self.hud_vao = ctx.vertex_array(self.hud_program, [])

    def on_key(self, window, key, scancode, action, modifiers):
        glfw = self.glfw
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_O and action == glfw.PRESS:
            self.show_occupancy = not self.show_occupancy
        elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
            self.azimuth += 3 if key == glfw.KEY_RIGHT else -3
        elif key in (glfw.KEY_UP, glfw.KEY_DOWN):
            self.elevation = float(np.clip(self.elevation + (3 if key == glfw.KEY_UP else -3), 18., 82.))

    def on_mouse(self, window, x, y):
        if self.glfw.get_mouse_button(window, self.glfw.MOUSE_BUTTON_LEFT) == self.glfw.PRESS:
            if self.drag is not None:
                self.azimuth += (x - self.drag[0]) * .26
                self.elevation = float(np.clip(self.elevation + (y - self.drag[1]) * .25, 18., 82.))
            self.drag = (x, y)
        else:
            self.drag = None

    def zoom(self, delta):
        self.distance = float(np.clip(self.distance - 2 * delta, 15., 75.))

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
            self._mesh = geometry(pred_dict, self.show_occupancy)
            if self.ctx:
                vertices = np.concatenate(self._mesh)
                if self.stream.size < vertices.nbytes:
                    self.stream.orphan(max(vertices.nbytes * 2, 2**20))
                self.stream.write(vertices.tobytes())
        opaque, blend = self._mesh
        transform = camera_matrix(self.size, self.azimuth, self.elevation, self.distance)
        key = (pred_dict.get("speed", 0), pred_dict.get("cruise", 0), pred_dict.get("intent", "Follow"),
               round(self.fps / 4))
        if key != self.hud_key:
            self.hud_image = hud(pred_dict, self.fps)
            self.hud_key = key
            if self.ctx:
                self.hud_texture.write(np.ascontiguousarray(self.hud_image[..., ::-1]).tobytes())
        if not self.ctx:
            return self.software(opaque, blend, transform, self.hud_image)
        ctx, gl = self.ctx, self.gl
        self.msaa.use()
        self.program["camera"].write(transform.T.copy().tobytes())
        with self.timer:
            self.msaa.clear(*BG)
            ctx.enable_only(gl.DEPTH_TEST)
            self.msaa.depth_mask = True
            self.draw(opaque, 0)
            ctx.enable(gl.BLEND)
            ctx.blend_func = gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA
            self.msaa.depth_mask = False
            self.draw(blend, len(opaque))
            self.msaa.depth_mask = True
            ctx.disable(gl.DEPTH_TEST | gl.BLEND)
            self.hud_texture.use()
            self.hud_vao.render(gl.TRIANGLE_STRIP, vertices=4)
            ctx.copy_framebuffer(self.resolved, self.msaa)
        # Read after submission so the value includes the completed GPU query.
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

    def software(self, opaque, blended, transform, hud_image):
        w, h = self.size
        canvas = np.full((h, w, 3), tuple(round(255 * c) for c in BG[:3][::-1]), np.uint8)
        verts = np.concatenate((opaque, blended)).reshape(-1, 3, 7)
        clips = np.concatenate((verts[..., :3], np.ones((*verts.shape[:2], 1))), axis=-1) @ transform.T
        good = (clips[..., 3] > .12).all(axis=1)
        verts, clips = verts[good], clips[good]
        xyz = clips[..., :3] / clips[..., 3:4]
        points = np.rint((xyz[..., :2] * [1., -1.] + 1.) * [w / 2, h / 2]).astype(np.int32)
        colors = verts[..., 3:].mean(axis=1)
        for i in np.argsort(-xyz[..., 2].mean(axis=1), kind="stable"):
            poly, color = points[i], colors[i]
            x0, y0 = poly.min(axis=0).clip(0, (w - 1, h - 1))
            x1, y1 = poly.max(axis=0).clip(0, (w - 1, h - 1))
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            rgb = tuple(map(int, np.rint(color[2::-1] * 255)))
            if color[3] >= .999:
                cv2.fillConvexPoly(canvas, poly, rgb, cv2.LINE_AA)
            else:
                roi = canvas[y0:y1 + 1, x0:x1 + 1]
                mask = np.zeros(roi.shape[:2], np.uint8)
                cv2.fillConvexPoly(mask, poly - [x0, y0], 1)
                inside = mask > 0
                roi[inside] = np.rint(roi[inside] * (1. - color[3]) + np.asarray(rgb) * color[3])
        overlay = hud_image[:, :max(w - 24, 1)]
        canvas[12:12 + len(overlay), 12:12 + overlay.shape[1]] = overlay
        return canvas


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
    x = np.linspace(.9, 25, 25)
    path = np.column_stack((x, 2.7 / (1 + np.exp(-(x - 11.) / 2.6)), np.full_like(x, 12.)))
    alt = np.column_stack((x, -path[:, 1]))
    occ = np.zeros((5, H, W), np.uint8)
    for i in range(5):
        occ[i, 66 - 4*i:72 - 4*i, 59:63] = 1
    return dict(lanes=mask, boxes=boxes, trajectory=path, speed=16., cruise=21., intent="Overtake left",
                alternatives=(alt,), future_occupancy=occ)


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
            # Rebuild the changing prediction mesh each iteration and measure full frame rate.
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
