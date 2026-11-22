"""Interactive CARLA spectator and sensor dashboard for the frozen Phase 2 driver."""
import argparse
import json
from pathlib import Path
import time

import carla
import cv2
import numpy as np

from . import phase2_live
from .collect import DT, matrix
from .geometry import pixels, projection_tables
from .phase2_demo_lane import AdjacentLaneRequest, LaneAssist
from .phase2_model import mounts
from .render_probe import bgr, locked_sizes


WINDOW = 'CARLA | Mapless driving'
SIZE = (1832, 930)
CHASE_SIZE = (1152, 648)
CHASE_FOV = 92
GAMMA_LUT = np.rint(np.sqrt(np.arange(256)/255.)*255).astype(np.uint8)
INK = (234, 241, 247)
MUTED = (154, 170, 185)
BACKGROUND = (21, 16, 14)
PANEL = (39, 31, 27)
CYAN = (244, 210, 73)
AMBER = (72, 180, 254)
RED = (121, 96, 253)
GREEN = (151, 226, 125)
BUTTONS = (
    ((1200, 802, 1388, 847), 'S   Speed -5', 'slower'),
    ((1403, 802, 1591, 847), 'W   Speed +5', 'faster'),
    ((1606, 802, 1794, 847), 'Space   Pause', 'pause'),
    ((1200, 859, 1388, 904), 'A   Lane left', 'left'),
    ((1403, 859, 1591, 904), 'D   Lane right', 'right'),
    ((1606, 859, 1794, 904), 'R   Replay', 'restart'),
)
KEYS = {ord('s'): 'slower', ord('w'): 'faster', ord('a'): 'left', ord('d'): 'right',
        ord('r'): 'restart', ord(' '): 'pause', 27: 'close', ord('q'): 'close'}
COMMANDS = ('Keep lane', 'Lane left', 'Lane right')


def label(canvas, value, x, y, scale=.56, color=INK, weight=1):
    cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def mark(canvas, box, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)


def draw_curve(canvas, points, color, width):
    if len(points) > 1:
        cv2.polylines(canvas, [np.rint(points).astype(np.int32)], False, color, width, cv2.LINE_AA)


def rgb_view(camera):
    return cv2.LUT(bgr(camera), GAMMA_LUT)


def fast_ground(scan):
    pts = scan[::4, :3]
    use = (pts[:, 0] >= 1.8) & (pts[:, 0] < 30.) & (np.abs(pts[:, 1]) < 3.4) & (np.abs(pts[:, 2]) < .55)
    pts = pts[use]
    if len(pts) < 12:
        return np.zeros(3)
    design = np.column_stack((pts[:, :2], np.ones(len(pts), dtype=np.float32)))
    z = pts[:, 2]
    keep = z <= np.quantile(z, 0.55)
    for _ in range(2):
        if keep.sum() < 8:
            return np.zeros(3)
        abc, _, rank, _ = np.linalg.lstsq(design[keep], z[keep], rcond=None)
        if rank != 3:
            return np.zeros(3)
        keep = np.abs(z - design @ abc) <= 0.08
    return abc.astype(np.float64)


class Dashboard:
    def __init__(self, destination, replay_minutes=0, spec=None, title_tag='STAGE 2.6', exit_on_finish=False):
        self.destination = destination
        self.replay_minutes = replay_minutes
        self.replay_until = None
        self.exit_on_finish = bool(exit_on_finish)
        self.spec = spec or {'name': 'town04_ring', 'town': 'Town04_Opt'}
        self.title_tag = title_tag
        self.camera_poses = mounts()
        self.chase_mount = carla.Transform(carla.Location(x=-8.2,z=4.1), carla.Rotation(pitch=-11.5))
        self.chase_pose = matrix(self.chase_mount)
        uv, seen = projection_tables(self.camera_poses, (0.,),sizes=locked_sizes())
        self.remaps = [np.ascontiguousarray(table) for table in uv[0]]
        self.camera_count = seen[0].sum(0).clip(1)
        self.coverage = seen[0].any(0)
        self.closed = False
        self.start_route(self.spec)
        self.canvas = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, dtype=np.uint8)
        label(self.canvas, f"Loading {self.spec.get('town', 'CARLA')} / {self.spec.get('name', '')}", 33, 70, .9, weight=2)
        label(self.canvas, f'{self.title_tag}  /  High  /  HIP Graph', 34, 104, .56, MUTED)
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(WINDOW, 24, 50)
        cv2.setMouseCallback(WINDOW, self.click)
        cv2.imshow(WINDOW, self.canvas)
        cv2.waitKey(1)

    def chase_camera(self, library):
        bp = library.find('sensor.camera.rgb')
        for name,value in dict(image_size_x=CHASE_SIZE[0],image_size_y=CHASE_SIZE[1],fov=CHASE_FOV,
                               sensor_tick=0,enable_postprocess_effects='True',use_ray_tracing='False',
                               motion_blur_intensity='0.0').items():
            if bp.has_attribute(name):
                bp.set_attribute(name,str(value))
        return bp,self.chase_mount

    def start_route(self, spec):
        self.route = spec['name']
        self.target_kmh = float(spec.get('cruise_kmh', 46))
        self.tick = 0
        self.intent = 0
        self.intent_until = 0
        self.intent_tick = None
        self.request = AdjacentLaneRequest()
        self.lane_assist = LaneAssist()
        self.lane_status = 'Waiting for lane'
        self.lane_side = 0
        self.feedback_until = 0
        self.restart = False
        self.paused = False
        self.finished = False
        self.last_tick = None
        self._fps_t = None
        self._fps_ema = 0.0

    @property
    def command(self):
        return self.intent if self.tick<self.intent_until else 0

    def observe_command(self, output):
        if self.command and self.request.observe(output):
            self.intent = 0
            self.lane_status = 'New lane detected / Keep lane'
            return True
        elif self.intent and not self.command:
            self.intent = 0
            self.lane_status = 'Request expired / Keep lane'
        return False

    def choose_path(self, pose, output, raw, speed):
        request,self.intent = self.intent,0
        path,status,cap,complete,self.lane_side = self.lane_assist.step(pose,output,raw,speed,request)
        if status in ('No adjacent lane','Target lane blocked','Lane change complete','Slow down to change'):
            self.feedback_until = self.tick + round(2 / DT)
        if status != 'Tracking lane' or self.tick >= self.feedback_until:
            self.lane_status = status
        return path,cap,dict(lane_assist=status,lane_change_complete=complete)

    def action(self, name):
        if name == 'close':
            self.closed = True
        elif name == 'restart':
            self.restart = True
        elif not self.finished:
            if name == 'pause':
                self.paused = not self.paused
            elif name in ('faster', 'slower'):
                delta = 5 if name == 'faster' else -5
                self.target_kmh = min(75, max(5, self.target_kmh + delta))
            elif name in ('left', 'right') and not self.command and not self.lane_side:
                self.intent = 1 if name == 'left' else 2
                self.intent_tick = self.tick
                self.request.begin(self.intent)
                self.intent_until=self.tick+round(3.5/DT)
                self.lane_status='Left requested' if self.intent==1 else 'Right requested'

    def click(self, event, x, y, flags, state):
        if event == cv2.EVENT_LBUTTONUP:
            for (xmin, ymin, xmax, ymax), _, name in BUTTONS:
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    self.action(name)
                    break

    def poll(self, delay=1):
        key = cv2.waitKeyEx(delay)
        if ord('A') <= key <= ord('Z'):
            key += ord('a')-ord('A')
        if key in KEYS:
            self.action(KEYS[key])
        try:
            self.closed |= cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            self.closed = True

    def advance(self, tick):
        if self.replay_until is None:
            self.replay_until = time.monotonic() + self.replay_minutes * 60
        self.tick = tick
        self.poll()
        while self.paused and not (self.closed or self.restart):
            self.paint_controls('PAUSED')
            cv2.imshow(WINDOW, self.canvas)
            self.poll(30)
            self.last_tick = None
        if self.last_tick is not None:
            time.sleep(max(0., self.last_tick + DT - time.monotonic()))
        self.last_tick = time.monotonic()
        return not (self.closed or self.restart)

    def project_chase(self, path, frame, color, thickness, ground):
        pose = self.chase_pose
        h,w = frame.shape[:2]
        z = (path @ ground[:2] + ground[2]).clip(-2.,2.)
        xyz = np.c_[path[:, 0], -path[:, 1], z]
        xyz = (xyz-pose[:3, 3])@pose[:3, :3]
        xyz = xyz[xyz[:, 0] > .75]
        focal = w/(2*np.tan(np.deg2rad(CHASE_FOV/2)))
        uv = np.c_[w/2+focal*xyz[:, 1]/xyz[:, 0], h/2-focal*xyz[:, 2]/xyz[:, 0]]
        draw_curve(frame, uv, (27,23,18), thickness+2)
        draw_curve(frame, uv, color, thickness)

    def render_bev(self, views, density, mask, nav, planned, occupancy, route=None, target=None):
        warped = [cv2.remap(img, table, None, cv2.INTER_LINEAR) for img, table in zip(views, self.remaps)]
        fused = np.sum(warped, axis=0) / self.camera_count[..., None]
        canvas = np.full((128, 128, 3), (31, 24, 20), np.float32)
        canvas[self.coverage] = .50 * fused[self.coverage] + (26, 20, 16)
        returns = np.clip(density.astype(np.float32)/4., 0., .7)[..., None]
        canvas = canvas*(1-returns)+returns*np.array((151, 176, 190))
        for cells, color, mix in ((mask[0]>0, GREEN, .20), (mask[3]>0, CYAN, .17),
                                  (mask[1]>0, (220, 222, 220), .55), (mask[2]>0, AMBER, .7),
                                  (occupancy.max(0)>0, RED, .65)):
            canvas[cells] = canvas[cells]*(1-mix)+np.array(color)*mix
        pane = cv2.resize(canvas[20:117, 3:125].astype(np.uint8), (368, 642), interpolation=cv2.INTER_NEAREST)
        draw = lambda points: (pixels(points)-[3, 20])*[368/122, 642/97]
        for meter in (0, 20, 40, 60):
            py = round(draw(np.array([[meter, 0.]]))[0, 1])
            cv2.line(pane, (0, py), (367, py), (73, 67, 63), 1)
            label(pane, f'{meter} m', 5, py-4, .40, MUTED)
        draw_curve(pane, draw(nav if route is None else route), CYAN, 2)
        draw_curve(pane, draw(planned), AMBER, 3)
        if target is not None:
            for point in draw(target[2::2]):
                cv2.circle(pane,tuple(np.rint(point).astype(int)),3,GREEN,-1)
        for point in draw(nav[1:]):
            cv2.circle(pane, tuple(np.rint(point).astype(int)), 4, CYAN, -1, cv2.LINE_AA)
        car = np.array([[2.3, -.95], [2.3, .95], [-2.3, .95], [-2.3, -.95]])
        cv2.fillConvexPoly(pane, np.rint(draw(car)).astype(np.int32), INK, cv2.LINE_AA)
        return pane

    def paint_controls(self, status):
        mark(self.canvas, (1188, 777, 1810, 918), PANEL)
        for box, text, name in BUTTONS:
            active = (name, self.command) in (('left', 1), ('right', 2))
            mark(self.canvas, box, (103, 70, 31) if active else (64, 49, 39))
            if name == 'pause':
                text = 'Space   Resume' if self.paused else 'Space   Pause'
            label(self.canvas, text, box[0]+15, box[1]+29, .55, CYAN if active else INK)
        mark(self.canvas, (29, 854, 1175, 914), BACKGROUND)
        label(self.canvas, status, 31, 883, .60, AMBER if self.finished or self.paused else GREEN)
        label(self.canvas, f'Target {self.target_kmh:.0f} km/h   /   {COMMANDS[self.command]}', 430, 883, .60)
        label(self.canvas, 'Esc exit', 1068, 883, .48, MUTED)

    def present(self, images, chase_image, density, raw, output, nav, path, occupancy, speed,
                throttle, brake, safety, progress, route=None, target=None, steer=None):
        now = time.perf_counter()
        if self._fps_t is not None and now > self._fps_t:
            inst = 1.0 / (now - self._fps_t)
            self._fps_ema = inst if self._fps_ema == 0.0 else 0.85 * self._fps_ema + 0.15 * inst
        self._fps_t = now
        views = [rgb_view(image) for image in images]
        nav = np.vstack(([0., 0.], nav))
        planned = np.vstack(([0., 0.], path))
        chase = np.frombuffer(chase_image.raw_data, np.uint8).reshape(chase_image.height, chase_image.width, 4)[:, :, :3].copy()
        ground = fast_ground(raw)
        boundaries = output['lines'][0,1:3].float().cpu().numpy()
        exists = output['exists'][0,1:3].float().cpu().numpy()
        for points,confidence in zip(boundaries,exists):
            visible = np.isfinite(points).all(1)&(points[:,0]>=0.)&(points[:,0]<45.)
            if confidence>0 and visible.sum()>=3:
                self.project_chase(points[visible],chase,INK,2,ground)
        self.project_chase(nav if route is None else route,chase,CYAN,4,ground)
        self.project_chase(planned,chase,AMBER,4,ground)
        if target is not None:
            for i in range(2,len(target)-1,3):
                self.project_chase(target[i:i+2],chase,GREEN,2,ground)
        mask = output['mask'][0].float().cpu().numpy()
        canvas = self.canvas
        canvas[:130] = BACKGROUND
        canvas[772:] = BACKGROUND
        canvas[130:778, :24] = BACKGROUND
        canvas[130:778, 1176:1200] = BACKGROUND
        canvas[130:778, 1568:] = BACKGROUND
        label(canvas, f"{self.spec.get('town', 'CARLA').upper()}  /  {self.route.upper()}", 29, 49, .95, weight=2)
        label(canvas, f'{self.title_tag}   HIGH   HIP GRAPH   {self._fps_ema:4.1f} FPS', 31, 75, .49, MUTED)
        label(canvas, f'{speed:04.1f} km/h', 29, 833, 1.05, weight=2)
        label(canvas, f'{progress:04.0f} m   Throttle {throttle:.0%}   Brake {brake:.0%}', 282, 832, .63,
              RED if safety.get('safety_brake') else MUTED)
        if steer is not None and 'curve_kmh' in safety:
            side = 'L' if steer<0 else 'R'
            label(canvas,f"Limit {safety['curve_kmh']:.0f} km/h   Steer {side} {abs(steer):.0%}",
                  747,832,.63,MUTED)
        label(canvas, 'CHASE CAMERA', 29, 112, .62, weight=2)
        if target is not None:
            label(canvas,'Selected lane',575,113,.49,GREEN)
        for x,text,color in ((708,'Navigation',CYAN),(849,'Tracking path',AMBER),(1008,'Current lane edges',INK)):
            mark(canvas,(x,105,x+13,109),color)
            label(canvas,text,x+20,113,.49,color)
        canvas[130:778, 24:1176] = chase if chase.shape[:2] == (648, 1152) else cv2.resize(chase,(1152,648),interpolation=cv2.INTER_LINEAR)
        label(canvas, 'COMPOSITE BEV', 1200, 112, .62, weight=2)
        canvas[130:772, 1200:1568] = self.render_bev(views,density,mask,nav,planned,occupancy,route=route,target=target)
        label(canvas, 'RGB   768 / 384 / 384', 1584, 112, .55, weight=2)
        for y,caption,view in zip((130,307,484),('FRONT','FRONT LEFT','FRONT RIGHT'),views):
            canvas[y:y+133, 1584:1812] = cv2.resize(view,(228,133),interpolation=cv2.INTER_LINEAR)
            label(canvas,caption,1590,y+153,.50,MUTED)
        label(canvas, 'Future occupancy', 1604, 684, .53, RED)
        label(canvas, 'LiDAR / ground projection', 1604, 713, .46, MUTED)
        self.paint_controls('LIDAR BRAKE' if safety.get('safety_brake') else self.lane_status.upper())
        cv2.imshow(WINDOW, canvas)
        if self.tick in (0, 100):
            cv2.imwrite(str(self.destination/'dashboard.jpg'), canvas)

    def wait_for_restart(self, result):
        self.finished = True
        if self.restart or self.closed or self.exit_on_finish:
            if self.exit_on_finish:
                self.closed = True
            return
        status = ('ROUTE DONE' if result['reason'] == 'reached_end' else
                  result['reason'].replace('_', ' ').upper()) if result else 'STOPPED'
        replay = self.replay_until is not None and time.monotonic() < self.replay_until
        if replay:
            status += '   /   REPLAY IN 3 S'
        self.paint_controls(status)
        cv2.imshow(WINDOW, self.canvas)
        next_start = time.monotonic() + 3
        while not (self.closed or self.restart):
            self.poll(40)
            if replay and time.monotonic() >= next_start:
                self.restart = True


def main(args):
    route = json.loads(args.plan.read_text())
    if args.route:
        matches = [s for s in route if s['name'] == args.route]
    else:
        matches = [s for s in route if s['town'] == 'Town04_Opt' and s['name'] == 'town04_ring'] or route[:1]
    if len(matches) != 1:
        raise ValueError(f'Expected one demo route matching {args.route!r}, found {len(matches)}')
    selected = dict(matches[0])
    if args.npc is not None:
        selected['npc'] = args.npc
    if args.max_ticks is not None:
        selected['max_ticks'] = args.max_ticks
    if args.cruise_kmh is not None:
        selected['cruise_kmh'] = args.cruise_kmh
    if args.npc_along_route:
        selected['npc_along_route'] = True
    args.out.mkdir(parents=True, exist_ok=False)
    args.plan = args.out/'demo-plan.json'
    args.plan.write_text(json.dumps([selected]), encoding='utf-8')
    args.out = args.out/'live'
    title_tag = 'STAGE 2.6' if 'stage26' in str(args.weights).lower() else 'PHASE 2.5'
    view = Dashboard(args.out.parent, args.replay_minutes, spec=selected, title_tag=title_tag, exit_on_finish=args.exit_on_finish)
    try:
        phase2_live.main(args, demo=view)
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('carla_root', 'out', 'weights', 'plan', 'latency'):
        p.add_argument('--'+field.replace('_', '-'), type=Path, required=True)
    p.add_argument('--port', type=int, default=2460)
    p.add_argument('--replay-minutes', type=float, default=0,
                   help='Automatically repeat the live drive for this viewing period.')
    p.add_argument('--sd-atlas',type=Path)
    p.add_argument('--route', type=str, default=None)
    p.add_argument('--npc', type=int, default=None)
    p.add_argument('--max-ticks', type=int, default=None)
    p.add_argument('--cruise-kmh', type=float, default=None)
    p.add_argument('--npc-along-route', action='store_true')
    p.add_argument('--exit-on-finish', action='store_true')
    args = p.parse_args()
    args.survey = False
    args.carla_root = args.carla_root.resolve()
    main(args)
