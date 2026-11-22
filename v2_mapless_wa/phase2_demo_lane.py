"""Lane confirmation for the interactive demo, using only predictions, LiDAR and ego pose."""
import math
from collections import deque

import numpy as np

from .geometry import local, pixels, sample_arc, world_at


X = np.arange(0., 61.5, 1.5)
CHECK = np.array([6., 12., 18., 24.])
NEAR = (X >= 4.5) & (X <= 18.)


class AdjacentLaneRequest:
    """Release a relative button command when the ego-lane queries switch at the divider."""
    def __init__(self):
        self.begin(0)

    def begin(self, direction):
        self.direction = direction
        self.approach = 0.
        self.switched = 0

    def observe(self, output):
        if not self.direction:
            return False
        if not (output['exists'][0,:3]>0).all().item():
            self.switched = 0
            return False
        near = output['lines'][0,:3,0,1].float().cpu().numpy()
        center, left, right = near
        width = left-right
        if not 2.5<width<5. or not right<center<left:
            self.switched = 0
            return False
        offset = (center if self.direction==2 else -center)/width
        self.approach = max(self.approach, offset)
        changed = self.approach>.19 and offset<-.06 and self.approach-offset>.38
        self.switched = self.switched+1 if changed else 0
        if self.switched<2:
            return False
        self.begin(0)
        return True


def route_normals(coarse):
    tangent = np.gradient(coarse, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    ok = np.flatnonzero(norm > .05)
    if not len(ok):
        return None, None
    tangent = np.stack([np.interp(np.arange(len(coarse)), ok, c[ok] / norm[ok]) for c in tangent.T], axis=1)
    normal = tangent[:, ::-1] * [-1., 1.]
    return tangent, normal


def measure_route_lane_offset(coarse, lane_xy):
    tangent, normal = route_normals(coarse)
    if tangent is None or lane_xy is None or len(lane_xy) < 4:
        return None
    offsets = []
    for p, t, n in zip(coarse[:14], tangent[:14], normal[:14]):
        along = (lane_xy - p) @ t
        if np.all(np.diff(along) > 0) and along[0] <= 0. <= along[-1]:
            offsets.append(float(np.interp(0., along, (lane_xy - p) @ n)))
    if len(offsets) < 3:
        return None
    return float(np.clip(np.median(offsets), -9.0, 9.0))


def shift_route(coarse, offset):
    _, normal = route_normals(coarse)
    if normal is None or abs(offset) < 1e-4:
        return coarse
    return (coarse + normal * float(offset)).astype(np.float32)


class TargetLaneMemory:
    """Track the confirmed lane in GNSS coordinates and condition the next planning query."""
    stations = np.arange(0., 25., 1.5)

    def __init__(self):
        self.world = None
        self.command = 0
        self.missing = 0
        self.status = ''
        self.last_request = 0
        self.acquired = False
        self.confirming = False
        self.match_error = None
        self.recent_offsets = deque(maxlen=10)
        self.relative_lane = 0
        self.rejected = False

    def local_target(self, navigator):
        if self.world is None:
            return None
        points = (self.world-navigator.gnss_xy)@navigator.axes
        points = points[points[:,0]>-5.]
        if len(points)<3 or np.any(np.diff(points[:,0])<=0.) or points[0,0]>5. or points[-1,0]<5.:
            return None
        return points

    def guidance(self, coarse, navigator):
        """Join the selected physical lane to the directional SD preview beyond sensor range."""
        points = self.local_target(navigator)
        if points is None:
            return coarse
        tangent = np.gradient(coarse,axis=0)
        norm = np.linalg.norm(tangent,axis=1)
        ok = np.flatnonzero(norm>.05)
        if not len(ok):
            return coarse
        tangent = np.stack([np.interp(np.arange(len(coarse)),ok,c[ok]/norm[ok])
                            for c in tangent.T],axis=1)
        normal = tangent[:,::-1]*[-1.,1.]
        # For each preview point find the corresponding point on the observed
        # lane along the local road normal. Extend only the transverse offset.
        offsets, ids = [], []
        for i,(p,t,n) in enumerate(zip(coarse,tangent,normal)):
            along = (points-p)@t
            if np.all(np.diff(along)>0) and along[0]<=0<=along[-1]:
                offsets.append(np.interp(0.,along,(points-p)@n))
                ids.append(i)
        if not ids:
            return coarse
        return (coarse+normal*np.interp(np.arange(len(coarse)),ids,offsets)[:,None]).astype(np.float32)

    def extend_target(self, lane, navigator):
        samples = np.column_stack((self.stations,lane))
        if self.world is None:
            self.world = samples@navigator.axes.T+navigator.gnss_xy
            return
        local = (self.world-navigator.gnss_xy)@navigator.axes
        keep = local[:,0]>=-5.
        self.world = self.world[keep]
        local = local[keep]
        if len(local)<2 or np.any(np.diff(local[:,0])<=0):
            self.world = None
            return
        # Never rewrite a point that has already been published. Extend in
        # metre-sized steps so the tip does not integrate subpixel noise.
        commit_end = 19.5
        tail = np.arange(local[-1, 0] + 1.5, commit_end + .001, 1.5)
        if len(tail):
            mismatch = local[-1, 1] - np.interp(local[-1, 0], self.stations, lane)
            y = np.interp(tail, self.stations, lane) + mismatch * np.exp(-(tail - local[-1, 0]) / 4.)
            self.world = np.vstack((self.world, np.column_stack((tail, y)) @ navigator.axes.T + navigator.gnss_xy))

    def update(self, output, navigator, path, requested, confirmed, applied, scan, speed, new_request=False):
        new = requested and (new_request or requested!=self.last_request)
        self.last_request = requested
        self.command = 0
        self.match_error = None
        if new:
            self.world = None
            self.acquired = False
            self.confirming = False
            self.missing = 0
            self.status = 'Adjacent lane unconfirmed'
            self.recent_offsets.clear()
            self.relative_lane = 0
            self.rejected = False
        if not new and self.world is None:
            return
        if confirmed and self.world is not None:
            self.confirming = True
        lines = output['lines'][0].float().cpu().numpy()
        exists = output['exists'][0].float().cpu().numpy()
        if self.acquired and np.all(exists[:3]>0):
            center, left, right = lines[:3,0,1]
            width = left-right
            if 2.5<width<5. and right<center<left:
                offset = center/width
                if self.recent_offsets:
                    # A rightward crossing relabels the ego center from left to
                    # right of the car; the original physical lane is now left.
                    right_cross = (max(self.recent_offsets)>.18 and offset<-.12
                                   and max(self.recent_offsets)-offset>.38)
                    left_cross = (min(self.recent_offsets)<-.18 and offset>.12
                                  and offset-min(self.recent_offsets)>.38)
                    if right_cross or left_cross:
                        self.relative_lane += 1 if right_cross else -1
                        self.recent_offsets.clear()
                self.recent_offsets.append(offset)
            else:
                self.recent_offsets.clear()
        hint = None
        if self.world is not None:
            previous = (self.world-navigator.gnss_xy)@navigator.axes
            if np.all(np.diff(previous[:,0])>0):
                hint = np.interp(self.stations, previous[:,0], previous[:,1], left=np.nan, right=np.nan)
        best, source, cost = None, None, float('inf')
        if new:
            indices = (4+requested,)
        elif self.confirming:
            indices = (0,)
        elif self.acquired:
            indices = {0:(0,), 1:(5,), -1:(6,)}.get(self.relative_lane,())
        else:
            indices = (0,5,6)
        for idx in indices:
            if exists[idx]<=0:
                continue
            lane = lane_curve(lines[idx])
            if lane is None:
                continue
            lane = lane[:len(self.stations)]
            if new:
                own = lane_curve(lines[0]) if exists[0]>0 else None
                if own is not None:
                    separation = (lane[2:9]-own[2:9])*(1 if requested==1 else -1)
                    if np.all((separation>2.25)&(separation<5.25)):
                        projected = np.interp(X,self.stations,lane)
                        drive = output['mask'][0,0].float().cpu().numpy()
                        if in_drivable_area(projected,drive) and clear_to_merge(projected,scan,speed):
                            best, source = lane, idx
            elif hint is not None:
                overlap = np.isfinite(hint) & (self.stations<=(9. if self.confirming else 16.))
                if overlap.sum()<5:
                    continue
                gap = np.abs(lane[overlap]-hint[overlap]).mean()
                if self.match_error is None or gap<self.match_error:
                    self.match_error = float(gap)
                if gap<min(cost, 1.25):
                    best, source, cost = lane, idx, gap
        if best is None:
            self.command = 0
            self.missing += 1
            self.status = 'Target lane unconfirmed'
            if self.local_target(navigator) is None or new:
                self.world = None
                self.acquired = False
                self.confirming = False
                self.status = 'Target lost / Keep lane'
                if new:
                    self.rejected = True
                    self.status = 'No clear adjacent lane'
            return
        self.extend_target(best, navigator)
        if (self.acquired and source == 0 and self.relative_lane == 0
                and not requested and not new and not self.confirming
                and cost < 0.45 and abs(best[0]) < 0.65):
            loc = (self.world - navigator.gnss_xy) @ navigator.axes
            mask = (loc[:, 0] >= -2.0) & (loc[:, 0] <= 14.0)
            if mask.any():
                obs = _near_observation(lines, exists, best, self.stations)
                err = np.interp(loc[mask, 0], self.stations, obs) - loc[mask, 1]
                if np.all(np.abs(err) < 0.55):
                    alpha = 0.14 * np.clip(1.0 - np.maximum(loc[mask, 0], 0.0) / 16.0, 0.25, 1.0)
                    loc[mask, 1] += np.clip(alpha * err, -0.04, 0.04)
                    self.world = loc @ navigator.axes.T + navigator.gnss_xy
        target = self.local_target(navigator)
        if target is None:
            self.status = 'Target lost / Keep lane'
            self.world = None
            self.acquired = False
            return
        best = np.interp(self.stations,target[:,0],target[:,1])
        self.missing = 0
        centered = source==0 and abs(best[0])<.6 and abs(best[3]-best[0])<.55
        self.acquired |= self.confirming or (not requested and centered)
        self.confirming = False
        if not self.acquired:
            self.status = 'Selected adjacent lane'
            return
        self.status = 'Tracking selected lane'
        near = (self.stations>=7.5)&(self.stations<=13.5)
        if np.any(np.diff(path[:12,0])<=0) or path[11,0]<self.stations[near][-1]:
            return
        planned = np.interp(self.stations[near],path[:12,0],path[:12,1])
        error = float(np.mean(best[near]-planned))
        wanted = 1 if error>0 else 2
        # Usually alternate with a neutral query. If the vehicle is approaching a
        # boundary and the corrective path is still short, keep correcting.
        if abs(error)<=.40 or (applied and (applied!=wanted or abs(best[0])<.65)):
            return
        if source!=0:
            # Re-entering the selected lane is a merge; require observed free space.
            extended = np.interp(X,self.stations,best)
            drive = output['mask'][0,0].float().cpu().numpy()
            if not in_drivable_area(extended,drive) or not clear_to_merge(extended,scan,speed):
                self.status = 'Selected lane blocked'
                return
        self.command = wanted


def _near_observation(lines, exists, own_curve, stations):
    obs = own_curve
    if exists[1] > 0 and exists[2] > 0:
        left_c = lane_curve(lines[1])
        right_c = lane_curve(lines[2])
        if left_c is not None and right_c is not None:
            left_c = left_c[:len(stations)]
            right_c = right_c[:len(stations)]
            w = left_c[:11] - right_c[:11]
            if np.all((w > 2.6) & (w < 4.8)) and np.all((right_c[:11] < own_curve[:11]) & (own_curve[:11] < left_c[:11])):
                mid = 0.5 * (left_c + right_c)
                obs = own_curve.copy()
                obs[:11] = 0.5 * own_curve[:11] + 0.5 * mid[:11]
    return obs


def lane_curve(points):
    finite = np.isfinite(points).all(1)
    x, y = points[finite].T
    use = (x >= -3.) & (x <= 32.)
    x, y = x[use], y[use]
    if len(x) < 6 or x.min() > 10. or x.max() < 18.:
        return None
    w = 1.0 / (1.0 + np.maximum(x, 0.0) / 10.0)
    fit = np.polynomial.Polynomial.fit(x, y, 2, w=w).convert()
    x_used, y_used = x, y
    if np.median(np.abs(fit(x) - y)) > 0.55:
        near = (x >= -3.) & (x <= 20.)
        if near.sum() >= 5 and x[near].max() >= 12.0:
            x_near, y_near = x[near], y[near]
            w_near = 1.0 / (1.0 + np.maximum(x_near, 0.0) / 10.0)
            fit_near = np.polynomial.Polynomial.fit(x_near, y_near, 2, w=w_near).convert()
            if np.median(np.abs(fit_near(x_near) - y_near)) < np.median(np.abs(fit(x) - y)):
                fit, x_used, y_used = fit_near, x_near, y_near
    if np.median(np.abs(fit(x_used) - y_used)) > .85 or abs(fit(0.)) > 9.:
        return None
    end = min(float(x_used.max()), 30.)
    return fit(X.clip(0, end)) + np.clip(fit.deriv()(end), -.5, .5)*np.maximum(X-end, 0.)


def lane_at_pose(saved, pose, min_ahead=24.):
    if saved is None:
        return None
    points = local(saved,pose)
    order = np.argsort(points[:, 0])
    x,y = points[order,:2].T
    if x[0] > 4. or x[-1] < min_ahead:
        return None
    end = x[-1]
    slope = np.clip((y[-1]-np.interp(end-6.,x,y))/6.,-.5,.5)
    return np.interp(X,x,y)+slope*np.maximum(X-end,0.)


def mean_difference(a,b):
    return float(np.abs(a[NEAR]-b[NEAR]).mean())


def in_drivable_area(lane, drivable):
    xy = np.column_stack((CHECK,np.interp(CHECK,X,lane)))
    cells = np.rint(pixels(xy)).astype(int)
    inside = ((cells>=0)&(cells<128)).all(1)
    return bool(inside.all() and np.count_nonzero(drivable[cells[:,1],cells[:,0]]>0)>=3)


def clear_to_merge(lane, scan, speed):
    x,y,z = scan[:,:3].T
    x_max = min(26., max(15., 2.*speed))
    distance = abs(y-np.interp(x,X,lane))
    for near in np.arange(-7.,x_max,3.):
        tile = (x>=near)&(x<near+3.)&(distance<1.25)&(z<2.2)
        if tile.sum()<5:
            continue
        ground = min(float(np.quantile(z[tile],.15)), .18+.12*max(0.,near))
        if np.count_nonzero(tile&(z>ground+.38))>=8:
            return False
    return True


class LaneAssist:
    def __init__(self):
        self.current = None
        self.origin = None
        self.goal = None
        self.travel = 0.
        self.last_position = None
        self.side = 0
        self.confirmed = 0
        self.missing = 0

    @staticmethod
    def remember(lane,pose):
        return world_at(np.column_stack((X,lane,np.zeros(len(X)))),pose)

    @staticmethod
    def match(saved, detected, pose):
        hint = lane_at_pose(saved,pose)
        if hint is None:
            return None
        if detected:
            measurement = min(detected,key=lambda p:mean_difference(p,hint))
            if mean_difference(measurement,hint)<1.25:
                return .78*hint+.22*measurement
        return hint

    def step(self, pose, output, raw, speed, request):
        curves = output['lines'][0].float().cpu().numpy()
        exists = output['exists'][0].float().cpu().numpy()
        own = lane_curve(curves[0]) if exists[0]>0 else None
        if self.goal is not None:
            return self.continue_change(pose,own)
        drivable = output['mask'][0,0].float().cpu().numpy()
        detected = [own] if own is not None else []
        detected.extend(lane for i in (5,6) if exists[i]>0 and (lane:=lane_curve(curves[i])) is not None)
        current = self.match(self.current,detected,pose)
        if self.current is None:
            current = own if own is not None and abs(own[0])<=1.6 else None
        current_confirmed = (current is not None and detected and
                             min(mean_difference(lane,current) for lane in detected)<1.25)
        self.missing = 0 if current_confirmed else self.missing+1
        flat = np.column_stack((X[1:],np.zeros(40)))
        if current is None or self.missing>12:
            return flat,'Lane unconfirmed',0.,False,self.side
        if current_confirmed:
            self.current = self.remember(current,pose)
        if not request or not current_confirmed:
            return self.path(current),'Tracking lane',math.inf,False,0
        if speed>50/3.6:
            return self.path(current),'Slow down to change',math.inf,False,0
        side = 1 if request==1 else -1
        choices = [lane for lane in detected if ((side*(lane-current)[4:17]>=2.35)&
                                                  (side*(lane-current)[4:17]<=5.25)).all()]
        if not choices:
            return self.path(current),'No adjacent lane',math.inf,False,0
        adjacent = min(choices,key=lambda lane:mean_difference(lane,current))
        separation = float(np.median((adjacent-current)[4:17]))
        if abs(separation)>4.4 or np.ptp((adjacent-current)[4:17])>1.2:
            return self.path(current),'No adjacent lane',math.inf,False,0
        goal = current+separation
        if not in_drivable_area(goal,drivable) or not clear_to_merge(goal,raw,speed):
            return self.path(current),'Target lane blocked',math.inf,False,0
        self.origin = self.remember(current,pose)
        self.goal = self.remember(goal,pose)
        self.travel = 0.
        self.last_position = pose[:2,3].copy()
        self.start_position = self.last_position.copy()
        self.change_axis = -side*pose[:2,1].copy()
        self.lateral_limit = min(4.05,abs(separation)+.45)
        self.span = max(20.,3.2*speed)
        self.confirmed = 0
        self.side = request
        return self.continue_change(pose,own)

    def continue_change(self, pose, own):
        # Preserve one physical target across the model's ego-lane relabeling at the lane line.
        origin = lane_at_pose(self.origin,pose,min_ahead=4.)
        target = lane_at_pose(self.goal,pose,min_ahead=4.)
        if origin is None or target is None:
            flat = np.column_stack((X[1:],np.zeros(40)))
            return flat,'Change unconfirmed',0.,False,self.side
        self.travel += float(np.linalg.norm(pose[:2,3]-self.last_position))
        self.last_position = pose[:2,3].copy()
        lateral = float((self.last_position-self.start_position)@self.change_axis)
        ratio = np.clip((self.travel+X)/self.span,0.,1.)
        weight = ratio**2*(3.-2.*ratio)
        path = self.path(origin+(target-origin)*weight)
        observed_target = (lateral>=2.40 and own is not None and abs(own[0])<1.15 and
                           abs(own[3]-own[0])<.8)
        if observed_target:
            path = self.path(own)
        if observed_target and lateral>=2.65 and abs(own[0])<.75:
            self.confirmed += 1
        else:
            self.confirmed = 0
        if self.confirmed>=6:
            self.current = self.remember(own,pose)
            self.origin = self.goal = None
            self.side = 0
            return path,'Lane change complete',math.inf,True,0
        if lateral>self.lateral_limit or self.travel>self.span+20.:
            return path,'Change unconfirmed',0.,False,self.side
        return path,'Changing lane',45.,False,self.side

    @staticmethod
    def path(center):
        xy = np.column_stack((X,center))
        xy[0] = 0.
        return sample_arc(xy,X[1:]).astype(np.float32)
