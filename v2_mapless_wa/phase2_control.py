"""Physics-only steering, longitudinal IDM, and the live LiDAR safety envelope."""
import math

import numpy as np

from v1_dual_system.curve_controller import curvature_preview
from .geometry import X,Y


def _gaussian_smooth(points, sigma=6.0):
    radius = int(math.ceil(4.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(np.asarray(points, dtype=np.float64), ((radius, radius), (0, 0)), mode='edge')
    return np.column_stack([np.convolve(padded[:, c], kernel, mode='valid') for c in range(padded.shape[1])])


def _sd_curvature_preview(points, sigma=6.0, half_span=10.0):
    pts = _gaussian_smooth(points, sigma=sigma)
    chord = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.r_[True, chord > 1e-3]]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    centres = np.flatnonzero((arc >= half_span) & (arc <= arc[-1] - half_span))
    distance = arc[centres]
    if not len(distance):
        return distance, np.empty(0)
    left = np.searchsorted(arc, distance - half_span, side='right') - 1
    right = np.searchsorted(arc, distance + half_span).clip(0, len(pts) - 1)
    before, after = pts[centres] - pts[left], pts[right] - pts[centres]
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    lengths = (np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
               * np.linalg.norm(before + after, axis=1))
    return distance, 2.0 * cross / np.maximum(lengths, 1e-6)


def project_corridor(points, centerline):
    """Return arc stations and transverse distances to a finite, possibly turning sweep."""
    segments = np.diff(centerline,axis=0)
    lengths = np.linalg.norm(segments,axis=1)
    if np.any(lengths<1e-4):
        raise ValueError('Degenerate planning path in the first 12 m')
    relative = points[:,None,:]-centerline[None,:-1,:]
    along = (relative*segments).sum(-1)/lengths**2
    clipped = np.clip(along,0,1)
    residual = relative-clipped[...,None]*segments
    d2 = (residual**2).sum(-1)
    which = d2.argmin(1)
    rows = np.arange(len(points))
    fraction = along[rows,which]
    starts = np.r_[0.,lengths.cumsum()]
    station = starts[which]+clipped[rows,which]*lengths[which]
    bounded = ~(((which==0)&(fraction<0)) | ((which==len(segments)-1)&(fraction>1)))
    return station,np.sqrt(d2[rows,which]),bounded


def local_ground(station,dist,height,bounded,half_width,max_s=38.0):
    """Track the low road surface in arc-length bands, retaining an ego-height anchor."""
    support_s, support_z = [0.], [0.]
    low = bounded & (dist<half_width+.75) & (abs(height)<.65+.22*station)
    bands = list(np.arange(1.5,12.1,1.5)) + list(np.arange(14.5,max_s+0.1,2.5))
    for s in bands:
        half_w = 0.9 if s <= 12.1 else 1.4
        min_pts = 5 if s <= 15.0 else 3
        sample = low & (abs(station-s)<half_w)
        if sample.sum()<min_pts:
            continue
        z = float(np.quantile(height[sample],.20))
        if abs(z-support_z[-1])<=.16+.22*(s-support_s[-1]):
            support_s.append(float(s));support_z.append(z)
    return np.interp(station,support_s,support_z),len(support_s)-1


class SpatialController:
    def __init__(self, cruise_kmh, wheelbase, half_width, max_steer, front, rear, roof):
        self.cruise = cruise_kmh/3.6
        self.wheelbase = wheelbase
        self.half_width = half_width
        self.max_steer = max_steer
        self.front = front
        self.rear = rear
        self.roof = roof
        self.last_steer = 0.
        self.last_gap = None
        self.closing = 0.
        self._lane_state = None
        self._last_speed = 0.0
        self._smooth_kappa = None
        self._smooth_limit = None

    def lane_tracking(self, path, lane, changing, speed=None):
        """Join the sensed target with continuous position and heading at the ego."""
        self._lane_state = None
        if lane is None or len(lane)<4:
            self._smooth_kappa = None
            return path, self.cruise, None
        near = lane[(lane[:,0]>=-5.) & (lane[:,0]<=24.)]
        if len(near)<4 or np.any(np.diff(near[:,0])<=0) or np.ptp(near[:,0])<6.:
            near = lane[(lane[:,0]>=-5.) & (lane[:,0]<=18.)]
        if len(near)<4 or np.any(np.diff(near[:,0])<=0) or np.ptp(near[:,0])<6.:
            self._smooth_kappa = None
            return path, self.cruise, None
        w = 1.0/(1.0+np.maximum(near[:,0],0.)/8.0)
        fit = np.polynomial.Polynomial.fit(near[:,0],near[:,1],2,w=w).convert()
        offset, slope = float(fit(0)), float(fit.deriv()(0))
        raw_signed_k = float(fit.deriv(2)(0))/(1.0+slope*slope)**1.5
        raw_curvature = abs(raw_signed_k)
        self._smooth_kappa = raw_curvature if self._smooth_kappa is None else 0.75*self._smooth_kappa + 0.25*raw_curvature
        curvature = self._smooth_kappa
        signed_kappa = math.copysign(curvature, raw_signed_k)
        self._lane_state = (offset, slope, float(curvature), bool(changing))
        v_ref = max(6.0, float(self._last_speed if speed is None else speed))
        if changing:
            outward_slope = slope if offset>=0.0 else -slope
            div_s = float(np.clip((outward_slope-0.03)/0.05, 0.0, 1.0))
            conv_slope = (abs(slope)/0.10) if (outward_slope<=0.0) else 0.0
            b_off = float(np.clip(max((abs(offset)-0.8)/1.4, conv_slope), 0.0, 1.0))
            t_lc = (1.0-div_s)*(2.4+1.2*b_off)+div_s*2.2
            length = float(np.clip(t_lc*v_ref, 22.0, 72.0))
            eff_off, eff_slope = offset, slope
            a_lat_join = 3.2-0.4*div_s
        else:
            length = float(np.clip(1.8*v_ref, 18.0, 36.0))
            eff_off = math.copysign(max(0.0, abs(offset)-0.35), offset)
            eff_slope = math.copysign(max(0.0, abs(slope)-0.03), slope)
            a_lat_join = 2.8
        k_join_0 = (6.0*eff_off+4.0*length*eff_slope)/(length**2)
        k_join_1 = -(6.0*eff_off+2.0*length*eff_slope)/(length**2)
        demand = max(abs(signed_kappa+k_join_0), abs(signed_kappa+k_join_1))
        a_lat = 2.8
        limit = min(self.cruise, math.sqrt(a_lat/max(.001,curvature)), math.sqrt(a_lat_join/max(.001,demand)))
        x = path[:,0]
        if np.any(np.diff(x[:8])<=0):
            return path,limit,offset
        t = np.clip(x/length,0.,1.)
        x_end = float(near[-1,0])
        base = fit(np.clip(x,near[0,0],x_end))
        base += np.clip(fit.deriv()(x_end),-.6,.6)*np.maximum(x-x_end,0.)
        reference = base-(1.-t)**2*((1.+2*t)*offset+t*length*slope)
        delta = reference-path[:,1]
        corr = delta if changing else delta*(delta*delta/(delta*delta+0.02*0.02))
        u_near = np.clip(x/4.5,0.,1.)
        w_near = u_near*u_near*(3.-2.*u_near)
        adjusted = path.copy()
        near_mask = x<=x_end
        adjusted[near_mask,1] += w_near[near_mask]*corr[near_mask]
        if np.any(~near_mask):
            end_idx = max(0, min(int(np.searchsorted(x, x_end, side='right')) - 1, len(x) - 1))
            delta_end = float(w_near[end_idx] * corr[end_idx])
            slope_end = float(np.clip((w_near[end_idx] * corr[end_idx] - w_near[end_idx - 1] * corr[end_idx - 1]) / max(0.5, float(x[end_idx] - x[end_idx - 1])), -0.35, 0.35)) if end_idx >= 1 else 0.0
            far_dx = x[~near_mask] - x_end
            if changing:
                l_relax = 22.0
                adjusted[~near_mask, 1] += delta_end + slope_end * l_relax * (1.0 - np.exp(-far_dx / l_relax))
            else:
                l_relax = 30.0
                adjusted[~near_mask, 1] += (delta_end + (slope_end + delta_end / l_relax) * far_dx) * np.exp(-far_dx / l_relax)
        return adjusted, limit, offset

    def step(self,path,occ,returns,speed,dt=.05,speed_limit=None,lane_tracking=False,route_preview=None):
        if path.shape!=(40,2) or not np.isfinite(path).all() or not np.isfinite(occ).all():
            return 0.,1.,0.,{'safety':'invalid_model'}
        self._last_speed = float(speed)
        centerline = np.vstack(([0.,0.],path))
        distance, curvature = curvature_preview(centerline)
        limit = self.cruise if speed_limit is None else min(self.cruise,speed_limit)
        if len(curvature):
            step_a_lat = 3.2 if (lane_tracking and self._lane_state is not None and self._lane_state[3]) else 2.8
            cap_sq = step_a_lat/np.maximum(abs(curvature),.001)
            preview = np.maximum(distance-2.0,0.)
            limit = min(limit,math.sqrt(float(np.min(cap_sq+2*3.*preview))))
        if route_preview is not None and len(route_preview)>=8:
            sd_dist, sd_kappa = _sd_curvature_preview(route_preview, sigma=6.0, half_span=10.0)
            if len(sd_kappa):
                sd_k = np.where(np.abs(sd_kappa)<.034,0.,np.abs(sd_kappa))
                if np.any(sd_k>0):
                    cap_sq = 1.35/np.maximum(sd_k,1e-4)
                    preview = np.maximum(sd_dist-12.0,0.)
                    sd_limit = math.sqrt(float(np.min(cap_sq+2.0*1.8*preview)))
                    limit = min(limit,sd_limit)
        if limit <= 0.1:
            self._smooth_limit = 0.0
        elif self._smooth_limit is None:
            self._smooth_limit = float(limit)
        else:
            blend = 0.35 if limit < self._smooth_limit else 0.24
            self._smooth_limit = (1.0 - blend) * self._smooth_limit + blend * float(limit)
        limit = min(self.cruise, self._smooth_limit)
        near_kappa = float(np.max(np.abs(curvature[distance<=15.]))) if len(curvature) and np.any(distance<=15.) else 0.
        if lane_tracking and self._lane_state is not None:
            near_kappa = max(near_kappa, self._lane_state[2])
        base_ahead = np.clip(2.8+.43*speed,4.5,11.5) if lane_tracking else np.clip(3.3+.58*speed,4.,18.)
        ahead = float(np.clip(base_ahead/(1.0+14.0*near_kappa),3.2,18.0))
        stations = np.r_[0.,np.linalg.norm(np.diff(centerline,axis=0),axis=1).cumsum()]
        x,y = [np.interp(ahead,stations,np.r_[0.,coord]) for coord in path.T]
        desired_steer = -math.atan2(2*self.wheelbase*y,max(.5,x*x+y*y))/self.max_steer
        if lane_tracking and self._lane_state is not None:
            l_off, l_slope, _, l_chg = self._lane_state
            if not l_chg and abs(l_off)<1.35:
                trim = -(0.20*math.atan(l_slope)+math.atan2(0.40*l_off,max(4.5,speed)))/self.max_steer
                desired_steer += float(np.clip(trim,-.15,.15))
        alpha = .45 if near_kappa>.018 else .35
        steer = float(np.clip(self.last_steer+alpha*(desired_steer-self.last_steer),-1,1))

        # Occupancy keeps spatio-temporal alignment; LiDAR sweeps up to 38 m with clustered hits.
        stop = np.flatnonzero(np.diff(np.r_[0.,path[:,0]])<=0)
        fwd = path[:(int(stop[0]) if len(stop) else len(path))]
        if len(fwd)<2:
            fwd = centerline[:2]
        points = returns[:,:3]
        r_max = 38.+self.half_width
        px0,py0,pz0 = points[:,0],points[:,1],points[:,2]
        points = points[(px0>=-4.0)&(px0<=r_max)&(np.abs(py0)<=r_max)&(pz0>-3.5)&(pz0<2.30)]
        points = points[(points[:,:2]**2).sum(1)<r_max**2]
        px,py,pz = points.T
        ego_body = (px>=self.rear)&(px<=self.front)&(abs(py)<=self.half_width)&(pz<=self.roof)
        take = ~ego_body
        points = points[take]; py,pz = py[take],pz[take]
        step_d = np.r_[True, np.linalg.norm(np.diff(centerline[:26], axis=0), axis=1) > 0.05]
        sweep_cl = centerline[:26][step_d]
        if len(sweep_cl) < 2:
            sweep_cl = centerline[:2]
        if len(points):
            pad = self.half_width + 1.0
            lo_xy = sweep_cl.min(axis=0) - pad
            hi_xy = sweep_cl.max(axis=0) + pad
            in_box = (points[:,0]>=lo_xy[0])&(points[:,0]<=hi_xy[0])&(points[:,1]>=lo_xy[1])&(points[:,1]<=hi_xy[1])
            points = points[in_box]; py,pz = py[in_box],pz[in_box]
        if len(points):
            try:
                station, transverse, bounded = project_corridor(points[:,:2],sweep_cl.astype(np.float32))
            except ValueError:
                return 0.,1.,0.,{'safety':'invalid_swept_path'}
            ground,support_bins = local_ground(station,transverse,pz,bounded,self.half_width,max_s=38.0)
            corridor = bounded & (transverse<=self.half_width+.18) & (pz>ground+.30) & (pz<2.20)
            hits = station[corridor]
            close_hits = hits[hits<=12.0]
            far_hits = hits[hits>12.0]
            valid_s = []
            if len(close_hits):
                if np.any(close_hits<=4.0) or len(close_hits)>=2:
                    valid_s.append(float(close_hits.min()))
            if len(far_hits)>=4:
                bins = np.floor(far_hits/1.5).astype(int)
                uniq, counts = np.unique(bins, return_counts=True)
                good = uniq[counts>=4]
                if len(good):
                    valid_s.append(float(far_hits[bins==good.min()].min()))
            gap = max(.1,min(valid_s)-self.front) if valid_s else math.inf
        else:
            station = np.empty(0, dtype=np.float32)
            corridor = np.empty(0, dtype=bool)
            support_bins = 0
            gap = math.inf
        if math.isfinite(gap) and self.last_gap is not None:
            rate = float(np.clip((self.last_gap-gap)/dt,-12.,speed+5.))
            self.closing += .18*(rate-self.closing)
        else:
            self.closing = 0.
        self.last_gap = gap if math.isfinite(gap) else None
        headway = 2.5+max(0.,0.95*speed+speed*self.closing/(2*math.sqrt(1.6*2.5)))
        if speed <= limit:
            free_accel = 1.6*(1.-(speed/max(.5,limit))**2)
        else:
            free_accel = max(-2.4, -0.75*(speed-limit))
        idm = free_accel-1.6*(headway/gap)**2

        x_rows = X[:,0]
        predicted_y = np.interp(x_rows,fwd[:,0],fwd[:,1],left=0.)[:,None]
        occ_thresh = math.log(.70/.30)
        occ_risk = False
        for t,grid in zip((.5,1.,2.),occ[:3]):
            ego_x_t = speed*t
            lo = max(self.front+.5, ego_x_t-3.5)
            hi = min(38., ego_x_t+self.front+4.5)
            row_mask = (x_rows>lo)&(x_rows<hi)
            if row_mask.any():
                occ_risk |= np.count_nonzero((grid[row_mask]>occ_thresh)&(np.abs(Y[row_mask]-predicted_y[row_mask])<self.half_width+.12))>=4
        if occ_risk:
            idm = min(idm,-.6)

        side_band = corridor & (station<self.front+3.5)&(abs(py)>.25)
        clearance = np.abs(py)-self.half_width
        left = side_band & (py>0)
        right = side_band & (py<0)
        left_gap = float(clearance[left].min()) if left.any() else math.inf
        right_gap = float(clearance[right].min()) if right.any() else math.inf
        if left_gap < .85:
            steer = max(0.,steer)
        if right_gap < .85:
            steer = min(0.,steer)
        intrusion = gap<6.
        emergency = min(left_gap,right_gap)<.45 or intrusion
        brake = float(np.clip(-idm/5.,0.,1.))
        if emergency:
            brake = max(brake,.72 if gap<2.0 or min(left_gap,right_gap)<.22 else .42)
        if limit<=.1:
            brake = max(brake,.4)
        throttle = 0. if brake>.015 else float(np.clip(idm/2.35,0.,.67))
        self.last_steer = steer
        return throttle,brake,steer,dict(curve_kmh=limit*3.6, lidar_gap_m=gap,
              left_clearance_m=left_gap,right_clearance_m=right_gap,occupancy_risk=bool(occ_risk),
              safety_brake=bool(emergency),safety_steer=bool(min(left_gap,right_gap)<.85),
              swept_path_hits=int(corridor.sum()),ground_support_bins=support_bins)
