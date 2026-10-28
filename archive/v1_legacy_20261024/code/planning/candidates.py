"""Transparent costs from current BEV and motion-compensated observed actors."""
import cv2
import numpy as np
from configs.planner7 import Planner7Config
from mini_wa_jepa.geometry import box_corners, ego_to_pixels, ego_to_world, world_to_ego, wrap_angle


TERM_NAMES=('progress','speed','lateral_accel','curvature','static','dynamic','crash','offroad')


def actors(bev,state):
    count,_,stats,centroids=cv2.connectedComponentsWithStats((bev[2]>0).astype(np.uint8),8)
    points=[];radii=[]
    for i in range(1,count):
        if stats[i,cv2.CC_STAT_AREA]<2:continue
        u,v=centroids[i];xy=np.array([(63.5-v)*.5,(63.5-u)*.5])
        points.append(ego_to_world(xy,state[:2],state[2]))
        radii.append(max(.5,.25*np.hypot(stats[i,cv2.CC_STAT_WIDTH],stats[i,cv2.CC_STAT_HEIGHT])))
    return np.asarray(points).reshape(-1,2),np.asarray(radii)


def actor_tracks(bevs,states,dt=.1):
    current,radii=actors(bevs[-1],states[-1]);velocity=np.zeros_like(current)
    matched=np.zeros(len(current),bool)
    if len(bevs)>1 and len(current):
        past,_=actors(bevs[0],states[0]);elapsed=(len(bevs)-1)*dt
        if len(past):
            distance=np.linalg.norm(current[:,None]-past[None],axis=-1)
            near=distance.argmin(1);reverse=distance.argmin(0)
            for i,j in enumerate(near):
                if reverse[j]==i and distance[i,j]<=3+30*elapsed:
                    velocity[i]=(current[i]-past[j])/elapsed;matched[i]=True
    anchor=states[-1]
    local=world_to_ego(current,anchor[:2],anchor[2])
    local_v=world_to_ego(velocity,[0.,0.],anchor[2])
    return local,local_v,radii,matched


def footprint_mask(point,length,width):
    pixels=ego_to_pixels(box_corners(point[:2],point[3],length,width),128,64)
    mask=np.zeros((128,128),np.uint8)
    cv2.fillConvexPoly(mask,np.rint(pixels).astype(np.int32),1)
    return mask.astype(bool),pixels


def static_cost(bev,trajectory,length,width):
    ego,_=footprint_mask(np.zeros(4),length,width)
    known=(bev[3]>0)&~ego&~(bev[2]>0)
    values=[]
    for point in trajectory:
        mask,pixels=footprint_mask(point,length,width);count=mask.sum()
        if not count or np.any((pixels<0)|(pixels>127)):
            values.append(1.);continue
        available=mask&known;n=available.sum()
        # Ego footprint/occluded actor cells are unknown, not fabricated road.
        off=float(np.sum(available&(bev[0]==0))/n) if n else 0.
        boundary=float(np.sum(available&(bev[1]>0))/max(n,1))
        unknown=float(np.sum(mask&~(bev[3]>0))/count)
        values.append(off+2*boundary+.5*unknown)
    return float(np.mean(values))


def dynamic_cost(trajectory,tracks,length,width,config):
    position,velocity,radii,matched=tracks
    if not len(position):return 0.
    times=np.arange(1,11)*.2;future=position[None]+times[:,None,None]*velocity[None]
    delta=future-trajectory[:,None,:2];angle=trajectory[:,3,None]
    x=delta[...,0]*np.cos(angle)+delta[...,1]*np.sin(angle)
    y=-delta[...,0]*np.sin(angle)+delta[...,1]*np.cos(angle)
    distance=np.hypot(np.maximum(np.abs(x)-length/2,0),np.maximum(np.abs(y)-width/2,0))
    buffer=radii[None]+config.actor_margin_m+times[:,None]*config.actor_uncertainty_m_per_s+(~matched)[None]*.5
    penetration=np.maximum(buffer-distance,0.)/buffer
    return float(np.max(penetration))


def kinematic_reasons(trajectory,config=Planner7Config()):
    if trajectory.shape!=(10,4) or not np.isfinite(trajectory).all():return ['nonfinite_or_shape']
    reasons=[];v=trajectory[:,2]
    if np.any((v<0)|(v>config.max_speed_m_s)):reasons.append('speed')
    if np.any(np.abs(np.diff(v))/.2>config.max_accel_m_s2):reasons.append('acceleration')
    difference=np.diff(trajectory[:,:2],axis=0)
    moving=(np.linalg.norm(difference,axis=-1)>.25)&(v[1:]>1.)
    headings=np.arctan2(difference[:,1],difference[:,0])
    if np.any(np.abs(wrap_angle(trajectory[1:,3]-headings))[moving]>config.max_heading_error_rad):
        reasons.append('heading_position')
    return reasons


def emergency_trajectory(speed,config=Planner7Config()):
    t=np.arange(1,11)*.2;v=max(0.,float(speed));a=config.emergency_deceleration_m_s2
    moving=np.minimum(t,v/a);x=v*moving-.5*a*moving**2
    return np.column_stack([x,np.zeros(10),np.maximum(v-a*t,0),np.zeros(10)]).astype(np.float32)


def generate_lane_change_candidates(v_curr, dt=0.2, steps=10, T=2.0, lane_width=3.5):
    """Generate smooth cycloidal S-curve lane-change candidates (left/right, cruise/accelerate)."""
    t = np.arange(1, steps + 1) * dt
    tau = t / T
    s_curve = tau - np.sin(2 * np.pi * tau) / (2 * np.pi)
    v_c = max(0.5, float(v_curr))

    # 1. Left lane change (dy = +lane_width)
    x_c = v_c * t
    y_l = lane_width * s_curve
    dx_c = np.diff(np.concatenate([[0], x_c]))
    dy_l = np.diff(np.concatenate([[0], y_l]))
    h_l = np.arctan2(dy_l, dx_c)
    lc_l_cruise = np.column_stack([x_c, y_l, np.full(steps, v_c), h_l]).astype(np.float32)

    a_acc = 1.0
    v_a = np.clip(v_c + a_acc * t, 0.5, 15.0)
    x_a = v_c * t + 0.5 * a_acc * (t ** 2)
    dx_a = np.diff(np.concatenate([[0], x_a]))
    h_la = np.arctan2(dy_l, dx_a)
    lc_l_acc = np.column_stack([x_a, y_l, v_a, h_la]).astype(np.float32)

    # 2. Right lane change (dy = -lane_width)
    y_r = -lane_width * s_curve
    dy_r = np.diff(np.concatenate([[0], y_r]))
    h_r = np.arctan2(dy_r, dx_c)
    lc_r_cruise = np.column_stack([x_c, y_r, np.full(steps, v_c), h_r]).astype(np.float32)
    h_ra = np.arctan2(dy_r, dx_a)
    lc_r_acc = np.column_stack([x_a, y_r, v_a, h_ra]).astype(np.float32)

    return np.stack([lc_l_cruise, lc_l_acc, lc_r_cruise, lc_r_acc])


def score_candidates(candidates,bev,bevs,states,length,width,probability=None,thresholds=None,
                     config=Planner7Config()):
    tracks=actor_tracks(bevs,states);terms=[];reasons=[]
    pos,vel,_,_=tracks
    # Detect slow lead vehicle in ego corridor
    lead_blocked = False
    lead_speed = config.target_speed_m_s
    if len(pos):
        ego_corridor = (pos[:, 0] > 0) & (pos[:, 0] < 28.) & (np.abs(pos[:, 1]) < 1.8)
        if ego_corridor.any():
            idx_l = int(np.argmin(np.where(ego_corridor, pos[:, 0], np.inf)))
            v_l = max(0., float(vel[idx_l, 0]))
            if v_l < 0.75 * config.target_speed_m_s:
                lead_blocked = True
                lead_speed = v_l
    for i,trajectory in enumerate(candidates):
        reason=kinematic_reasons(trajectory,config)
        if not np.isfinite(trajectory).all():raise FloatingPointError('Nonfinite generated trajectory')
        yaw=np.abs(wrap_angle(np.diff(trajectory[:,3])))/.2
        mean_yaw=float(np.mean(yaw))
        is_lane_change=(np.abs(trajectory[-1,1])>=2.2)
        if is_lane_change:
            v_target=config.target_speed_m_s
        else:
            v_target=min(config.target_speed_m_s,3./max(mean_yaw,.428))
        if len(pos):
            corridor=(pos[:,0]>0)&(pos[:,0]<35.)&(np.abs(pos[:,1]-trajectory[-1,1])<2.0)
            if corridor.any():
                idx=int(np.argmin(np.where(corridor,pos[:,0],np.inf)))
                d_lead=float(pos[idx,0]); v_lead=max(0.,float(vel[idx,0]))
                d_safe=8.0
                d_desired=d_safe+1.2*v_lead
                v_lead_limit=0.0 if d_lead<=d_safe else max(0.,v_lead+(d_lead-d_desired)/1.5)
                v_target=min(v_target,v_lead_limit)
        speed_err=float(np.mean(((trajectory[:,2]-v_target)/3.)**2))
        lat_accel=float(np.mean(np.abs(trajectory[1:,2])*yaw))/4.
        stat_c=static_cost(bev,trajectory,length,width)
        dyn_c=dynamic_cost(trajectory,tracks,length,width,config)
        prog=-float(trajectory[-1,0])/20.
        if lead_blocked:
            is_same_lane=(np.abs(trajectory[-1,1])<1.8)
            if is_same_lane:
                speed_err+=1.2*max(0.,(config.target_speed_m_s-lead_speed)/config.target_speed_m_s)
            elif is_lane_change and stat_c<0.05 and dyn_c<0.1:
                prog-=0.4
        row=[prog,speed_err,lat_accel,mean_yaw/.35,
            stat_c,dyn_c,0.,0.]
        if probability is not None:
            row[-2:]=probability[i].tolist()
            for k,name in enumerate(('crash','offroad')):
                if thresholds[k] is not None and probability[i,k]>thresholds[k]:reason.append(name+'_threshold')
        terms.append(row);reasons.append(reason)
    weights=np.array([config.progress_weight,config.speed_weight,config.lateral_accel_weight,
        config.curvature_weight,config.static_weight,config.dynamic_weight,config.crash_weight,config.offroad_weight])
    terms=np.asarray(terms);scores=terms@weights
    eligible=np.array([not r for r in reasons])
    kin_valid=np.array([len(r)==0 or all('threshold' in item for item in r) for r in reasons])
    if eligible.any():
        index=int(np.argmin(np.where(eligible,scores,np.inf)))
    elif kin_valid.any():
        index=int(np.argmin(np.where(kin_valid,scores,np.inf)))
    else:
        best_idx=int(np.argmin(scores))
        index=best_idx if scores[best_idx]<8.0 else -1
    return dict(terms=terms,scores=scores,rejection_reasons=reasons,selected_index=index,
                track_count=len(tracks[0]),matched_track_count=int(tracks[3].sum()))
