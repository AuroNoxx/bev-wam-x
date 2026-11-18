"""Numpy-only native OCC and trajectory/controller bridge; no trajectory quality veto.

CARLA x-forward/y-right ego is converted to FLU once. Raster storage is Z,Y,X with
row zero at positive FLU Y, exactly as b2d_native_occ.align_native.
"""
import math
from functools import lru_cache
import numpy as np

SHAPE = (20, 128, 128)
DT = .2
SEMANTIC_GROUPS = [(1,[1,25,27]), (2,[9,10]), (4,[14,15,16,17,18,19,21]),
                   (5,[12,13]), (6,[24]), (7,[2,28])]


def compact_tags(tags):
    labels = np.full(np.shape(tags), 3, np.uint8)
    for value, ids in SEMANTIC_GROUPS:
        labels[np.isin(tags, ids)] = value
    return labels


def merge_semantic_scans(previous,current,previous_sensor2world,current_sensor2world):
    previous=np.asarray(previous);current=np.asarray(current)
    if previous.ndim!=2 or current.ndim!=2 or previous.shape[1]!=6 or current.shape[1]!=6:
        raise ValueError('Semantic scans must be Nx6 raw float32 words')
    transform=np.linalg.inv(current_sensor2world)@previous_sensor2world
    corrected=previous.copy()
    corrected[:,:3]=(np.column_stack([previous[:,:3],np.ones(len(previous))])@transform.T)[:,:3]
    return np.concatenate([corrected,current])


@lru_cache(maxsize=1)
def carla_grid_centers():
    z,y,x = np.indices(SHAPE)
    return np.stack([-25.6+(x+.5)*.4, 25.6-(y+.5)*.4,
                     -5+(z+.5)*.4, np.ones_like(x)], -1).reshape(-1,4)


def native_points_grid(points_carla_ego, tags):
    p = np.asarray(points_carla_ego)
    ix = np.floor((p[:,0]+25.6)/.4).astype(int)
    iy = 127-np.floor((p[:,1]+25.6)/.4).astype(int)
    iz = np.floor((p[:,2]+5)/.4).astype(int)
    valid = (ix>=0)&(ix<128)&(iy>=0)&(iy<128)&(iz>=0)&(iz<20)
    grid = np.zeros(SHAPE, np.uint8)  # unfilled, never certified free
    grid[iz[valid],iy[valid],ix[valid]] = compact_tags(np.asarray(tags)[valid])
    return grid


def align_history_grid(grid, source2world, current2world):
    p = carla_grid_centers() @ (np.linalg.inv(source2world)@current2world).T
    ix = np.floor((p[:,0]+25.6)/.4).astype(int)
    iy = 127-np.floor((p[:,1]+25.6)/.4).astype(int)
    iz = np.floor((p[:,2]+5)/.4).astype(int)
    ok = (ix>=0)&(ix<128)&(iy>=0)&(iy<128)&(iz>=0)&(iz<20)
    out = np.full(len(ix),255,np.uint8)
    out[ok] = grid[iz[ok],iy[ok],ix[ok]]
    return out.reshape(SHAPE)


def history_payload(frames):
    if len(frames)<7:
        raise ValueError('Need seven real 10Hz observations, no duplicated warm-up')
    chosen = [frames[-7],frames[-5],frames[-3],frames[-1]]
    ts = np.array([x['timestamp'] for x in chosen])
    if not np.allclose(np.diff(ts), .2, atol=.002):
        raise ValueError('History must be oldest->newest at 0.2s cadence')
    history = np.stack([align_history_grid(x['grid'],x['pose'],chosen[-1]['pose']) for x in chosen])
    return history, {'frames':[x['frame'] for x in chosen], 'timestamps':ts.tolist(),
                     'ages_s':(ts[-1]-ts).tolist()}


def state20(speed, acceleration_carla_local, gyro_carla_rad_s, pose, near, far, commands):
    local = np.linalg.inv(pose)
    state = [speed/30, acceleration_carla_local[0]/10,
             -acceleration_carla_local[1]/10, -gyro_carla_rad_s]
    for goal in (near,far):
        # The training contract projects navigation at ego altitude.
        q = local @ np.array([goal[0],goal[1],pose[2,3],1.])
        state.extend([q[0]/50,-q[1]/50])
    for cmd in commands:
        state.extend([float(int(cmd)==j) for j in range(1,7)])
    out=np.asarray(state,np.float32)
    if out.shape!=(20,) or not np.isfinite(out).all():
        raise ValueError('Invalid state20')
    return out


def trajectory_to_world(xy_flu, pose):
    xy=np.asarray(xy_flu,float)
    if xy.shape!=(25,2) or not np.isfinite(xy).all():
        raise ValueError('Expected finite physical 25x2 FLU trajectory')
    return (np.column_stack([xy[:,0],-xy[:,1],np.zeros(25),np.ones(25)])@pose.T)[:,:3]


def vehicle_geometry(wheel_positions_world_cm, max_steer_degrees, pose):
    positions=np.asarray(wheel_positions_world_cm,float)/100.
    local=np.column_stack([positions,np.ones(len(positions))])@np.linalg.inv(pose).T
    if np.max(np.abs(local[:,:3]))>10:
        raise ValueError('Wheel/vehicle transform mismatch, possibly pre-tick stale actor pose')
    steering=np.asarray(max_steer_degrees)>0
    if steering.sum()!=2 or (~steering).sum()!=2:raise ValueError('Expected front-steering four-wheel vehicle')
    front=local[steering,:3].mean(0);rear=local[~steering,:3].mean(0)
    wheelbase=float(front[0]-rear[0])
    if not wheelbase>0:raise ValueError('Invalid wheelbase coordinate conversion')
    return wheelbase, {'wheel_positions_world_cm':np.asarray(wheel_positions_world_cm).tolist(),
                       'wheel_positions_vehicle_m':local[:,:3].tolist(),
                       'front_axle_vehicle_m':front.tolist(),'rear_axle_vehicle_m':rear.tolist(),
                       'wheelbase_m':wheelbase,'contract':'CARLA0.9.15 wheel.position is WORLD centimeters; transform to vehicle frame before axle separation'}


class TrajectoryController:
    def __init__(self,wheelbase,wheel_limit,steering_curve,kp=.2,ki=.1):
        self.wheelbase=wheelbase;self.wheel_limit=wheel_limit
        self.curve=sorted(steering_curve);self.kp=kp;self.ki=ki;self.integral=0.

    def control(self,world_path,vel_flu,pose,speed,age,dt=.05):
        velocity=np.asarray(vel_flu,float)
        if velocity.shape!=(25,2) or not np.isfinite(velocity).all():
            raise ValueError('Expected finite physical velocities')
        path=np.asarray(world_path,float)
        local=np.column_stack([path,np.ones(25)])@np.linalg.inv(pose).T
        local=local[:,:2];local[:,1]*=-1
        # Pure pursuit arc-distance lookahead is controller geometry, no path rejection.
        lookahead=max(2.,speed*.7)
        nearest=int(np.argmin(np.linalg.norm(local,axis=1)))
        choices=np.arange(nearest,25)
        index=int(choices[np.argmin(abs(np.linalg.norm(local[choices],axis=1)-lookahead))])
        target=local[index]
        angle=-math.atan2(2*self.wheelbase*target[1], max(float(target@target),1e-9))
        # CARLA steering curve's speed axis is km/h.
        scale=float(np.interp(speed*3.6,[p[0] for p in self.curve],[p[1] for p in self.curve]))
        steer=angle/max(self.wheel_limit*scale,1e-9)
        # Signed forward velocity, no artificial minimum/cap or jerk veto.
        target_speed=float(np.interp(age+.05,np.arange(1,26)*DT,velocity[:,0]))
        error=target_speed-speed;proposed=self.integral+dt*error
        effort=self.kp*error+self.ki*proposed
        if abs(effort)<=1 or effort*error<0:self.integral=proposed
        effort=self.kp*error+self.ki*self.integral
        applied={'throttle':float(np.clip(effort,0,1)), 'brake':float(np.clip(-effort,0,1)),
                 'steer':float(np.clip(steer,-1,1))}
        return applied, {'target_speed_mps':target_speed,'speed_error_mps':error,
                         'raw_longitudinal':effort,'raw_steer':steer,'target_index':index,
                         'target_flu_m':target.tolist(),'lookahead_m':lookahead,
                         'steering_speed_scale':scale,'integral':self.integral,
                         'trajectory_age_sim_s':age,'clipped':abs(steer)>1 or abs(effort)>1}
