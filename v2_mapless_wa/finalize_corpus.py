"""Rebuild policy inputs from the frozen SD atlas and recorded sensors before publication."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import carla
import cv2
import numpy as np
from scipy.spatial import cKDTree
from .labels import ends
from .scenarios import xyz

from .sd_policy import RoadNavigator, route_vertices
from .sd_replay import verify
from .maneuver_qc import certify


class DirectionAudit:
    """Audit a target in every overlapping driving polygon at a junction."""
    def __init__(self,road):
        self.road=road
        self.waypoints=road.generate_waypoints(2.)
        self.tree=cKDTree(np.array([xyz(w) for w in self.waypoints]))

    def contains_forward_lane(self,point,axis):
        for i in self.tree.query_ball_point(point,6):
            w=self.waypoints[i]
            if abs(w.transform.location.z-point[2])>1.5:continue
            v=w.transform.get_forward_vector()
            if np.array([v.x,v.y,v.z])@axis<=.5:continue
            for before in w.previous(1.05):
                for after in w.next(1.05):
                    a,b=ends(before);c,d=ends(after)
                    polygon=np.asarray([a,b,d,c],np.float32)[:,:2].copy()
                    if cv2.pointPolygonTest(polygon,tuple(point[:2]),False)>=0:return True
        return False


def lane_direction(audit,xy,zs,tangents):
    states=[]
    for (x,y),z,axis in zip(xy,zs,tangents):
        w=audit.road.get_waypoint(carla.Location(float(x),float(y),float(z)),project_to_road=False)
        if w is None or abs(w.transform.location.z-z)>3.5:states.append('outside');continue
        f=w.transform.get_forward_vector()
        dot=axis@np.array([f.x,f.y,f.z])
        if dot<-.5 and w.is_junction and audit.contains_forward_lane(np.array([x,y,z]),axis):
            states.append('overlap_same_and_opposite')
        else:
            states.append('same' if dot>.5 else 'opposite' if dot<-.5 else 'crossing')
    return states


def finalize(raw,output,plan,road,write):
    meta=json.loads((raw/'episode.json').read_text())
    spec=plan
    line=route_vertices(spec)
    policy=RoadNavigator(line,spec['georef'],meta['drift_phase'],spec['events'],spec['sd_accuracy'])
    log_path=raw/'drive-log.npz'
    with np.load(log_path) as pack:log={k:pack[k] for k in pack}
    rows=meta['frames']
    poses=[]
    for r in rows:
        with np.load(raw/f'{r["frame"]:07d}'/'index.npz') as index:poses.append(index['world_from_ego'])
    poses=np.array(poses)
    meta['verified_change_frames']=certify(rows,poses,road.road)
    z=poses[:,2,3]
    arc=np.r_[0,np.linalg.norm(np.diff(poses[:,:2,3],axis=0),axis=1).cumsum()]
    if 'expert_pose' in log:
        recorded=log['expert_pose']
        arc=np.r_[0,np.linalg.norm(np.diff(recorded[:,:2,3],axis=0),axis=1).cumsum()]
        z=recorded[:,2,3]
    audit=Counter()
    sd_err=[];hint=[];hint_dist=[]
    dest=output/'episodes'/raw.name
    stage=dest.with_name(dest.name+'.partial-'+str(time.time_ns()))
    if write:stage.mkdir(parents=True)
    for i,fix in enumerate(log['fix']):
        nav=policy.update(fix[0],fix[1],fix[3],i*float(log['dt']))
        dist,instruction=policy.instruction()
        hint.append(instruction);hint_dist.append(dist)
        if i>=len(rows):continue
        r=rows[i];pose=poses[i]
        src=raw/f'{r["frame"]:07d}'
        with np.load(src/'labels.npz') as lab:expert=lab['path_gt']
        dd=np.arange(1,41)*1.5
        def at(s):return np.stack([np.interp(s,dd,c) for c in expert.T],axis=1)
        h=np.array([20.,40.,60.]);center=at(h)
        tangent=at(np.minimum(h+3,60))-at(h-3)
        tangent/=np.linalg.norm(tangent,axis=1,keepdims=True).clip(1e-6)
        lateral=((nav-center)*tangent[:,::-1]*[-1,1]).sum(1)
        axis=np.column_stack((tangent[:,0],-tangent[:,1],np.zeros(3)))@pose[:3,:3].T
        zz=np.interp(arc[i]+h,arc,z)
        # For old recordings with no tail pose archive use the actual measured
        # path in the local road plane as a height estimate only for the audit.
        far=arc[i]+h>arc[-1]
        zz[far]=(np.column_stack((center[:,0],-center[:,1],np.zeros(3)))@pose[:3,:3].T+pose[:3,3])[far,2]
        world_xy=nav@policy.axes.T+policy.xy
        ego_xy=(np.column_stack((nav[:,0],-nav[:,1],np.zeros(3)))@pose[:3,:3].T+pose[:3,3])[:,:2]
        nominal=lane_direction(road,world_xy,zz,axis)
        actual=lane_direction(road,ego_xy,zz,axis)
        r.update(sd_road_direction=nominal,ego_anchor_direction=actual,sd_station_m=policy.station,
                 sd_drift_m=policy.drift,nav_lateral_m=lateral.tolist(),route_instruction=instruction,
                 instruction_distance_m=float(dist))
        for key,states in [('atlas',nominal),('ego',actual)]:
            audit[key+'_opposite_frames']+=int('opposite' in states)
            for state in states:audit[key+'_'+state]+=1
        sd_err.append(lateral)
        if write:
            frame=stage/src.name;frame.mkdir()
            for name in ('front.png','front_left.png','front_right.png','lidar.npz','labels.npz'):
                os.link(src/name,frame/name)
            with np.load(src/'index.npz') as idx:index={k:idx[k] for k in idx}
            index['sd_drift_m']=policy.drift
            np.savez_compressed(frame/'index.npz',**index)
            np.savez_compressed(frame/'input.npz',sd_nav=nav,cmd_one_hot=np.eye(3,dtype=np.uint8)[log['tactical'][i]])
    e=np.abs(sd_err);audit=dict(audit)
    audit.update(any_ge8=int((e.max(1)>=8).sum()),p95_m=np.percentile(e,95,axis=0).tolist(),
                 max_m=e.max(0).tolist(),frames=len(rows))
    meta.update(plan_world_xy=line.tolist(),georef=spec['georef'],events=spec['events'],
                sd_accuracy=spec['sd_accuracy'],sd_navigation='independent-SD-GNSS-IMU-v1',qc=audit)
    if write:
        log.update(route_hint=hint,hint_distance=hint_dist)
        np.savez_compressed(stage/'drive-log.npz',**log)
        (stage/'episode.json').write_text(json.dumps(meta,indent=2))
        verify(stage)
        stage.rename(dest)
    return audit


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--write',action='store_true')
    p.add_argument('--episodes',nargs='*')
    args=p.parse_args()
    data=json.loads(args.plan.read_text());maps={}
    args.output.mkdir(exist_ok=True,parents=True)
    for raw in sorted((args.root/'episodes').glob('ep_*')):
        if not raw.name[3:].isdigit() or args.episodes and raw.name not in args.episodes:continue
        if args.write and (args.output/'episodes'/raw.name).exists():continue
        meta=json.loads((raw/'episode.json').read_text())
        town=meta['frames'][0]['map']
        if town not in maps:
            maps[town]=DirectionAudit(carla.Map(town,(args.root/(town+'.xodr')).read_text()))
        spec=next(s for s in data[town]['missions'] if s['name']==meta['mission'])
        result=finalize(raw,args.output,spec,maps[town],args.write)
        print(raw.name,json.dumps(result),flush=True)
        (args.output/(raw.name+'-qc.json')).write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
