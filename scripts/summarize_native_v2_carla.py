"""Read-only per-run evaluation and schematic replay. No driving guard or scoring feedback."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import math
import numpy as np
import cv2
import carla

ROOT=Path(__file__).resolve().parents[1]


def read_rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stats(x):
    a=np.asarray(x,float)
    return {'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),
            'max':float(a.max()),'min':float(a.min())} if len(a) else None


def occupancy_path_hits(plan,folder,report):
    with np.load(folder/plan['input_file'],allow_pickle=False) as z:grid=z['history'][-1]
    xy=np.asarray(plan['prediction']['xy_flu_m'])
    ix=np.floor((xy[:,0]+25.6)/.4).astype(int)
    iy=np.floor((25.6-xy[:,1])/.4).astype(int)
    valid=(ix>=0)&(ix<128)&(iy>=0)&(iy<128)
    extent=report.get('ego_bounding_box',{}).get('extent',[2.5,1.1,1.])
    valid&=~((np.abs(xy[:,0])<extent[0]+.4)&(np.abs(xy[:,1])<extent[1]+.4))
    hits=np.zeros(25,bool)
    hits[valid]=np.any(np.isin(grid[:,iy[valid],ix[valid]],[4,5]),axis=0)
    return {'center_path_hits_dynamic_occupied_voxel':bool(hits.any()),
            'first_hit_time_s':float((np.argmax(hits)+1)*.2) if hits.any() else None,
            'center_xy_support_fraction':float(valid.mean()),
            'note':'diagnostic current occupied vehicle/pedestrian columns; exclude current ego footprint+one voxel (self LiDAR returns); no future prediction or swept footprint; never feeds controls'}


def replay(folder,steps,plans,report):
    route=np.asarray(report['route']['route_xyz'])[:,:2]
    ego=np.array([r['location'][:2] for r in steps]);allpts=np.concatenate([route,ego])
    lo=allpts.min(0)-8;hi=allpts.max(0)+8
    scale=560/max(hi-lo);center=(lo+hi)/2
    def pixel(p):
        q=(np.asarray(p)[:,:2]-center)*scale+np.array([320,330])
        q[:,1]=660-q[:,1]
        return np.round(q).astype(np.int32)
    writer=cv2.VideoWriter(str(folder/'trajectory-replay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,720))
    last=None
    for i,row in enumerate(steps):
        if i%2:continue
        image=np.full((720,640,3),245,np.uint8)
        cv2.polylines(image,[pixel(route)],False,(160,160,160),8)
        cv2.polylines(image,[pixel(ego[:i+1])],False,(30,30,220),2)
        if row['plan_id'] is not None:
            plan=plans[row['plan_id']]
            cv2.polylines(image,[pixel(np.asarray(plan['path_world']))],False,(220,120,20),2)
        for actor in row['traffic_truth_evaluation']:
            q=pixel(np.array([actor['location']]))[0]
            cv2.circle(image,tuple(q),5,(60,170,30),-1)
        p=pixel(ego[i:i+1])[0];cv2.circle(image,tuple(p),4,(30,30,220),-1)
        ctl=row['control_applied_for_next_tick'];t=row['sim_t'] or 0.
        text=[f"{report['scenario']} | {report['mode']} | t={t:.2f}s",
              f"speed={row['speed_mps']:.2f} m/s   route offset={row['route_error_m']:.2f}m",
              f"throttle={ctl['throttle']:.2f} brake={ctl['brake']:.2f} steer={ctl['steer']:.2f}"]
        for j,label in enumerate(text):cv2.putText(image,label,(10,22+23*j),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1)
        cv2.putText(image,'SCHEMATIC LOG REPLAY - NO CAMERA / NO RENDER',(10,690),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1)
        cv2.putText(image,'gray: route | red: actual | blue: model | green: actors',(10,710),cv2.FONT_HERSHEY_SIMPLEX,.43,(20,20,20),1)
        if writer.isOpened():writer.write(image)
        last=image
    writer.release()
    if last is not None:cv2.imwrite(str(folder/'trajectory-overview.png'),last)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--port',type=int,default=2125)
    p.add_argument('--no-map-check',action='store_true');a=p.parse_args();root=Path(a.root)
    m=None
    if not a.no_map_check:
        c=carla.Client('127.0.0.1',a.port);c.set_timeout(10);m=c.get_world().get_map()
    summaries=[]
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder/'report.json').exists():continue
        report=json.loads((folder/'report.json').read_text())
        if report['status']!='completed_actual_closed_loop':continue
        rows=read_rows(folder/'steps.jsonl');steps=[r for r in rows if r['phase']=='active'];plans=read_rows(folder/'plans.jsonl')
        times=np.array([r['timestamp'] for r in steps]);tracking={};per_plan=[]
        for plan in plans:
            row={'id':plan['id'],'sim_t':plan['sim_t'],**occupancy_path_hits(plan,folder,report)}
            pose=np.asarray(plan['ego_pose']);prediction=np.asarray(plan['prediction']['xy_flu_m'])
            for horizon,index in [(.2,0),(1.,4),(2.,9)]:
                goal=plan['timestamp']+horizon;j=int(np.argmin(abs(times-goal)))
                if abs(times[j]-goal)>.01:continue
                actual=np.linalg.inv(pose)@np.r_[steps[j]['location'],1.]
                actual=np.array([actual[0],-actual[1]])
                error=actual-prediction[index]
                tracking.setdefault(str(horizon),[]).append(float(np.linalg.norm(error)))
                row['actual_minus_pred_'+str(horizon)+'s_flu_m']=error.tolist()
            per_plan.append(row)
        collision_events=report['collision_events'];episodes=[]
        for ev in sorted(collision_events,key=lambda r:r['timestamp']):
            if not episodes or ev['other_type']!=episodes[-1]['other_type'] or ev['timestamp']-episodes[-1]['last_timestamp']>.5:
                episodes.append({'other_type':ev['other_type'],'start_timestamp':ev['timestamp'],'last_timestamp':ev['timestamp'],
                                 'ego_location':ev['ego_world_location'],'event_count':1})
            else:episodes[-1]['last_timestamp']=ev['timestamp'];episodes[-1]['event_count']+=1
        road=[]
        if m:
            for row in steps:
                loc=carla.Location(*row['location']);wp=m.get_waypoint(loc,project_to_road=False,lane_type=carla.LaneType.Driving)
                direction=None
                if wp:
                    pose=np.asarray(row['ego_pose']);forward=pose[:3,0];lane=wp.transform.get_forward_vector()
                    direction=float(np.dot(forward,[lane.x,lane.y,lane.z]))
                road.append({'frame':row['frame'],'timestamp':row['timestamp'],'inside_driving_lane':wp is not None,
                             'lane_alignment_cosine':direction,'road_id':None if wp is None else wp.road_id,'lane_id':None if wp is None else wp.lane_id})
            (folder/'road-check.json').write_text(json.dumps(road,indent=2))
        latency=[r['prediction']['inference_seconds'] for r in plans]
        speeds=[r['speed_mps'] for r in steps]
        controls=[r['control_applied_for_next_tick'] for r in steps]
        targets=[r['controller']['target_speed_mps'] for r in steps if 'target_speed_mps' in r['controller']]
        summary={'scenario':report['scenario'],'folder':str(folder),'simulated_seconds':report['simulated_seconds'],
                 'termination':report['termination'],'distance_m':report['distance_travelled_m'],
                 'route_completion_fraction':report['route_completion_fraction'],'speed_mps':stats(speeds),
                 'route_center_offset_m':stats([r['route_error_m'] for r in steps]),
                 'outside_driving_lane_fraction':float(np.mean([not r['inside_driving_lane'] for r in road])) if road else None,
                 'opposite_lane_alignment_fraction':float(np.mean([r['lane_alignment_cosine'] is not None and r['lane_alignment_cosine']<0 for r in road])) if road else None,
                 'collision_callback_count':len(collision_events),'collision_objects':dict(Counter(r['other_type'] for r in collision_events)),
                 'collision_episodes':episodes,'takeovers':report['takeovers'],'inference_seconds':stats(latency),
                 'ipc_wall_seconds':stats([r['prediction'].get('ipc_wall_seconds',0) for r in plans]),
                 'one_plan_tracking_error_m':{k:stats(v) for k,v in tracking.items()},
                 'tracking_limits':'.2s is one held plan with four control ticks;1/2s includes replanning, not isolated controller error',
                 'target_forward_speed_mps':stats(targets),'brake_nonzero_fraction':float(np.mean([r['brake']>0 for r in controls])),
                 'controls_saturated_fraction':float(np.mean([r['controller'].get('clipped',False) for r in steps])),
                 'model_paths_enter_current_vehicle_or_pedestrian_voxel_fraction':float(np.mean([r['center_path_hits_dynamic_occupied_voxel'] for r in per_plan])),
                 'first_collision':collision_events[0] if collision_events else None,
                 'scene_traffic':report.get('traffic'),'jerk_veto':False,'low_speed_veto':False,
                 'route_metric_limit':'nearest-route progress can be high despite cutting corners, off-road driving or collision; not pass/fail'}
        (folder/'analysis.json').write_text(json.dumps(summary,indent=2));(folder/'plan-analysis.json').write_text(json.dumps(per_plan,indent=2))
        replay(folder,rows,plans,report);summaries.append(summary)
    (root/'analysis-summary.json').write_text(json.dumps(summaries,indent=2));print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
