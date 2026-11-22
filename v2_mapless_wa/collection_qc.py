"""Full corpus QC, including policy-only replay, directional audits and boundary provenance."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree
from .render_probe import locked_sizes
from .sd_replay import verify


def scan_episode(job):
    root,ep_id,ep,partial,camera_sizes,samples=job
    coordinates=[]
    disk=0
    folder=root/'episodes'/ep_id
    meta=json.loads((folder/'episode.json').read_text())
    if meta['frames']!=ep or len(ep)!=2000:raise ValueError(f'Invalid manifest for {ep_id}')
    if meta['final_frame_future_m']<60 or meta['final_frame_future_ticks']<80:
        raise ValueError(f'Missing measured future at {ep_id}')
    replay=verify(folder)
    key=ep[0]['map']+'/'+ep[0]['split']
    group=dict(episodes=[],err=[],directions=Counter(),ego_directions=Counter(),
        frames_opposite=0,ego_frames_opposite=0,ego_frames_outside=0,
        source_pts=np.zeros((2,3),int),valid_frames=np.zeros(2,int),
        scenes=Counter(),lane_events=Counter(),episodes_weather=Counter(),recovery_episodes=0,
        frames_with_future_occupancy=0,frames_with_measured_boundary=np.zeros(2,int),
        min_future_m=float('inf'),min_future_ticks=2**31)
    first,last=[(folder/f"{r['frame']:07d}"/'front.png').stat().st_mtime for r in (ep[0],ep[-1])]
    period=(samples[:,0]>=np.floor(first))&(samples[:,0]<=np.ceil(last))
    if not period.any() or last<=first:raise ValueError(f'Invalid capture time window at {ep_id}')
    group['episodes'].append(dict(episode=ep_id,mission=meta['mission'],replay=replay,
        png_first_unix=first,png_last_unix=last,retained_fps=(len(ep)-1)/(last-first),
        ue_peak_during_retention_gib=float(samples[period,1].max()/2**30)))
    group['episodes_weather'][meta['weather']]+=1
    group['recovery_episodes']+=any(r['recovery'] for r in ep)
    group['min_future_m']=min(group['min_future_m'],meta['final_frame_future_m'])
    group['min_future_ticks']=min(group['min_future_ticks'],meta['final_frame_future_ticks'])
    previous=None
    disk+=sum(f.stat().st_size for f in folder.iterdir() if f.is_file())
    for r in ep:
        frame=folder/f'{r["frame"]:07d}'
        if not partial:
            for name,(w,h) in zip(('front','front_left','front_right'),camera_sizes):
                image=cv2.imread(str(frame/(name+'.png')),cv2.IMREAD_COLOR)
                if image is None or image.shape!=(h,w,3):raise ValueError(f'Invalid RGB at {frame}/{name}')
            with np.load(frame/'lidar.npz') as lidar:
                bev=lidar['bev']
                if bev.shape!=(6,128,128) or not np.isfinite(bev).all():raise ValueError(f'Invalid LiDAR {frame}')
        if r['map']=='Town12':
            with np.load(frame/'index.npz') as index:coordinates.append(index['pose_xyzyaw'][:2])
        group['err'].append(r['nav_lateral_m'])
        group['directions'].update(r['sd_road_direction'])
        group['ego_directions'].update(r['ego_anchor_direction'])
        group['frames_opposite']+=('opposite' in r['sd_road_direction'])
        group['ego_frames_opposite']+=('opposite' in r['ego_anchor_direction'])
        group['ego_frames_outside']+=('outside' in r['ego_anchor_direction'])
        for field in ('straight','sharp_curve','junction','following','uphill','downhill','roadside_pole','recovery'):
            group['scenes'][field]+=bool(r[field])
        group['scenes']['requested_but_gap_occupied']+=bool(r.get('wait_reason'))
        side=r['lane_change']
        if side:
            group['scenes'][side+'_change']+=1
            if side!=previous:group['lane_events'][side]+=1
        previous=side
        with np.load(frame/'labels.npz') as lab:
            if not partial:
                shapes={'masks':(4,128,128),'map_lines':(7,20,2),'point_valid':(7,20),'exists':(7,),
                        'path_gt':(40,2),'p_occ':(5,128,128)}
                for k,shape in shapes.items():
                    a=lab[k]
                    if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f'{frame}/{k}')
                    if k in ('masks','p_occ') and a.max()>1:raise ValueError(f'Invalid binary target at {frame}/{k}')
                    if k=='p_occ':group['frames_with_future_occupancy']+=int(a.any())
                if not np.array_equal(lab['exists'],lab['point_valid'].any(1)):
                    raise ValueError(f'Polyline existence mismatch at {frame}')
            src=lab['hard_sources']
            if not np.array_equal(src>0,lab['point_valid'][3:5]):raise ValueError(f'Boundary provenance {frame}')
            for i in range(2):group['source_pts'][i]+=np.bincount(src[i],minlength=3)
            group['valid_frames']+=np.any(src>0,axis=1)
            group['frames_with_measured_boundary']+=np.any(src==2,axis=1)
        disk+=sum(f.stat().st_size for f in frame.iterdir())
    print(f'{ep_id} verified {len(ep)} frames ({key})',flush=True)
    return key, group, disk, coordinates


def quality(root,partial=False,workers=4):
    rows=[json.loads(v) for v in (root/'manifest.jsonl').read_bytes().splitlines()]
    episodes=defaultdict(list)
    for r in rows:episodes[r['episode']].append(r)
    counts=Counter((r['map'],r['split']) for r in rows)
    expected={('Town12','val'):2000,('Town12','train'):22000,('Town10HD_Opt','train'):16000}
    if not partial and counts!=expected:raise ValueError(f'Map and split mismatch: {counts}')
    report={}
    disk=0
    town12_xy=defaultdict(list)
    camera_sizes=locked_sizes()
    memory=[]
    for line in (root/'gpu.jsonl').read_bytes().splitlines():
        record=json.loads(line)
        memory.append((record['unix'],record['ue_bytes']))
    samples=np.asarray(memory,dtype=float)
    if not len(samples):raise ValueError('Missing sampled UE process memory')
    jobs=[(root,ep_id,ep,partial,camera_sizes,samples) for ep_id,ep in episodes.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for key,group,episode_bytes,coordinates in pool.map(scan_episode,jobs):
            disk+=episode_bytes
            if coordinates:town12_xy[key.split('/')[1]].extend(coordinates)
            if key not in report:
                report[key]=group
                continue
            for field,value in group.items():
                if field in ('min_future_m','min_future_ticks'):
                    report[key][field]=min(report[key][field],value)
                elif isinstance(value,Counter):
                    report[key][field].update(value)
                else:
                    report[key][field]+=value
    all_errors=[]
    for group in report.values():
        err=np.abs(np.array(group.pop('err')));all_errors.extend(err)
        n=len(err);group['frames']=n
        group['abs_lateral_p50_p95_max_m']=np.percentile(err,[50,95,100],axis=0).tolist()
        group['any_ge8m']=int((err.max(1)>=8).sum())
        group['boundary_valid_frames']=group.pop('valid_frames').tolist()
        group['boundary_frame_fraction']=(np.asarray(group['boundary_valid_frames'])/n).tolist()
        group['hard_points_missing_opendrive_semantic']=group.pop('source_pts').tolist()
        group['frames_with_measured_boundary']=group['frames_with_measured_boundary'].tolist()
        if partial:group.pop('frames_with_future_occupancy')
        group['scene_fractions']={k:v/n for k,v in group['scenes'].items()}
    disk+=sum(f.stat().st_size for f in root.iterdir() if f.is_file() and f.name not in ('qc.json','qc-progress.json'))
    start=datetime.fromisoformat((root/'start-time.txt').read_text().strip()).timestamp()
    distance=None
    if set(town12_xy)=={'train','val'}:
        distance=float(cKDTree(town12_xy['train']).query(town12_xy['val'])[0].min())
        if distance<=150:raise ValueError(f'Town12 splits overlap: {distance} m')
    if not partial and (len(episodes)!=20 or any(g['frames_opposite'] or g['ego_frames_opposite'] for g in report.values())):
        raise ValueError('Final corpus contains an opposing SD anchor or an incorrect episode count')
    errors=np.asarray(all_errors)
    result=dict(frames=len(rows),size_bytes=disk,by_map_and_split=report,town12_min_train_val_distance_m=distance,
        horizon_m=[20,40,60],at_or_above_8m_by_horizon=(errors>=8).sum(axis=0).tolist(),
        frames_with_opposite_ego_anchor=sum(g['ego_frames_opposite'] for g in report.values()),
        abs_lateral_p50_p95_max_m=
        np.percentile(all_errors,[50,95,100],axis=0).tolist() if len(rows) else [],any_ge8m=sum(g['any_ge8m'] for g in report.values()),
        frames_with_opposite_atlas_anchor=sum(g['frames_opposite'] for g in report.values()),
        valid_change_frames=sum(g['scenes']['left_change']+g['scenes']['right_change'] for g in report.values()),
        ue_peak_gib=float(samples[:,1].max()/2**30),ue_peak_unix=float(samples[samples[:,1].argmax(),0]),
        ue_samples_over_14gib=int((samples[:,1]>14*2**30).sum()),
        ue_peak_during_retention_gib=max(e['ue_peak_during_retention_gib'] for g in report.values() for e in g['episodes']),
        elapsed_including_survey_and_qc_s=time.time()-start)
    dest=root/('qc-progress.json' if partial else 'qc.json')
    dest.write_text(json.dumps(result,indent=2))
    print('QC',result['frames'],'frames',result['valid_change_frames'],'change frames',flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path);p.add_argument('--partial',action='store_true')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();quality(args.root,args.partial,args.workers)
