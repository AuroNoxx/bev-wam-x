"""Build actual two-stage datasets; retain source archives, extract RGB once."""
import gzip,hashlib,json,random,tarfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from .real_data_geometry import ego,relative_poses,labels,world_pose

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/v5-qwen-drive-world'
CAMERAS=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
RGB={c:c.replace('CAM_','rgb_').lower() for c in CAMERAS}
VIEWS=['<FRONT VIEW>','<FRONT LEFT VIEW>','<FRONT RIGHT VIEW>','<BACK VIEW>','<BACK LEFT VIEW>','<BACK RIGHT VIEW>']


def read_rows(route,source):
    if source=='highway':
        folder=OUT/'highway'/route
        return [json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines()]
    folder=ROOT/'data/public/bench2drive/clips'/route/'anno'
    paths=sorted(folder.glob('*.json.gz'));rows=[]
    for p in paths:
        with gzip.open(p,'rt',encoding='utf-8') as f:r=json.load(f)
        r.update(index=int(p.name.split('.')[0]),timestamp=int(p.name.split('.')[0])*.1)
        rows.append(r)
    if [r['index'] for r in rows]!=list(range(len(rows))):raise ValueError(f'Noncontiguous source frames: {route}')
    return rows


def select(records,count,seed=71):
    rng=random.Random(seed);buckets=defaultdict(list)
    for r in records:buckets[min(5,int(r['speed_kmh']//20))].append(r)
    for group in buckets.values():rng.shuffle(group)
    chosen=[]
    while len(chosen)<count:
        nonempty=[b for b in sorted(buckets) if buckets[b]]
        if not nonempty:raise ValueError(f'Need{count} distinct samples; only{len(chosen)} available')
        for bucket in nonempty:
            chosen.append(buckets[bucket].pop())
            if len(chosen)==count:break
    return chosen


def actor(rows,index,route,source):
    now=rows[index];past=rows[index-15:index+1]
    history=relative_poses(past,now)
    velocity=np.gradient(history[:,:2],.1,axis=0)
    acceleration=np.gradient(velocity,.1,axis=0)
    images={}
    for c,v in zip(CAMERAS,VIEWS):
        indices=[index-15,index-10,index-5,index] if c=='CAM_FRONT' else [index]
        if source=='bench2drive':images[v]=[str((OUT/'rgb'/route/RGB[c]/f'{i:05d}.jpg').relative_to(ROOT)) for i in indices]
        else:images[v]=[str((OUT/'highway'/route/rows[i]['images'][c]).relative_to(ROOT)) for i in indices]
    inv=np.asarray(ego(now)['world2ego'],np.float64)
    if source=='highway':nav=np.asarray(now['nav_world'],np.float64);target=float(now['target_speed_kmh'])
    else:
        # Use the actual recorded navigation point; coordinates are CARLA GPS-converted world XY.
        nav=np.array([now['x_command_near'],now['y_command_near'],ego(now)['location'][2]])
        target=float(now['desired_speed'])*3.6
    local=inv@np.r_[nav,1];nav_xy=[float(local[0]),float(-local[1])]
    command=int(now.get('command_near',4));nav_command={1:1,2:2}.get(command,0)
    return {'source':source,'route':route,'index':index,'timestamp':now['timestamp'],'images':images,
            'history':history.tolist(),'history_velocity':velocity.tolist(),'history_acceleration':acceleration.tolist(),
            'ego_velocity':[float(now['speed']),0.0],'ego_acceleration':acceleration[-1].tolist(),
            'driving_command':[0,0,0,0],'driving_command_note':'unmapped official four-channel command; zeros explicitly unknown',
            'nav_command':nav_command,'navigation_flu':nav_xy,'target_speed_kmh':target,'speed_kmh':float(now['speed'])*3.6,
            'world_from_ego':world_pose(now).tolist(),'history_timestamps':[r['timestamp'] for r in past],
            'camera_calibration':now.get('sensors') if source=='bench2drive' else str((OUT/'highway'/route/'calibration.json').relative_to(ROOT))}


def main():
    old=json.loads((ROOT/'data/occ_v3/manifest.json').read_text());candidates=defaultdict(list)
    for route in old['routes']:
        if route['split'] not in ('train','validation'):continue
        rows=read_rows(route['route'],'bench2drive')
        for i in range(15,len(rows)-50):
            speed=float(rows[i]['speed'])*3.6
            if not 0<=speed<=120:continue
            candidates['b2d_'+route['split']].append({'source':'bench2drive','route':route['route'],'index':i,'speed_kmh':speed,'split':route['split']})
    for folder in sorted((OUT/'highway').glob('*')):
        if not (folder/'status.json').exists():continue
        status=json.loads((folder/'status.json').read_text())
        if status.get('phase')!='complete' or status.get('probe_only'):continue
        rows=read_rows(folder.name,'highway')
        for i in range(15,len(rows)-50):
            speed=rows[i]['speed']*3.6
            if 80<=speed<=120 and not any(r['collision'] for r in rows[i-15:i+51]):
                candidates['highway_train'].append({'source':'highway','route':folder.name,'index':i,'speed_kmh':speed,'split':'train'})
    inventory={k:len(v) for k,v in candidates.items()};print(json.dumps(inventory),flush=True)
    (OUT/'candidate-inventory.json').write_text(json.dumps(inventory,indent=2))
    train=select(candidates['b2d_train'],1400)+select(candidates['highway_train'],600)
    validation=select(candidates['b2d_validation'],min(128,len(candidates['b2d_validation'])))
    selected=train+validation;by_route=defaultdict(list)
    for r in selected:by_route[(r['source'],r['route'])].append(r)
    for (source,route),members in by_route.items():
        rows=read_rows(route,source)
        actor_indices=set()
        for m in members:
            i=m['index'];actor_indices.update([i,i+10,i+20,i+30])
        needed_rgb=set()
        for i in actor_indices:
            for c in CAMERAS:
                for j in ([i-15,i-10,i-5,i] if c=='CAM_FRONT' else [i]):needed_rgb.add((RGB[c],j))
        if source=='bench2drive':
            archive=ROOT/'data/public/bench2drive/archives'/f'{route}.tar.gz';found=set()
            with tarfile.open(archive,'r|gz') as tar:
                for member in tar:
                    parts=Path(member.name).parts
                    if not member.isfile() or len(parts)!=4 or parts[0]!=route or parts[1]!='camera' or not parts[-1].endswith('.jpg'):continue
                    key=(parts[2],int(parts[3].split('.')[0]))
                    if key not in needed_rgb:continue
                    target=OUT/'rgb'/route/parts[2]/parts[3];target.parent.mkdir(parents=True,exist_ok=True)
                    payload=tar.extractfile(member).read()
                    if not target.exists():target.write_bytes(payload)
                    elif target.read_bytes()!=payload:raise ValueError('Existing RGB differs from source archive')
                    found.add(key)
            if found!=needed_rgb:raise ValueError(f'Missing required RGB: {route}: {len(needed_rgb-found)}')
        for i in actor_indices:
            path=OUT/'actors'/route/f'{i:05d}.json';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(actor(rows,i,route,source)),encoding='utf-8')
        for m in members:
            i=m['index'];path=OUT/'labels'/route/f'{i:05d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,**labels(rows,i,source))
            inv=np.asarray(ego(rows[i])['world2ego'])
            front=False
            for b in rows[i]['bounding_boxes']:
                if b['class']!='vehicle':continue
                q=inv@np.r_[np.asarray(b['location']),1]
                if 3<q[0]<50 and abs(q[1])<2.5:front=True
            m['has_front_vehicle']=front
            m['actual_highway_lane_change']=source=='highway' and rows[i]['road_id']==rows[i+30]['road_id'] and rows[i]['lane_id']!=rows[i+30]['lane_id']
            m.update(actor_path=str((OUT/'actors'/route/f'{i:05d}.json').relative_to(ROOT)),label_path=str(path.relative_to(ROOT)),
                     future_actor_paths=[str((OUT/'actors'/route/f'{i+k:05d}.json').relative_to(ROOT)) for k in (10,20,30)],
                     current_timestamp=rows[i]['timestamp'],history_timestamps=[rows[j]['timestamp'] for j in range(i-15,i+1)],
                     future_timestamps=[rows[i+k]['timestamp'] for k in (10,20,30)],
                     initial_condition_family=route if source=='bench2drive' else 'Town04_collection_route_family')
        print(json.dumps({'prepared':route,'samples':len(members)}),flush=True)
    low=[r for r in train if r['source']=='bench2drive' and r['speed_kmh']<40]
    high=[r for r in train if r['source']=='highway' and r['speed_kmh']>=100]
    follow=[r for r in low if r['has_front_vehicle']]
    low_selected=select(follow,min(45,len(follow)))
    if len(low_selected)<45:low_selected+=select([r for r in low if r not in low_selected],45-len(low_selected))
    changes=[r for r in high if r['actual_highway_lane_change']]
    if not changes:raise ValueError('No actual100-120km/h lane-change windows; acquire before overfit64')
    high_selected=select(changes,min(10,len(changes)))
    high_selected+=select([r for r in high if r not in high_selected],19-len(high_selected))
    pilot=low_selected+high_selected
    manifest={'version':5,'scope':'real supervised two-stage workflow; measured speeds; no pseudo highway labels',
              'splits':{'train':train,'validation':validation,'test':[]},'overfit64':pilot,
              'source_counts':{'bench2drive':1400,'highway':600},'validation_highway_samples':0,
              'coverage_note':'Town04 training acquired; current independent validation is held-out B2D routes, high-speed validation still absent',
              'occupancy_labels':'vehicle/walker footprint ground truth; future-frame support transformed to current FLU; not full static scene occupancy',
              'b2d_timestamps':'10Hz source tools/data_collect.py; matching annotation and RGB filenames',
              'source_collector_sha256':hashlib.sha256((ROOT/'runtime/Bench2Drive-0.0.4/tools/data_collect.py').read_bytes()).hexdigest()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('PREPARED_REAL_MANIFEST',len(train),len(validation),len(pilot),flush=True)


if __name__=='__main__':main()
