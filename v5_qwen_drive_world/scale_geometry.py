"""Label-side projection and scale geometry for the offline lead-motion replay."""
import gzip
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/v5-qwen-drive-world/geometry-repair/lead-motion/scale-replay'
DATA=ROOT/'data/v5-qwen-drive-world'


def project(vertices,origin,camera):
    x,y,z=(camera['world2cam']@np.c_[vertices,np.ones(len(vertices))].T)[:3]
    assert x.min()>.1
    k=np.asarray(camera['intrinsic'])
    u=k[0,2]+k[0,0]*y/x
    v=k[1,2]-k[1,1]*z/x
    return dict(box=[u.min(),v.min(),u.max(),v.max()],center_depth=float((np.asarray(camera['world2cam'])@np.r_[origin,1])[0]),
                near_depth=float(np.sort(x)[:4].mean()))


def point(box,z,cam):
    u=(box[0]+box[2])/2
    v=(box[1]+box[3])/2
    k=np.asarray(cam['intrinsic'])
    return np.linalg.solve(cam['world2cam'],[z,z*(u-k[0,2])/k[0,0],-z*(v-k[1,2])/k[1,1],1])


def displacement(prev,cur,depth,record):
    cameras=record['cameras']
    width=lambda box,cam:(box[2]-box[0])/cam['intrinsic'][0][0]
    zprev=depth*width(cur,cameras[-1])/width(prev,cameras[0])
    world_delta=point(cur,depth,cameras[-1])-point(prev,zprev,cameras[0])
    return float(np.linalg.solve(record['ego'],world_delta)[0])


def carla(row,local):
    path=ROOT/row['actor']
    actor=json.loads(path.read_text())
    physics=json.loads((path.parent/'physics.json').read_text())
    frames=actor['observations'][1:]
    labels=[]
    for f,i in zip(frames,[36,41,46],strict=True):
        snap=physics[i]
        assert abs(snap['time']-f['time'])<.005
        pose=np.asarray(snap['lead'])
        vertices=(pose@local.T).T[:,:3]
        labels.append(project(vertices,pose[:3,3],f['camera']))
    return dict(route=row['route'],index=46,branch=row['branch'],target=row['target'],ego=actor['world_from_ego'],
                paths=[f['path'] for f in frames],cameras=[f['camera'] for f in frames],labels=labels)


def bench(row):
    actor=json.loads((ROOT/row['actor_path']).read_text())
    folder=DATA.parent/'public/bench2drive/clips'/row['route']/'anno'
    frames=[json.load(gzip.open(folder/f'{row["index"]+offset:05d}.json.gz','rt')) for offset in (-10,-5,0)]
    inv=np.linalg.inv(actor['world_from_ego'])
    leads=[]
    for b in frames[-1]['bounding_boxes']:
        if b['class']=='vehicle':
            x,y,*_=inv@np.r_[b['location'],1.]
            if 0<x<35 and abs(y)<2:leads.append((x,b['id']))
    car_id=min(leads)[1]
    labels=[];cams=[]
    for f in frames:
        b=next(b for b in f['bounding_boxes'] if b['class']=='vehicle' and b['id']==car_id)
        cam=f['sensors']['CAM_FRONT']
        labels.append(project(np.array(b['world_cord']),b['location'],cam))
        cams.append(cam)
    return dict(route=row['route'],index=row['index'],branch='observed',target=row['target'],ego=actor['world_from_ego'],
                paths=actor['images']['<FRONT VIEW>'][1:],cameras=cams,labels=labels)


def main():
    local=np.c_[json.loads((RUN/'vehicle-box.json').read_text())['local_vertices'],np.ones(8)]
    files={'fit':DATA/'lead-readout-refit/fit.json','acceptance':DATA/'lead-readout-refit/acceptance.json',
           'regression':DATA/'lead-pairs-aligned/manifest.json'}
    splits={}
    for split,file in files.items():
        source=json.loads(file.read_text())
        rows=source['windows'] if isinstance(source,dict) else source
        splits[split]=[carla(r,local) for r in rows]
    cohort=json.loads((RUN.parent/'labels-and-split.json').read_text())['windows']
    splits['internal']=[bench(r) for r in cohort if r['dev']]
    # A single fixed metric width estimated only on CARLA training roads.
    widths=[(v['box'][2]-v['box'][0])*v['near_depth']/r['cameras'][-1]['intrinsic'][0][0]
            for r in splits['fit'] for v in [r['labels'][-1]]]
    width=float(np.median(widths))
    for rows in splits.values():
        for r in rows:
            a,b=r['labels'][0],r['labels'][-1]
            for key in ('center_depth','near_depth'):
                r['oracle_'+key]=displacement(a['box'],b['box'],b[key],r)
            r['oracle_box_width_prior']=displacement(a['box'],b['box'],width*r['cameras'][-1]['intrinsic'][0][0]/(b['box'][2]-b['box'][0]),r)
    (RUN/'geometry.json').write_text(json.dumps(dict(width_prior_m=width,splits=splits)))
    for split,rows in splits.items():
        print(split,len(rows),{key:round(np.mean([abs(r[key]-r['target']) for r in rows]),4) for key in
                               ('oracle_center_depth','oracle_near_depth','oracle_box_width_prior')})


if __name__=='__main__':main()
