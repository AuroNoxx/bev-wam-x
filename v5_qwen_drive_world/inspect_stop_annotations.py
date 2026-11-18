"""Check whether stop/launch labels follow a later change in the expert's command."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import gzip
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=128)
def annotation(route,index):
    path = ROOT/'data/public/bench2drive/clips'/route/'anno'/f'{index:05d}.json.gz'
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        r=json.load(stream)
    ego=next(b for b in r['bounding_boxes'] if b['class']=='ego_vehicle')
    transform=np.asarray(ego['world2ego'])
    vehicles=[]
    for b in r['bounding_boxes']:
        if b['class']=='vehicle':
            x,y,*_=transform@np.r_[b['location'],1]
            if 0<x<50 and abs(y)<2:
                vehicles.append((float(x),b['id'],float(b['speed'])))
    return float(r['desired_speed']),min(vehicles) if vehicles else None, bool(r['disappear_flag'] or r['teleport_flag'])


def main():
    manifest=json.loads((ROOT/'data/v5-qwen-drive-world/manifest.json').read_text())
    rows=[]
    for split in ('train','validation'):
        for r in sorted(manifest['splits'][split],key=lambda r:(r['route'],r['index'])):
            if r['speed_kmh']>3 or r['source']!='bench2drive':
                continue
            route,index=r['route'],r['index']
            target,lead,_=annotation(route,index)
            if target!=0:
                continue
            actor=json.loads((ROOT/r['actor_path']).read_text())
            assert actor['target_speed_kmh']==0
            with np.load(ROOT/r['label_path']) as labels:
                d=np.linalg.norm(labels['trajectory'],axis=1).max()
            future=[annotation(route,index+t) for t in range(1,51)]
            release=next((i/10 for i,(speed,_,_) in enumerate(future,1) if speed*3.6>1),None)
            previous=annotation(route,index-10)[1]
            rows.append(dict(split=split,route=route,index=index,group='hold' if d<.5 else 'launch' if d>=3 else 'intermediate',
                             lead_x=lead[0] if lead else None,lead_speed=lead[2] if lead else None,
                             lead_dx_past_1s=lead[0]-previous[0] if lead and previous and lead[1]==previous[1] else None,
                             later_target_gt_1_kmh_s=release,any_future_scene_event=any(flag for _,_,flag in future)))
    folder=ROOT/'artifacts/v5-qwen-drive-world/subset2000/final-window-audit'
    (folder/'stop-annotations.json').write_text(json.dumps(rows,indent=2))
    groups=defaultdict(list)
    for row in rows:
        groups[row['split'],row['group']].append(row)
    for key,group in sorted(groups.items()):
        print(key,len(group),'lead found',sum(r['lead_x'] is not None for r in group),
              'future command release',sum(r['later_target_gt_1_kmh_s'] is not None for r in group),
              'flagged',sum(r['any_future_scene_event'] for r in group))


if __name__=='__main__':
    main()
