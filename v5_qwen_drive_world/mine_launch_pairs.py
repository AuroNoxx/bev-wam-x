"""Mine new, route-matched stop/launch windows from the original Bench2Drive training archives."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from .build_real_manifest import ROOT, OUT, actor, read_rows
from .real_data_geometry import ego, world_pose
from .scale_profile_inputs import DEST
from .scale_geometry import DATA

SOURCE = DEST / 'history-association'
PAIR = SOURCE / 'paired-launch'
DATASET = DATA / 'paired-launch'


def load(folder, name):
    return json.loads((folder/name).read_text())


def scan():
    identity = load(SOURCE,'identity.json')
    manifest = load(DATA,'manifest.json')
    prior = {(r['route'],r['index']) for r in manifest['splits']['train']}
    excluded = set(identity['development_routes']) | {r['route'] for r in manifest['splits']['validation']}
    routes = sorted({r['route'] for r in manifest['splits']['train'] if r['source']=='bench2drive'}-excluded)
    inventory = {}
    pairs = []
    for route in routes:
        rows = read_rows(route,'bench2drive')
        poses = np.array([world_pose(row)[:3,3] for row in rows])
        groups = defaultdict(list)
        for i in range(15,len(rows)-50):
            r = rows[i]
            if (route,i) in prior or r['desired_speed'] != 0 or not 0 <= r['speed']*3.6 <= 3:
                continue
            inv = np.asarray(ego(r)['world2ego'])
            future = (poses[i+1:i+51] - poses[i]) @ inv[:2,:3].T
            distance = np.linalg.norm(future,axis=1).max()
            leads = []
            for car in r['bounding_boxes']:
                if car['class']=='vehicle':
                    x,y,*_ = inv @ np.r_[car['location'],1]
                    if 0<x<35 and abs(y)<2:
                        leads.append((x,car['id'],car['location']))
            if not leads:continue
            gap,lead,loc=min(leads)
            history = [next((b for b in rows[t]['bounding_boxes'] if b.get('id')==lead and b['class']=='vehicle'),None)
                       for t in (i-10,i-5)]
            if not all(history):continue
            dx=float(inv[0,:3] @ (np.asarray(loc)-history[0]['location']))
            kind = 'launch' if distance>=3 and dx>.3 else 'hold' if distance<.5 and abs(dx)<.15 else None
            if kind:
                groups[kind].append(dict(index=i,lead_id=lead,lead_dx=dx,gap=gap,distance=distance))
        chosen=[]
        for pos in sorted(groups['launch'],key=lambda r:-r['lead_dx']):
            if any(abs(pos['index']-p['launch']['index'])<10 for p in chosen):continue
            matches=[h for h in groups['hold'] if h['lead_id']==pos['lead_id'] and
                     abs(h['index']-pos['index'])>=50 and np.linalg.norm(poses[h['index']]-poses[pos['index']])<6 and
                     all(abs(h['index']-p['hold']['index'])>=10 for p in chosen)]
            if matches:
                h=min(matches,key=lambda h:abs(h['index']-pos['index']))
                chosen.append(dict(route=route,hold=h,launch=pos))
        pairs.extend(chosen)
        inventory[route]=dict(launch=len(groups['launch']),hold=len(groups['hold']),pairs=len(chosen))
    DATASET.mkdir(parents=True,exist_ok=True)
    out=dict(excluded_routes=sorted(excluded),inventory=inventory,pairs=pairs)
    (DATASET/'mined.json').write_text(json.dumps(out,indent=2))
    print('routes',len(routes),'pairs',len(pairs),'paired routes',len(set(p['route'] for p in pairs)),flush=True)
    print({k:v for k,v in inventory.items() if v['launch']},flush=True)


if __name__=='__main__':scan()
