"""Recompute actor-only lead cues, then check pair eligibility on the label side."""
import gzip
import json
import numpy as np
from PIL import Image
import cv2
from .mine_launch_pairs import PAIR,load,ROOT
from .scale_track import track, iou
from .history_association import association
from .box_seeded_flow import match_pair as seeded_match
from .scale_geometry import RUN,displacement,project,DATA


def verify_identity(record, observed_boxes):
    """Label-side check of object identity at all three image exposures."""
    index=record['index']
    if record['source']=='carla':
        physics=load((ROOT/record['actor_path']).parent,'physics.json')
        local=np.c_[load(RUN,'vehicle-box.json')['local_vertices'],np.ones(8)]
        poses=[np.asarray(physics[index+k]['lead']) for k in (-10,-5,0)]
        objects=[((pose@local.T).T[:,:3],pose[:3,3]) for pose in poses]
    else:
        objects=[]
        for k in (-10,-5,0):
            path=DATA.parent/'public/bench2drive/clips'/record['route']/'anno'/f'{index+k:05d}.json.gz'
            with gzip.open(path,'rt') as stream:annotation=json.load(stream)
            obj=next(b for b in annotation['bounding_boxes'] if b['class']=='vehicle' and b['id']==record['lead_id'])
            objects.append((np.asarray(obj['world_cord']),obj['location']))
    scores=[]
    for (vertices,center),cam,path,b in zip(objects,record['cameras'],record['paths'],observed_boxes,strict=True):
        if b is None:
            scores.append(0.)
            continue
        target=project(vertices,center,cam)['box']
        with Image.open(ROOT/path) as im:w,h=im.size
        scores.append(iou(np.clip(target,0,[w,h,w,h]),np.clip(b,0,[w,h,w,h])))
    return scores


def observe(r, boxes, width):
    path=r['paths'][-1]
    with Image.open(ROOT/path) as im:w=im.width
    choices=[b['box'] for b in boxes[path] if w/3<=(b['box'][0]+b['box'][2])/2<=2*w/3]
    anchor=max(choices,key=lambda b:(b[0]<=w/2<=b[2],b[3]),default=None)
    result=None
    sequence=[None,None,anchor]
    if anchor:
        old=track(r['paths'],anchor)
        if old:
            half=track(r['paths'][-2:],anchor)
            sequence=[old[0],half[0] if half else None,anchor]
            depth=width*r['cameras'][-1]['intrinsic'][0][0]/(anchor[2]-anchor[0])
            result=dict(depth=depth,displacement=displacement(old[0],anchor,depth,r))
        else:
            result=association(r,anchor,boxes,width) or association(r,anchor,boxes,width,seeded_match)
            if result:
                sequence=[e['box'] for e in reversed(result['evidence'])]+[anchor]
    return result, sequence


def main(folder=PAIR):
    cv2.setNumThreads(3);cv2.setRNGSeed(9147)
    rows=load(folder,'cohort.json')
    boxes=load(folder,'historical-boxes.json')['predictions']
    width=load(RUN,'scale-results.json')['width_m']
    obs=[]
    audit=[]
    for r in rows:
        result,sequence=observe(r,boxes,width)
        audit.append(verify_identity(r,sequence))
        obs.append(dict(actor_path=r['actor_path'],depth=result['depth'] if result else None,
                        displacement=result['displacement'] if result else None))
    eligible={}
    for r,o,overlap in zip(rows,obs,audit,strict=True):
        v=o['displacement']
        eligible.setdefault(r['route'],True)
        eligible[r['route']] &= min(overlap)>=.5 and v is not None and (v>.3 if r['branch']=='launch' else abs(v)<=.3)
    (folder/'pair-observations.json').write_text(json.dumps(dict(observations=obs,eligible=eligible,three_frame_iou=audit),allow_nan=False))
    print('visible on both branches',sum(eligible.values()),'/',len(eligible))
    for r,o in zip(rows,obs):
        print(r['route'],r['branch'],round(r['lead_dx'],2),o['displacement'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--independent',action='store_true')
    main(PAIR/'independent-roads' if parser.parse_args().independent else PAIR)
