"""Pixel-domain audit of the stationary-lead errors on internal development routes."""
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/v5-qwen-drive-world/geometry-repair/lead-motion'


def pixels(transform,intrinsic,points):
    cam=(transform@points.T).T
    assert np.all(cam[:,0]>0)
    return np.stack((intrinsic[0,2]+intrinsic[0,0]*cam[:,1]/cam[:,0],
                     intrinsic[1,2]-intrinsic[1,1]*cam[:,2]/cam[:,0]),1)


def main():
    cv2.setNumThreads(2)
    rows=[]
    for r in json.loads((RUN/'result.json').read_text())['windows']:
        if abs(r['target'])>1e-4:
            continue
        actor=json.loads((ROOT/r['actor_path']).read_text())
        past,now=[json.load(gzip.open(ROOT/'data/public/bench2drive/clips'/r['route']/'anno'/f'{i:05d}.json.gz','rt'))
                  for i in (r['index']-10,r['index'])]
        ego=np.array(actor['world_from_ego'])
        inv=np.linalg.inv(ego)
        leads=[]
        for obj in now['bounding_boxes']:
            x,y,*_=inv@np.r_[obj['location'],1]
            if obj['class']=='vehicle' and 0<x<35 and abs(y)<2:
                leads.append((x,obj['id'],obj))
        _,_,lead=min(leads)
        box=np.c_[lead['world_cord'],np.ones(8)]
        old_box=next(b for b in past['bounding_boxes'] if b['id']==lead['id'])
        static_error=np.max(np.linalg.norm(box[:,:3]-np.array(old_box['world_cord']),axis=1))
        s0=now['sensors']['CAM_FRONT']; s1=past['sensors']['CAM_FRONT']
        k=np.array(s0['intrinsic'])
        current=pixels(np.array(s0['world2cam']),k,box)
        exact=pixels(np.array(s1['world2cam']),k,box)
        x,y,yaw=actor['history'][5]
        c,s=np.cos(yaw),np.sin(yaw)
        rel=np.array([[c,s,0,x],[-s,c,0,-y],[0,0,1,0],[0,0,0,1]])
        approximate=pixels(np.array(s0['world2cam'])@ego@np.linalg.inv(rel)@inv,k,box)
        paths=actor['images']['<FRONT VIEW>']
        a,b=[cv2.imread(str(ROOT/paths[i]),cv2.IMREAD_GRAYSCALE) for i in (1,3)]
        assert all(im.shape==(s0['image_size_y'],s0['image_size_x']) for im in (a,b))
        lo,hi=exact.min(0),exact.max(0)
        lo,hi=lo+.25*(hi-lo),hi-.25*(hi-lo)
        mask=np.zeros_like(a)
        cv2.rectangle(mask,tuple(lo.astype(int)),tuple(hi.astype(int)),255,-1)
        corners=cv2.goodFeaturesToTrack(a,100,.015,5,mask=mask)
        movement=None
        valid=0
        if corners is not None:
            nxt,ok,_=cv2.calcOpticalFlowPyrLK(a,b,corners,None)
            rev,back,_=cv2.calcOpticalFlowPyrLK(b,a,nxt,None)
            good=(ok&back).ravel().astype(bool)&(np.linalg.norm((rev-corners)[:,0],axis=1)<1)
            valid=int(good.sum())
            if valid:
                movement=float(np.linalg.norm((nxt-corners)[good,0],axis=1).mean())
        rows.append(dict(route=r['route'],index=r['index'],false_positive=r['prediction']>.3,
              bbox_motion_m=float(static_error),geometry_error_px=float(np.linalg.norm(exact-approximate,axis=1).mean()),
              unaligned_px=float(np.linalg.norm(current-exact,axis=1).mean()),tracked_points=valid,
              raw_rgb_shift_px=movement))
    (RUN/'alignment-audit.json').write_text(json.dumps(rows,indent=2))
    for flag in (True,False):
        group=[r for r in rows if '23128' in r['route'] and r['false_positive']==flag]
        print(flag,len(group),{k:float(np.mean([r[k] for r in group if r[k] is not None]))
              for k in ('bbox_motion_m','geometry_error_px','unaligned_px','raw_rgb_shift_px','tracked_points')})


if __name__=='__main__':
    main()
