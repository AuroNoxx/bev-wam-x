"""Digitize road-level routes from aerial road area and independently sketched missions."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


def sample(line,stations):
    arc=np.r_[0,np.linalg.norm(np.diff(line,axis=0),axis=1).cumsum()]
    return np.stack([np.interp(stations,arc,x) for x in line.T],axis=1)


def digitize(xy,cloud,town,freeway):
    guide=np.asarray(xy,float)
    distance=np.linalg.norm(np.diff(guide,axis=0),axis=1).sum()
    p=sample(guide,np.arange(0,distance+2,4))
    tangent=np.gradient(gaussian_filter1d(p,1.5,axis=0),axis=0)
    tangent/=np.linalg.norm(tangent,axis=1,keepdims=True).clip(.01)
    right=tangent[:,::-1]*[-1,1]
    # Query aerial ground returns in a narrow longitudinal window. Retain alternative
    # carriageways until the whole route is matched, particularly near interchanges.
    tree=cKDTree(cloud[:,:2])
    candidates=[]
    prior=[]
    previous=None
    for i,point in enumerate(p):
        delta=cloud[tree.query_ball_point(point,82),:2]-point
        lateral=delta[abs(delta@tangent[i])<2.8]@right[i]
        bins=np.unique(np.rint(lateral).astype(int))
        spans=np.split(bins,np.flatnonzero(np.diff(bins)>2)+1)
        choices=[]
        for span in spans:
            if len(span)<3:continue
            width=span[-1]-span[0]+1
            if width>57:continue
            # A divided highway may render as a single broad paved strip. On
            # undivided streets use the right half; on narrow ramps use the center.
            frac=.74 if town=='Town10HD_Opt' or width>=22 or width<=12 or not freeway else .64
            loc=span[0]+frac*(width-1)
            merit=.025*abs(loc)+(max(0,17-width)*.20 if freeway else 0)
            choices.append((loc,merit))
        vals=np.arange(-80.,81.,1.)
        # Bridges, trees and intersections interrupt visible pavement. Permit a
        # continuous surveyed road through short gaps without snapping to a ramp.
        obs=np.full(len(vals),4.5)
        for loc,merit in choices:
            obs=np.minimum(obs,merit+.24*np.abs(vals-loc))
        if previous is None:
            cost=obs;back=np.zeros(len(vals),int)
        else:
            shift=np.abs(vals[:,None]-vals[None,:])
            total=cost[None,:]+.10*shift+1.5*shift**2
            back=total.argmin(1)
            cost=obs+total[np.arange(len(vals)),back]
        candidates.append(vals);prior.append(back);previous=vals
    choice=cost.argmin()
    offset=[]
    for vals,back in zip(reversed(candidates),reversed(prior)):
        offset.append(vals[choice]);choice=back[choice]
    corrected=p+right*gaussian_filter1d(np.array(offset[::-1]),1.4)[:,None]
    corrected=gaussian_filter1d(corrected,.8,axis=0,mode='nearest')
    line=cv2.approxPolyDP(corrected.astype(np.float32),.95,False).reshape(-1,2)
    line=np.round(line)
    return line[np.r_[True,np.linalg.norm(np.diff(line,axis=0),axis=1)>0]].tolist()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path);a=p.parse_args()
    data=json.loads((a.root/'sketches.json').read_text())
    for town,section in data.items():
        folder=a.root/town
        files=sorted(folder.glob('tile_*.npz'))
        points=np.concatenate([np.load(f)['xyz'] for f in files])
        for mission in section['missions']:
            verts=np.array(mission['xy'])
            subset=points[((points[:,:2]>verts.min(0)-85)&(points[:,:2]<verts.max(0)+85)).all(1)]
            line=digitize(verts,subset,town,town=='Town12' and 'rural' not in mission['name'])
            if mission.get('laps'):line[-1]=line[0]
            mission['xy']=line
            mission['source']='nadir-road-area'
            mission['georef']=json.loads((folder/'georef.json').read_text())
            mission['events']=[]
            if 'exit' in mission['name'] or 'interchange' in mission['name']:
                # Store the explicit route instruction separately from tactical intent.
                arc=np.r_[0,np.linalg.norm(np.diff(line,axis=0),axis=1).cumsum()]
                target=verts[min(len(verts)-2,2)]
                mission['events']=[[float(arc[np.linalg.norm(line-target,axis=1).argmin()]),'exit']]
            print(town,mission['name'],len(line),flush=True)
    (a.root/'missions.json').write_text(json.dumps(data,indent=2))


if __name__=='__main__':main()
