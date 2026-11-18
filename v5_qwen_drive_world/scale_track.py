"""RGB-only scale tracking. Fit a metric size prior on the designated train roads."""
import json
from functools import lru_cache
import cv2
import numpy as np
from .scale_geometry import RUN,ROOT,displacement


@lru_cache(maxsize=12)
def image(path):
    data=cv2.imread(str(ROOT/path),cv2.IMREAD_GRAYSCALE)
    if data is None:raise FileNotFoundError(path)
    return data


def track(paths,box):
    current=image(paths[-1])
    x1,y1,x2,y2=box
    w,h=x2-x1,y2-y1
    mask=np.zeros_like(current)
    cv2.rectangle(mask,(round(x1+.12*w),round(y1+.12*h)),(round(x2-.12*w),round(y2-.12*h)),255,-1)
    pts=cv2.goodFeaturesToTrack(current,180,.01,4,mask=mask)
    if pts is None:return None
    origin=pts.copy()
    for path in paths[-2::-1]:
        past=image(path)
        dest,ok,_=cv2.calcOpticalFlowPyrLK(current,past,pts,None,winSize=(25,25),maxLevel=4)
        back,rev,_=cv2.calcOpticalFlowPyrLK(past,current,dest,None,winSize=(25,25),maxLevel=4)
        valid=(ok&rev).ravel().astype(bool)&(np.linalg.norm((back-pts)[:,0],axis=1)<1.5)
        if valid.sum()<6:return None
        origin,pts,current=origin[valid],dest[valid],past
    affine,inliers=cv2.estimateAffinePartial2D(origin[:,0],pts[:,0],method=cv2.RANSAC,ransacReprojThreshold=2)
    if affine is None or inliers.sum()<6:return None
    cx,cy=affine@[(x1+x2)/2,(y1+y2)/2,1]
    scale=float(np.linalg.norm(affine[:,0]))
    if scale<=0:return None
    return [cx-w*scale/2,cy-h*scale/2,cx+w*scale/2,cy+h*scale/2],int(inliers.sum())


def iou(a,b):
    area=lambda k:max(0,k[2]-k[0])*max(0,k[3]-k[1])
    inter=area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])
    return inter/(area(a)+area(b)-inter)


def metrics(rows,key):
    known=[r for r in rows if r[key] is not None]
    y=np.array([r['target'] for r in known])
    p=np.array([r[key] for r in known])
    tp=int(((p>.3)&(y>.3)).sum())
    total_positive=sum(r['target']>.3 for r in rows)
    return dict(total=len(rows),valid=len(known),tp=tp,fp=int(((p>.3)&(y<=.3)).sum()),positive=total_positive,
                precision=tp/max(1,int((p>.3).sum())),recall=tp/max(1,total_positive),
                mae=float(abs(p-y).mean()) if len(y) else None,
                zero_same_coverage=float(abs(y).mean()) if len(y) else None)


def main():
    cv2.setNumThreads(3)
    cv2.setRNGSeed(9147)
    sets=json.loads((RUN/'geometry.json').read_text())['splits']
    boxes=json.loads((RUN/'rgb-boxes.json').read_text())['predictions']
    widths=[]
    for r in sets['fit']:
        box=boxes[r['paths'][-1]]
        if box:
            b=box['box']
            widths.append((b[2]-b[0])*r['labels'][-1]['near_depth']/r['cameras'][-1]['intrinsic'][0][0])
    width=float(np.median(widths))
    result={'width_m':width,'width_fit_count':len(widths),'splits':{}}
    for split,records in sets.items():
        rows=[]
        for r in records:
            b=boxes[r['paths'][-1]]
            cur=r['labels'][-1];prior=r['labels'][0]
            row={k:r[k] for k in ('route','index','branch','target','oracle_center_depth','oracle_near_depth','oracle_box_width_prior')}
            row.update(rgb=None,rgb_true_depth=None,box_iou=None,depth_error=None,inliers=0)
            if b:
                box=b['box']
                obs=track(r['paths'],box)
                f=r['cameras'][-1]['intrinsic'][0][0]
                depth=width*f/(box[2]-box[0])
                row.update(box_iou=iou(box,cur['box']),depth_error=depth-cur['near_depth'])
                if obs:
                    prev,n=obs
                    # Keep teacher-assisted intermediate diagnostics out of the RGB estimator.
                    row.update(rgb=displacement(prev,box,depth,r),rgb_true_depth=displacement(prev,box,cur['near_depth'],r),inliers=n)
            rows.append(row)
        result['splits'][split]=dict(windows=rows,metrics={k:metrics(rows,k) for k in
                                      ('oracle_center_depth','oracle_near_depth','oracle_box_width_prior','rgb_true_depth','rgb')},
                                      groups={g:metrics([r for r in rows if r['branch']==g],'rgb') for g in sorted({r['branch'] for r in rows})})
        print(split,result['splits'][split]['metrics']['rgb'],flush=True)
    (RUN/'scale-results.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
