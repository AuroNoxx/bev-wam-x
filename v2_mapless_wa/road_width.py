"""Carriageway widths from an independent aerial survey, with no simulator client."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d,median_filter,minimum_filter1d,maximum_filter1d
from scipy.spatial import cKDTree


def center_city_route(spec,cloud):
    """Estimate urban carriageway centers between intersections using the road-area ribbon."""
    vertices=np.asarray(spec['xy'],dtype=float)
    if not np.allclose(vertices[0],vertices[-1]):raise ValueError('Expected a closed urban circuit')
    arc=np.r_[0.,np.linalg.norm(np.diff(vertices,axis=0),axis=1).cumsum()]
    station=np.linspace(0,arc[-1],round(arc[-1]/1.5),endpoint=False)
    p=np.stack([np.interp(station,arc,c) for c in vertices.T],axis=1)
    smooth=gaussian_filter1d(p,1.,axis=0,mode='wrap')
    fwd=np.roll(smooth,-1,axis=0)-np.roll(smooth,1,axis=0)
    fwd/=np.linalg.norm(fwd,axis=1,keepdims=True).clip(.01)
    right=fwd[:,::-1]*[-1,1]
    tree=cKDTree(cloud[:,:2])
    offset=np.full(len(p),np.nan)
    widths=np.full(len(p),np.nan)
    junction=np.zeros(len(p),bool)
    for i,(point,d,n) in enumerate(zip(p,fwd,right)):
        q=cloud[tree.query_ball_point(point,34.),:2]-point
        bins=np.unique(np.rint(2*(q[abs(q@d)<2.]@n)))/2
        spans=[v for v in np.split(bins,np.flatnonzero(np.diff(bins)>1.55)+1) if len(v)>5]
        strip=next((v for v in spans if v[0]<=1 and v[-1]>=-1),None)
        if strip is None:continue
        width=strip[-1]-strip[0]
        junction[i]=width>25
        if not 7<=width<=23:continue
        divided=any(1.5<strip[0]-v[-1]<12 and v[-1]-v[0]>5 for v in spans)
        # On the city arterial the visible paved strip also contains curbside
        # parking / lay-bys. Favor the inner portion of the directional half.
        target=strip[0]+(.5 if divided else 2/3)*width
        offset[i]=np.clip(target,-5,5)
        widths[i]=width if divided else width*.5
    valid=np.flatnonzero(np.isfinite(offset))
    if len(valid)<len(p)/4:raise ValueError('Insufficient aerial coverage for a city route')
    def periodic(values):
        return np.interp(station,station[valid],values[valid],period=arc[-1])
    movement=gaussian_filter1d(median_filter(periodic(offset),11,mode='wrap'),2.5,mode='wrap')
    line=p+right*movement[:,None]
    # At a crossing the combined road mask opens sideways into the intersecting
    # street. Continue straight between the independent approaches on each side.
    opened=maximum_filter1d(junction.astype(np.uint8),35,mode='wrap').astype(bool)
    for a in np.flatnonzero(opened & ~np.roll(opened,1)):
        gap=next((k for k in range(1,len(p)) if not opened[(a+k)%len(p)]),None)
        if gap is None:continue
        before=(a-1)%len(p);after=(a+gap)%len(p)
        chord=line[after]-line[before]
        if np.linalg.norm(chord)<1:continue
        direction=chord/np.linalg.norm(chord)
        if min(direction@fwd[before],direction@fwd[after])<.9:continue
        ids=np.arange(a,a+gap)%len(p)
        line[ids]=line[before]+(np.arange(1,gap+1)/(gap+1))[:,None]*chord
    mid=len(line)//2
    first=cv2.approxPolyDP(line[:mid+1].astype(np.float32),.4,False).reshape(-1,2)
    second=cv2.approxPolyDP(np.vstack((line[mid:],line[:1])).astype(np.float32),.4,False).reshape(-1,2)
    simple=np.vstack((first,second[1:]))
    length=np.linalg.norm(np.diff(simple,axis=0),axis=1).sum()
    # Reserve alignment uncertainty for raster digitization and intersections.
    available=np.clip(.5*periodic(widths)-3.,.20,2.5)
    available[~np.isfinite(offset)]=.20
    cap=minimum_filter1d(available,13,mode='wrap')
    ds=np.r_[np.arange(0.,length,8.),length]
    margin=np.stack((ds,np.interp(ds*arc[-1]/length,station,cap,period=arc[-1])),axis=1)
    laps=spec.get('laps') or 1
    limits=np.vstack([margin[:-1]+[k*length,0] for k in range(laps)]+[margin[-1:]+[(laps-1)*length,0]])
    return dict(xy=simple.round(3).tolist(),sd_accuracy=limits.tolist(),
                events=[[s*length/arc[-1],k] for s,k in spec['events']])


def center_undivided_approaches(spec,cloud):
    """Digitize the right-hand half of narrow, undivided approach roads in the aerial survey."""
    xy=np.asarray(spec['xy'],dtype=float)
    arc=np.r_[0.,np.linalg.norm(np.diff(xy,axis=0),axis=1).cumsum()]
    station=np.linspace(0,arc[-1],round(arc[-1]/2.)+1)
    p=np.stack([np.interp(station,arc,c) for c in xy.T],axis=1)
    along=np.gradient(gaussian_filter1d(p,1.,axis=0),axis=0)
    along/=np.linalg.norm(along,axis=1,keepdims=True).clip(.01)
    right=along[:,::-1]*[-1,1]
    tree=cKDTree(cloud[:,:2])
    offset=np.full(len(p),np.nan)
    for i,(center,fwd,normal) in enumerate(zip(p,along,right)):
        nearby=cloud[tree.query_ball_point(center,32.),:2]-center
        bins=np.unique(np.rint(2*(nearby[abs(nearby@fwd)<2.5]@normal)))/2
        spans=[v for v in np.split(bins,np.flatnonzero(np.diff(bins)>1.55)+1) if len(v)>3]
        spans_at_route=[v for v in spans if v[0]<=.5 and v[-1]>=-.5]
        if not spans_at_route:continue
        strip=spans_at_route[0]
        width=strip[-1]-strip[0]
        divided=any(1.5<strip[0]-v[-1]<13 and v[-1]-v[0]>4 for v in spans)
        offset[i]=np.clip(strip[0]+.75*width,-2.,2.) if 7.5<=width<=12.5 and not divided else 0.
    valid=np.flatnonzero(np.isfinite(offset))
    if len(valid)<2:raise ValueError('Insufficient aerial coverage of the access road')
    offset=np.interp(station,station[valid],offset[valid])
    # The first ordinary navigation instruction marks the end of the approach.
    # A narrow one-way ramp beyond that point uses the already surveyed ramp center.
    end=min(s for s,_ in spec['events'])
    taper=np.clip((end-station)/20.,0,1)
    line=p+right*(taper*gaussian_filter1d(median_filter(offset,7),3.))[:,None]
    idx=int(np.searchsorted(station,end))
    first=cv2.approxPolyDP(line[:idx+1].astype(np.float32),.25,False).reshape(-1,2)
    simple=np.vstack((first,xy[arc>station[idx]+.1]))
    new_length=np.linalg.norm(np.diff(simple,axis=0),axis=1).sum()
    ratio=new_length/arc[-1]
    return dict(xy=simple.round(2).tolist(),sd_accuracy=[[s*ratio,v] for s,v in spec['sd_accuracy']],
                events=[[s*ratio,k] for s,k in spec['events']])


def highway_ring_survey(line, cloud):
    """Follow the directional half of a divided highway from road-surface observations."""
    arc = np.r_[0., np.linalg.norm(np.diff(line, axis=0), axis=1).cumsum()]
    station = np.linspace(0., arc[-1], int(arc[-1]/2)+1)
    sketch = np.stack([np.interp(station, arc, axis) for axis in line.T], axis=1)
    tangent = np.gradient(gaussian_filter1d(sketch, 3., axis=0), axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normals = tangent[:, ::-1]*[-1, 1]
    tree = cKDTree(cloud[:, :2])
    entry = cloud[tree.query_ball_point(sketch[0], 15)]
    if not len(entry):
        raise ValueError('Missing highway survey at entry')
    # This ring section stays on the entry deck. Exclude crossing decks.
    level = np.median(entry[:, 2])
    sections = []
    for point, along, right in zip(sketch, tangent, normals):
        samples = cloud[tree.query_ball_point(point, 65)]
        delta = samples[np.abs(samples[:, 2]-level)<2., :2]-point
        counts, edges = np.histogram(delta[np.abs(delta@along)<1.8]@right,
                                     bins=np.arange(-65.25, 65.5, .5))
        occupied = edges[:-1][counts>=2]+.25
        spans = np.split(occupied, np.flatnonzero(np.diff(occupied)>1.5)+1)
        sections.append([(v[0], v[-1]) for v in spans if len(v)>4 and v[0]<0<v[-1]])
    widths = [hi-lo for spans in sections[:30] for lo, hi in spans]
    if not widths:
        raise ValueError('Unable to measure highway width at entry')
    width = np.median(widths)
    indices, centers = [], []
    for i, spans in enumerate(sections):
        matches = [(lo, hi) for lo, hi in spans if abs(hi-lo-width)<2.]
        if len(matches)==1:
            lo, hi = matches[0]
            indices.append(i)
            centers.append(sketch[i]+normals[i]*(lo+.75*(hi-lo)))
    stations, centers = station[indices], np.asarray(centers)
    keep = np.ones(len(indices), bool)
    for _ in range(2):
        for j in np.flatnonzero(keep):
            before = np.flatnonzero(keep & (stations<stations[j]-8))
            after = np.flatnonzero(keep & (stations>stations[j]+8))
            if before.size and after.size:
                a, b = before[-1], after[0]
                t = (stations[j]-stations[a])/(stations[b]-stations[a])
                keep[j] = np.linalg.norm(centers[j]-(centers[a]*(1-t)+centers[b]*t))<=3.
    s, centers = stations[keep], centers[keep]
    if len(s)<4 or s[0]>50 or station[-1]-s[-1]>50 or np.diff(s).max()>250:
        raise ValueError('Insufficient independent highway cross sections')
    xy = np.stack([np.interp(station, s, axis) for axis in centers.T], axis=1)
    for mask, j, k in ((station<s[0], 0, 3), (station>s[-1], -1, -4)):
        xy[mask] = centers[j]+(station[mask, None]-s[j])*(centers[j]-centers[k])/(s[j]-s[k])
    xy[2:-2] = gaussian_filter1d(xy, 1.2, axis=0)[2:-2]
    coarse = cv2.approxPolyDP(xy.astype(np.float32), .5, False).reshape(-1, 2)
    coarse = np.round(2*coarse)/2
    length = float(np.linalg.norm(np.diff(coarse, axis=0), axis=1).sum())
    return coarse.tolist(), [[0., 2.5], [length, 2.5]]


def survey(spec,cloud,town,search_radius=12):
    line=np.array(spec['xy'],float)
    if town=='Town04_Opt' and spec['name']=='town04_ring':
        return highway_ring_survey(line, cloud)
    closed=np.array_equal(line[0],line[-1])
    arc=np.r_[0,np.linalg.norm(np.diff(line,axis=0),axis=1).cumsum()]
    station=np.linspace(0,arc[-1],round(arc[-1]/1.8)+1)
    p=np.stack([np.interp(station,arc,v) for v in line.T],axis=1)
    tangent=np.gradient(gaussian_filter1d(p,1.,axis=0),axis=0)
    tangent/=np.linalg.norm(tangent,axis=1,keepdims=True).clip(.01)
    right=tangent[:,::-1]*[-1,1]
    tree=cKDTree(cloud[:,:2])
    offsets=[];caps=[]
    for pos,d,n in zip(p,tangent,right):
        q=cloud[tree.query_ball_point(pos,34),:2]-pos
        bins=np.unique(np.rint(2*(q[np.abs(q@d)<2.]@n)).astype(int))/2
        spans=[a for a in np.split(bins,np.flatnonzero(np.diff(bins)>1.6)+1) if len(a)>3]
        near=[v for v in spans if v[0]<search_radius and v[-1]>-search_radius]
        if not near:
            offsets.append(0.);caps.append(.2);continue
        v=min(near,key=lambda v:abs(v.mean()) if search_radius<=3 else max(0,v[0],-v[-1])*2+.2*abs(v.mean()))
        width=v[-1]-v[0]
        if width>49 or width<3:
            offsets.append(0.);caps.append(.2);continue
        # A physically divided strip has another ribbon immediately to its left.
        divided=any(1.5<v[0]-q[-1]<12 and q[-1]-q[0]>4 for q in spans)
        ratio=.54 if divided else .76
        if 'interchange' in spec['name'] and width<13:ratio=.55
        target=v[0]+ratio*width
        bound=max(4,search_radius)
        offsets.append(float(np.clip(target,-bound,bound)))
        directional_width=width if divided else width/2
        caps.append(float(np.clip(directional_width*.5-2.2,.18,2.5)))
    # Keep a road-level polyline. Fine aerial samples and pavement edges stay offline.
    off=gaussian_filter1d(median_filter(offsets,size=9),2.)
    refined=p+right*off[:,None]
    poly=cv2.approxPolyDP(refined.astype(np.float32),.55,False).reshape(-1,2)
    poly=np.round(2*poly)/2
    poly=poly[np.r_[True,np.linalg.norm(np.diff(poly,axis=0),axis=1)>.1]]
    if closed and not np.array_equal(poly[0],poly[-1]):poly=np.vstack((poly,poly[0]))
    new_len=np.linalg.norm(np.diff(poly,axis=0),axis=1).sum()
    caps=minimum_filter1d(median_filter(caps,size=13),size=15)
    ds=np.r_[np.arange(0,new_len,10),new_len]
    margin=np.stack((ds,np.interp(ds/new_len*arc[-1],station,caps)),axis=1)
    repeat=spec.get('laps') or 1
    joined=np.vstack([margin[:-1]+[lap*new_len,0] for lap in range(repeat)]+[margin[-1:]+[(repeat-1)*new_len,0]])
    return poly.tolist(),joined.round(3).tolist()


def main(root):
    data=json.loads((root/'missions.json').read_text())
    for town,section in data.items():
        pts=np.concatenate([np.load(f)['xyz'] for f in sorted((root/town).glob('tile_*.npz'))])
        for spec in section['missions']:
            guide=np.array(spec['xy'])
            sel=((pts[:,:2]>guide.min(0)-40)&(pts[:,:2]<guide.max(0)+40)).all(1)
            poly,limits=survey(spec,pts[sel],town)
            spec['xy']=poly;spec['sd_accuracy']=limits
    (root/'road-atlas-final.json').write_text(json.dumps(data,indent=2))


if __name__=='__main__':main(Path(sys.argv[1]))
