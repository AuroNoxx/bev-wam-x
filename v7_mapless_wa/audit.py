"""Measure calibration against live RGB and independent simulator semantic returns."""
import argparse,json
from datetime import datetime
from pathlib import Path
import cv2,numpy as np
from .geometry import X,Y,Z,NAMES,pixels,local,projection_tables

COLORS=((0,220,255),(255,215,45),(55,150,255),(210,65,245),(185,100,255),(250,160,75),(110,210,250))

def p95(values):
    return round(float(np.percentile(values,95)),3) if len(values) else None

def dist(mask):
    return cv2.distanceTransform((~mask.astype(bool)).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)

def paint_mask(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    chroma=np.ptp(image.astype(np.float32),axis=2)
    # Local contrast detects light paint even under wet or sunset exposure.
    return ((gray-cv2.GaussianBlur(gray,(0,0),3)>11)&(gray>80)&(chroma<85)).astype(np.uint8)

def fused(rgb,uv,visible):
    parts=np.stack([cv2.remap(im,grid,None,cv2.INTER_LINEAR) for im,grid in zip(rgb,uv)])
    return np.rint((parts*visible[...,None]).sum(0)/np.maximum(1,visible.sum(0))[...,None]).astype(np.uint8)

def panel(im,caption,width,height):
    canvas=np.full((height+25,width,3),26,np.uint8)
    scale=min(width/im.shape[1],height/im.shape[0])
    view=cv2.resize(im,(round(scale*im.shape[1]),round(scale*im.shape[0])),interpolation=cv2.INTER_NEAREST if im.shape[1]<200 else cv2.INTER_AREA)
    x=(width-view.shape[1])//2;canvas[:view.shape[0],x:x+view.shape[1]]=view
    cv2.putText(canvas,caption,(6,height+17),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1,cv2.LINE_AA)
    return canvas

def draw(out,key,fid,rgb,cal,lidar,labels,ground):
    rows=[np.hstack([panel(rgb[k],NAMES[k],400,228) for k in (1,0,2)])]
    rows.append(np.hstack([panel(fused(rgb,u,v),f'z = {height:+.1f} m',240,170) for height,u,v in zip(Z,cal['uv'],cal['visible'])]))
    road=labels['masks']
    overlay=np.zeros((128,128,3),np.uint8);overlay[road[0]>0]=(45,45,45);overlay[road[3]>0]=(50,65,45)
    overlay[lidar[2]>0]=(200,200,200);overlay[road[2]>0]=(130,52,95)
    # Render on metric axes (equal physical scale per metre).
    meter=cv2.resize(overlay,(256,480),interpolation=cv2.INTER_NEAREST)
    def screen(xy):return np.rint((pixels(xy)+.5)*[2,3.75]-.5).astype(np.int32)
    for j,(line,valid) in enumerate(zip(labels['map_lines'],labels['point_valid'])):
        pts=screen(line)
        for k in np.flatnonzero(valid):
            cv2.circle(meter,pts[k],2,COLORS[j],-1)
            if k and valid[k-1]:cv2.line(meter,pts[k-1],pts[k],COLORS[j],1,cv2.LINE_AA)
    cv2.polylines(meter,[screen(labels['path_gt'])],False,(65,255,80),2,cv2.LINE_AA)
    cv2.drawMarker(meter,screen(np.array([0.,0.])),(255,255,255),cv2.MARKER_CROSS,9,1)
    bottom=np.full((365,1200,3),26,np.uint8)
    lower=panel(meter,'Map / expert / LiDAR',300,335);bottom[:360,:300]=lower
    lower=panel(ground,'IPM ground z=0',275,280);bottom[:305,300:575]=lower
    names=['Ego lane center','Left lane marking','Right lane marking','Left hard boundary','Right hard boundary','Adjacent left center','Adjacent right center']
    for j,(c,name) in enumerate(zip(COLORS,names)):
        y=34+j*23;cv2.line(bottom,(608,y),(636,y),c,3)
        cv2.putText(bottom,name+('  (absent)' if not labels['exists'][j] else ''),(645,y+5),0,.5,(230,230,230),1,cv2.LINE_AA)
    for j,msg in enumerate(['Green: measured future path, ds=1.5 m','White: LiDAR low returns (0.2-1.2 m)','Gaps: no supported ground-truth vertex','Forward up / left left',f'Town10HD_Opt  |  {key}  |  frame {fid}']):
        cv2.putText(bottom,msg,(608,227+j*23),0,.53,(110,250,150) if not j else (230,230,230),1,cv2.LINE_AA)
    rows.append(bottom)
    if not cv2.imwrite(str(out/f'{key}-{fid}.png'),np.vstack(rows)):raise IOError(key)

def inside(pts):
    uv=np.rint(pixels(pts[:,:2])).astype(int)
    keep=((uv>=0)&(uv<128)).all(1)
    return keep,uv[keep]

def main(out):
    reports=out.parent/'verification';reports.mkdir(exist_ok=True)
    cal=dict(np.load(out/'calibration.npz')); sim=json.loads((out/'sim.json').read_text())
    # Independent zero-height ground projection for measuring paint, in addition to five requested slices.
    uv0,vis0=projection_tables(cal['extrinsics'],heights=(0.,))
    uv0=uv0[0];vis0=vis0[0]
    gates={'straight_boundary_cross_section':[],'semantic_static_to_low_lidar':[],'live_semantic_to_survey_static':[],
           'all_paint_to_ego_lane_diagnostic':[],'overlap_rgb_paint':[],'path_raster_quantization':[],'path_arc_error_m':[]}
    cross_sections=0
    sizes=[]; durations=[];episodes=[];pairs=0; occ=np.zeros(5,int); t_sample=[]; total_ticks=0;windows=[]
    static_ids=[sim['tags'][k] for k in ('Poles','GuardRail','Walls','Fences','TrafficSigns')]
    align_roi=(X>=3)&(X<=25)&(abs(Y)<8)
    for ep in sorted(out.glob('*/episode.json')):
        d=json.loads(ep.read_text());folder=ep.parent;frames=d['frames']; assert len(frames)==25 and np.all(np.diff(frames)==1)
        durations.append(d['elapsed']);total_ticks+=d['sim_ticks'];t_sample.extend(d['retained_tick_s'])
        end=ep.stat().st_mtime
        windows.append((end-d['elapsed'],end))
        with np.load(folder/'teacher.npz') as tf:
            sem=tf['sem_xyz'];tag=tf['sem_tag'];fid_arr=np.repeat(tf['sem_fid'],tf['sem_count'])
            trace=tf['pose'];ids=tf['fid']
        pairs+=len(set(frames)&{j-20 for j in frames})
        for n,fid in enumerate(frames):
            fd=folder/f'{fid:07d}'; sample_files=[*fd.glob('*.png'),fd/'index.npz',fd/'input.npz',fd/'labels.npz',fd/'lidar.npz']
            sizes.append(sum(f.stat().st_size for f in sample_files))
            rgb=[cv2.imread(str(fd/(name+'.png'))) for name in NAMES]; assert all(im.shape==(224,384,3) for im in rgb)
            with np.load(fd/'labels.npz') as f: lab=dict(f)
            with np.load(fd/'input.npz') as f: assert set(f.files)=={'sd_nav','cmd_one_hot'} and f['sd_nav'].shape==(3,2) and np.isfinite(f['sd_nav']).all()
            with np.load(fd/'lidar.npz') as f: lidar=f['bev']; assert lidar.shape==(6,128,128) and lidar.dtype==np.float16
            assert lab['map_lines'].shape==(7,20,2) and lab['exists'].shape==(7,) and lab['path_gt'].shape==(40,2)
            assert lab['masks'].shape==(4,128,128) and lab['p_occ'].shape==(5,128,128)
            occ+=(lab['p_occ'].sum((1,2))>0)
            frame_i=int(fid-ids[0]);pose=trace[frame_i];take=fid_arr==fid
            pts=local(sem[take],pose); tags=tag[take]
            low=pts[np.isin(tags,static_ids)&(pts[:,2]>=.2)&(pts[:,2]<1.2)]
            _,pix=inside(low)
            if len(pix):
                gates['semantic_static_to_low_lidar'].extend(dist(lidar[2]>0)[pix[:,1],pix[:,0]].tolist())
                gates['live_semantic_to_survey_static'].extend(dist(lab['masks'][2])[pix[:,1],pix[:,0]].tolist())
            # The unassociated paint population also contains neighboring lines and crosswalks.
            paint_pts=pts[(tags==sim['tags']['RoadLines'])&(pts[:,0]>3)&(pts[:,0]<25)&(abs(pts[:,1])<7)]
            if d['scenario']=='straight':
                for j,sign in ((1,1),(2,-1)):
                    for q,center,ok in zip(lab['map_lines'][j],lab['map_lines'][0],lab['point_valid'][j]):
                        if not ok or not 4<q[0]<20:continue
                        cross_sections+=1
                        # Assign paint to the closest divider, using half the distance to
                        # the adjacent divider (half lane width), independent of the 1-cell gate.
                        candidates=paint_pts[(abs(paint_pts[:,0]-q[0])<.5)&(sign*(paint_pts[:,1]-center[1])>0)&
                                             (abs(paint_pts[:,1]-q[1])<abs(q[1]-center[1]))]
                        if len(candidates):gates['straight_boundary_cross_section'].append(float(np.min(abs(candidates[:,1]-q[1]))/.3))
            _,pix=inside(paint_pts)
            if len(pix):
                line_mask=np.zeros((128,128),np.uint8)
                for j in (1,2):
                    vs=lab['point_valid'][j]
                    ps=np.rint(pixels(lab['map_lines'][j])).astype(np.int32)
                    for t in range(1,len(ps)):
                        if vs[t-1] and vs[t]:cv2.line(line_mask,ps[t-1],ps[t],1)
                if line_mask.any():
                    distance=dist(line_mask)[pix[:,1],pix[:,0]]
                    gates['all_paint_to_ego_lane_diagnostic'].extend(distance.tolist())
            cam_paint=[]
            for im,grid,visible in zip(rgb,uv0,vis0):
                m=cv2.remap(paint_mask(im),grid,None,cv2.INTER_NEAREST)>0
                cam_paint.append(m&visible)
            for j in (1,2):
                overlap=align_roi&vis0[0]&vis0[j]
                if len(pix):
                    observed=np.zeros((128,128),np.uint8);observed[pix[:,1],pix[:,0]]=1
                    overlap &= cv2.dilate(observed,np.ones((3,3),np.uint8)).astype(bool)
                else:continue
                for a,b in ((0,j),(j,0)):
                    origin=cam_paint[a]&overlap
                    if origin.any():gates['overlap_rgb_paint'].extend(dist(cam_paint[b])[origin].tolist())
            xy=lab['path_gt']
            sub=pixels(xy);gates['path_raster_quantization'].extend(np.linalg.norm(sub-np.rint(sub),axis=1).tolist())
            # Independent sequential walk along actual odometry, stopping at each 1.5 m crossing.
            lengths=0.;target=1
            for a,b in zip(trace[frame_i:,:3,3],trace[frame_i+1:,:3,3]):
                seg=np.linalg.norm(b[:2]-a[:2])
                while target<=40 and lengths+seg>=1.5*target:
                    point=a+(b-a)*(1.5*target-lengths)/seg
                    gates['path_arc_error_m'].append(float(np.linalg.norm(local(point,pose)[:2]-xy[target-1])))
                    target+=1
                lengths+=seg
                if target==41:break
            assert target==41
            if n==12 or (ep.parent.name=='fork' and n==24):draw(reports,ep.parent.name,fid,rgb,cal,lidar,lab,fused(rgb,uv0,vis0))
        subset=trace[np.isin(ids,frames)]
        yaw_range=float(np.ptp(np.rad2deg(np.unwrap(np.arctan2(subset[:,1,0],subset[:,0,0])))))
        episodes.append(dict(retained_yaw_range_deg=yaw_range,**{k:d[k] for k in ('scenario','weather','npc_count','perturb_m','episode_group','planned_peak_kappa_first_46m','first_choices_m','driven_m','sim_ticks')}))
    # Near zones in FLU: outside nominal bumper x=2.5 plus side strips along front half.
    visible=vis0.any(0)
    cover={}
    for name,zone in {'front_0_6':(X>=0)&(X<6)&(abs(Y)<1.),'front_beyond_bumper':(X>=2.5)&(X<6)&(abs(Y)<1.),
                      'left_adjacent':(X>=0)&(X<6)&(Y>=1)&(Y<3.5),'right_adjacent':(X>=0)&(X<6)&(Y<=-1)&(Y>-3.5)}.items():
        cover[name]=dict(visible=int((zone&visible).sum()),cells=int(zone.sum()))
    gpu=[json.loads(i) for f in out.glob('gpu*.jsonl') for i in f.read_text(encoding='utf-8-sig').splitlines() if i.strip()]
    def in_episode(g):
        ts=datetime.fromisoformat(g['utc'].split('.')[0]+'+00:00').timestamp()
        return any(start<=ts<=end for start,end in windows)
    gpu=[g for g in gpu if in_episode(g)]
    vram=np.array([g['dedicated_bytes']/2**30 for g in gpu if g['instances'] and g['dedicated_bytes']])
    count=len(sizes); offline=float((out/'offline-seconds.txt').read_text())
    scores={k:dict(n=len(v),p95=p95(v),below_one=float(np.mean(np.array(v)<1)) if v else None) for k,v in gates.items() if v}
    metrics=dict(sim=sim,episodes=episodes,count=count,pairs_1s=pairs,nonempty_occ_frames=occ.tolist(),coverage=cover,alignment=scores,
                 straight_boundary_cross_sections_tested=cross_sections,
                 mean_bytes=float(np.mean(sizes)),size_40k_gib=np.mean(sizes)*40000/2**30,
                 gpu_samples=len(vram),gpu_dedicated_peak_gib=float(vram.max()),gpu_dedicated_median_gib=float(np.median(vram)),
                 wall_sim_fps=total_ticks/sum(durations),retained_fps=count/sum(t_sample),save_and_label_fps=count/(sum(t_sample)+offline),
                 four_short_episodes_fps=count/(sum(durations)+offline),offline_s=offline,
                 hours_40k_stream=40000*(sum(t_sample)+offline)/count/3600,hours_40k_short=40000*(sum(durations)+offline)/count/3600)
    (reports/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('sim','episodes')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);main(p.parse_args().out)
