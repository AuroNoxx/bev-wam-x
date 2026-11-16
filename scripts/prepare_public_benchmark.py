"""Build small real-data training windows with explicit source-grid semantics."""
from pathlib import Path
import gzip,json,hashlib,time
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
CAMS=['FRONT','FRONT_LEFT','FRONT_RIGHT','BACK','BACK_LEFT','BACK_RIGHT']
HISTORY=[-6,-4,-2,0]
FUTURE=[10,20,30,40,50]


def compact(occ,seg):
    # Zero is source-unfilled, NOT verified free. 255 is unknown/invalid.
    a=np.zeros(occ.shape,np.uint8)
    a[occ==2]=255
    hit=occ==1
    a[hit]=3  # static/other
    a[hit & np.isin(seg,[1,24,25,27])]=1  # ground/road
    a[hit & np.isin(seg,[9,10])]=2  # vegetation/terrain
    a[hit & np.isin(seg,[14,15,16,17,18,19,21])]=4  # vehicle/dynamic
    a[hit & np.isin(seg,[12,13])]=5
    # Conservative union within five original height cells, dynamic priority.
    blocks=a.reshape(128,128,4,5)
    out=np.where(blocks==255,0,blocks).max(-1)
    out[(blocks==255).all(-1)]=255
    return out.transpose(2,0,1)


def anchors():
    z,y,x=np.meshgrid(np.arange(4),np.arange(128),np.arange(128),indexing='ij')
    return np.stack([-25.6+(x+.5)*.4,-25.6+(y+.5)*.4,-5+(z+.5)*2.,np.ones_like(x)],-1).reshape(-1,4)


def align_history(grid,world2source,ego2world):
    p=anchors() @ (world2source @ ego2world).T
    x=np.floor((p[:,0]+25.6)/.4).astype(int)
    y=np.floor((p[:,1]+25.6)/.4).astype(int)
    z=np.floor((p[:,2]+5)/2).astype(int)
    valid=(x>=0)&(x<128)&(y>=0)&(y<128)&(z>=0)&(z<4)
    out=np.full(len(p),255,np.uint8)
    out[valid]=grid[z[valid],y[valid],x[valid]]
    return out.reshape(4,128,128)


def main():
    started=time.monotonic()
    output=ROOT/'data/benchmark_cache'
    output.mkdir(exist_ok=True)
    clips=sorted((ROOT/'data/public/bench2drive/clips').iterdir())
    audit=[]
    for clip_index,clip in enumerate(clips):
        split='train' if clip_index==0 else 'validation'
        target=output/split
        target.mkdir(exist_ok=True)
        annos=[json.load(gzip.open(p,'rt')) for p in sorted((clip/'anno').glob('*.gz'))]
        original_annotation_count=len(annos)
        matched=set(range(len(annos))) & {int(p.stem) for p in (clip/'occ').glob('*.npz')}
        for cam in CAMS:
            matched &= {int(p.stem) for p in (clip/'camera'/('rgb_'+cam.lower())).glob('*.jpg')}
        excluded=sorted(set(range(len(annos)))-matched)
        if sorted(matched)!=list(range(len(matched))):
            raise RuntimeError('Non-contiguous source frames; explicit gap-aware windowing required')
        annos=annos[:len(matched)]
        poses=[np.array(next(b for b in a['bounding_boxes'] if b['class']=='ego_vehicle')['world2ego']) for a in annos]
        grids=[]
        for i in range(len(annos)):
            with np.load(clip/'occ'/f'{i:05d}.npz') as a:
                assert a['occ'].shape==(128,128,20)
                grids.append(compact(a['occ'],a['seg']))
        count=0
        for i in range(6,len(annos)-50):
            annotation=annos[i]
            ego2world=np.linalg.inv(poses[i])
            history=np.stack([align_history(grids[i+h],poses[i+h],ego2world) for h in HISTORY])
            future=np.stack([grids[i+f] for f in FUTURE])
            trajectory=[]
            for offset in range(2,51,2):
                rel=poses[i] @ np.linalg.inv(poses[i+offset])
                yaw=-np.arctan2(rel[1,0],rel[0,0])
                trajectory.append([rel[0,3]/50,-rel[1,3]/10,np.sin(yaw),np.cos(yaw),annos[i+offset]['speed']/30])
            goal_world=np.array([annotation['x_command_far'],annotation['y_command_far'],ego2world[2,3],1.])
            goal=poses[i] @ goal_world
            state=np.array([annotation['speed']/30,annotation['acceleration'][0]/10,annotation['acceleration'][1]/10,
                            goal[0]/50,-goal[1]/50,annotation['command_far']/6],np.float32)
            rgb=[]; projection=[]; visible=[]
            for cam in CAMS:
                sensor=annotation['sensors']['CAM_'+cam]
                image=Image.open(clip/'camera'/('rgb_'+cam.lower())/f'{i:05d}.jpg').convert('RGB').resize((448,256),Image.Resampling.BILINEAR)
                rgb.append(np.asarray(image))
                p=anchors() @ (np.array(sensor['world2cam']) @ ego2world).T
                k=np.array(sensor['intrinsic'])
                u=k[0,0]*p[:,1]/np.maximum(p[:,0],1e-6)+k[0,2]
                v=k[1,1]*(-p[:,2])/np.maximum(p[:,0],1e-6)+k[1,2]
                u=u/sensor['image_size_x']*448; v=v/sensor['image_size_y']*256
                valid=(p[:,0]>.1)&(u>=0)&(u<=447)&(v>=0)&(v<=255)
                projection.append(np.stack([u/447*2-1,v/255*2-1],-1).astype(np.float32))
                visible.append(valid)
            filename=f'{clip_index:02d}_{i:05d}.npz'
            np.savez_compressed(target/filename,history=history,future=future,current=grids[i],
                                trajectory=np.array(trajectory,np.float32),state=state,rgb=np.stack(rgb),
                                projection=np.clip(np.stack(projection),-3,3),visible=np.stack(visible),
                                controls=np.array([[annos[i+j][k] for k in ['throttle','steer','brake']] for j in range(2,51,2)],np.float32))
            count+=1
        audit.append({'clip':clip.name,'split':split,'frames':len(annos),'annotation_frames':original_annotation_count,'unpaired_frame_ids':excluded,'usable_windows':count,
                      'maximum_speed_m_s':max(a['speed'] for a in annos)})
        print(json.dumps(audit[-1]),flush=True)
    report={'source':'Bench2Drive-V0.0.4','clips':audit,'history_seconds':.6,'future_seconds':5,
            'original_occ_shape_yxz':[128,128,20],'original_voxel_m':.4,
            'benchmark_occ_shape_zyx':[4,128,128],'height_pooling':'5 original voxels -> one 2m layer, occupied class priority',
            'xy_range_m':[-25.6,25.6],'z_range_m':[-5,3],
            'class_names':['source_unfilled_NOT_verified_free','road_ground','vegetation_terrain','static_other','vehicle_dynamic','pedestrian_rider'],
            'unknown_label':255,'history_reference':'current ego frame','future_reference':'each future ego frame conditioned on executed candidate trajectory',
            'notes':['Short throughput pilot, not a trained driving policy','No source self-collected samples','Separate routes for training and validation','5-second waypoints can extend beyond available OCC range','No dense visibility/free-space mask in these source NPZs'],
            'preprocess_seconds':time.monotonic()-started}
    (output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PREPARED',json.dumps(report),flush=True)


if __name__=='__main__': main()
