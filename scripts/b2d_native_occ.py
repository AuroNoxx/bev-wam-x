"""Native Bench2Drive OCC input. No spatial downsample or enlarged canvas.

Nearest-neighbor rigid alignment is not new observation. Out-of-source cells
stay unknown. Source provenance/temporal causality remains uncertified upstream.
"""
from pathlib import Path
import gzip,json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
SHAPE=(20,128,128)

def semantic_grid(occ,seg):
    if occ.shape!=(128,128,20) or seg.shape!=occ.shape: raise ValueError('native shape mismatch')
    if not np.isin(occ,[0,1,2]).all(): raise ValueError('unknown source occupancy code')
    labels=np.zeros(occ.shape,np.uint8) # unfilled, NOT verified-free
    labels[occ==2]=255
    hit=occ==1;labels[hit]=3
    for cls,ids in [(1,[1,25,27]),(2,[9,10]),(4,[14,15,16,17,18,19,21]),(5,[12,13]),(6,[24]),(7,[2,28])]:
        labels[hit & np.isin(seg,ids)]=cls
    # Native source storage Y/X/Z -> internal Z/Y/X; Y is still CARLA right here.
    return labels.transpose(2,0,1)

def align_native(grid,world2source,current2world):
    z,y,x=np.indices(SHAPE)
    # Fixed current ego FLU canvas, same native extent and 0.4m spacing.
    points=np.stack([-25.6+(x+.5)*.4,25.6-(y+.5)*.4,-5+(z+.5)*.4,np.ones_like(x)],-1).reshape(-1,4)
    p=points@(world2source@current2world).T
    ix=np.floor((p[:,0]+25.6)/.4).astype(np.int64)
    iy=np.floor((p[:,1]+25.6)/.4).astype(np.int64)
    iz=np.floor((p[:,2]+5)/.4).astype(np.int64)
    valid=(ix>=0)&(ix<128)&(iy>=0)&(iy<128)&(iz>=0)&(iz<20)
    out=np.full(len(ix),255,np.uint8);out[valid]=grid[iz[valid],iy[valid],ix[valid]]
    return out.reshape(SHAPE),valid.reshape(SHAPE)

def load_sample(cache_name,include_labels=True):
    parts=Path(cache_name).parts
    if len(parts)!=4 or parts[0]!='cache' or parts[1] not in ['train','validation'] or '..' in parts:
        raise ValueError('only canonical train/validation cache paths are allowed')
    path=ROOT/'data/occ_v3'/cache_name
    with np.load(path,allow_pickle=False) as f:
        assert int(f['preprocess_version'])==3
        frame=int(f['frame']);world2current=f['world2ego'].copy()
        # Reuse scalar state/trajectory contract, NEVER old resampled OCC.
        state=f['state'].copy();trajectory=f['trajectory'].copy() if include_labels else None
        h=f['history_pose_ids'].copy();future=f['future_pose_ids'][:2].copy() if include_labels else []
        assert np.array_equal(f['history_occ_ids'],h-1)
        assert np.array_equal(h,frame+np.array([-6,-4,-2,0])), 'history cadence mismatch'
        assert (h<=frame).all()
        assert world2current.shape==(4,4) and np.isfinite(world2current).all()
        assert state.shape==(20,) and np.isfinite(state).all()
        if include_labels:
            assert np.array_equal(f['future_occ_ids'][:2],future-1)
            assert np.array_equal(future,frame+np.array([10,20])), 'future target cadence mismatch'
            assert (future>frame).all()
            assert trajectory.shape==(25,5) and np.isfinite(trajectory).all()
    clip=ROOT/'data/public/bench2drive/clips'/Path(cache_name).parent.name
    result={'state':state};sources=[]
    for key,ids in [('history',h)]+([('future',future)] if include_labels else []):
        grids=[];masks=[]
        for i in ids:
            anno=clip/'anno'/f'{i:05d}.json.gz';source=clip/'occ'/f'{i-1:05d}.npz'
            with gzip.open(anno,'rt') as f: a=json.load(f)
            pose=np.asarray(next(x for x in a['bounding_boxes'] if x['class']=='ego_vehicle')['world2ego'])
            with np.load(source,allow_pickle=False) as f: grid=semantic_grid(f['occ'],f['seg'])
            aligned,valid=align_native(grid,pose,np.linalg.inv(world2current))
            if i==frame: assert np.array_equal(aligned,grid[:,::-1,:]),'identity current-frame FLU conversion failed'
            grids.append(aligned);masks.append(valid)
            sources.append({'role':key,'pose_id':int(i),'occ_id':int(i-1),'path':str(source.relative_to(ROOT)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
        result[key]=np.stack(grids)
        result[key+'_source_support']=np.stack(masks)
    if include_labels: result['trajectory']=trajectory
    tensors={k:torch.from_numpy(v.copy()) for k,v in result.items()}
    return tensors,{'sources':sources,'shape_zyx':SHAPE,'voxel_xyz':[.4,.4,.4],'bounds_xyz':[[-25.6,25.6],[-25.6,25.6],[-5,3]],'spatial_downsample':False,'expanded_canvas':False,'source_causality_certified':False,'raw_semantics':'source files untouched; occupied classes grouped into existing 7 semantic classes; 0 unfilled,255 unknown'}


