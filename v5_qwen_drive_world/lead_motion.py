"""Ordered, ego-aligned native front-camera features and label-side lead displacement."""
import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from .geometry_repair_features import ROOT,BASE,RUN,RealData,load_base,load_actor,bev_grid,torch,math_attention,F
from .real_data_geometry import ego

DEST=RUN/'lead-motion'
CONFIG=json.loads((ROOT/'configs/v5-qwen-drive-world/lead-motion.json').read_text())


@lru_cache(maxsize=128)
def frame(route,index):
    path=ROOT/'data/public/bench2drive/clips'/route/'anno'/f'{index:05d}.json.gz'
    with gzip.open(path,'rt',encoding='utf8') as stream:
        return json.load(stream)


def displacement(route,index):
    now=frame(route,index)
    transform=np.array(ego(now)['world2ego'])
    leads=[]
    for b in now['bounding_boxes']:
        if b['class']=='vehicle':
            x,y,*_=transform@np.r_[b['location'],1.]
            if 0<x<35 and abs(y)<2:
                leads.append((x,b['id'],b['location']))
    if not leads:
        return None
    _,identity,position=min(leads)
    past=frame(route,index-10)
    for b in past['bounding_boxes']:
        if b['id']==identity and b['class']=='vehicle':
            return float(transform[0,:3]@(np.array(position)-b['location']))
    return None


@torch.no_grad()
def observation(base,actor_path,cameras=None,directed=False):
    scene,actor=load_actor(actor_path)
    return front_observation(base,scene.views['<FRONT VIEW>'],actor,cameras,directed)


@torch.no_grad()
def front_observation(base,frames,actor,cameras=None,directed=False):
    processor=base.model.processor
    assert len(frames)==4
    patches=[];shapes=[]
    for i,f in enumerate(frames):
        budget=processor.config.current_image_pixels if i==3 else processor.config.history_image_pixels
        patch,(h,w)=processor._patchify(f,budget)
        patches.append(patch);shapes.append((1,h,w))
    with math_attention(),torch.autocast('cuda',dtype=torch.bfloat16):
        tokens=base.model.vlm.model.visual(torch.cat(patches).cuda(),grid_thw=torch.tensor(shapes,device='cuda')).pooler_output
    if cameras is None:
        cal=actor['camera_calibration']
        if isinstance(cal,str):
            cal=json.loads((ROOT/cal).read_text())
    maps=[]
    native_views=[]
    grids=[]
    for i,(piece,(_,h,w)) in enumerate(zip(tokens.split([h*w//4 for _,h,w in shapes]),shapes)):
        if cameras is None:
            grid,weight=bev_grid(cal['CAM_FRONT'],np.array(actor['world_from_ego']),
                                 actor['history'][5*i] if i<3 else None)
        else:
            # Points stay in the current ego frame. Each image owns its exposure-time extrinsics.
            grid,weight=bev_grid(cameras[i],np.array(actor['world_from_ego']))
        grid=grid[:,12:20,9:24]
        native=piece.T.reshape(1,-1,h//2,w//2).float()
        if directed:
            native_views.append(native)
            grids.append((grid,weight[:,12:20,9:24]))
        warped=F.grid_sample(native.expand(2,-1,-1,-1),grid,align_corners=False)
        maps.append(F.normalize(warped,dim=1)*weight[:,None,12:20,9:24].gt(0))
    # Within each chronological pair retain spatial change at two heights.
    signals=[]
    for a,b in ((0,1),(1,2),(2,3),(1,3)):
        diff=(maps[a]-maps[b]).square().sum(1)
        both=(grids[a][1]*grids[b][1]).gt(0) if directed else None
        signals.append(F.adaptive_avg_pool2d(diff*both if directed else diff,(4,5)).flatten())
        if directed:
            # Evaluate signed longitudinal shifts in the past camera. Interpolate
            # in the native feature image using exact perspective coordinates.
            transform=np.asarray(cameras[a]['world2cam'])@np.asarray(actor['world_from_ego'])
            intr=np.asarray(cameras[a]['intrinsic'])
            yy,xx=np.meshgrid(-19.2+(np.arange(12,20)+.5)*1.2,-16+(np.arange(9,24)+.5)*2,indexing='ij')
            offsets=np.array([-2.,-.5,.5,2.,4.])*(b-a)*.5
            x=np.broadcast_to(xx,(len(offsets),2,*xx.shape))-offsets[:,None,None,None]
            xyz=np.stack([x,np.broadcast_to(-yy,x.shape),np.broadcast_to(np.array([.35,1.3])[None,:,None,None],x.shape),np.ones_like(x)],-1)
            camera=xyz@transform.T
            depth=camera[...,0]
            u=intr[0,2]+intr[0,0]*camera[...,1]/depth.clip(.01)
            v=intr[1,2]-intr[1,1]*camera[...,2]/depth.clip(.01)
            w_img,h_img=cameras[a]['image_size_x'],cameras[a]['image_size_y']
            coord=np.stack([2*(u+.5)/w_img-1,2*(v+.5)/h_img-1],-1)
            allowed=(depth>.2)&(u>=0)&(u<w_img)&(v>=0)&(v<h_img)
            grid=torch.as_tensor(coord.clip(-2,2),device=native.device,dtype=torch.float32).flatten(0,1)
            shifted=F.normalize(F.grid_sample(native_views[a].expand(len(grid),-1,-1,-1),grid,align_corners=False),dim=1)
            cost=(shifted.unflatten(0,(len(offsets),2))-maps[b][None]).square().sum(2)
            gain=(diff[None]-cost)*both*torch.as_tensor(allowed,device=native.device)
            signals.append(F.adaptive_avg_pool2d(gain,(4,5)).flatten())
    return torch.cat(signals).cpu()


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--full-pose',action='store_true')
    parser.add_argument('--directed',action='store_true')
    args=parser.parse_args()
    dest=DEST/'directed' if args.directed else DEST/'full-pose' if args.full_pose else DEST
    dest.mkdir(exist_ok=True)
    if (dest/'labels-and-split.json').exists():
        raise FileExistsError(f'Feature set already exists: {dest}')
    data=RealData('subset2000')
    records=[r for r in data.records if r['route'].startswith('HardBreakRoute') and r['speed_kmh']<=3]
    dev=json.loads((RUN/'motion-profile/identity.json').read_text())['development_routes']
    torch.set_num_threads(4)
    base=load_base(data)
    features=[];rows=[]
    for r in records:
        d=displacement(r['route'],r['index'])
        if d is None:
            continue
        cameras=None
        if args.full_pose or args.directed:
            cameras=[]
            for offset in (-15,-10,-5,0):
                sensor=frame(r['route'],r['index']+offset)['sensors']['CAM_FRONT']
                cameras.append({k:sensor[k] for k in ('world2cam','intrinsic','image_size_x','image_size_y')})
        features.append(observation(base,r['actor_path'],cameras,directed=args.directed))
        rows.append(dict(route=r['route'],index=r['index'],actor_path=r['actor_path'],target=d,dev=r['route'] in dev))
        if len(rows)%40==0:
            print('extracted',len(rows),flush=True)
    # Teacher targets are kept separately from actor inputs.
    torch.save(torch.stack(features),dest/'observations.pt')
    meta=dict(config=CONFIG,manifest=data.identity,source=json.loads((RUN/'feature-identity.json').read_text())['base_sha256'],
              extractor=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),windows=rows)
    meta['historical_projection']='exposure_time_full_pose' if args.full_pose or args.directed else 'planar_history'
    meta['signed_displacement_matching']=args.directed
    (dest/'labels-and-split.json').write_text(json.dumps(meta,indent=2))
    print('complete',len(rows),flush=True)


if __name__=='__main__':
    main()
