"""Actor-only feature extraction for the local stop/start and BEV repair."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .runtime import ROOT, configure, math_attention

configure()
import torch
from torch.nn import functional as F
from .real_dataset import RealData, load_actor
from .build_real_manifest import CAMERAS, VIEWS
from .train_disk_cached import DiskCachedTask

RUN = ROOT/'artifacts/v5-qwen-drive-world/geometry-repair'
BASE = ROOT/'artifacts/v5-qwen-drive-world/subset2000'
CFG = json.loads((ROOT/'configs/v5-qwen-drive-world/geometry-repair.json').read_text())


def split_records(data):
    rng = np.random.default_rng(CFG['seed'])
    order = rng.permutation(len(data.records))
    selected = [int(i) for i in order if data.records[i]['speed_kmh'] <= 3]
    buckets = [[] for _ in range(6)]
    for i in order:
        speed = data.records[i]['speed_kmh']
        if speed > 3:
            buckets[min(5,int(speed//20))].append(int(i))
    while len(selected)<CFG['train_windows']:
        for bucket in buckets:
            if bucket and len(selected)<CFG['train_windows']:
                selected.append(bucket.pop())
    rng.shuffle(selected)
    return {'train':[data.records[i] for i in selected],'validation':data.validation}


def projection(sensor, world_from_ego):
    if 'world2cam' in sensor:
        transform=np.asarray(sensor['world2cam'])@world_from_ego
        return transform,np.asarray(sensor['intrinsic']),sensor['image_size_x'],sensor['image_size_y']
    return (np.linalg.inv(sensor['ego_from_sensor']),np.asarray(sensor['intrinsics']),
            sensor['width'],sensor['height'])


def bev_grid(sensor,world_from_ego,prior_pose=None):
    y,x=np.meshgrid(-19.2+(np.arange(32)+.5)*1.2,-16+(np.arange(64)+.5)*2.,indexing='ij')
    if prior_pose is not None:
        dx=x-prior_pose[0];dy=y-prior_pose[1]
        c,s=np.cos(prior_pose[2]),np.sin(prior_pose[2])
        x,y=c*dx+s*dy,-s*dx+c*dy
    tr,k,w,h=projection(sensor,world_from_ego)
    grids=[]; weights=[]
    for z in (.35,1.3):
        xyz=np.stack([x,-y,np.full_like(x,z),np.ones_like(x)],-1)@tr.T
        depth=xyz[...,0]
        u=k[0,0]*xyz[...,1]/np.maximum(depth,.01)+k[0,2]
        v=k[1,2]-k[1,1]*xyz[...,2]/np.maximum(depth,.01)
        valid=(depth>.2)&(u>=0)&(u<w)&(v>=0)&(v<h)
        grids.append(np.stack([2*(u+.5)/w-1,2*(v+.5)/h-1],-1).clip(-2,2))
        weights.append(valid*np.clip(1-np.abs(2*u/w-1),.05,1))
    return torch.tensor(np.stack(grids),device='cuda',dtype=torch.float32),torch.tensor(np.stack(weights),device='cuda',dtype=torch.float32)


def project_visual(task,scene,actor):
    processor=task.model.processor
    patches=[]; shapes=[]; refs=[]
    for cam,view in zip(CAMERAS,VIEWS):
        frames=scene.views[view]
        for index,frame in enumerate(frames):
            current=index==len(frames)-1
            pixels=processor.config.current_image_pixels if current else processor.config.history_image_pixels
            patch,(h,w)=processor._patchify(frame,pixels)
            patches.append(patch);shapes.append((1,h,w));refs.append((cam,current,index))
    visual=task.model.vlm.model.visual(torch.cat(patches).to('cuda'),grid_thw=torch.tensor(shapes,device='cuda')).pooler_output
    device=visual.device
    generator=torch.Generator(device='cpu').manual_seed(CFG['seed'])
    matrix=torch.randn(visual.shape[-1],CFG['projection_channels'],generator=generator)/visual.shape[-1]**.5
    reduced=F.layer_norm(visual.float(),(visual.shape[-1],))@matrix.to(device)
    cal=actor['camera_calibration']
    cal=cal if isinstance(cal,dict) else json.loads((ROOT/cal).read_text())
    pose=np.asarray(actor['world_from_ego'])
    history=actor['history']
    maps=[];masks=[];past=[]
    for piece,(_,h,w),(cam,current,index) in zip(reduced.split([h*w//4 for _,h,w in shapes]),shapes,refs):
        grid,weight=bev_grid(cal[cam],pose,None if current else history[index*5])
        feature=piece.T.reshape(1,-1,h//2,w//2).float()
        with torch.autocast('cuda',enabled=False):
            features=F.grid_sample(feature.expand(2,-1,-1,-1),grid,align_corners=False)
            weighted=features*weight[:,None]
        if current:
            maps.append(weighted);masks.append(weight)
            if cam==CAMERAS[0]:
                front=features*weight.gt(0)[:,None]
        else:
            past.append((features,weight))
    denom=torch.stack(masks).sum(0).clamp_min(.0001)
    current_map=(torch.stack(maps).sum(0)/denom[:,None]).flatten(0,1)
    motion=torch.stack([(front-old)*wt.gt(0)[:,None]*(masks[0]>0)[:,None] for old,wt in past]).mean((0,1))
    state=torch.tensor([actor['speed_kmh']/120,actor['target_speed_kmh']/120,
                        *np.clip(np.asarray(actor['ego_acceleration'])/8,-3,3),
                        *np.clip(np.asarray(actor['navigation_flu'])/80,-3,3),
                        *np.clip(np.asarray(history)[[0,5,10],:2].ravel()/20,-3,3)],device=device,dtype=torch.float32)
    return torch.cat([current_map,motion]),state


def actor_features(task,record,seed):
    scene,actor=load_actor(record['actor_path'])
    projected,state=project_visual(task,scene,actor)
    native,inp=task.encode_path(record['actor_path'])
    ctx=task.planner.context(native,inp)
    speed=torch.tensor([actor['speed_kmh']/3.6],device='cuda',dtype=torch.float32)
    with torch.random.fork_rng(devices=[task.device.index]):
        torch.manual_seed(seed)
        sample=task.planner.sample(ctx,speed,torch.randn(1,50,2,device='cuda'),steps=10)
    return dict(bev=projected.half().cpu(),state=state.cpu(),ctrl=sample['controls'][0].float().cpu(),
                latent=task.world.encode_current(native['hidden'])[0].cpu(),v0=speed.cpu())


def load_base(data):
    checkpoint=torch.load(BASE/'latest.ckpt',map_location='cpu',weights_only=False,mmap=True)
    assert checkpoint['global_step']==CFG['base_step']
    task=DiskCachedTask(checkpoint['task_state']['config'],data.identity,CFG['base_phase']).cuda()
    task.on_load_checkpoint(checkpoint)
    task.load_state_dict(checkpoint['state_dict'])
    return task.eval().requires_grad_(False)


def main():
    torch.manual_seed(71);torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(.8)
    data=RealData('subset2000')
    task=load_base(data)
    RUN.mkdir(exist_ok=True)
    with (BASE/'latest.ckpt').open('rb') as handle:
        source=hashlib.file_digest(handle,'sha256').hexdigest()
    identity=dict(base_sha256=source,manifest=data.identity,config=CFG,
                  extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    meta=RUN/'feature-identity.json'
    if meta.exists() and json.loads(meta.read_text())!=identity:
        raise ValueError('Existing repair cache belongs to a different model, manifest or extractor')
    meta.write_text(json.dumps(identity,indent=2))
    torch.save(task.world.state_dict(),RUN/'base-world.pt')
    with torch.no_grad(),math_attention(),torch.autocast('cuda',dtype=torch.bfloat16):
        for split,records in split_records(data).items():
            folder=RUN/'features'/split;folder.mkdir(parents=True,exist_ok=True)
            for i,r in enumerate(records):
                path=folder/f'{i:04d}.pt'
                if not path.exists():
                    item=actor_features(task,r,(1907 if split=='validation' else 43001)+i)
                    item['actor_path']=r['actor_path']
                    torch.save(item,path.with_suffix('.pending'))
                    path.with_suffix('.pending').replace(path)
                if (i+1)%32==0: print(split,i+1,'/',len(records),flush=True)


if __name__=='__main__':
    main()
