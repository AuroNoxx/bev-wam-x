"""Fixed-pose native-resolution sensor and BF16 HIP Graph benchmark."""
import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import queue
import time

import carla
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .collect import DT, matrix, read
from .geometry import CAMS, X, Y, Z, local, raster
from .phase2_infer import Inference, fold_for_inference, load_model, model_input
from .phase2_live import gpu_monitor, server
from .phase2_model import GAMMA, configure_sdpa, mounts, rgb_preprocess
from .render_probe import camera_blueprint, lidar_blueprint, locked_tier


SHAPES = {'A': ((384, 224),)*3, 'B': ((640, 360),)*3,
          'C': ((768, 448), (384, 224), (384, 224))}


def project(points, pose, fov, size):
    w,h = size
    q = (points-pose[:3,3])@pose[:3,:3]
    depth = np.maximum(q[...,0], 1e-6)
    f = w/(2*np.tan(np.deg2rad(fov/2)))
    uv = np.stack((w/2+f*q[...,1]/depth, h/2-f*q[...,2]/depth),axis=-1)
    visible = (q[...,0]>0)&(uv[...,0]>=0)&(uv[...,0]<w-1)&(uv[...,1]>=0)&(uv[...,1]<h-1)
    return uv,visible


class NativeResolution(nn.Module):
    """Groups identically sized sensors and projects each using its own native intrinsics."""
    def __init__(self, model, sizes):
        super().__init__()
        self.model = model
        groups = defaultdict(list)
        grids, visible = [], []
        points = np.stack((np.broadcast_to(X,(5,128,128)),np.broadcast_to(-Y,(5,128,128)),
                           np.broadcast_to(np.array(Z)[:,None,None],(5,128,128))),axis=-1)
        for i,(size,cam,pose) in enumerate(zip(sizes,CAMS,mounts())):
            groups[size].append(i)
            uv, mask = project(points,pose,cam[-1],size)
            uv = np.where(mask[...,None],uv,-1).astype(np.float32)
            uv = (uv+.5)*np.array([2/size[0],2/size[1]],np.float32)-1
            grids.append(uv.reshape(640,128,2))
            visible.append(mask.reshape(1,640,128))
        self.groups = tuple(tuple(indices) for indices in groups.values())
        for j,indices in enumerate(self.groups):
            self.register_buffer(f'grid{j}',torch.from_numpy(np.stack([grids[i] for i in indices])))
            self.register_buffer(f'mask{j}',torch.from_numpy(np.stack([visible[i] for i in indices])))
        self.register_buffer('count',torch.from_numpy(np.stack(visible).sum(0).clip(1).astype(np.float32)))

    def forward(self, front, left, right, lidar, nav, command):
        cameras = (front,left,right)
        lifted = []
        for i, indices in enumerate(self.groups):
            rgb = torch.cat([cameras[j] for j in indices],0)
            features = self.model.pyramid(rgb_preprocess(rgb).contiguous(memory_format=torch.channels_last))
            planes = F.grid_sample(features,getattr(self,f'grid{i}'), align_corners=False)
            lifted.append((planes*getattr(self,f'mask{i}')).sum(0,keepdim=True))
        fused = sum(lifted)/self.count
        return self.model.predict(self.model.lift.mix(fused.reshape(1,320,128,128)),lidar,nav,command)


def world_pose(values):
    return carla.Transform(carla.Location(*values[:3]),carla.Rotation(*values[3:]))


def camera_image(image, size):
    w,h = size
    if (image.width,image.height)!=(w,h):
        raise RuntimeError(f'Expected {size}, received {image.width}x{image.height}')
    return np.frombuffer(image.raw_data,np.uint8).reshape(h,w,4)[:,:,:3].copy()


def capture(world, scene, sizes, config, weights, case_out):
    settings = world.get_settings()
    settings.synchronous_mode=True
    settings.fixed_delta_seconds=DT
    settings.no_rendering_mode=False
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    if world.get_map().name != scene['road_name']:
        raise RuntimeError(world.get_map().name)
    target = world_pose(scene['pose'])
    lib = world.get_blueprint_library()
    spawn = carla.Transform(target.location+carla.Location(z=.38),target.rotation)
    ego = world.spawn_actor(lib.find('vehicle.lincoln.mkz'),spawn)
    ego.set_simulate_physics(False)
    ego.set_transform(target)
    world.get_spectator().set_transform(carla.Transform(target.location+carla.Location(z=5),target.rotation))
    sensors, queues = [], []

    def attach(bp, mount):
        sensor = world.spawn_actor(bp,mount,attach_to=ego)
        q = queue.Queue()
        sensor.listen(q.put)
        sensors.append(sensor)
        queues.append(q)

    def tick():
        fid = world.tick(45)
        snap = world.get_snapshot()
        bundle = [read(q,fid) for q in queues]
        if snap.frame!=fid or any(sensor.frame!=fid for sensor in bundle):
            raise RuntimeError(f'Frame misalignment at {fid}')
        pose = matrix(snap.find(ego.id).get_transform())
        if not np.allclose(pose,matrix(target),rtol=0,atol=.001):
            raise RuntimeError('Frozen pose changed')
        return fid,bundle,pose

    def convert(bundle):
        rgbs = tuple(camera_image(image,size) for image,size in zip(bundle[:3],sizes))
        raw = np.frombuffer(bundle[3].raw_data,np.float32).reshape(-1,4).copy()
        raw[:,1]*=-1
        raw[:,2]+=2.2
        return rgbs,raster(raw)

    try:
        for cam,(w,h) in zip(CAMS,sizes):
            x,y,z,yaw,pitch,fov=cam
            bp=camera_blueprint(lib,fov)
            bp.set_attribute('image_size_x',str(w))
            bp.set_attribute('image_size_y',str(h))
            attach(bp,carla.Transform(carla.Location(x,y,z),carla.Rotation(yaw=yaw,pitch=pitch)))
        attach(lidar_blueprint(lib,False),carla.Transform(carla.Location(z=2.2)))
        for _ in range(40):
            tick()
        lead_start=time.time()
        for _ in range(50):
            tick()
        samples,frames,elapsed=[],[],[]
        capture_start=time.time()
        for _ in range(20):
            start = time.perf_counter()
            fid,bundle,pose = tick()
            samples.append(convert(bundle))
            frames.append(fid)
            elapsed.append(time.perf_counter()-start)
        capture_end=time.time()
        if not np.all(np.diff(frames)==1):
            raise RuntimeError('Retained frame numbers are not consecutive')
        mid = samples[10][0][0]
        if not cv2.imwrite(str(case_out/'front-native.png'),mid):
            raise IOError(case_out/'front-native.png')
        rgb_sizes = [sum(cv2.imencode('.png',img,[cv2.IMWRITE_PNG_COMPRESSION,1])[1].nbytes for img in imgs)
                     for imgs,_ in samples]
        lidar_sizes=[]
        for _,bev in samples:
            buf=io.BytesIO()
            np.savez_compressed(buf,bev=bev)
            lidar_sizes.append(buf.tell())
        capture_metrics=dict(frames=frames,first=frames[0],last=frames[-1],pose=pose.tolist(),
                           lead_start=lead_start,capture_start=capture_start,capture_end=capture_end,
                           retained_fps=20/sum(elapsed),tick_p95_ms=float(1000*np.percentile(elapsed,95)),
                           estimated_png_bytes_median=float(np.median(rgb_sizes)),
                           estimated_bev_bytes_median=float(np.median(lidar_sizes)),
                           estimated_sensor_bytes_median=float(np.median(np.array(rgb_sizes)+lidar_sizes)))
        (case_out/'capture.json').write_text(json.dumps(capture_metrics,indent=2))
        print(config,'capture',json.dumps(capture_metrics),flush=True)
        infer=benchmark(tick,convert,samples,config,sizes,pose,scene,weights)
        return capture_metrics,infer
    finally:
        for s in reversed(sensors):
            s.stop()
            s.destroy()
        ego.destroy()


@torch.inference_mode()
def benchmark(tick,convert,samples,config,sizes,pose,scene,weights):
    configure_sdpa('aotriton')
    torch.cuda.tunable.tuning_enable(False)
    torch.set_num_threads(3)
    model=load_model(weights)
    parameters=sum(p.numel() for p in model.parameters())
    folded=fold_for_inference(model)
    effective=model if config=='A' else NativeResolution(model,sizes).eval()
    effective.cuda().to(memory_format=torch.channels_last)
    world_points=np.asarray(scene['coordinates'])
    nav=np.stack([np.interp([20,40,60],scene['arc_m'],column) for column in world_points.T],axis=1)
    nav=local(nav,pose)[:,:2].astype(np.float32)
    cmd=np.array([1,0,0],np.uint8)

    def inputs(sample):
        rgb,bev=sample
        rgb=tuple(np.ascontiguousarray(x[:,:,::-1].transpose(2,0,1)) for x in rgb)
        if config=='A':
            return model_input(dict(rgb=np.stack(rgb),lidar=bev,nav=nav,cmd=cmd))
        return tuple(torch.from_numpy(np.ascontiguousarray(x))[None] for x in (*rgb,bev,nav,cmd))

    observation=inputs(samples[10])
    if config=='A':
        alternative=NativeResolution(model,sizes).eval().cuda()
        rgb,bev,nav_,cmd_=observation
        with torch.autocast('cuda',torch.bfloat16,cache_enabled=False):
            base=model(*(x.cuda() for x in observation))
            groups=alternative(*(x.cuda() for x in (*rgb.unbind(1),bev,nav_,cmd_)))
        for name in base:
            torch.testing.assert_close(groups[name].float(),base[name].float(),atol=.002,rtol=.002)
        del groups,base,alternative
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    engine=Inference(effective,observation,graph=True)
    if engine.graph is None:
        raise RuntimeError('HIP Graph was not captured')
    with torch.autocast('cuda',torch.bfloat16,cache_enabled=False):
        eager=effective(*(x.cuda() for x in observation))
    tested=engine(*observation)
    for name in eager:
        torch.testing.assert_close(tested[name].float(),eager[name].float(),atol=.002,rtol=.002)
    del eager,tested
    for _ in range(12):
        engine(*observation)['path'].cpu()
    begin=time.time()
    host,gpu,loop=[],[],[]
    first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for _ in range(128):
        step_start=time.perf_counter()
        _,bundle,_=tick()
        sample=inputs(convert(bundle))
        start=time.perf_counter()
        first.record()
        out=engine(*sample)
        last.record()
        out['path'].float().cpu()
        out['occ'].float().cpu()
        host.append(1000*(time.perf_counter()-start))
        gpu.append(first.elapsed_time(last))
        loop.append(1000*(time.perf_counter()-step_start))
    return dict(start=begin,end=time.time(),iterations=len(host),parameters=parameters,folded_bn=folded,
                input_shapes=[list(x.shape) for x in observation],hip_version=torch.version.hip,
                p95_wall_ms=float(np.percentile(host,95)),p95_device_ms=float(np.percentile(gpu,95)),
                p95_tick_to_output_ms=float(np.percentile(loop,95)),median_wall_ms=float(np.median(host)),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)


def memory(log, capture, inference):
    meas=[]
    for row in log.read_text(encoding='utf-8-sig').splitlines():
        try:
            record=json.loads(row)
        except json.JSONDecodeError:
            continue
        adapter=max(record['adapters'],key=lambda a:a['ue_bytes']+a['model_bytes'],default=None)
        if adapter and adapter['ue_bytes']:
            meas.append((record['unix'],adapter))
    def stats(begin,end,concurrent):
        use=[r for t,r in meas if begin<=t<=end and (not concurrent or r['model_bytes'])]
        if not use:
            return dict(samples=0)
        return dict(samples=len(use),peak_ue_gib=max(r['ue_bytes'] for r in use)/2**30,
                    peak_model_gib=max(r['model_bytes'] for r in use)/2**30,
                    peak_joint_gib=max(r['ue_bytes']+r['model_bytes'] for r in use)/2**30,
                    peak_card_gib=max(r['card_bytes'] for r in use)/2**30)
    return dict(retained=stats(capture['capture_start'],capture['capture_end'],False),
                camera_only_warm_and_retained=stats(capture['lead_start'],capture['capture_end'],False),
                active_model=stats(inference['start'],inference['end'],True))


def run(args):
    if locked_tier()!='High' or GAMMA != .5:
        raise RuntimeError('The locked rendering or photometric setting changed')
    sizes=SHAPES[args.config]
    args.out.mkdir(parents=True,exist_ok=False)
    scene=json.loads(args.pose.read_text())
    with server(args.carla_root.resolve(),args.out,args.port) as (client,pid):
        world=client.load_world('Town12')
        with gpu_monitor(args.out,pid):
            sample,infer=capture(world,scene,sizes,args.config,args.weights,args.out)
    result=dict(config=args.config,cameras=sizes,gamma=GAMMA,capture=sample,inference=infer,
                vram=memory(args.out/'gpu.jsonl',sample,infer))
    (args.out/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('carla_root','out','pose','weights'):
        p.add_argument('--'+field.replace('_','-'),type=Path,required=True)
    p.add_argument('--config',choices=SHAPES,required=True)
    p.add_argument('--port',type=int,default=2460)
    run(p.parse_args())
