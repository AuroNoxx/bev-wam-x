"""Bounded causal stock B2D OCC-generation probe; never drives or trains.

The upstream file imports two symbols it never references. Only these unused
imports are removed from an in-memory AST; the generator class stays unchanged.
No fake dependency modules and no modifications to the bundled upstream files.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def load_upstream_generator():
    source=ROOT/'runtime/Bench2Drive-0.0.4/tools/occ_online_carla.py'
    tree=ast.parse(source.read_text(encoding='utf-8-sig'));removed=[]
    for node in list(tree.body):
        if isinstance(node,ast.Import) and any(x.name=='open3d' for x in node.names):name='o3d'
        elif isinstance(node,ast.ImportFrom) and node.module=='leaderboard.utils.occ_label_gen':name='OccDatasetGen'
        else:continue
        assert not any(isinstance(x,ast.Name) and x.id==name for x in ast.walk(tree)), 'Dependency is actually used'
        tree.body.remove(node);removed.append(ast.unparse(node))
    namespace={'__name__':'bounded_upstream_occ_reference'}
    exec(compile(tree,str(source),'exec'),namespace)
    return namespace['OccOnlineCARLA'](history_length=10), {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'removed_unused_imports_only':removed,'generator_body_modified':False,'causal_frames_only':True}


def reference(out):
    import numpy as np
    import torch
    from b2d_native_occ import semantic_grid
    torch.set_num_threads(4)
    generator,identity=load_upstream_generator()
    rows=json.loads((out/'capture.json').read_text());results=[]
    for i,row in enumerate(rows):
        with np.load(out/row['file']) as z:raw=z['raw'].copy()
        start=time.perf_counter()
        occ,seg=generator.occ_from_carla_tick(raw,row['bbox'],{'LIDAR_0':{'world2lidar':row['world2lidar']}})
        grid=semantic_grid(occ,seg)[:,::-1,:].copy()
        np.savez_compressed(out/f'upstream-{i}.npz',history_current=grid,occ=occ,seg=seg)
        results.append({'frame':row['frame'],'seconds':time.perf_counter()-start,
                        'native_counts':{str(k):int(v) for k,v in zip(*np.unique(grid,return_counts=True))}})
        (out/'reference-result.json').write_text(json.dumps({**identity,'results':results},indent=2))
        print(results[-1],flush=True)


def capture(out,port):
    import carla
    import numpy as np
    import queue
    c=carla.Client('127.0.0.1',port);c.set_timeout(10);world=c.get_world();old=world.get_settings();owned=[]
    try:
        assert not old.synchronous_mode and not world.get_actors().filter('vehicle.*')
        settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;settings.no_rendering_mode=True
        world.apply_settings(settings)
        tr=world.get_map().get_waypoint(carla.Location(x=83.8,y=-93.5,z=8)).transform;tr.location.z+=.4
        lib=world.get_blueprint_library();ego=world.spawn_actor(lib.find('vehicle.lincoln.mkz_2020'),tr);owned.append(ego)
        ego.apply_control(carla.VehicleControl(brake=1.));bp=lib.find('sensor.lidar.ray_cast_semantic')
        for k,v in {'range':'85','channels':'64','points_per_second':'600000','rotation_frequency':'10','upper_fov':'10','lower_fov':'-30','sensor_tick':'0.0'}.items():bp.set_attribute(k,v)
        sensor=world.spawn_actor(bp,carla.Transform(carla.Location(x=-.39,z=1.84)),attach_to=ego);owned.append(sensor)
        q=queue.Queue();sensor.listen(q.put);rows=[]
        for tick in range(8):
            frame=world.tick();p=q.get(timeout=10)
            while p.frame<frame:p=q.get(timeout=10)
            assert p.frame==frame
            if tick not in [5,7]:continue
            raw=np.frombuffer(p.raw_data,dtype='<f4').reshape(-1,6).copy()
            file=f'raw-{len(rows)}.npz';np.savez_compressed(out/file,raw=raw)
            pose=ego.get_transform();center=pose.transform(ego.bounding_box.location)
            bbox={'id':ego.id,'class':'ego_vehicle','location':[pose.location.x,pose.location.y,pose.location.z],
                  'center':[center.x,center.y,center.z],'extent':[ego.bounding_box.extent.x,ego.bounding_box.extent.y,ego.bounding_box.extent.z],
                  'rotation':[pose.rotation.pitch,pose.rotation.roll,pose.rotation.yaw],
                  'semantic_tags':list(ego.semantic_tags),'world2ego':pose.get_inverse_matrix()}
            rows.append({'frame':frame,'timestamp':p.timestamp,'file':file,'bbox':[bbox],'world2lidar':p.transform.get_inverse_matrix()})
        (out/'capture.json').write_text(json.dumps(rows,indent=2))
    finally:
        for actor in owned[::-1]:
            if hasattr(actor,'stop'):actor.stop()
            actor.destroy()
        world.apply_settings(old)
    with (out/'reference.stdout.log').open('w') as log:
        try:
            proc=subprocess.run([str(ROOT/'.venv-train/Scripts/python.exe'),'-u',str(Path(__file__).resolve()),'--reference','--output',str(out.resolve())],stdout=log,stderr=subprocess.STDOUT,timeout=90)
            result={'returncode':proc.returncode}
        except subprocess.TimeoutExpired:result={'timeout_seconds':90,'note':'bounded CPU generation probe timed out; not a model/drive failure'}
    (out/'probe-status.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--port',type=int,default=2125);p.add_argument('--reference',action='store_true');args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if args.reference:reference(output)
    else:capture(output,args.port)
