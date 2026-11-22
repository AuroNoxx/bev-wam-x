"""Independent road-surface cartography from nadir images and surveyed GNSS control points."""
import argparse
import json
from pathlib import Path
import queue

import carla
import cv2
import numpy as np

from .collect import check_versions, read


def control_points(world):
    library=world.get_blueprint_library()
    actors=[]
    fixes=[]
    try:
        for x,y in ((0,0),(3000,0),(0,3000),(3000,3000)):
            tf=carla.Transform(carla.Location(x=x,y=y,z=600))
            sensor=world.spawn_actor(library.find('sensor.other.gnss'),tf)
            q=queue.Queue()
            sensor.listen(q.put)
            actors.append((sensor,q))
        for _ in range(4):
            frame=world.tick(40)
            fixes=[read(q,frame) for _,q in actors]
        return [[f.latitude,f.longitude] for f in fixes]
    finally:
        for s,_ in actors:s.stop();s.destroy()


def survey(world,town,plan,out):
    size=840
    altitude=650 if town=='Town12' else 255
    folder=out/town
    folder.mkdir(exist_ok=True)
    library=world.get_blueprint_library()
    spectator=world.get_spectator()
    georef=control_points(world)
    (folder/'georef.json').write_text(json.dumps(georef))
    if town=='Town10HD_Opt':
        tiles={(0,0)}
    else:
        tiles=set()
        for spec in plan['missions']:
            for a,b in zip(spec['xy'],spec['xy'][1:]):
                for p in np.linspace(a,b,2+int(np.linalg.norm(np.subtract(b,a))/110)):
                    tiles.add(tuple((np.rint(p/275)*275).astype(int)))
    tag=int(carla.CityObjectLabel.Roads)
    lane=int(carla.CityObjectLabel.RoadLines)
    for cx,cy in sorted(tiles):
        dest=folder/f'tile_{cx}_{cy}.npz'
        if dest.exists():continue
        x,y=int(cx),int(cy)
        spectator.set_transform(carla.Transform(carla.Location(x,y,altitude-200),carla.Rotation(pitch=-80)))
        for _ in range(26):world.tick(45)
        actors=[]
        try:
            for kind in ('semantic_segmentation','depth'):
                bp=library.find('sensor.camera.'+kind)
                for k,v in dict(image_size_x=size,image_size_y=size,fov=90).items():bp.set_attribute(k,str(v))
                s=world.spawn_actor(bp,carla.Transform(carla.Location(x,y,altitude),carla.Rotation(pitch=-90)))
                q=queue.Queue()
                s.listen(q.put)
                actors.append((s,q))
            for _ in range(8):
                frame=world.tick(45)
                seg,depth=[read(q,frame) for _,q in actors]
            tags=np.frombuffer(seg.raw_data,np.uint8).reshape(size,size,4)[:,:,2]
            channels=np.frombuffer(depth.raw_data,np.uint8).reshape(size,size,4).astype(np.float32)
            zdist=(channels[...,2]+256*channels[...,1]+65536*channels[...,0])*(1000/16777215.)
            r,c=np.nonzero(np.isin(tags,[tag,lane])&(zdist>5)&(zdist<900))
            d=zdist[r,c]
            xyz=np.column_stack((x-(r+.5-size/2)*d/(size/2),y+(c+.5-size/2)*d/(size/2),altitude-d))
            np.savez_compressed(dest,xyz=xyz.astype(np.float32),frame=frame)
            print(town,dest.name,len(d),flush=True)
        finally:
            for s,_ in actors:s.stop();s.destroy()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path);p.add_argument('town',choices=['Town12','Town10HD_Opt'])
    a=p.parse_args()
    c=carla.Client('localhost',2460);c.set_timeout(160);check_versions(c)
    w=c.get_world()
    if w.get_map().name.rsplit('/',1)[-1]!=a.town:w=c.load_world(a.town)
    orig=w.get_settings();cfg=w.get_settings();cfg.synchronous_mode=True;cfg.fixed_delta_seconds=.05
    cfg.spectator_as_ego=True;w.apply_settings(cfg)
    try:survey(w,a.town,json.loads((a.out/'sketches.json').read_text())[a.town],a.out)
    finally:w.apply_settings(orig)


if __name__=='__main__':main()
