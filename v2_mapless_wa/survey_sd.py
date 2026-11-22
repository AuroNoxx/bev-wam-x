"""Independent SD survey: one nadir camera, no map or driving trajectory input."""
import argparse,json,queue
from pathlib import Path
import carla,cv2,numpy as np
from .collect import read

def main(port,out,centers,altitude,size,step,expected_map,evaluation=False):
    c=carla.Client('127.0.0.1',port);c.set_timeout(35);w=c.get_world()
    name=w.get_map().name.rsplit('/',1)[-1]
    allowed = ('Town04_Opt',) if evaluation else ('Town12','Town10HD_Opt')
    if name not in allowed or expected_map and expected_map!=name:
        raise ValueError(f'SD survey cannot run on {name}')
    out.mkdir(parents=True,exist_ok=True)
    old=w.get_settings(); cam=None; spectator=w.get_spectator(); old_spectator=spectator.get_transform()
    try:
        s=w.get_settings();s.synchronous_mode=True;s.fixed_delta_seconds=.05;s.spectator_as_ego=True;w.apply_settings(s)
        bp=w.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        for k,v in dict(image_size_x=size,image_size_y=size,fov=90).items():bp.set_attribute(k,str(v))
        for n,(x,y,z) in enumerate(centers):
            spectator.set_transform(carla.Transform(carla.Location(x,y,z+20),carla.Rotation(pitch=-35)))
            for _ in range(45 if name=='Town12' else 3):w.tick(45)
            cam=w.spawn_actor(bp,carla.Transform(carla.Location(x,y,z+altitude),carla.Rotation(pitch=-90)))
            q=queue.Queue();cam.listen(q.put)
            for _ in range(10):
                fid=w.tick(45);img=read(q,fid)
            cam.stop();cam.destroy();cam=None
            tags=np.frombuffer(img.raw_data,np.uint8).reshape(size,size,4)[:,:,2]
            prefix='sd-aerial' if len(centers)==1 else f'sd-aerial-{n:02d}'
            cv2.imwrite(str(out/(prefix+'-tags.png')),tags)
            vis=np.full((size,size,3),22,np.uint8);vis[tags==int(carla.CityObjectLabel.Roads)]=(170,170,170)
            vis[tags==int(carla.CityObjectLabel.RoadLines)]=(240,240,240)
            vis[tags==int(carla.CityObjectLabel.Sidewalks)]=(65,85,80)
            for offset in range((-int(altitude)//step+1)*step,int(altitude),step):
                i=round(size/2+offset*size/(2*altitude))
                cv2.line(vis,(i,0),(i,size-1),(55,80,105),1);cv2.line(vis,(0,i),(size-1,i),(55,80,105),1)
                cv2.putText(vis,'y'+str(round(y+offset)),(i+1,size//2-2),0,.43,(0,250,255),1)
                cv2.putText(vis,'x'+str(round(x-offset)),(size//2+3,i-2),0,.43,(0,250,255),1)
            cv2.imwrite(str(out/(prefix+'-grid.jpg')),vis)
            (out/(prefix+'.json')).write_text(json.dumps(dict(map=name,center=[x,y,z],altitude=altitude,
                image_size=size,meters_per_pixel_at_reference_height=2*altitude/size,frame=fid,
                purpose='closed_loop_only' if evaluation else 'road_survey'),indent=2))
            print('survey',name,n,(x,y,z),'road pixels',int(np.count_nonzero(tags==int(carla.CityObjectLabel.Roads))),flush=True)
    finally:
        if cam:cam.stop();cam.destroy()
        spectator.set_transform(old_spectator)
        w.apply_settings(old)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--port',default=2460,type=int)
    p.add_argument('--center',nargs=3,type=float,action='append');p.add_argument('--altitude',type=float,default=150)
    p.add_argument('--size',type=int,default=1200);p.add_argument('--grid-step',type=int,default=25);p.add_argument('--map')
    a=p.parse_args();main(a.port,a.out,a.center or [(0,0,0)],a.altitude,a.size,a.grid_step,a.map)
