"""Independent SD survey: one nadir camera, no map or driving trajectory input."""
import argparse,queue
from pathlib import Path
import carla,cv2,numpy as np

def main(port,out):
    c=carla.Client('127.0.0.1',port);c.set_timeout(35);w=c.get_world()
    old=w.get_settings(); cam=None
    try:
        s=w.get_settings();s.synchronous_mode=True;s.fixed_delta_seconds=.05;w.apply_settings(s)
        bp=w.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        for k,v in dict(image_size_x=1200,image_size_y=1200,fov=90).items():bp.set_attribute(k,str(v))
        # 0.25 m/px on z=0; map axes x runs up and y right.
        cam=w.spawn_actor(bp,carla.Transform(carla.Location(z=150),carla.Rotation(pitch=-90)))
        q=queue.Queue();cam.listen(q.put);img=None
        for _ in range(5):
            fid=w.tick();img=q.get(timeout=15);assert img.frame==fid
        tags=np.frombuffer(img.raw_data,np.uint8).reshape(1200,1200,4)[:,:,2]
        cv2.imwrite(str(out/'sd-aerial-tags.png'),tags)
        vis=np.full((1200,1200,3),22,np.uint8);vis[tags==int(carla.CityObjectLabel.Roads)]=(170,170,170)
        vis[tags==int(carla.CityObjectLabel.RoadLines)]=(240,240,240)
        vis[tags==int(carla.CityObjectLabel.Sidewalks)]=(65,85,80)
        for n in range(-125,150,25):
            i=600+4*n
            cv2.line(vis,(i,0),(i,1199),(55,80,105),1);cv2.line(vis,(0,i),(1199,i),(55,80,105),1)
            cv2.putText(vis,'y'+str(n),(i+1,598),0,.43,(0,250,255),1)
            cv2.putText(vis,'x'+str(-n),(603,i-2),0,.43,(0,250,255),1)
        cv2.imwrite(str(out/'sd-aerial-grid.jpg'),vis)
    finally:
        if cam:cam.stop();cam.destroy()
        w.apply_settings(old)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--port',default=2460,type=int);a=p.parse_args();main(a.port,a.out)
