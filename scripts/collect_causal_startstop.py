"""Bounded causal sensor collection. Simulator truth is explicitly identified."""
import json,time,queue,math,hashlib
from pathlib import Path
import numpy as np
import carla

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/carla_causal_miniloop_20261116_startstop_v1'
DTYPE=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('cos','<f4'),('object','<u4'),('tag','<u4')])
def vec(v):return [v.x,v.y,v.z]
def packet(q,frame):
 while True:
  p=q.get(timeout=15)
  if p.frame==frame:return p
  if p.frame>frame:raise RuntimeError('sensor ahead of world')
def main():
 OUT.mkdir(exist_ok=False)
 client=carla.Client('127.0.0.1',2100);client.set_timeout(5)
 deadline=time.monotonic()+150
 while True:
  try:w=client.get_world();break
  except RuntimeError:
   if time.monotonic()>deadline:raise
   time.sleep(2)
 if w.get_actors().filter('vehicle.*') or w.get_actors().filter('sensor.*'):raise RuntimeError('foreign actors present')
 original=w.get_settings();weather=w.get_weather();owned=[];sensors=[]
 manifest={'schema':'causal-miniloop-v1','status':'running','server':client.get_server_version(),'map':w.get_map().name,'dt':.1,'controller':'map-pure-pursuit-speedP-startstop-v1','episodes':[], 'sources':{'rgb':'online camera','lidar_xyz':'online semantic lidar rays','lidar_tags':'simulator semantic truth','ego':'simulator snapshot truth','navigation':'HD map','traffic':'one scripted lead; not read by ego controller'},'future_leakage':'only same-frame raw observations; future labels not in actor files','occ':'raw xyz/tags retained; no free-space inferred; 0/255 adapter not yet constructed','limitations':'scripted following-distance drive-brake-stop, no reactive lead policy, not expert interaction or driving acceptance'}
 try:
  if original.synchronous_mode:raise RuntimeError('foreign sync owner')
  settings=w.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.1;settings.substepping=True;settings.max_substep_delta_time=.01;settings.max_substeps=10;w.apply_settings(settings);client.set_timeout(20)
  lib=w.get_blueprint_library();m=w.get_map()
  for ep,(road,split,target,cloud) in enumerate([(388,'train',8.,0.),(392,'train',11.,50.),(713,'dev',9.,20.)]):
   folder=OUT/f'episode{ep}';folder.mkdir();wp=m.get_waypoint_xodr(road,-3,100.)
   if wp is None:raise RuntimeError(f'road unavailable {road}')
   tr=wp.transform;tr.location.z+=.3
   ego=w.spawn_actor(lib.find('vehicle.tesla.model3'),tr);owned.append(ego)
   leadtr=wp.next(25.)[0].transform;leadtr.location.z+=.3
   lead=w.spawn_actor(lib.find('vehicle.tesla.model3'),leadtr);owned.append(lead)
   w.set_weather(carla.WeatherParameters(cloudiness=cloud,sun_altitude_angle=60.))
   qs={}; specs={'rgb':('sensor.camera.rgb',{'image_size_x':'640','image_size_y':'360','fov':'90','sensor_tick':'0.0'},carla.Transform(carla.Location(x=1.5,z=1.8))), 'lidar':('sensor.lidar.ray_cast_semantic',{'range':'60','channels':'32','points_per_second':'160000','rotation_frequency':'10','upper_fov':'15','lower_fov':'-30','sensor_tick':'0.0'},carla.Transform(carla.Location(z=2.2)))}
   for key,(name,attrs,transform) in specs.items():
    bp=lib.find(name)
    for k,v in attrs.items():bp.set_attribute(k,v)
    s=w.spawn_actor(bp,transform,attach_to=ego);owned.append(s);sensors.append(s);qs[key]=queue.Queue();s.listen(qs[key].put)
   events=[];col=w.spawn_actor(lib.find('sensor.other.collision'),carla.Transform(),attach_to=ego);owned.append(col);sensors.append(col);col.listen(lambda e:events.append({'frame':e.frame,'other_type':e.other_actor.type_id,'impulse':vec(e.normal_impulse)}))
   rows=[];start=time.monotonic()
   for i in range(400):
    tr=ego.get_transform();vel=ego.get_velocity();speed=math.sqrt(vel.x**2+vel.y**2+vel.z**2);wp=m.get_waypoint(tr.location);nxt=wp.next(max(5.,speed*.8));goal=nxt[0].transform.location if nxt else wp.transform.location;local=np.array(tr.get_inverse_matrix())@np.r_[vec(goal),1.];steer=float(np.clip(2*2.8*local[1]/max(25.,local[0]**2+local[1]**2)/.6,-.6,.6))
    desired=0. if i<70 else (target if i<230 else max(0.,target*(290-i)/60.))
    error=desired-speed;control=carla.VehicleControl(throttle=float(np.clip(error*.25,0,.6)),brake=float(np.clip(-error*.25,0,.8)) if desired>0 else 1.,steer=steer)
    lt=lead.get_transform();lv=lead.get_velocity();ls=math.sqrt(lv.x**2+lv.y**2+lv.z**2);lw=m.get_waypoint(lt.location);ln=lw.next(max(5.,ls*.8));lp=ln[0].transform.location if ln else lw.transform.location;ll=np.array(lt.get_inverse_matrix())@np.r_[vec(lp),1.];lsteer=float(np.clip(2*2.8*ll[1]/max(25.,ll[0]**2+ll[1]**2)/.6,-.6,.6));ld=0. if i<70 else (target if i<220 else max(0.,target*(280-i)/60.));le=ld-ls
    lead.apply_control(carla.VehicleControl(throttle=float(np.clip(le*.25,0,.6)),brake=float(np.clip(-le*.25,0,.8)) if ld>0 else 1.,steer=lsteer))
    ego.apply_control(control);frame=w.tick(20);snap=w.get_snapshot();actor=snap.find(ego.id);tr=actor.get_transform();rgb=packet(qs['rgb'],frame);lidar=packet(qs['lidar'],frame)
    assert frame==snap.frame and abs(rgb.timestamp-snap.timestamp.elapsed_seconds)<1e-4 and abs(lidar.timestamp-rgb.timestamp)<1e-4
    raw=np.frombuffer(lidar.raw_data,DTYPE);np.savez_compressed(folder/f'{i:05}.npz',rgb_bgra=np.frombuffer(rgb.raw_data,np.uint8).reshape(360,640,4),lidar_xyz=np.stack([raw['x'],raw['y'],raw['z']],-1),lidar_tag=raw['tag'],lidar_world=np.array(lidar.transform.get_matrix()),ego_world=np.array(tr.get_matrix()))
    rows.append({'index':i,'frame':frame,'timestamp':snap.timestamp.elapsed_seconds,'rgb_frame':rgb.frame,'lidar_frame':lidar.frame,'ego_world':tr.get_matrix(),'velocity_world':vec(actor.get_velocity()),'acceleration_world':vec(actor.get_acceleration()),'angular_velocity_deg':vec(actor.get_angular_velocity()),'control_applied':{'throttle':control.throttle,'brake':control.brake,'steer':control.steer},'target_speed':desired,'road':wp.road_id,'lane':wp.lane_id,'navigation_world':vec(goal),'phase':'settle_stop' if i<70 else 'drive' if i<230 else 'decelerate' if i<290 else 'stop'})
    if i%50==0:print(ep,i,frame,flush=True)
   (folder/'frames.json').write_text(json.dumps(rows));(folder/'events.json').write_text(json.dumps(events))
   manifest['episodes'].append({'episode':ep,'split':split,'road':road,'lane':-3,'spawn_s':100.,'seed':0,'frames':len(rows),'target_speed_mps':target,'weather_cloudiness':cloud,'wall_seconds':time.monotonic()-start,'collision_callbacks':len(events),'collision_coverage':'registered callback only, not calibrated','max_speed_mps':max(np.linalg.norm(r['velocity_world']) for r in rows),'final_speed_mps':float(np.linalg.norm(rows[-1]['velocity_world']))})
   for s in sensors:s.stop()
   for a in reversed(owned):a.destroy()
   sensors.clear();owned.clear();(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
  manifest['status']='completed'
 except Exception as e:manifest.update(status='failed',error=repr(e));raise
 finally:
  for s in sensors:
   try:s.stop()
   except Exception:pass
  for a in reversed(owned):
   try:a.destroy()
   except Exception:pass
  w.apply_settings(original);w.set_weather(weather);manifest['settings_restored']=True;(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()



