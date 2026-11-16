"""Stationary CARLA lead-car visual counterfactual, no driving, no training."""
from pathlib import Path
import carla,numpy as np,json,time,queue,subprocess,threading,hashlib,math
R=Path(__file__).resolve().parents[1];O=R/'artifacts/qwen-driving-v3-lead-counterfactual';O.mkdir(exist_ok=True)
client=carla.Client('localhost',2102);client.set_timeout(30);world=client.get_world();old=world.get_settings();actors=[];worker=None;result={'conditions':{},'map':world.get_map().name,'simulator':client.get_server_version(),'scope':'stationary image response, not causal driving competence or safety'}
try:
 settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;settings.no_rendering_mode=False;world.apply_settings(settings);maps=world.get_map();bps=world.get_blueprint_library();bp=bps.find('vehicle.dodge.charger');ego=None
 for tr in maps.get_spawn_points():
  wp=maps.get_waypoint(tr.location);n=wp.next(40)
  if wp.is_junction or not n or n[0].is_junction:continue
  diff=(n[0].transform.rotation.yaw-tr.rotation.yaw+180)%360-180
  if abs(diff)>3:continue
  ego=world.try_spawn_actor(bp,tr)
  if ego:break
 if ego is None:raise RuntimeError('No40m straight spawn found')
 actors.append(ego);ego.apply_control(carla.VehicleControl(brake=1))
 for _ in range(60):world.tick()
 pose=ego.get_transform();ego.set_simulate_physics(False);ego.set_transform(pose)
 cam=bps.find('sensor.camera.rgb');cam.set_attribute('image_size_x','448');cam.set_attribute('image_size_y','252');cam.set_attribute('fov','70');cam.set_attribute('sensor_tick','0');camera=world.spawn_actor(cam,carla.Transform(carla.Location(x=.8,z=1.6)),attach_to=ego);actors.append(camera);iq=queue.Queue();camera.listen(iq.put)
 inv=np.array(pose.get_inverse_matrix());wp=maps.get_waypoint(pose.location);state=[0.,0.,0.,0.]
 for d in [5,30]:
  n=wp.next(d);loc=(n[0] if n else wp).transform.location;pt=inv@np.array([loc.x,loc.y,loc.z,1]);state.extend([float(pt[0]/50),float(-pt[1]/50)])
 state.extend([0,0,0,1,0,0]*2)
 result.update(state20=state,state_sha=hashlib.sha256(json.dumps(state).encode()).hexdigest(),ego_pose={'location':[pose.location.x,pose.location.y,pose.location.z],'rotation':[pose.rotation.pitch,pose.rotation.yaw,pose.rotation.roll]},camera={'fov':70,'position_ego':[.8,0,1.6],'resolution':[448,252]},vehicle=ego.type_id,weather=str(world.get_weather()),history_interval_s=.2,noise_seed=9188)
 worker=subprocess.Popen([str(R/'.venv-train/Scripts/python.exe'),'-u',str(R/'scripts/qwen_carla_worker_v3.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(O/'worker.stderr.log').open('w'),text=True);oq=queue.Queue()
 def read():
  for line in worker.stdout:oq.put(line)
 threading.Thread(target=read,daemon=True).start();assert json.loads(oq.get(timeout=120))['ready']
 def packet():
  f=world.tick();im=iq.get(timeout=10)
  while im.frame<f:im=iq.get(timeout=10)
  assert im.frame==f;return im
 for label,d in [('none',None),('lead15',15),('lead35',35)]:
  lead=None
  if d is not None:
   loc=pose.transform(carla.Location(x=d));lt=carla.Transform(loc,pose.rotation);lead=world.try_spawn_actor(bp,lt)
   if lead is None:raise RuntimeError('Lead spawn failed '+label)
   actors.append(lead);lead.set_simulate_physics(False);lead.set_transform(lt)
  for _ in range(20):packet()
  frames=[]
  for tick in range(13):
   im=packet()
   if tick in [0,4,8,12]:
    path=O/f'{label}-{im.frame}.png';im.save_to_disk(str(path));frames.append({'path':str(path),'frame':im.frame,'timestamp':im.timestamp})
  assert all(abs(frames[i+1]['timestamp']-frames[i]['timestamp']-.2)<.005 for i in range(3));now=ego.get_transform();assert now.location.distance(pose.location)<1e-5
  worker.stdin.write(json.dumps({'images':[x['path'] for x in frames],'state':state})+'\n');worker.stdin.flush();out=json.loads(oq.get(timeout=60));assert 'error' not in out,out;xy=np.array(out['xy']);v=np.array(out['velocity'])
  if label=='none':reference=xy
  result['conditions'][label]={'lead_distance_m':d,'frames':frames,'ego_position_error_m':now.location.distance(pose.location),'prediction':out,'first_target_vx_mps':float(v[0,0]),'mean_predicted_speed_mps':float(np.linalg.norm(v,axis=1).mean()),'final5s_xy_m':xy[-1].tolist(),'trajectory_difference_vs_none_m':float(np.linalg.norm(xy-reference,axis=1).mean()),'endpoint_difference_vs_none_m':float(np.linalg.norm(xy[-1]-reference[-1]))};(O/'results.json').write_text(json.dumps(result,indent=2));print('CONDITION',label,result['conditions'][label]['first_target_vx_mps'],flush=True)
  if lead:lead.destroy();actors.remove(lead)
 result['completed']=True
finally:
 if worker:worker.terminate();worker.wait(timeout=10)
 for actor in actors[::-1]:
  try:actor.destroy()
  except Exception:pass
 world.apply_settings(old);result['cleanup']='own actors destroyed; world settings restored; server left running';(O/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:{z:v[z] for z in ['first_target_vx_mps','mean_predicted_speed_mps','final5s_xy_m','trajectory_difference_vs_none_m']} for k,v in result['conditions'].items()}),flush=True)
