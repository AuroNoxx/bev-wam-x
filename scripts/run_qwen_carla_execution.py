"""New Qwen RGB loop, synchronous guarded smoke. Not a safety validation."""
from pathlib import Path
import json,time,queue,subprocess,math,sys,threading
import numpy as np
import carla
MODE=sys.argv[1] if len(sys.argv)>1 else 'model'
assert MODE in ['model','baseline']
R=Path(__file__).resolve().parents[1];O=R/('artifacts/qwen-visual-pilot-v1/carla-execution-v1-'+MODE);O.mkdir(exist_ok=True)
c=carla.Client('127.0.0.1',2102);c.set_timeout(20);world=c.get_world();old=world.get_settings();actors=[];worker=None;records=[];report={'simulator':c.get_server_version(),'official_B2D':False,'mode':MODE+' actual trajectory execution, no quality veto','model_control_count':0,'fallback_count':0,'collisions':None};q=queue.Queue()
try:
 settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;settings.no_rendering_mode=False;world.apply_settings(settings);bp=world.get_blueprint_library();maps=world.get_map();report['map']=maps.name
 cars=[b for b in bp.filter('vehicle.*') if b.has_attribute('number_of_wheels') and b.get_attribute('number_of_wheels').as_int()==4];spawns=maps.get_spawn_points();vehicle=None
 for tr in spawns:
  wp=maps.get_waypoint(tr.location)
  if wp.is_junction:continue
  future=wp.next(20)
  if not future or future[0].is_junction:continue
  vehicle=world.try_spawn_actor(cars[0],tr)
  if vehicle:break
 if vehicle is None:raise RuntimeError('No bounded nonjunction spawn')
 actors.append(vehicle);vehicle.apply_control(carla.VehicleControl(brake=1));cam=bp.find('sensor.camera.rgb');cam.set_attribute('image_size_x','448');cam.set_attribute('image_size_y','252');cam.set_attribute('fov','70');cam.set_attribute('sensor_tick','0.0');sensor=world.spawn_actor(cam,carla.Transform(carla.Location(x=.8,z=1.6)),attach_to=vehicle);actors.append(sensor);sensor.listen(q.put)
 iq=queue.Queue();imu=world.spawn_actor(bp.find('sensor.other.imu'),carla.Transform(),attach_to=vehicle);actors.append(imu);imu.listen(iq.put);hits=[];collision=world.spawn_actor(bp.find('sensor.other.collision'),carla.Transform(),attach_to=vehicle);actors.append(collision);collision.listen(lambda ev:hits.append(ev.frame))
 report.update(camera={'fov':70,'position_ego':[.8,0,1.6],'resolution':[448,252]},navigation='map lane-follow near5m/far30m, not original B2D route planner',vehicle_type=vehicle.type_id)
 worker=subprocess.Popen([str(R/'.venv-train/Scripts/python.exe'),'-u',str(R/'scripts/qwen_carla_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(O/'worker.stderr.log').open('w'),text=True);ready=json.loads(worker.stdout.readline());assert ready.get('ready');history=[];response_queue=queue.Queue()
 def read_worker():
  for line in worker.stdout:response_queue.put(line)
 threading.Thread(target=read_worker,daemon=True).start()
 for settle in range(60):
  ff=world.tick();im=q.get(timeout=10)
  while im.frame<ff:im=q.get(timeout=10)
  imu_packet=iq.get(timeout=10)
  while imu_packet.frame<ff:imu_packet=iq.get(timeout=10)
 startloc=vehicle.get_location();start=time.perf_counter()
 for tick in range(200):
  if time.perf_counter()-start>180:raise TimeoutError('Wall-time limit180s')
  frame=world.tick();startloc=vehicle.get_location() if startloc is None else startloc;packet=q.get(timeout=10)
  while packet.frame<frame:packet=q.get(timeout=10)
  assert packet.frame==frame
  imu_packet=iq.get(timeout=10)
  while imu_packet.frame<frame:imu_packet=iq.get(timeout=10)
  assert imu_packet.frame==frame
  path=O/f'rgb-{frame}.png';packet.save_to_disk(str(path));history.append((frame,packet.timestamp,str(path)))
  if len(history)<13 or tick%4:continue
  tr=vehicle.get_transform();inv=np.array(tr.get_inverse_matrix());v=vehicle.get_velocity();speed=math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z);acc=vehicle.get_acceleration();local=inv[:3,:3]@np.array([acc.x,acc.y,acc.z]);state=[speed/30,imu_packet.accelerometer.x/10,-imu_packet.accelerometer.y/10,-imu_packet.gyroscope.z];wp=maps.get_waypoint(tr.location)
  for d in [5,30]:
   nxt=wp.next(d);loc=(nxt[0] if nxt else wp).transform.location;goal=inv@np.array([loc.x,loc.y,loc.z,1]);state.extend([float(goal[0]/50),float(-goal[1]/50)])
  state.extend([0,0,0,1,0,0]*2) # LANEFOLLOW command4, from explicit straight-road scenario
  hist=[history[-13],history[-9],history[-5],history[-1]];assert all(abs((hist[j+1][1]-hist[j][1])-.2)<.005 for j in range(3))
  if MODE=='model':
   worker.stdin.write(json.dumps({'images':[p[2] for p in hist],'state':state})+'\n');worker.stdin.flush();out=json.loads(response_queue.get(timeout=30))
  else:
   points=[]
   for i in range(25):
    nxt=wp.next(6*.2*(i+1));loc=(nxt[0] if nxt else wp).transform.location;pt=inv@np.array([loc.x,loc.y,loc.z,1]);points.append([float(pt[0]),float(-pt[1])])
   out={'xy':points,'velocity':[[6.,0.]]*25,'model_seconds':0.,'baseline':'map-lane-follow fixed6mps, not trained model'}
  reason=[]
  if 'error' in out:reason=['worker_error']
  else:
   vel=np.asarray(out['velocity']);xy=np.asarray(out['xy'])
   if xy.shape!=(25,2) or vel.shape!=(25,2) or not np.isfinite(xy).all() or not np.isfinite(vel).all():reason=['invalid_numerics']
  if reason:ctl=carla.VehicleControl(brake=1);report['fallback_count']+=1
  else:
   target=xy[4];wheel_limit=math.radians(max(w.max_steer_angle for w in vehicle.get_physics_control().wheels));raw_steer=-math.atan2(2*2.8*target[1],max(float(np.dot(target,target)),.01))/wheel_limit;targetspeed=float(vel[0,0]);raw_throttle=.2*(targetspeed-speed);raw_brake=.3*(speed-targetspeed)
   ctl=carla.VehicleControl(steer=float(np.clip(raw_steer,-1,1)),throttle=float(np.clip(raw_throttle,0,1)),brake=float(np.clip(raw_brake,0,1)));report['model_control_count']+=1
   out['controller']={'raw_steer':raw_steer,'raw_throttle':raw_throttle,'raw_brake':raw_brake,'target_speed_mps':targetspeed,'physical_control_bounds_only':True}
  vehicle.apply_control(ctl);records.append({'frame':frame,'timestamp':packet.timestamp,'history_frame_ids':[p[0] for p in hist],'speed_mps':speed,'state20':state,'ego_location':[tr.location.x,tr.location.y,tr.location.z],'fallback_reasons':reason,'output':out,'applied':{'steer':ctl.steer,'throttle':ctl.throttle,'brake':ctl.brake}});(O/'steps.json').write_text(json.dumps(records,indent=2));print('STEP',tick,reason,flush=True)
 report.update(status='completed_actual_execution',wall_seconds=time.perf_counter()-start,ticks=200,simulated_seconds=10.,displacement_m=vehicle.get_location().distance(startloc),steps=len(records),collisions=len(hits),collision_check='sensor event count in this tiny noNPC smoke only',latency_note='synchronous simulation waits for inference, not real-time validation')
except Exception as e:report.update(status='blocked_or_failed',error=repr(e));raise
finally:
 if worker:worker.terminate();worker.wait(timeout=10)
 for a in actors[::-1]:
  try:a.destroy()
  except:pass
 world.apply_settings(old);(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
