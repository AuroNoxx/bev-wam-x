"""New Qwen RGB loop, synchronous guarded smoke. Not a safety validation."""
from pathlib import Path
import json,time,queue,subprocess,math,sys,threading
import numpy as np
import carla
MODE=sys.argv[1] if len(sys.argv)>1 else 'model'
assert MODE in ['model','baseline']
SCENARIO=sys.argv[2] if len(sys.argv)>2 else 'follow'
SEED=int(sys.argv[3]) if len(sys.argv)>3 else 0
assert SCENARIO in ['follow','brake','lanechange']
R=Path(__file__).resolve().parents[1];O=R/('artifacts/b2d-0915/smoke-scenarios/'+SCENARIO+'-'+str(SEED)+'-'+MODE);O.mkdir(parents=True,exist_ok=True)
c=carla.Client('127.0.0.1',2115);c.set_timeout(20);world=c.get_world();old=world.get_settings();actors=[];worker=None;records=[];lead=None;report={'simulator':c.get_server_version(),'official_B2D':False,'mode':MODE+' actual trajectory execution, no quality veto','model_control_count':0,'fallback_count':0,'collisions':None,'scenario':SCENARIO,'seed':SEED,'initial_ego_speed_mps':3};q=queue.Queue()
try:
 settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;settings.no_rendering_mode=False;world.apply_settings(settings);bp=world.get_blueprint_library();maps=world.get_map();report['map']=maps.name
 cars=[b for b in bp.filter('vehicle.*') if b.has_attribute('number_of_wheels') and b.get_attribute('number_of_wheels').as_int()==4];spawns=maps.get_spawn_points();vehicle=None
 eligible=[]
 for tr in spawns:
  wp=maps.get_waypoint(tr.location);left=wp.get_left_lane();future=wp.next(65)
  if wp.is_junction or not future or future[0].is_junction:continue
  if SCENARIO=='lanechange' and (left is None or left.lane_type!=carla.LaneType.Driving or left.transform.get_forward_vector().dot(wp.transform.get_forward_vector())<.8):continue
  eligible.append(tr)
 for tr in eligible[SEED:]:
  wp=maps.get_waypoint(tr.location)
  if wp.is_junction:continue
  future=wp.next(20)
  if not future or future[0].is_junction:continue
  vehicle=world.try_spawn_actor(cars[0],tr)
  if vehicle:break
 if vehicle is None:raise RuntimeError('No bounded nonjunction spawn')
 actors.append(vehicle);vehicle.apply_control(carla.VehicleControl(brake=1));route=[maps.get_waypoint(tr.location)];route_index=0
 if SCENARIO=='lanechange':route=[route[0].get_left_lane()]
 for _ in range(300):
  nxt=route[-1].next(1.)
  if not nxt:break
  route.append(nxt[0])
 if SCENARIO in ['follow','brake']:
  leadwp=maps.get_waypoint(tr.location).next(20)[0];lt=leadwp.transform;lt.location.z+=.5;lead=world.try_spawn_actor(cars[0],lt)
  if lead is None:raise RuntimeError('Lead spawn failed')
  actors.append(lead)
 cam=bp.find('sensor.camera.rgb');cam.set_attribute('image_size_x','448');cam.set_attribute('image_size_y','252');cam.set_attribute('fov','70');cam.set_attribute('sensor_tick','0.0');sensor=world.spawn_actor(cam,carla.Transform(carla.Location(x=.8,z=1.6)),attach_to=vehicle);actors.append(sensor);sensor.listen(q.put)
 iq=queue.Queue();imu=world.spawn_actor(bp.find('sensor.other.imu'),carla.Transform(),attach_to=vehicle);actors.append(imu);imu.listen(iq.put);hits=[];collision=world.spawn_actor(bp.find('sensor.other.collision'),carla.Transform(),attach_to=vehicle);actors.append(collision);collision.listen(lambda ev:hits.append({'frame':ev.frame,'other_actor':ev.other_actor.type_id,'impulse':[ev.normal_impulse.x,ev.normal_impulse.y,ev.normal_impulse.z]}))
 report.update(camera={'fov':70,'position_ego':[.8,0,1.6],'resolution':[448,252]},navigation='anchored monotonic lane route near5m/far30m; not original B2D route planner',vehicle_type=vehicle.type_id)
 history=[];response_queue=queue.Queue()
 if MODE=='model':
  worker=subprocess.Popen([str(R/'.venv-train/Scripts/python.exe'),'-u',str(R/'scripts/qwen_carla_worker_v4.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(O/'worker.stderr.log').open('w'),text=True);ready=json.loads(worker.stdout.readline());assert ready.get('ready')
  def read_worker():
   for line in worker.stdout:response_queue.put(line)
  threading.Thread(target=read_worker,daemon=True).start()
 for settle in range(60):
  ff=world.tick();im=q.get(timeout=10)
  while im.frame<ff:im=q.get(timeout=10)
  imu_packet=iq.get(timeout=10)
  while imu_packet.frame<ff:imu_packet=iq.get(timeout=10)
 startloc=vehicle.get_location();start=time.perf_counter();speed_integral=0.;fw=vehicle.get_transform().get_forward_vector();vehicle.apply_control(carla.VehicleControl());vehicle.set_target_velocity(carla.Vector3D(x=fw.x*3,y=fw.y*3,z=0))
 for tick in range(400):
  if time.perf_counter()-start>180:raise TimeoutError('Wall-time limit180s')
  if lead:
   lwp=maps.get_waypoint(lead.get_location());lf=lwp.transform.get_forward_vector();ls=0. if SCENARIO=='brake' and tick>=80 else 3.;lead.set_target_velocity(carla.Vector3D(x=lf.x*ls,y=lf.y*ls,z=0));lead.apply_control(carla.VehicleControl(brake=1 if ls==0 else 0))
  frame=world.tick();startloc=vehicle.get_location() if startloc is None else startloc;packet=q.get(timeout=10)
  while packet.frame<frame:packet=q.get(timeout=10)
  assert packet.frame==frame
  imu_packet=iq.get(timeout=10)
  while imu_packet.frame<frame:imu_packet=iq.get(timeout=10)
  assert imu_packet.frame==frame
  path=O/f'rgb-{frame}.png';packet.save_to_disk(str(path));history.append((frame,packet.timestamp,str(path)))
  if len(history)<13 or tick%4:continue
  tr=vehicle.get_transform();inv=np.array(tr.get_inverse_matrix());v=vehicle.get_velocity();speed=math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z);acc=vehicle.get_acceleration();local=inv[:3,:3]@np.array([acc.x,acc.y,acc.z]);state=[speed/30,imu_packet.accelerometer.x/10,-imu_packet.accelerometer.y/10,-imu_packet.gyroscope.z];wp=maps.get_waypoint(tr.location)
  candidate_indices=range(route_index,min(route_index+31,len(route)))
  route_index=min(candidate_indices,key=lambda k:route[k].transform.location.distance(tr.location))
  for d in [5,30]:
   loc=route[min(route_index+d,len(route)-1)].transform.location;goal=inv@np.array([loc.x,loc.y,loc.z,1]);state.extend([float(goal[0]/50),float(-goal[1]/50)])
  state.extend(([0,0,0,0,1,0] if SCENARIO=='lanechange' else [0,0,0,1,0,0])*2) # LANEFOLLOW command4, from explicit straight-road scenario
  hist=[history[-13],history[-9],history[-5],history[-1]];assert all(abs((hist[j+1][1]-hist[j][1])-.2)<.005 for j in range(3))
  gap=None;ttc=None
  if lead:
   rel=inv@np.array([lead.get_location().x,lead.get_location().y,lead.get_location().z,1]);gap=float(rel[0]-vehicle.bounding_box.extent.x-lead.bounding_box.extent.x);lv=lead.get_velocity();closing=speed-math.sqrt(lv.x*lv.x+lv.y*lv.y);ttc=gap/closing if closing>0 and gap>0 else None
  if MODE=='model':
   worker.stdin.write(json.dumps({'images':[p[2] for p in hist],'state':state})+'\n');worker.stdin.flush();out=json.loads(response_queue.get(timeout=30))
  else:
   target_rule_speed=6. if gap is None else min(6.,max(0.,(gap-5)*.5))
   points=[]
   for i in range(25):
    loc=route[min(route_index+max(1,round(target_rule_speed*.2*(i+1))),len(route)-1)].transform.location;pt=inv@np.array([loc.x,loc.y,loc.z,1]);points.append([float(pt[0]),float(-pt[1])])
   out={'xy':points,'velocity':[[target_rule_speed,0.]]*25,'model_seconds':0.,'baseline':'privileged map lane path + actor gap rule max6mps + PI; no learned model'}
  reason=[]
  if 'error' in out:reason=['worker_error']
  else:
   vel=np.asarray(out['velocity']);xy=np.asarray(out['xy'])
   if xy.shape!=(25,2) or vel.shape!=(25,2) or not np.isfinite(xy).all() or not np.isfinite(vel).all():reason=['invalid_numerics']
  if reason:ctl=carla.VehicleControl(brake=1);report['fallback_count']+=1
  else:
   target=xy[4];wheel_limit=math.radians(max(w.max_steer_angle for w in vehicle.get_physics_control().wheels));raw_steer=-math.atan2(2*2.8*target[1],max(float(np.dot(target,target)),.01))/wheel_limit;targetspeed=float(vel[0,0]);speed_error=targetspeed-speed;candidate_integral=speed_integral+.2*speed_error;candidate_command=.2*speed_error+.1*candidate_integral
   if abs(candidate_command)<=1 or candidate_command*speed_error<0:speed_integral=candidate_integral
   longitudinal=.2*speed_error+.1*speed_integral;raw_throttle=max(0.,longitudinal);raw_brake=max(0.,-longitudinal)
   ctl=carla.VehicleControl(steer=float(np.clip(raw_steer,-1,1)),throttle=float(np.clip(raw_throttle,0,1)),brake=float(np.clip(raw_brake,0,1)));report['model_control_count']+=1
   out['controller']={'raw_steer':raw_steer,'raw_throttle':raw_throttle,'raw_brake':raw_brake,'target_speed_mps':targetspeed,'speed_integral':speed_integral,'speed_error':speed_error,'Kp':.2,'Ki':.1,'dt':.2,'physical_control_bounds_only':True}
  vehicle.apply_control(ctl);records.append({'frame':frame,'timestamp':packet.timestamp,'history_frame_ids':[p[0] for p in hist],'speed_mps':speed,'route_index':route_index,'nearest_lane_distance_m':math.hypot(tr.location.x-wp.transform.location.x,tr.location.y-wp.transform.location.y),'lead_gap_m':gap,'TTC_s':ttc,'state20':state,'ego_location':[tr.location.x,tr.location.y,tr.location.z],'fallback_reasons':reason,'output':out,'applied':{'steer':ctl.steer,'throttle':ctl.throttle,'brake':ctl.brake}});(O/'steps.json').write_text(json.dumps(records,indent=2));print('STEP',tick,reason,flush=True)
 report.update(status='completed_actual_execution',wall_seconds=time.perf_counter()-start,ticks=400,simulated_seconds=20.,displacement_m=vehicle.get_location().distance(startloc),steps=len(records),collisions=len(hits),collision_events=hits,collision_check='sensor events with controlled lead actor (follow/brake), no uncontrolled traffic',latency_note='synchronous simulation waits for inference, not real-time validation')
except Exception as e:report.update(status='blocked_or_failed',error=repr(e));raise
finally:
 if worker:worker.terminate();worker.wait(timeout=10)
 for a in actors[::-1]:
  try:a.destroy()
  except:pass
 world.apply_settings(old);(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
