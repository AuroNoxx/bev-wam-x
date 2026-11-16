"""Bounded owned CARLA low-speed collision calibration and paired rule probe."""
import carla,json,time,math,queue
from pathlib import Path
from driving_metric_contract import scenario_fingerprint,evaluate_episode,paired_comparison
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/carla-rule-event-calibration-20261116';OUT.mkdir(exist_ok=False)
c=carla.Client('127.0.0.1',2100);c.set_timeout(5)
for attempt in range(30):
 try:w=c.get_world();break
 except RuntimeError:
  if attempt==29:raise
  time.sleep(2)
owned=[];orig=None;changed=False;report={'status':'starting','runs':[]}
try:
 if w.get_actors().filter('vehicle.*') or w.get_actors().filter('sensor.*'):raise RuntimeError('occupied actors')
 orig=w.get_settings()
 if orig.synchronous_mode:raise RuntimeError('occupied sync')
 s=w.get_settings();s.synchronous_mode=True;s.fixed_delta_seconds=.05;w.apply_settings(s);changed=True
 lib=w.get_blueprint_library();mp=w.get_map();chosen=None
 for sp in mp.get_spawn_points():
  wp=mp.get_waypoint(sp.location);n=wp.next(15)
  if n and not wp.is_junction and not n[0].is_junction and abs((n[0].transform.rotation.yaw-wp.transform.rotation.yaw+180)%360-180)<2:
   chosen=(sp,n[0].transform);break
 if chosen is None:raise RuntimeError('no straight 15m nonjunction spawn')
 sp,ob=chosen;sp.location.z+=.2;ob.location.z+=.2
 config={'simulator_version':c.get_server_version(),'map':mp.name,'weather':str(w.get_weather()),'traffic_seed':0,'route':{'start':str(sp),'obstacle':str(ob),'type':'straight15m'},'initial_actors':[{'blueprint':'vehicle.tesla.model3','transform':str(sp)},{'blueprint':'vehicle.tesla.model3','transform':str(ob)}],'fixed_delta_seconds':.05,'sensor_config':{'collision':'default','imu':'default'},'controller_version':'paired-rule-protocol-v1'}
 fingerprint=scenario_fingerprint(config)
 for mode in ['calibration_contact','gap_brake','crawl_brake']:
  events=[];q=queue.Queue();rows=[];ticks=[];matched=[]
  bp=lib.find('vehicle.tesla.model3');bp.set_attribute('role_name','rule_calibration_owned')
  ego=w.spawn_actor(bp,sp);owned.append(ego);obs=w.spawn_actor(bp,ob);owned.append(obs);obs.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
  col=w.spawn_actor(lib.find('sensor.other.collision'),carla.Transform(),attach_to=ego);owned.append(col)
  col.listen(lambda e:events.append({'id':f'{e.frame}/{e.other_actor.id}','kind':'collision','frame':e.frame,'other_id':e.other_actor.id,'impulse':e.normal_impulse.length()}))
  imu=w.spawn_actor(lib.find('sensor.other.imu'),carla.Transform(),attach_to=ego);owned.append(imu);imu.listen(q.put)
  ego.apply_control(carla.VehicleControl(brake=1))
  for _ in range(20):w.tick()
  startframe=w.get_snapshot().frame;events.clear();origin=ego.get_location();forward=ego.get_transform().get_forward_vector();wall=time.perf_counter();stop_reason='duration'
  for i in range(160):
   tr=ego.get_transform();v=ego.get_velocity();speed=v.length();delta=obs.get_location()-tr.location;gap=delta.x*forward.x+delta.y*forward.y-ego.bounding_box.extent.x-obs.bounding_box.extent.x
   target=3.0 if mode!='crawl_brake' else 1.5
   braking=mode!='calibration_contact' and gap<max(4.0,speed*1.8)
   ego.apply_control(carla.VehicleControl(throttle=min(.3,max(0,(target-speed)*.25)) if not braking else 0,brake=.6 if braking else 0))
   frame=w.tick();snap=w.get_snapshot();a=snap.find(ego.id);v=a.get_velocity();tr=a.get_transform();acc=a.get_acceleration();right=tr.get_right_vector();deadline=time.monotonic()+3
   while True:
    pkt=q.get(timeout=max(.001,deadline-time.monotonic()))
    if pkt.frame==frame:matched.append(frame);break
    if pkt.frame>frame or time.monotonic()>deadline:raise RuntimeError('IMU frame mismatch')
   pos=tr.location-origin;relative=obs.get_location()-tr.location;gap=relative.x*forward.x+relative.y*forward.y-ego.bounding_box.extent.x-obs.bounding_box.extent.x
   rows.append({'t_s':snap.timestamp.elapsed_seconds,'speed_mps':v.length(),'route_progress_m':pos.x*forward.x+pos.y*forward.y,'lateral_accel_mps2':acc.x*right.x+acc.y*right.y,'lead_valid':True,'lead_in_path':abs(relative.x*right.x+relative.y*right.y)<.5,'bumper_gap_m':gap,'closing_speed_mps':v.x*forward.x+v.y*forward.y})
   ticks.append(frame)
   if events:stop_reason='collision_callback';break
  time.sleep(.1);active=[e for e in events if e['frame']>startframe]
  # Positive stimulus only certifies callback response for this obstacle; no global event completeness.
  ep={'schema':'driving-metrics-v1','scenario_fingerprint':fingerprint,'status':'completed','samples':rows,'coverage':{},'events':active,'route_completed':None}
  run={'mode':mode,'stop_reason':stop_reason,'frames':len(ticks),'imu_matched':len(matched),'wall_s':time.perf_counter()-wall,'raw_collision_callbacks':active,'metrics':evaluate_episode(ep),'limits':'privileged straight static-obstacle geometry, no learned OCC/traffic/LLM. Counts remain unknown, calibration is not full coverage.'}
  report['runs'].append(run);(OUT/(mode+'.json')).write_text(json.dumps({'episode':ep,'run':run},indent=2))
  col.stop();imu.stop()
  for actor in reversed(owned):actor.destroy()
  owned.clear()
  for _ in range(5):w.tick()
 report.update(status='completed',scenario=config,scenario_fingerprint=fingerprint,paired=paired_comparison(report['runs'][1]['metrics'],report['runs'][2]['metrics']))
except Exception as e:report.update(status='failed',error=repr(e))
finally:
 for a in reversed(owned):
  try:
   if a.type_id.startswith('sensor.'):a.stop()
   a.destroy()
  except Exception:pass
 if changed:w.apply_settings(orig)
 report['settings_restored']=changed;report['remaining_vehicles']=len(w.get_actors().filter('vehicle.*'));report['remaining_sensors']=len(w.get_actors().filter('sensor.*'))
 (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
