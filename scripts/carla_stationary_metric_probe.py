"""Two-second stationary owned-actor instrumentation probe, not driving acceptance."""
import json,time,queue,argparse
from pathlib import Path
import carla
from driving_metric_contract import scenario_fingerprint,evaluate_episode
from carla_episode_contract import validate_ticks,event_coverage
ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',default='artifacts/carla-stationary-metric-20261116');args=parser.parse_args();out=ROOT/args.out;out.mkdir(exist_ok=False);client=carla.Client('127.0.0.1',2100);client.set_timeout(10);world=client.get_world();owned=[];sensors=[];original=None;changed=False;report={'status':'starting'};events=[];imuq=queue.Queue();rows=[];ticks=[];matched=[]
 try:
  if world.get_actors().filter('vehicle.*') or world.get_actors().filter('sensor.*'):raise RuntimeError('Existing actors: refusing')
  original=world.get_settings()
  if original.synchronous_mode:raise RuntimeError('Existing synchronous owner: refusing')
  settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;world.apply_settings(settings);changed=True
  lib=world.get_blueprint_library();spawn=world.get_map().get_spawn_points()[0];spawn.location.z+=.2;bp=lib.find('vehicle.tesla.model3');bp.set_attribute('role_name','metric_probe_owned');ego=world.spawn_actor(bp,spawn);owned.append(ego);ego.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
  collision=world.spawn_actor(lib.find('sensor.other.collision'),carla.Transform(),attach_to=ego);owned.append(collision);sensors.append(collision)
  collision.listen(lambda e:events.append({'id':f'{collision.id}/{e.frame}/{e.other_actor.id}','kind':'collision','frame':e.frame,'other_type':e.other_actor.type_id,'impulse_norm':(e.normal_impulse.x**2+e.normal_impulse.y**2+e.normal_impulse.z**2)**.5}))
  imu=world.spawn_actor(lib.find('sensor.other.imu'),carla.Transform(),attach_to=ego);owned.append(imu);sensors.append(imu);imu.listen(imuq.put)
  for _ in range(20):world.tick(10)
  while not imuq.empty():imuq.get_nowait()
  startframe=world.get_snapshot().frame;started=bool(collision.is_listening() if callable(collision.is_listening) else collision.is_listening) and bool(imu.is_listening() if callable(imu.is_listening) else imu.is_listening);start=time.perf_counter()
  for _ in range(40):
   frame=world.tick(10);snap=world.get_snapshot();assert frame==snap.frame;actor=snap.find(ego.id);vel=actor.get_velocity();acc=actor.get_acceleration();tr=actor.get_transform();right=tr.get_right_vector();ticks.append({'frame':frame,'timestamp_s':snap.timestamp.elapsed_seconds});deadline=time.monotonic()+3;packet=None
   while time.monotonic()<deadline:
    p=imuq.get(timeout=max(.001,deadline-time.monotonic()))
    if p.frame==frame:packet=p;break
    if p.frame>frame:raise RuntimeError('IMU ahead of synchronous snapshot')
   if packet is None:raise RuntimeError('missing same-frame IMU')
   matched.append(frame);rows.append({'t_s':snap.timestamp.elapsed_seconds,'speed_mps':(vel.x**2+vel.y**2+vel.z**2)**.5,'lateral_accel_mps2':acc.x*right.x+acc.y*right.y+acc.z*right.z})
  validate_ticks(ticks);ended=bool(collision.is_listening() if callable(collision.is_listening) else collision.is_listening) and bool(imu.is_listening() if callable(imu.is_listening) else imu.is_listening);coverage=event_coverage([x['frame'] for x in ticks],matched,started,ended,[])
  config={'simulator_version':client.get_server_version(),'map':world.get_map().name,'weather':str(world.get_weather()),'traffic_seed':0,'route':{'type':'stationary-health-only'},'initial_actors':[{'blueprint':ego.type_id,'spawn':str(spawn)}],'fixed_delta_seconds':.05,'sensor_config':{'collision':'default','imu':'default'},'controller_version':'constant-brake1-handbrake'}
  active=[e for e in events if e['frame']>startframe]
  episode={'schema':'driving-metrics-v1','scenario_fingerprint':scenario_fingerprint(config),'status':'completed','samples':rows,'coverage':{},'events':active,'route_completed':None}
  report.update(status='completed',ticks=ticks,imu_recording=coverage,raw_collision_callbacks=len(active),collision_sensor_listening_start_end=[started,ended],wall_seconds=time.perf_counter()-start,scenario=config,metrics=evaluate_episode(episode),limits='IMU cadence and callback registration tested; collision response not stimulated/calibrated, so collision coverage remains unknown. Stationary health, no traffic/route policy/world/LLM.')
 except Exception as e:report.update(status='failed',error=repr(e),recorded_frames=len(rows))
 finally:
  for s in sensors:
   try:s.stop()
   except Exception:pass
  destroyed=[]
  for a in reversed(owned):
   try:destroyed.append({'id':a.id,'destroyed':a.destroy()})
   except Exception as e:destroyed.append({'id':a.id,'error':repr(e)})
  report['owned_actor_cleanup']=destroyed
  if changed:
   try:world.apply_settings(original);report['settings_restored']=True
   except Exception as e:report['settings_restored']=False;report['restore_error']=repr(e)
  (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2))
 return 0 if report['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
