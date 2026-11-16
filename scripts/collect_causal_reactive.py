"""Bounded causal sensor collection. Simulator truth is explicitly identified."""
import json,time,queue,math,hashlib
from pathlib import Path
import numpy as np
import carla

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/carla_causal_reactive_20261116_v1'
DTYPE=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('cos','<f4'),('object','<u4'),('tag','<u4')])
def vec(v):return [v.x,v.y,v.z]
def packet(q,frame):
 while True:
  p=q.get(timeout=15)
  if p.frame==frame:return p
  if p.frame>frame:raise RuntimeError('sensor ahead of world')
def teacher(speed,lead_speed,gap,cruise,permitted):
 # Current observation only: no elapsed time, phase, lead target or schedule.
 desired=max(0.,min(cruise,lead_speed+.6*(gap-(4.+1.3*speed)))) if permitted else 0.
 error=desired-speed
 return desired,float(np.clip(error*.25,0,.6)),float(np.clip(-error*.25,0,.8)) if desired>0 else 1.
def relative_bbox(ego,lead):
 et=ego.get_transform();lt=lead.get_transform();f=np.array(vec(et.get_forward_vector()));r=np.array(vec(et.get_right_vector()));o=np.array(vec(et.location))
 ec=np.array([vec(x) for x in ego.bounding_box.get_world_vertices(et)])-o;lc=np.array([vec(x) for x in lead.bounding_box.get_world_vertices(lt)])-o
 gap=float(min(lc@f)-max(ec@f));overlap=bool(min(max(ec@r),max(lc@r))>=max(min(ec@r),min(lc@r)));ev=np.array(vec(ego.get_velocity()));lv=np.array(vec(lead.get_velocity()));closing=float((ev-lv)@f)
 ttc=gap/closing if gap>0 and closing>0 and overlap else None
 return dict(gap_m=gap,lateral_overlap=overlap,closing_mps=closing,ttc_s=ttc,ttc_null_reason=None if ttc is not None else 'not_positive_gap_closing_and_overlap',ego_speed_mps=float(np.linalg.norm(ev[:2])),lead_forward_speed_mps=float(lv@f))
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
 manifest={'schema':'causal-miniloop-v1','status':'running','server':client.get_server_version(),'map':w.get_map().name,'dt':.1,'controller':'causal-current-gap-speed-following-v1','episodes':[], 'sources':{'rgb':'online camera','lidar_xyz':'online semantic lidar rays','lidar_tags':'simulator semantic truth','ego':'simulator snapshot truth','navigation':'HD map','traffic':'exogenous randomized lead schedule; current bbox gap and actual relative velocity are oracle teacher inputs'},'future_leakage':'only same-frame raw observations; future labels not in actor files','occ':'raw xyz/tags retained; no free-space inferred; 0/255 adapter not yet constructed','limitations':'reactive rule teacher with oracle lead state and HD map; not learned perception, not driving acceptance; scheduled lead is exogenous and schedule never enters ego teacher'}
 try:
  if original.synchronous_mode:raise RuntimeError('foreign sync owner')
  settings=w.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.1;settings.substepping=True;settings.max_substep_delta_time=.01;settings.max_substeps=10;w.apply_settings(settings);client.set_timeout(20)
  lib=w.get_blueprint_library();m=w.get_map()
  for ep,(road,split,target,cloud,sign) in enumerate([(388,'train',8.,0.,0),(392,'train',9.,50.,0),(713,'dev',8.5,20.,0)]):
   folder=OUT/f'episode{ep}';folder.mkdir();wp=m.get_waypoint_xodr(road,-3,100.)
   if wp is None:raise RuntimeError(f'road unavailable {road}')
   seed=[3101,3102,4101][ep];rng=np.random.default_rng(seed);lead_start=float(rng.uniform(7.5,10.));lead_decel=float(rng.uniform(23.,26.));lead_restart=float(rng.uniform(35.,39.));lead_final_decel=float(rng.uniform(49.,52.));schedule=dict(start=lead_start,decel=lead_decel,restart=lead_restart,final_decel=lead_final_decel,brake_duration=4.)
   route_s=np.arange(90.,550.001,.1);route_wp=[m.get_waypoint_xodr(road,-3,float(x)) for x in route_s]
   if any(x is None for x in route_wp):raise RuntimeError('fixed route unavailable')
   route_xyz=np.array([vec(x.transform.location) for x in route_wp])
   def match(location,last_s):
    d=np.linalg.norm(route_xyz[:,:2]-np.array(vec(location))[:2],axis=1);d[np.abs(route_s-last_s)>10]=np.inf
    idx=int(np.argmin(d))
    if not np.isfinite(d[idx]):raise RuntimeError('route projection unavailable')
    return route_wp[idx]
   ego_s=100.;lead_s=125.
   tr=wp.transform;right=tr.get_right_vector();tr.location.x+=sign*.3*right.x;tr.location.y+=sign*.3*right.y;tr.rotation.yaw+=sign*2.;tr.location.z+=.3
   ego=w.spawn_actor(lib.find('vehicle.tesla.model3'),tr);owned.append(ego)
   center=m.get_waypoint_xodr(road,-3,100).transform;rv=np.array(vec(center.get_right_vector()));cv=np.array(vec(center.location));corners=np.array([vec(x) for x in ego.bounding_box.get_world_vertices(tr)]);margin=1.75-float(np.max(abs((corners-cv)@rv)))
   if margin<.1:raise RuntimeError(f'unsafe spawn bbox margin {margin}')
   leadtr=wp.next(25.)[0].transform;leadtr.location.z+=.3
   lead=w.spawn_actor(lib.find('vehicle.tesla.model3'),leadtr);owned.append(lead)
   w.set_weather(carla.WeatherParameters(cloudiness=cloud,sun_altitude_angle=60.))
   qs={}; specs={'rgb':('sensor.camera.rgb',{'image_size_x':'640','image_size_y':'360','fov':'90','sensor_tick':'0.0'},carla.Transform(carla.Location(x=1.5,z=1.8))), 'lidar':('sensor.lidar.ray_cast_semantic',{'range':'60','channels':'32','points_per_second':'160000','rotation_frequency':'10','upper_fov':'15','lower_fov':'-30','sensor_tick':'0.0'},carla.Transform(carla.Location(z=2.2)))}
   for key,(name,attrs,transform) in specs.items():
    bp=lib.find(name)
    for k,v in attrs.items():bp.set_attribute(k,v)
    s=w.spawn_actor(bp,transform,attach_to=ego);owned.append(s);sensors.append(s);qs[key]=queue.Queue();s.listen(qs[key].put)
   events=[];col=w.spawn_actor(lib.find('sensor.other.collision'),carla.Transform(),attach_to=ego);owned.append(col);sensors.append(col);col.listen(lambda e:events.append({'frame':e.frame,'other_type':e.other_actor.type_id,'impulse':vec(e.normal_impulse)}))
   rows=[];start=time.monotonic();episode_guard=None
   for i in range(600):
    tr=ego.get_transform();vel=ego.get_velocity();speed=math.sqrt(vel.x**2+vel.y**2+vel.z**2);wp=match(tr.location,ego_s);ego_s=wp.s;fixed=wp;goalwp=m.get_waypoint_xodr(road,-3,ego_s+max(5.,speed*.8))
    if fixed is None or goalwp is None:raise RuntimeError('fixed route missing')
    goal=goalwp.transform.location;local=np.array(tr.get_inverse_matrix())@np.r_[vec(goal),1.];steer=float(np.clip(2*2.8*local[1]/max(25.,local[0]**2+local[1]**2)/.6,-.6,.6))
    pre_snap=w.get_snapshot();observation=relative_bbox(ego,lead);permitted=i>=70
    desired,throttle,brake=teacher(observation['ego_speed_mps'],observation['lead_forward_speed_mps'],observation['gap_m'],target,permitted)
    safety_takeover=observation['lateral_overlap'] and (observation['gap_m']<2. or (observation['ttc_s'] is not None and observation['ttc_s']<1.))
    control=carla.VehicleControl(throttle=0. if safety_takeover else throttle,brake=1. if safety_takeover else brake,steer=steer)
    lt=lead.get_transform();lv=lead.get_velocity();ls=math.sqrt(lv.x**2+lv.y**2+lv.z**2);lw=match(lt.location,lead_s);lead_s=lw.s;lgoal=m.get_waypoint_xodr(road,-3,lead_s+max(5.,ls*.8));lp=lgoal.transform.location;ll=np.array(lt.get_inverse_matrix())@np.r_[vec(lp),1.];lsteer=float(np.clip(2*2.8*ll[1]/max(25.,ll[0]**2+ll[1]**2)/.6,-.6,.6));t=i*.1;ld=0. if t<lead_start else target if t<lead_decel else target*max(0.,1.-(t-lead_decel)/4.) if t<lead_restart else target if t<lead_final_decel else target*max(0.,1.-(t-lead_final_decel)/4.);le=ld-ls
    lead.apply_control(carla.VehicleControl(throttle=float(np.clip(le*.25,0,.6)),brake=float(np.clip(-le*.25,0,.8)) if ld>0 else 1.,steer=lsteer))
    ego.apply_control(control);frame=w.tick(20);snap=w.get_snapshot();actor=snap.find(ego.id);tr=actor.get_transform();rgb=packet(qs['rgb'],frame);lidar=packet(qs['lidar'],frame)
    assert frame==snap.frame and abs(rgb.timestamp-snap.timestamp.elapsed_seconds)<1e-4 and abs(lidar.timestamp-rgb.timestamp)<1e-4
    raw=np.frombuffer(lidar.raw_data,DTYPE);np.savez_compressed(folder/f'{i:05}.npz',rgb_bgra=np.frombuffer(rgb.raw_data,np.uint8).reshape(360,640,4),lidar_xyz=np.stack([raw['x'],raw['y'],raw['z']],-1),lidar_tag=raw['tag'],lidar_world=np.array(lidar.transform.get_matrix()),ego_world=np.array(tr.get_matrix()))
    rows.append({'index':i,'frame':frame,'timestamp':snap.timestamp.elapsed_seconds,'rgb_frame':rgb.frame,'lidar_frame':lidar.frame,'ego_world':tr.get_matrix(),'velocity_world':vec(actor.get_velocity()),'acceleration_world':vec(actor.get_acceleration()),'angular_velocity_deg':vec(actor.get_angular_velocity()),'control_applied':{'throttle':control.throttle,'brake':control.brake,'steer':control.steer},'target_speed':desired,'road':wp.road_id,'lane':wp.lane_id,'navigation_world':vec(goal),'phase':'settle_stop' if not permitted else 'reactive_follow', 'teacher_observation_frame':pre_snap.frame,'teacher_observation_timestamp':pre_snap.timestamp.elapsed_seconds,'teacher_inputs':dict(**observation,cruise_mps=target,movement_permitted=permitted),'lead_target_eval_only':ld,'lead_actual_post_eval_only':relative_bbox(ego,lead),'teacher_controller':'current-gap-speed-only','safety_takeover':bool(safety_takeover)})
    applied_goal=vec(goal);wp=match(tr.location,ego_s);ego_s=wp.s;goal=m.get_waypoint_xodr(road,-3,ego_s+max(5.,speed*.8)).transform.location
    rows[-1]['navigation_world']=vec(goal)
    center=wp.transform;right=np.array(vec(center.get_right_vector()));err=float(np.dot(np.array(vec(tr.location))-vec(center.location),right));yawerr=(tr.rotation.yaw-center.rotation.yaw+180)%360-180
    corners=np.array([vec(x) for x in ego.bounding_box.get_world_vertices(tr)]);edge=1.75-float(np.max(abs((corners-np.array(vec(center.location)))@right)));actualwp=m.get_waypoint(tr.location);navlocal=np.array(tr.get_inverse_matrix())@np.r_[vec(goal),1.]
    episode_guard='rule_takeover_gap_or_ttc' if safety_takeover else 'collision' if events else ('lane_or_bbox_boundary' if edge<0 else ('overspeed' if speed>14 else None))
    rows[-1].update(source_route=road,desired_lane=-3,projection_s=ego_s,projection_method='fixed-road-lane-xy-nearest-0.1m-continuity10m',global_nearest_road_diagnostic=actualwp.road_id,global_nearest_lane_diagnostic=actualwp.lane_id,controller_goal_previous_frame=applied_goal,near_goal_world=vec(goal),near_goal_local_carla=navlocal[:3].tolist(),lane_error_right_m=err,heading_error_deg=yawerr,bbox_lane_margin_m=edge,guard=episode_guard,valid_for_training=episode_guard is None)
    if i%50==0:print(ep,i,frame,err,edge,flush=True)
    if episode_guard:ego.apply_control(carla.VehicleControl(brake=1));break
   (folder/'frames.json').write_text(json.dumps(rows));(folder/'events.json').write_text(json.dumps(events))
   manifest['episodes'].append({'episode':ep,'split':split,'initial_lateral_right_m':sign*.3,'initial_yaw_deg':sign*2.,'spawn_bbox_margin_m':margin,'status':'guard_stop' if episode_guard else 'completed','guard':episode_guard,'road':road,'lane':-3,'spawn_s':100.,'seed':seed,'lead_schedule_eval_only':schedule,'frames':len(rows),'target_speed_mps':target,'weather_cloudiness':cloud,'wall_seconds':time.monotonic()-start,'collision_callbacks':len(events),'collision_coverage':'registered callback only, not calibrated','max_speed_mps':max(np.linalg.norm(r['velocity_world']) for r in rows),'final_speed_mps':float(np.linalg.norm(rows[-1]['velocity_world']))})
   for s in sensors:s.stop()
   for a in reversed(owned):a.destroy()
   sensors.clear();owned.clear();(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
  manifest['status']='completed' if all(e['status']=='completed' for e in manifest['episodes']) else 'partial_with_guard_stops'
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



