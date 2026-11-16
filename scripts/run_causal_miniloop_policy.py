"""Bounded CARLA learned-trajectory smoke. Requires an explicit inference worker.
No invocation at import. Simulator semantic/pose truth is not real perception.
"""
import argparse,json,math,queue,subprocess,threading,time
from pathlib import Path
import numpy as np
import carla
from collect_causal_miniloop import DTYPE,packet,vec
from learned_policy_bridge_contract import candidate_to_physical
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--seconds',type=float,default=25);a=p.parse_args()
 if not 1<=a.seconds<=30:raise ValueError('bounded 1..30 seconds only')
 out=ROOT/a.out;out.mkdir(exist_ok=False);live=out/'live';live.mkdir()
 stderr=(out/'worker.stderr.log').open('w');worker=subprocess.Popen([str(ROOT/'.venv-train/Scripts/python.exe'),'-B','-u',a.worker,'--checkpoint',a.checkpoint],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,bufsize=1,creationflags=subprocess.CREATE_NO_WINDOW)
 answers=queue.Queue()
 def reader():
  for line in worker.stdout:
   try:answers.put(json.loads(line))
   except Exception:answers.put({'error':'worker stdout is not JSON','line':line[:300]})
 threading.Thread(target=reader,daemon=True).start()
 c=carla.Client('127.0.0.1',2100);c.set_timeout(20);w=c.get_world();original=None;oldweather=None;owned=[];sensors=[];records=[];collisions=[];paths=[];rows=[]
 report={'status':'starting','checkpoint':a.checkpoint,'worker':a.worker,'route':{'map':'Town12','road':713,'lane':-3,'s':100},'seed':0,'dt':.1,'source':'online semantic LiDAR tags + simulator ego pose + HD map','controller':'trajectory-pure-pursuit-speedP-smoke-v1','limits':'K1 fixed noise/16 Euler; synchronous paused wall-time; guards are interventions, not learned success; no driving safety acceptance'}
 try:
  if 'Town12' not in w.get_map().name:raise RuntimeError('Expected Town12; no implicit map change')
  if w.get_actors().filter('vehicle.*') or w.get_actors().filter('sensor.*'):raise RuntimeError('foreign actors present')
  original=w.get_settings();oldweather=w.get_weather()
  if original.synchronous_mode:raise RuntimeError('foreign sync owner')
  st=w.get_settings();st.synchronous_mode=True;st.fixed_delta_seconds=.1;st.substepping=True;st.max_substep_delta_time=.01;st.max_substeps=10;w.apply_settings(st);w.set_weather(carla.WeatherParameters(cloudiness=20,sun_altitude_angle=60))
  m=w.get_map();lib=w.get_blueprint_library();wp=m.get_waypoint_xodr(713,-3,100);tr=wp.transform;tr.location.z+=.3;ego=w.spawn_actor(lib.find('vehicle.tesla.model3'),tr);owned.append(ego)
  lt=wp.next(25)[0].transform;lt.location.z+=.3;lead=w.spawn_actor(lib.find('vehicle.tesla.model3'),lt);owned.append(lead)
  q=queue.Queue();bp=lib.find('sensor.lidar.ray_cast_semantic')
  for k,v in {'range':'60','channels':'32','points_per_second':'160000','rotation_frequency':'10','upper_fov':'15','lower_fov':'-30','sensor_tick':'0'}.items():bp.set_attribute(k,v)
  lidar=w.spawn_actor(bp,carla.Transform(carla.Location(z=2.2)),attach_to=ego);owned.append(lidar);sensors.append(lidar);lidar.listen(q.put)
  collision=w.spawn_actor(lib.find('sensor.other.collision'),carla.Transform(),attach_to=ego);owned.append(collision);sensors.append(collision);collision.listen(lambda e:collisions.append({'frame':e.frame,'other_type':e.other_actor.type_id}))
  cmd=carla.VehicleControl(brake=1);start=time.monotonic()
  for i in range(int(a.seconds/.1)+36):
   elapsed=max(0,(i-36)*.1);lt=lead.get_transform();lv=lead.get_velocity();ls=np.linalg.norm(vec(lv));lwp=m.get_waypoint(lt.location);nxt=lwp.next(max(5,ls*.8));goal=nxt[0].transform.location if nxt else lwp.transform.location;local=np.array(lt.get_inverse_matrix())@np.r_[vec(goal),1.];steer=float(np.clip(2*2.8*local[1]/max(25,local[0]**2+local[1]**2)/.6,-.6,.6));target=0 if i<36 else (9 if elapsed<15 else max(0,9*(21-elapsed)/6));err=target-ls
   lead.apply_control(carla.VehicleControl(throttle=float(np.clip(err*.25,0,.6)),brake=float(np.clip(-err*.25,0,.8)) if target else 1.,steer=steer));ego.apply_control(cmd)
   f=w.tick(20);snap=w.get_snapshot();obs=packet(q,f);actor=snap.find(ego.id);tr=actor.get_transform();v=actor.get_velocity();speed=float(np.linalg.norm(vec(v)));wp=m.get_waypoint(tr.location);nxt=wp.next(max(5,speed*.8));goal=nxt[0].transform.location if nxt else wp.transform.location
   if abs(obs.timestamp-snap.timestamp.elapsed_seconds)>1e-4:raise RuntimeError('sensor timestamp mismatch')
   raw=np.frombuffer(obs.raw_data,DTYPE);path=live/f'{i:05}.npz';np.savez_compressed(path,lidar_xyz=np.stack([raw['x'],raw['y'],raw['z']],-1),lidar_tag=raw['tag'],lidar_world=np.array(obs.transform.get_matrix()),ego_world=np.array(tr.get_matrix()))
   row={'index':i,'frame':f,'timestamp':snap.timestamp.elapsed_seconds,'lidar_frame':obs.frame,'ego_world':tr.get_matrix(),'velocity_world':vec(v),'acceleration_world':vec(actor.get_acceleration()),'angular_velocity_deg':vec(actor.get_angular_velocity()),'navigation_world':vec(goal),'road':wp.road_id,'lane':wp.lane_id};paths.append(str(path.resolve()));rows.append(row)
   guard=None;trajectory=None;latency=None
   if collisions:guard='collision callback'
   elif speed>14:guard='speed exceeds 14m/s'
   elif wp.road_id!=713 or wp.lane_id!=-3:guard='outside fixed road/lane'
   if i>=36 and guard is None:
    started=time.monotonic();worker.stdin.write(json.dumps({'op':'infer','frames':paths[-7:],'rows':rows[-7:]})+'\n');worker.stdin.flush();result=answers.get(timeout=30);latency=time.monotonic()-started
    if result.get('error'):raise RuntimeError(str(result['error']))
    trajectory=candidate_to_physical(result['trajectory'])['trajectory_flu_m_sin_cos_mps'];look=max(4,speed*.8);idx=int(np.argmin(abs(np.linalg.norm(trajectory[:,:2],axis=1)-look)));x,y=trajectory[idx,:2]
    if x<-.5 or np.max(abs(trajectory[:,1]))>15 or trajectory[:,4].max()>20:guard='invalid or excessive candidate'
    else:
     steer=float(np.clip(-2*2.8*y/max(16,x*x+y*y)/.6,-.5,.5));desired=float(trajectory[min(2,idx),4]);err=desired-speed;cmd=carla.VehicleControl(throttle=float(np.clip(err*.25,0,.6)),brake=float(np.clip(-err*.25,0,.8)),steer=steer)
   records.append({'frame':f,'sim_elapsed':elapsed,'speed_mps':speed,'pose':tr.get_matrix(),'inference_wall_seconds':latency,'trajectory_physical':trajectory.tolist() if trajectory is not None else None,'guard':guard,'next_control':{'throttle':cmd.throttle,'brake':cmd.brake,'steer':cmd.steer}})
   if guard:ego.apply_control(carla.VehicleControl(brake=1));report.update(status='guard_stop',guard=guard);break
  else:report['status']='completed'
  report['wall_seconds']=time.monotonic()-start
 except Exception as e:report.update(status='failed',error=repr(e))
 finally:
  for s in sensors:
   try:s.stop()
   except Exception:pass
  for actor in reversed(owned):
   try:actor.destroy()
   except Exception:pass
  if original is not None:w.apply_settings(original)
  if oldweather is not None:w.set_weather(oldweather)
  worker.terminate();worker.wait(timeout=10);stderr.close();(out/'frames.json').write_text(json.dumps(rows));(out/'records.json').write_text(json.dumps(records));report['collision_callbacks']=collisions;report['records']=len(records);(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()

