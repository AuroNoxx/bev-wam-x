"""CARLA 0.9.15 sensor contract probe. Actor truth is diagnostic only, never policy input."""
import argparse,json,math,queue,time
from pathlib import Path
import numpy as np
import carla

def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=2115);p.add_argument('--out',default='artifacts/b2d-0915/sensor_contract.json');a=p.parse_args()
 c=carla.Client('127.0.0.1',a.port);c.set_timeout(30);w=c.get_world();old=w.get_settings();actors=[];report={'client':c.get_client_version(),'server':c.get_server_version(),'map':w.get_map().name,'diagnostic_only':True,'samples':[]}
 try:
  settings=w.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;w.apply_settings(settings)
  bp=w.get_blueprint_library();car=bp.find('vehicle.lincoln.mkz_2020');ego=None
  for tr in w.get_map().get_spawn_points():
   ego=w.try_spawn_actor(car,tr)
   if ego:break
  if ego is None:raise RuntimeError('No free spawn')
  actors.append(ego);ego.apply_control(carla.VehicleControl(brake=1));report['vehicle']=ego.type_id
  phys=ego.get_physics_control();report['wheel_max_steer_deg']=[x.max_steer_angle for x in phys.wheels];report['wheel_positions_cm']=[[x.position.x,x.position.y,x.position.z] for x in phys.wheels]
  queues={}
  for name,kind in [('gps','sensor.other.gnss'),('imu','sensor.other.imu')]:
   b=bp.find(kind)
   for attr in ('noise_lat_stddev','noise_lon_stddev','noise_alt_stddev','noise_accel_stddev_x','noise_accel_stddev_y','noise_accel_stddev_z','noise_gyro_stddev_x','noise_gyro_stddev_y','noise_gyro_stddev_z'):
    if b.has_attribute(attr):b.set_attribute(attr,'0')
   q=queue.Queue();s=w.spawn_actor(b,carla.Transform(),attach_to=ego);actors.append(s);s.listen(q.put);queues[name]=q
  def tick():
   f=w.tick();pack={}
   for name,q in queues.items():
    item=q.get(timeout=10)
    while item.frame<f:item=q.get(timeout=10)
    if item.frame!=f:raise RuntimeError('Sensor frame mismatch')
    pack[name]=item
   return f,pack
  for _ in range(30):tick()
  origin=w.get_map().transform_to_geolocation(carla.Location(0,0,0));scale=math.cos(math.radians(origin.latitude));R=6378137.
  def project(lat,lon):return np.array([scale*R*math.radians(lon-origin.longitude),-scale*R*(math.log(math.tan(math.pi/4+math.radians(lat)/2))-math.log(math.tan(math.pi/4+math.radians(origin.latitude)/2)))])
  report['georef_origin']=[origin.latitude,origin.longitude]
  for i in range(80):
   if i==0:ego.apply_control(carla.VehicleControl(throttle=.3,steer=.15))
   f,d=tick();tr=ego.get_transform();gps=d['gps'];imu=d['imu'];ang=ego.get_angular_velocity();acc=ego.get_acceleration();inv=np.asarray(tr.get_inverse_matrix())[:3,:3]
   yaw=(imu.compass-math.pi/2);dyaw=(yaw-math.radians(tr.rotation.yaw)+math.pi)%(2*math.pi)-math.pi
   xy=project(gps.latitude,gps.longitude);actor_gyro=inv@np.array([ang.x,ang.y,ang.z])*math.pi/180
   report['samples'].append({'frame':f,'gps_xy':xy.tolist(),'actor_xy':[tr.location.x,tr.location.y],'gps_error_m':float(np.linalg.norm(xy-[tr.location.x,tr.location.y])),'compass_yaw_error_deg':math.degrees(dyaw),'imu_gyro_rad_s':[imu.gyroscope.x,imu.gyroscope.y,imu.gyroscope.z],'actor_local_gyro_rad_s':actor_gyro.tolist(),'actor_local_accel_mps2':(inv@np.array([acc.x,acc.y,acc.z])).tolist(),'imu_accel_mps2':[imu.accelerometer.x,imu.accelerometer.y,imu.accelerometer.z]})
  rows=report['samples'];report['max_gps_error_m']=max(x['gps_error_m'] for x in rows);report['max_compass_yaw_error_deg']=max(abs(x['compass_yaw_error_deg']) for x in rows);report['max_gyro_z_difference_rad_s']=max(abs(x['imu_gyro_rad_s'][2]-x['actor_local_gyro_rad_s'][2]) for x in rows)
  first,last=rows[0],rows[-1];report['gps_delta_xy']=(np.array(last['gps_xy'])-first['gps_xy']).tolist();report['actor_delta_xy']=(np.array(last['actor_xy'])-first['actor_xy']).tolist();report['status']='completed';report['notes']='Motion is only a low-speed diagnostic. Compare gyro at meaningful nonzero yaw rate; actor/IMU timing differences can remain. Acceleration z includes gravity. No model or benchmark policy receives truth.'
 except Exception as e:report['status']='failed';report['error']=repr(e);raise
 finally:
  cleanup=[]
  for actor in reversed(actors):
   try:
    if hasattr(actor,'stop'):actor.stop()
    actor.destroy()
   except Exception as e:cleanup.append(repr(e))
  try:w.apply_settings(old)
  except Exception as e:cleanup.append(repr(e))
  report['cleanup_errors']=cleanup;out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='samples'}))
if __name__=='__main__':main()
