"""Bounded UE5 RPC/physics/sensor probe; no training or dataset collection."""
from pathlib import Path
import argparse, json, math, queue, time, traceback
import numpy as np
import psutil
import carla

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/carla-ue5'

def speed(actor):
    v=actor.get_velocity()
    return math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=2100)
    parser.add_argument('--seconds',type=int,default=120)
    parser.add_argument('--tm-port',type=int,default=8100)
    parser.add_argument('--camera-mode',choices=['free','follow'],default='free')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    report={'status':'running','port':args.port,'requested_seconds':args.seconds,'camera_mode':args.camera_mode}
    client=carla.Client('127.0.0.1',args.port); client.set_timeout(15)
    world=None; original=None; actors=[]; sensors=[]; tm=None
    try:
        report['client_version']=client.get_client_version()
        report['server_version']=client.get_server_version()
        report['maps']=client.get_available_maps()
        if report['client_version']!=report['server_version']:
            raise RuntimeError('Client/server version mismatch')
        world=client.get_world(); report['map']=world.get_map().name
        original=world.get_settings()
        settings=world.get_settings(); settings.synchronous_mode=True; settings.fixed_delta_seconds=.05
        world.apply_settings(settings)
        spawns=world.get_map().get_spawn_points()
        report['spawn_points']=len(spawns)
        roads=world.get_map().generate_waypoints(5)
        report['driving_waypoints']=len(roads)
        report['nonjunction_waypoints']=sum(not w.is_junction for w in roads)
        bps=world.get_blueprint_library()
        candidates=[b for b in bps.filter('vehicle.*') if b.has_attribute('number_of_wheels') and b.get_attribute('number_of_wheels').as_int()==4]
        if not candidates: candidates=list(bps.filter('vehicle.*'))
        vehicle=None
        for transform in spawns:
            vehicle=world.try_spawn_actor(candidates[0],transform)
            if vehicle: break
        if vehicle is None: raise RuntimeError('No vehicle spawn succeeded')
        actors.append(vehicle); report['vehicle']=vehicle.type_id
        report['physics_wheels']=len(vehicle.get_physics_control().wheels)
        # UE5 exposes separate vehicle classes, not the old Vehicles group.
        report['level_vehicle_boxes']={name:len(world.get_level_bbs(getattr(carla.CityObjectLabel,name)))
                                       for name in ['Car','Truck','Bus','Motorcycle','Bicycle']}
        report['level_road_boxes']=len(world.get_level_bbs(carla.CityObjectLabel.Roads))
        packets={name:queue.Queue(maxsize=5) for name in ['rgb','semantic','lidar']}
        counts={k:0 for k in packets}
        def callback(name):
            def put(data):
                counts[name]+=1
                if packets[name].full():
                    try: packets[name].get_nowait()
                    except queue.Empty: pass
                packets[name].put_nowait(data)
            return put
        for name,kind in [('rgb','sensor.camera.rgb'),('semantic','sensor.camera.semantic_segmentation'),('lidar','sensor.lidar.ray_cast_semantic')]:
            bp=bps.find(kind); bp.set_attribute('sensor_tick','0.2')
            if name!='lidar':
                bp.set_attribute('image_size_x','480'); bp.set_attribute('image_size_y','270')
            else:
                bp.set_attribute('range','60'); bp.set_attribute('channels','16')
                bp.set_attribute('points_per_second','16000'); bp.set_attribute('rotation_frequency','5')
            sensor=world.spawn_actor(bp,carla.Transform(carla.Location(x=1.5,z=2.4)),attach_to=vehicle)
            actors.append(sensor); sensors.append(sensor); sensor.listen(callback(name))
        # RPC+2 is CARLA's secondary port; Traffic Manager needs its own port.
        tm=client.get_trafficmanager(args.tm_port); tm.set_synchronous_mode(True)
        start=time.monotonic(); samples=[]; saved=set(); start_loc=vehicle.get_location()
        with (OUT/'smoke-steps.jsonl').open('w',encoding='utf-8') as stream:
            for step in range(args.seconds*20):
                deadline=start+(step+1)*.05
                if step<80: vehicle.apply_control(carla.VehicleControl(throttle=.35))
                elif step<120: vehicle.apply_control(carla.VehicleControl(brake=.6))
                elif step<160: vehicle.apply_control(carla.VehicleControl(throttle=.2,steer=.12))
                elif step==160: vehicle.set_autopilot(True,args.tm_port)
                frame=world.tick(15)
                t=vehicle.get_transform()
                # Position once in free mode; do not overwrite user camera input.
                if step==0 or args.camera_mode=='follow':
                    world.get_spectator().set_transform(carla.Transform(t.location+carla.Location(z=15),carla.Rotation(pitch=-70,yaw=t.rotation.yaw)))
                for name,q in packets.items():
                    if name not in saved and not q.empty():
                        packet=q.get_nowait()
                        if name in ['rgb','semantic']:
                            packet.save_to_disk(str(OUT/f'{name}.png'))
                            if name=='semantic': report['semantic_ids']=np.unique(np.frombuffer(packet.raw_data,dtype=np.uint8).reshape(-1,4)[:,2]).tolist()
                        else: report['semantic_lidar_points_first']=len(packet)
                        saved.add(name)
                if step%20==0:
                    available=psutil.virtual_memory().available/1024**3
                    row={'step':step,'frame':frame,'elapsed_s':time.monotonic()-start,'speed_m_s':speed(vehicle),'x':t.location.x,'y':t.location.y,'sensor_counts':dict(counts),'available_ram_gib':available}
                    samples.append(row); stream.write(json.dumps(row)+'\n'); stream.flush()
                    if step%200==0: print(json.dumps(row),flush=True)
                    if available<4: raise RuntimeError('Available system RAM below 4GiB; stopping bounded test')
                time.sleep(max(0,deadline-time.monotonic()))
        report.update(status='passed_basic_smoke',wall_seconds=time.monotonic()-start,
                      simulated_seconds=args.seconds,sensor_counts=counts,
                      maximum_speed_m_s=max(r['speed_m_s'] for r in samples),
                      minimum_available_ram_gib=min(r['available_ram_gib'] for r in samples),
                      displacement_m=vehicle.get_location().distance(start_loc))
        if not all(counts.values()) or report['maximum_speed_m_s']<1:
            raise RuntimeError('Vehicle movement or sensor stream failed')
        report['limitations']=['Not a learned driving policy','No OCC equivalence or highway map acceptance yet','Single vehicle low-resolution bounded test','Not a long-duration stability or training benchmark']
    except Exception as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        for sensor in sensors:
            try: sensor.stop()
            except Exception: pass
        for actor in reversed(actors):
            try: actor.destroy()
            except Exception: pass
        if tm:
            try: tm.set_synchronous_mode(False)
            except Exception: pass
        if world and original:
            try: world.apply_settings(original)
            except Exception: pass
        (OUT/'smoke-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)

if __name__=='__main__': main()
