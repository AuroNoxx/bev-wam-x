"""Project-local CARLA RGB/physics-label acquisition; run with the CARLA venv.

Traffic Manager is the offline data expert, not a deployed actor fallback.
Measured velocity determines bins. No forced velocity or relabeled low-speed RGB.
"""
import argparse,json,math,queue,time,traceback,sys
from pathlib import Path
import carla
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
CAMERAS={'CAM_FRONT':(0,1920,1080,80),'CAM_FRONT_LEFT':(-60,960,540,100),
         'CAM_FRONT_RIGHT':(60,960,540,100),'CAM_BACK':(180,960,540,100),
         'CAM_BACK_LEFT':(-120,960,540,100),'CAM_BACK_RIGHT':(120,960,540,100)}


def write(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def box(actor,ego=False):
    tr=actor.get_transform();v=actor.get_velocity();a=actor.get_acceleration()
    return {'class':'ego_vehicle' if ego else 'vehicle','id':str(actor.id),'type_id':actor.type_id,
            'location':[tr.location.x,tr.location.y,tr.location.z],
            'rotation':[tr.rotation.pitch,tr.rotation.roll,tr.rotation.yaw],
            'world2ego':tr.get_inverse_matrix(),'world_from_ego':tr.get_matrix(),
            'velocity':[v.x,v.y,v.z],'acceleration_world':[a.x,a.y,a.z],
            'speed':math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z),
            'extent':[actor.bounding_box.extent.x,actor.bounding_box.extent.y,actor.bounding_box.extent.z],
            'world_cord':[[p.x,p.y,p.z] for p in actor.bounding_box.get_world_vertices(tr)]}


def route(map_):
    candidates=[]
    for wp in map_.generate_waypoints(30):
        adjacent=[wp.get_left_lane(),wp.get_right_lane()]
        if wp.is_junction or wp.lane_width<3 or not any(a and a.lane_type==carla.LaneType.Driving and a.lane_id*wp.lane_id>0 for a in adjacent):continue
        candidates.append(wp)
    choices=[]
    for start in candidates[::max(1,len(candidates)//80)]:
        path=[start]
        for _ in range(160):
            nxt=[w for w in path[-1].next(10) if not w.is_junction]
            if not nxt:break
            wp=min(nxt,key=lambda w:abs((w.transform.rotation.yaw-path[-1].transform.rotation.yaw+180)%360-180))
            path.append(wp)
        if len(path)>65:
            turns=sum(abs((b.transform.rotation.yaw-a.transform.rotation.yaw+180)%360-180) for a,b in zip(path,path[1:]))
            choices.append((len(path)-turns/10,path))
    if not choices:raise RuntimeError('No sufficiently long multilane highway route found')
    return max(choices,key=lambda x:x[0])[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');p.add_argument('--episode',default='train-cruise-118-0')
    p.add_argument('--speed',type=float,default=118);p.add_argument('--maneuver',choices=['cruise','lane_change'],default='cruise')
    p.add_argument('--seconds',type=float,default=65);p.add_argument('--port',type=int,default=2135)
    args=p.parse_args();folder=ROOT/'data/v5-qwen-drive-world/highway'/args.episode
    folder.mkdir(parents=True,exist_ok=False)
    status={'phase':'connecting','formal_training':False,'episode':args.episode,'target_speed_kmh':args.speed,
            'maneuver':args.maneuver,'map':'Town04','source':'CARLA0.9.15 TrafficManager physical rollout',
            'split':'validation' if args.episode.startswith('validation') else 'train','probe_only':args.probe}
    write(folder/'status.json',status);client=carla.Client('127.0.0.1',args.port);client.set_timeout(90)
    actors=[];sensors=[];original=None;world=None;tm=None;rows=[]
    try:
        world=client.get_world()
        if not world.get_map().name.endswith('Town04'):
            status['phase']='loading_map';write(folder/'status.json',status);world=client.load_world('Town04')
        map_=world.get_map();original=world.get_settings();settings=world.get_settings()
        settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;settings.no_rendering_mode=False
        world.apply_settings(settings);world.set_weather(carla.WeatherParameters.ClearNoon)
        tm=client.get_trafficmanager(args.port+1000);tm.set_synchronous_mode(True);tm.set_random_device_seed(71)
        path=route(map_);library=world.get_blueprint_library();bp=library.find('vehicle.tesla.model3')
        tr=path[0].transform;tr.location.z+=.4
        ego=world.spawn_actor(bp,tr);actors.append(ego)
        for offset in [10,20]:
            wp=path[offset].get_left_lane()
            if wp is None or wp.lane_type!=carla.LaneType.Driving or wp.lane_id*path[offset].lane_id<=0:wp=path[offset].get_right_lane()
            if wp is None or wp.lane_type!=carla.LaneType.Driving:continue
            pos=wp.transform;pos.location.z+=.4
            other=world.try_spawn_actor(bp,pos)
            if other:
                actors.append(other);other.set_autopilot(True,args.port+1000);tm.set_desired_speed(other,args.speed-10);tm.auto_lane_change(other,False)
        ego.apply_control(carla.VehicleControl(brake=1))
        for _ in range(30):world.tick(30)
        calibration={};queues={}
        for name,(yaw,width,height,fov) in CAMERAS.items():
            bp=library.find('sensor.camera.rgb');bp.set_attribute('image_size_x',str(width));bp.set_attribute('image_size_y',str(height))
            bp.set_attribute('fov',str(fov));bp.set_attribute('sensor_tick','.1');bp.set_attribute('enable_postprocess_effects','True')
            pose=carla.Transform(carla.Location(x=1.3,z=2.0),carla.Rotation(yaw=yaw))
            camera=world.spawn_actor(bp,pose,attach_to=ego);sensors.append(camera)
            q=queue.Queue();queues[name]=q;camera.listen(q.put)
            f=width/(2*math.tan(math.radians(fov)/2))
            calibration[name]={'width':width,'height':height,'fov':fov,'ego_from_sensor':pose.get_matrix(),
                               'intrinsics':[[f,0,width/2],[0,f,height/2],[0,0,1]]}
            (folder/'camera'/name).mkdir(parents=True)
        collisions=[];collision=world.spawn_actor(library.find('sensor.other.collision'),carla.Transform(),attach_to=ego)
        sensors.append(collision);collision.listen(lambda event:collisions.append(event.frame))
        write(folder/'calibration.json',calibration)
        write(folder/'route.json',[{'x':w.transform.location.x,'y':w.transform.location.y,'road':w.road_id,'lane':w.lane_id} for w in path])
        tm.auto_lane_change(ego,False);tm.set_desired_speed(ego,args.speed);tm.distance_to_leading_vehicle(ego,20)
        tm.set_path(ego,[w.transform.location for w in path]);ego.set_autopilot(True,args.port+1000)
        status['phase']='collecting';write(folder/'status.json',status);start=time.monotonic();changed=False;lane_agent=None
        pending={name:{} for name in CAMERAS};poses={}
        with (folder/'frames.jsonl').open('w',encoding='utf-8') as stream:
            for step in range(16 if args.probe else round(args.seconds/.05)):
                if lane_agent is not None:ego.apply_control(lane_agent.run_step())
                frame=world.tick(30)
                own=box(ego,True);snapshot=world.get_snapshot();wp=map_.get_waypoint(ego.get_location())
                ahead=wp.next(40);nav=ahead[0].transform.location if ahead else wp.transform.location
                poses[frame]={'frame':frame,'timestamp':snapshot.timestamp.elapsed_seconds,'speed':own['speed'],
                     'target_speed_kmh':args.speed,'bounding_boxes':[own]+[box(a) for a in actors if a.id!=ego.id],
                     'nav_world':[nav.x,nav.y,nav.z],'road_id':wp.road_id,'lane_id':wp.lane_id,
                     'collision':bool(collisions),'actor_support_radius_m':200,'current_world_from_ego':ego.get_transform().get_matrix()}
                # RGB readback may trail simulation ticks. Match exact frame IDs
                # against captured same-tick poses instead of assigning later poses.
                time.sleep(.03)
                for name,q in queues.items():
                    while not q.empty():
                        im=q.get_nowait();pending[name][im.frame]=im
                common=set(poses)
                for images_by_frame in pending.values():common.intersection_update(images_by_frame)
                if not common:
                    if step>=40 and not rows:raise RuntimeError('No synchronized RGB frames after40 ticks')
                    continue
                sample_frame=min(common)
                images={}
                for name in queues:
                    im=pending[name].pop(sample_frame)
                    array=np.frombuffer(im.raw_data,np.uint8).reshape(im.height,im.width,4)[:,:,:3][:,:,::-1]
                    if array.std()<1:raise RuntimeError(f'Empty/constant camera image: {name}')
                    rel=f'camera/{name}/{len(rows):05d}.jpg';Image.fromarray(array).save(folder/rel,quality=95)
                    images[name]=rel
                row={**poses.pop(sample_frame),'index':len(rows),'images':images}
                rows.append(row);stream.write(json.dumps(row)+'\n')
                if args.maneuver=='lane_change' and own['speed']*3.6>=100 and not changed and len(rows)>120:
                    adjacent=wp.get_left_lane();right=False
                    if adjacent is None or adjacent.lane_type!=carla.LaneType.Driving or adjacent.lane_id*wp.lane_id<=0:
                        adjacent=wp.get_right_lane();right=True
                    permitted=carla.LaneChange.Right if right else carla.LaneChange.Left
                    if adjacent and adjacent.lane_type==carla.LaneType.Driving and adjacent.lane_id*wp.lane_id>0 and int(wp.lane_change)&int(permitted):
                        # Use the bundled official lane-change planner for the
                        # acquisition maneuver; no hand-authored driving policy.
                        sys.path.insert(0,str(ROOT/'runtime/CARLA-0.9.15/WindowsNoEditor/PythonAPI/carla'))
                        from agents.navigation.basic_agent import BasicAgent
                        ego.set_autopilot(False,args.port+1000)
                        lane_agent=BasicAgent(ego,target_speed=args.speed,opt_dict={'dt':.05},map_inst=map_)
                        lane_agent.lane_change('right' if right else 'left',same_lane_time=.3,other_lane_time=15,lane_change_time=3)
                        changed=True
                        status['maneuver_expert']='bundled CARLA BasicAgent.lane_change; NPCs remain independent TrafficManager'
                        status['lane_change_command']={'frame':frame,'road':wp.road_id,'from_lane':wp.lane_id,'to_lane':adjacent.lane_id,'speed_kmh':own['speed']*3.6}
                if len(rows)%50==0:
                    stream.flush();status.update(frames=len(rows),speed_kmh=own['speed']*3.6,elapsed_s=time.monotonic()-start)
                    write(folder/'status.json',status);print(json.dumps(status),flush=True)
                if collisions:break
        if not rows:raise RuntimeError('No aligned nonconstant RGB observations collected')
        speeds=[r['speed']*3.6 for r in rows]
        status.update(phase='complete',frames=len(rows),max_speed_kmh=max(speeds,default=0),
                      actual_100_120_frames=sum(100<=s<=120 for s in speeds),lane_change_requested=changed,
                      collision_frames=collisions,elapsed_s=time.monotonic()-start)
    except Exception:
        status.update(phase='failed',error=traceback.format_exc());raise
    finally:
        for a in actors:
            try:a.set_autopilot(False,args.port+1000)
            except Exception:pass
        for s in sensors:
            try:s.stop()
            except Exception:pass
        if actors or sensors:
            try:client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in sensors+actors],True)
            except Exception:pass
        if tm:
            try:tm.set_synchronous_mode(False)
            except Exception:pass
        if world and original:
            try:world.apply_settings(original)
            except Exception:pass
        time.sleep(.5)
        write(folder/'status.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':main()
