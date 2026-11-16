"""Separate traffic setup process; never consumes ego decisions or schedules NPC actions.

Private sampled configuration is audit-only. Do not inspect it until the episode ends.
CARLA Traffic Manager owns ongoing NPC controls, following and lane-change decisions.
"""
import argparse
import json
import random
import secrets
import time
import traceback
from pathlib import Path
import carla


def read_json(path):
    for attempt in range(100):
        try:return json.loads(path.read_text())
        except PermissionError:
            if attempt==99:raise
            time.sleep(.01)


def write(path, value):
    temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,indent=2))
    for attempt in range(50):
        try:temporary.replace(path);break
        except PermissionError:
            if attempt==49:raise
            time.sleep(.002)


def main(folder):
    c=carla.Client('127.0.0.1',2100);c.set_timeout(60)
    w=c.get_world();m=w.get_map();cars=[];tm=None
    setup=json.loads((folder/'scenario.json').read_text())
    previous=setup.get('background_retry_from')
    seed=json.loads(Path(previous).read_text())['seed'] if previous else secrets.randbelow(2**31)
    rng=random.Random(seed)
    slots=[(lane,s) for lane in (-1,-2,-3) for s in (20,80,140,200,260)
           if not (lane==-2 and abs(s-80)<55)]
    rng.shuffle(slots)
    private={'seed':seed,'policy':'CARLA Traffic Manager with autonomous lane changes',
             'no_schedules':True,'no_ego_command_input':True,'vehicles':[]}
    try:
        for i,(lane,s) in enumerate(slots[:8]):
            bp=w.get_blueprint_library().find('vehicle.tesla.model3')
            bp.set_attribute('role_name',f'independent_npc_{i}')
            bp.set_attribute('color','30,70,200')
            tr=m.get_waypoint_xodr(388,lane,s).transform;tr.location.z+=.4
            car=w.spawn_actor(bp,tr);cars.append(car);car.apply_control(carla.VehicleControl(brake=1))
            private['vehicles'].append({'name':f'npc_{i}','id':car.id,'lane':lane,'s':s,
                                        'desired_speed_kmh':rng.uniform(100,130)})
        write(folder/'traffic-private-audit.json',private)
        write(folder/'traffic-spawned.json',{'actors':{v['name']:v['id'] for v in private['vehicles']}})
        deadline=time.monotonic()+180
        while not (folder/'traffic-start.json').exists():
            if (folder/'traffic-stop.json').exists() or time.monotonic()>deadline:return
            time.sleep(.1)
        tm=c.get_trafficmanager(8100);tm.set_synchronous_mode(True);tm.set_random_device_seed(seed)
        for car,parameters in zip(cars,private['vehicles']):
            tm.auto_lane_change(car,True)
            tm.set_desired_speed(car,parameters['desired_speed_kmh'])
            # Standard TM collision avoidance and following remain enabled.
            car.apply_control(carla.VehicleControl())
            car.set_target_velocity(car.get_transform().get_forward_vector()*(parameters['desired_speed_kmh']/3.6))
            car.set_autopilot(True,8100)
        write(folder/'traffic-active.json',{'active':True,'count':len(cars),'autonomous_lane_change_enabled':True})
        deadline=time.monotonic()+1800
        # This process owns the synchronous TM. The pinned CARLA remote TM's
        # SynchronousTick is a no-op, so this owner must advance the world.
        last_tick=0
        while not (folder/'traffic-stop.json').exists() and time.monotonic()<deadline:
            request=folder/'clock'/f'request-{last_tick+1:06d}.json'
            if request.exists():
                tick_id=read_json(request)['id']
                if tick_id>last_tick:
                    frame=w.tick(30);last_tick=tick_id
                    write(folder/'clock'/f'done-{tick_id:06d}.json',{'id':tick_id,'frame':frame})
            time.sleep(.001)
    except BaseException as error:
        write(folder/'traffic-error.json',{'error':repr(error),'traceback':traceback.format_exc()})
        raise
    finally:
        for car in cars:
            try:car.destroy()
            except Exception:pass
        if tm:tm.set_synchronous_mode(False)
        write(folder/'traffic-stopped.json',{'stopped':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('folder',type=Path)
    main(parser.parse_args().folder)
