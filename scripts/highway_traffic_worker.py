"""Independent Traffic Manager process. Only start/tick/stop commands accepted.

Driving parameters are sampled ONCE, before the episode. No event schedule,
force_lane_change, per-tick desired-speed writes, or ego decision input.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import traceback
import carla


def reply(value):print(json.dumps(value),flush=True)


def main(folder):
    cfg=json.loads((folder/'episode-config.json').read_text());rng=random.Random(cfg['traffic_seed'])
    c=carla.Client('127.0.0.1',cfg['rpc_port']);c.set_timeout(12);w=c.get_world();m=w.get_map()
    cars=[];tm=None;parameters=[]
    try:
        slots=[(lane,s+rng.uniform(-6,6)) for lane in cfg['lanes'] for s in range(40,520,60)
               if not (lane==cfg['ego_lane'] and abs(s-cfg['ego_start_s'])<35)]
        rng.shuffle(slots)
        ranges={'low':(12,42),'mixed':(35,115),'fast':(85,130),'dense':(5,90)}
        lo,hi=ranges[cfg['traffic_profile']]
        for i,(lane,s) in enumerate(slots[:cfg['traffic_count']]):
            bp=w.get_blueprint_library().find('vehicle.tesla.model3')
            bp.set_attribute('role_name',f'random_highway_{i}');bp.set_attribute('color','35,110,210')
            wp=m.get_waypoint_xodr(cfg['road'],lane,s)
            if wp is None:continue
            tr=wp.transform;tr.location.z+=.4
            car=w.try_spawn_actor(bp,tr)
            if car is None:continue
            cars.append(car);car.apply_control(carla.VehicleControl(brake=1))
            parameters.append({'id':car.id,'lane':lane,'initial_s':s,'desired_speed_kmh':rng.uniform(lo,hi),
                               'lanechange_probability':rng.uniform(.3,2.),'following_distance_m':rng.uniform(3,8)})
        (folder/'traffic-private.json').write_text(json.dumps({'seed':cfg['traffic_seed'],'parameters':parameters,
            'runtime_parameter_writes':0,'future_paths_prescribed':False,'ego_decisions_consumed':False},indent=2))
        reply({'status':'spawned','count':len(cars)})
        for line in sys.stdin:
            op=json.loads(line)['op']
            if op=='start':
                tm=c.get_trafficmanager(cfg['tm_port']);tm.set_synchronous_mode(True);tm.set_random_device_seed(cfg['traffic_seed'])
                for car,p in zip(cars,parameters):
                    tm.auto_lane_change(car,True);tm.set_desired_speed(car,p['desired_speed_kmh'])
                    tm.random_left_lanechange_percentage(car,p['lanechange_probability'])
                    tm.random_right_lanechange_percentage(car,p['lanechange_probability'])
                    tm.distance_to_leading_vehicle(car,p['following_distance_m'])
                    car.apply_control(carla.VehicleControl());car.set_autopilot(True,cfg['tm_port'])
                reply({'status':'active','count':len(cars)})
            elif op=='tick':reply({'frame':w.tick(12)})
            elif op=='stop':reply({'status':'stopping'});break
            else:raise ValueError('Traffic process accepts no driving commands')
    except BaseException as error:
        (folder/'traffic-error.json').write_text(json.dumps({'error':repr(error),'traceback':traceback.format_exc()},indent=2))
        reply({'status':'error','error':repr(error)})
    finally:
        c.set_timeout(2)
        if tm:
            try:tm.set_synchronous_mode(False)
            except Exception:pass
        for car in cars:
            try:car.destroy()
            except Exception:pass


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);main(p.parse_args().folder)
