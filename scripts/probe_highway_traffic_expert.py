"""Visible bounded multi-vehicle CARLA expert candidate screen. Never trains."""
import argparse
import hashlib
import json
import math
import queue
import time
import traceback
from datetime import datetime
from pathlib import Path

import carla
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DT = .05


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def state(actor, m):
    tr = actor.get_transform()
    wp = m.get_waypoint(tr.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    v, a = actor.get_velocity(), actor.get_acceleration()
    f, right = wp.transform.get_forward_vector(), wp.transform.get_right_vector()
    delta = tr.location - wp.transform.location
    return {'id': actor.id, 'x': tr.location.x, 'y': tr.location.y, 'z': tr.location.z,
            'yaw': tr.rotation.yaw, 's': wp.s, 'road': wp.road_id, 'lane': wp.lane_id,
            'speed': math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z),
            'long_speed': v.x*f.x+v.y*f.y, 'acceleration': a.x*f.x+a.y*f.y,
            'lateral_acceleration': a.x*right.x+a.y*right.y,
            'lateral_error': delta.x*right.x+delta.y*right.y,
            'half_length': actor.bounding_box.extent.x, 'half_width': actor.bounding_box.extent.y}


def gaps(states, lane):
    e = states['ego']
    result = []
    for name, other in states.items():
        if name == 'ego' or other['road'] != e['road'] or other['lane'] != lane:
            continue
        delta = other['s'] - e['s']
        gap = abs(delta) - e['half_length'] - other['half_length']
        closing = e['long_speed']-other['long_speed'] if delta >= 0 else other['long_speed']-e['long_speed']
        result.append({'actor': name, 'side': 'front' if delta >= 0 else 'rear', 'gap': gap,
                       'ttc': max(0, gap)/closing if closing > .1 else None,
                       'follower_speed': e['speed'] if delta >= 0 else other['speed'],
                       'headway': max(0, gap)/max(.1, e['speed'] if delta >= 0 else other['speed'])})
    return result


def inside_driving_union(m, point, road_id=1157, lanes=(-2,-3,-4)):
    """Check lane union explicitly: nearest-waypoint queries can miss boundaries.

    Restricted to the explicitly inspected road/lanes. No shoulder/opposite
    lanes. A 2 cm numerical tolerance applies only to lateral lane boundaries.
    """
    nearest=m.get_waypoint(point,project_to_road=True,lane_type=carla.LaneType.Driving)
    if nearest is None or nearest.road_id!=road_id:
        return False
    for lane in lanes:
        wp=m.get_waypoint_xodr(road_id,lane,nearest.s)
        if wp is None or wp.lane_type!=carla.LaneType.Driving:
            continue
        delta=point-wp.transform.location
        right=wp.transform.get_right_vector();forward=wp.transform.get_forward_vector()
        lateral=delta.x*right.x+delta.y*right.y
        longitudinal=delta.x*forward.x+delta.y*forward.y
        if abs(lateral)<=wp.lane_width/2+.02 and abs(longitudinal)<1 and abs(delta.z)<2:
            return True
    return False


def change_allowed(states, wp, target, rules):
    left = target > wp.lane_id
    marking = wp.left_lane_marking if left else wp.right_lane_marking
    permitted = carla.LaneChange.Left if left else carla.LaneChange.Right
    if not (int(wp.lane_change) & int(permitted)) or 'Solid' in str(marking.type):
        return False, 'marking_prohibits', []
    checks = gaps(states, target)
    for g in checks:
        headway = rules['lane_change_front_headway_s'] if g['side'] == 'front' else rules['lane_change_rear_headway_s']
        if g['gap'] < max(rules['lane_change_min_gap_m'], headway*g['follower_speed']):
            return False, 'insufficient_gap', checks
        if g['ttc'] is not None and g['ttc'] < rules['lane_change_min_ttc_s']:
            return False, 'closing_vehicle', checks
    return True, 'gap_check_passed', checks


def summarize(rows, events, collisions, markings, case, stop_reason, gates):
    min_or_none = lambda xs: min(xs) if xs else None
    ttc, headways, front_gaps = [], [], []
    following_match = 0
    for row in rows:
        e = row['actors']['ego']
        for g in row['current_lane_gaps']:
            if g['side'] == 'front':
                front_gaps.append(g['gap'])
                if g['ttc'] is not None and e['speed'] > 10/3.6:
                    ttc.append(g['ttc'])
                if e['speed'] > 10/3.6:
                    headways.append(g['headway'])
        lead = row['actors']['lead']
        if row['time'] >= 15 and e['lane'] == lead['lane'] and lead['s'] > e['s'] and abs(e['speed']-lead['speed'])*3.6 < 5:
            following_match += DT
    solid = [x for x in markings if any('Solid' in t for t in x['markings'])]
    min_ttc, min_headway = min_or_none(ttc), min_or_none(headways)
    completion = any(x['kind'] == 'maneuver_completed' for x in events)
    rejected = any(x['kind'] == 'change_rejected' for x in events)
    commands = [x for x in events if x['kind'] == 'change_command']
    if case['maneuver'] == 'following':
        task_ok = following_match >= gates['following_speed_match_seconds_abs_error_below_5kmh'] and all(r['actors']['ego']['lane']==-3 for r in rows)
    elif case['maneuver'] == 'blocked':
        task_ok = rejected and not commands and all(r['actors']['ego']['lane']==-3 for r in rows)
    else:
        task_ok = completion
    offroad = sum(r['off_driving_road'] for r in rows)
    safety = not collisions and not solid and not offroad and (min_ttc is None or min_ttc >= gates['min_ttc_when_closing_s']) and (min_headway is None or min_headway >= gates['min_headway_at_speed_above_10kmh_s'])
    accel = np.array([r['actors']['ego']['acceleration'] for r in rows])
    smoothed = np.convolve(accel, np.ones(11)/11, mode='valid') if len(accel)>=11 else accel
    jerk = np.diff(smoothed)/DT
    return {'task_completed': bool(task_ok), 'screening_safety_gates_passed': bool(safety),
            'initial_screen_passed': bool(task_ok and safety and stop_reason=='duration_complete'),
            'stop_reason': stop_reason, 'simulated_seconds': rows[-1]['time'] if rows else 0,
            'maximum_speed_kmh': max(r['actors']['ego']['speed'] for r in rows)*3.6,
            'min_front_gap_m': min_or_none(front_gaps), 'min_front_ttc_s': min_ttc,
            'min_front_time_headway_s': min_headway, 'following_speed_match_seconds': following_match,
            'collision_events': collisions, 'lane_marking_events': markings,
            'solid_marking_events': len(solid), 'off_driving_road_frames': offroad,
            'max_abs_lateral_acceleration_mps2': float(max(abs(r['actors']['ego']['lateral_acceleration']) for r in rows)),
            'p95_abs_long_acceleration_mps2': float(np.percentile(abs(accel),95)),
            'max_abs_long_acceleration_mps2': float(max(abs(accel))),
            'p95_abs_jerk_0p55s_boxcar_mps3': float(np.percentile(abs(jerk),95)) if len(jerk) else None,
            'max_abs_jerk_0p55s_boxcar_mps3': float(max(abs(jerk))) if len(jerk) else None,
            'events': events}


def run_case(client, world, m, tm, case, protocol, folder):
    folder.mkdir()
    (folder/'frames').mkdir()
    actors, sensors, vehicles = [], [], {}
    rows, events, collisions, markings = [], [], [], []
    report = {'case': case, 'candidate': protocol['candidate'], 'status': 'starting'}
    rules = protocol['decision_rules']
    images = queue.Queue(maxsize=8)
    try:
        tm.set_random_device_seed(protocol['seed'])
        base = 150 + case['shift']
        kind = case['maneuver']
        placements = [('ego', -3, base, case['speed']),
                      ('lead', -3, base+(200 if kind in ('lane_change','blocked') else 65), case['lead_speed']),
                      ('left_front', -2, base+180, case['speed']+5),
                      ('left_rear', -2, base-110, max(40,case['speed']-10)),
                      ('right_car', -4, base+45, max(40,case['speed']-10))]
        if kind == 'blocked':
            placements[3] = ('left_rear', -2, base, case['speed'])
        library = world.get_blueprint_library()
        for name, lane, s, target in placements:
            wp = m.get_waypoint_xodr(1157,lane,s)
            if wp is None or wp.is_junction:
                raise RuntimeError('Unexpected route geometry')
            transform = wp.transform
            transform.location.z += .4
            bp = library.find('vehicle.tesla.model3')
            bp.set_attribute('role_name','hero' if name=='ego' else name)
            if bp.has_attribute('color'):
                bp.set_attribute('color','255,255,255' if name=='ego' else ('200,30,30' if name=='lead' else '30,70,200'))
            car = world.spawn_actor(bp,transform)
            actors.append(car); vehicles[name]=car
            car.apply_control(carla.VehicleControl(brake=1))
            if name=='ego':
                world.get_spectator().set_transform(carla.Transform(transform.location+carla.Location(z=8),carla.Rotation(pitch=-25,yaw=transform.rotation.yaw)))
        for _ in range(30): world.tick(90)
        for name,car in vehicles.items():
            sensor=world.spawn_actor(library.find('sensor.other.collision'),carla.Transform(),attach_to=car)
            actors.append(sensor); sensors.append(sensor)
            sensor.listen(lambda e,n=name: collisions.append({'frame':e.frame,'vehicle':n,'other':e.other_actor.type_id,'other_id':e.other_actor.id}))
        ego=vehicles['ego']
        sensor=world.spawn_actor(library.find('sensor.other.lane_invasion'),carla.Transform(),attach_to=ego)
        actors.append(sensor); sensors.append(sensor)
        sensor.listen(lambda e: markings.append({'frame':e.frame,'markings':[str(x.type) for x in e.crossed_lane_markings]}))
        bp=library.find('sensor.camera.rgb');bp.set_attribute('image_size_x','640');bp.set_attribute('image_size_y','360');bp.set_attribute('sensor_tick','.5')
        camera=world.spawn_actor(bp,carla.Transform(carla.Location(x=-11,z=7),carla.Rotation(pitch=-20)),attach_to=ego)
        actors.append(camera);sensors.append(camera)
        def on_image(im):
            try: images.put_nowait(im)
            except queue.Full: pass
        camera.listen(on_image)
        for name,lane,s,target in placements:
            car=vehicles[name]
            tm.auto_lane_change(car,False)
            tm.distance_to_leading_vehicle(car,10)
            tm.set_desired_speed(car,target)
            car.set_autopilot(True,8100)
        phase='following' if kind=='following' else 'waiting_gap'
        desired_lane=-3
        stable=0
        requested=False
        lead_slowed=lead_restored=False
        last_reject=-100
        stop_reason='duration_complete'
        start=time.monotonic()
        print(json.dumps({'case':case['name'],'status':'running','vehicles':len(vehicles),'target_speed':case['speed']}),flush=True)
        with (folder/'telemetry.jsonl').open('w',encoding='utf-8') as stream:
            for step in range(int(case['seconds']/DT)):
                if time.monotonic()-start>600: raise RuntimeError('Case wall-clock watchdog')
                frame=world.tick(30)
                states={n:state(car,m) for n,car in vehicles.items()}
                e=states['ego']; lead=states['lead']; now=(step+1)*DT
                tr=ego.get_transform();wp=m.get_waypoint(tr.location)
                world.get_spectator().set_transform(carla.Transform(tr.location-tr.get_forward_vector()*12+carla.Location(z=8),carla.Rotation(pitch=-22,yaw=tr.rotation.yaw)))
                if step%20==0:
                    world.debug.draw_string(tr.location+carla.Location(z=3),f"{case['name']} | {e['speed']*3.6:.0f} km/h | {phase}",draw_shadow=True,color=carla.Color(255,230,0),life_time=1.1,persistent_lines=False)
                if step%4==0:
                    tm.distance_to_leading_vehicle(ego,max(rules['minimum_following_distance_m'],rules['desired_following_time_s']*e['speed']))
                if kind=='following':
                    if now>=20 and not lead_slowed:
                        tm.set_desired_speed(vehicles['lead'],case['lead_speed']-25);lead_slowed=True
                        events.append({'time':now,'kind':'lead_target_deceleration','target_kmh':case['lead_speed']-25})
                    if now>=35 and not lead_restored:
                        tm.set_desired_speed(vehicles['lead'],case['lead_speed']);lead_restored=True
                        events.append({'time':now,'kind':'lead_target_acceleration','target_kmh':case['lead_speed']})
                elif now>=rules['lane_change_request_after_s']:
                    target=None
                    if phase=='waiting_gap': target=-2
                    elif kind=='overtake' and phase=='passing' and e['s']-lead['s']-e['half_length']-lead['half_length']>=rules['overtake_return_clearance_m']:
                        target=-3
                    if target is not None:
                        ok,why,checks=change_allowed(states,wp,target,rules)
                        if ok:
                            tm.force_lane_change(ego,target<wp.lane_id)
                            requested=True;desired_lane=target;stable=0
                            phase='changing_left' if target==-2 else 'returning'
                            events.append({'time':now,'kind':'change_command','target_lane':target,'checks':checks})
                        elif now-last_reject>=2:
                            events.append({'time':now,'kind':'change_rejected','target_lane':target,'reason':why,'checks':checks});last_reject=now
                    if phase in ('changing_left','returning'):
                        stable=stable+DT if e['lane']==desired_lane and abs(e['lateral_error'])<.35 else 0
                        if stable>=rules['lane_settle_s']:
                            if phase=='changing_left' and kind=='overtake':
                                phase='passing';events.append({'time':now,'kind':'left_lane_settled'})
                            else:
                                phase='completed';events.append({'time':now,'kind':'maneuver_completed','lane':e['lane'],'clearance_from_original_lead_m':e['s']-lead['s']-e['half_length']-lead['half_length']})
                offroad=False
                for x in (-ego.bounding_box.extent.x,ego.bounding_box.extent.x):
                    for y in (-ego.bounding_box.extent.y,ego.bounding_box.extent.y):
                        corner=tr.transform(ego.bounding_box.location+carla.Location(x=x,y=y))
                        if not inside_driving_union(m,corner): offroad=True
                control=ego.get_control()
                row={'frame':frame,'timestamp':world.get_snapshot().timestamp.elapsed_seconds,'time':now,'phase':phase,'actors':states,
                     'current_lane_gaps':gaps(states,e['lane']),'off_driving_road':offroad,
                     'control':{'throttle':control.throttle,'brake':control.brake,'steer':control.steer}}
                if case.get('trace_tm_actions') and 11.5 <= now <= 16:
                    row['tm_actions']=[{'action':str(action),'lane':point.lane_id,'road':point.road_id,'s':point.s,
                                       'distance':tr.location.distance(point.transform.location)}
                                      for action,point in tm.get_all_actions(ego)]
                rows.append(row);stream.write(json.dumps(row)+'\n')
                while not images.empty():
                    im=images.get_nowait();im.save_to_disk(str(folder/'frames'/f'{im.frame:08d}.png'))
                if step%200==0:
                    stream.flush();print(json.dumps({'case':case['name'],'time':now,'speed':e['speed']*3.6,'lane':e['lane'],'phase':phase,'front':row['current_lane_gaps']}),flush=True)
                if collisions or offroad or e['road']!=1157 or e['s']>1620:
                    stop_reason='collision' if collisions else ('off_driving_road' if offroad else 'route_boundary');break
                time.sleep(max(0,start+(step+1)*DT-time.monotonic()))
        time.sleep(.2)
        report.update(status='completed',wall_seconds=time.monotonic()-start,
                      **summarize(rows,events,collisions,markings,case,stop_reason,protocol['screening_gates_not_certification']))
    except Exception as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        print(traceback.format_exc(),flush=True)
    finally:
        errors=[]
        for sensor in sensors:
            try: sensor.stop()
            except Exception as error: errors.append(repr(error))
        for actor in reversed(actors):
            try: actor.destroy()
            except Exception as error: errors.append(repr(error))
        report['cleanup_errors']=errors
        dump(folder/'report.json',report)
        print(json.dumps({'case':case['name'],'result':{k:v for k,v in report.items() if k not in ('events','lane_marking_events')}}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--cases',nargs='*')
    p.add_argument('--config',type=Path,default=ROOT/'configs/highway_tm_expert_probe.json');args=p.parse_args()
    config=args.config
    protocol=json.loads(config.read_text())
    cases=[c for c in protocol['cases'] if not args.cases or c['name'] in args.cases]
    if not cases: raise ValueError('No cases selected')
    out=ROOT/'logs/highway-expert-probe'/datetime.now().strftime('%Y%m%d-%H%M%S');out.mkdir(parents=True)
    dump(out/'protocol.json',protocol)
    dump(out/'source.json',{'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'script':str(Path(__file__))})
    print(json.dumps({'output':str(out),'planned_cases':[c['name'] for c in cases]}),flush=True)
    c=carla.Client('127.0.0.1',2100);c.set_timeout(100);w=c.get_world();m=w.get_map()
    if not m.name.endswith('/Town12') or w.get_actors().filter('vehicle.*'): raise RuntimeError('Need unoccupied Town12')
    if c.get_server_version()!=c.get_client_version(): raise RuntimeError('Version mismatch')
    original=w.get_settings();tm=None;results=[]
    try:
        settings=w.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=DT;w.apply_settings(settings)
        tm=c.get_trafficmanager(8100);tm.set_synchronous_mode(True)
        for case in cases:
            result=run_case(c,w,m,tm,case,protocol,out/case['name']);results.append(result)
            dump(out/'summary.json',{'status':'running','version':c.get_server_version(),'results':results})
            if result['status']=='failed' or result['cleanup_errors']: break
            for _ in range(5):w.tick(30)
        dump(out/'summary.json',{'status':'completed' if len(results)==len(cases) else 'incomplete','version':c.get_server_version(),'results':results,'expert_qualified':False,'reason':'Initial limited scenario screen only; sensor/OCC, emergency and distribution coverage not qualified'})
    finally:
        if tm: tm.set_synchronous_mode(False)
        w.apply_settings(original)
        dump(out/'post-run-state.json',{'vehicles':len(w.get_actors().filter('vehicle.*')),'sensors':len(w.get_actors().filter('sensor.*')),'synchronous_mode':w.get_settings().synchronous_mode})


if __name__=='__main__':main()
