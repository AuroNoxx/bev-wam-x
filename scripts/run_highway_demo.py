"""Bounded CARLA Town12 road-following demo, independent of learned Flow weights.

Uses Traffic Manager's privileged map-based controller. Writes a unique run folder;
does not collect training data, qualify an expert, or alter training settings.
"""
import argparse
import json
import math
import queue
import time
import traceback
from datetime import datetime
from pathlib import Path

import carla

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--port', type=int, default=2100)
    parser.add_argument('--speed-kmh', type=float, default=100)
    parser.add_argument('--seconds', type=float, default=55)
    args = parser.parse_args()
    if not 0 < args.speed_kmh <= 100 or not 10 <= args.seconds <= 60:
        parser.error('This bounded probe supports 0 < speed <= 100 and 10..60 seconds.')
    out = ROOT / 'logs/highway-demo' / datetime.now().strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'frames').mkdir()
    report = {'status': 'starting', 'controller': 'CARLA Traffic Manager',
              'learned_flow_used': False, 'target_speed_kmh': args.speed_kmh,
              'output': str(out), 'requested_cruise_and_acceleration_s': args.seconds,
              'limitations': ['One ego vehicle; no following, cut-in or overtaking test',
                             'Map/actor state is privileged, not observed OCC',
                             'Not expert qualification or a learned-model evaluation',
                             'Lane markings and corner checks are sampled, not swept-volume certification']}
    actors, sensors, rows, collisions, invasions = [], [], [], [], []
    world = original = tm = None
    try:
        client = carla.Client('127.0.0.1', args.port)
        client.set_timeout(100)
        world = client.get_world()
        m = world.get_map()
        if not m.name.endswith('/Town12'):
            raise RuntimeError('Load Town12 before this route-specific demo.')
        if world.get_actors().filter('vehicle.*'):
            raise RuntimeError('Existing vehicles detected; refusing to disrupt another session.')
        report.update(map=m.name, client_version=client.get_client_version(),
                      server_version=client.get_server_version())
        if report['client_version'] != report['server_version']:
            raise RuntimeError('Client/server version mismatch')
        # Inspected Town12 road 1157: three same-direction lanes, gentle curve,
        # no junction. Do not infer a legal speed limit from its generic road type.
        route = [m.get_waypoint_xodr(1157, -3, s) for s in range(100, 1801, 5)]
        if any(w is None or w.is_junction or w.lane_width < 3.4 for w in route):
            raise RuntimeError('Inspected route geometry no longer matches.')
        for w in route:
            for side in (w.get_left_lane(), w.get_right_lane()):
                if side is None or side.lane_type != carla.LaneType.Driving or side.lane_id * w.lane_id <= 0:
                    raise RuntimeError('Expected three same-direction driving lanes.')
        route_rows = [{'road': w.road_id, 'lane': w.lane_id, 's': w.s,
                       'x': w.transform.location.x, 'y': w.transform.location.y,
                       'z': w.transform.location.z, 'yaw_deg': w.transform.rotation.yaw,
                       'width_m': w.lane_width} for w in route]
        (out / 'route.json').write_text(json.dumps(route_rows, indent=2))
        report['route'] = {'road': 1157, 'lane': -3, 'start_s': 100, 'end_s': 1800,
                           'same_direction_lanes': 3, 'junctions': 0, 'lane_width_m': 3.5}
        original = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = .05
        world.apply_settings(settings)
        spawn = route[0].transform
        spawn.location.z += .4
        world.get_spectator().set_transform(carla.Transform(
            spawn.location + carla.Location(z=10), carla.Rotation(pitch=-30, yaw=spawn.rotation.yaw)))
        library = world.get_blueprint_library()
        bp = library.find('vehicle.tesla.model3')
        bp.set_attribute('role_name', 'hero')
        ego = world.spawn_actor(bp, spawn)
        actors.append(ego)
        report['vehicle'] = ego.type_id
        ego.apply_control(carla.VehicleControl(brake=1))
        for _ in range(20):
            world.tick(90)
        for kind, callback in [
            ('sensor.other.collision', lambda e: collisions.append({
                'frame': e.frame, 'other': e.other_actor.type_id,
                'impulse': math.sqrt(e.normal_impulse.x**2 + e.normal_impulse.y**2 + e.normal_impulse.z**2)})),
            ('sensor.other.lane_invasion', lambda e: invasions.append({
                'frame': e.frame, 'markings': [str(v.type) for v in e.crossed_lane_markings]}))]:
            sensor = world.spawn_actor(library.find(kind), carla.Transform(), attach_to=ego)
            actors.append(sensor)
            sensors.append(sensor)
            sensor.listen(callback)
        images = queue.Queue(maxsize=4)
        camera_bp = library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '360')
        camera_bp.set_attribute('sensor_tick', '0.5')
        camera = world.spawn_actor(camera_bp, carla.Transform(
            carla.Location(x=-8, z=4), carla.Rotation(pitch=-13)), attach_to=ego)
        actors.append(camera)
        sensors.append(camera)
        def on_image(image):
            try:
                images.put_nowait(image)
            except queue.Full:
                pass
        camera.listen(on_image)
        tm = client.get_trafficmanager(8100)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(20261115)
        tm.auto_lane_change(ego, False)
        tm.set_path(ego, [w.transform.location for w in route[1:]])
        tm.set_desired_speed(ego, args.speed_kmh)
        ego.set_autopilot(True, 8100)
        start = time.monotonic()
        braking = False
        brake_step = None
        reason = 'time_limit'
        frame_count = 0
        print(json.dumps({'status': 'driving', 'output': str(out), 'route': report['route']}), flush=True)
        with (out / 'telemetry.jsonl').open('w', encoding='utf-8') as log:
            for step in range(int((args.seconds + 20) * 20)):
                if time.monotonic() - start > 600:
                    raise RuntimeError('600-second wall-time watchdog')
                frame = world.tick(30)
                transform = ego.get_transform()
                loc = transform.location
                velocity = ego.get_velocity()
                v = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                wp = m.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                forward = transform.get_forward_vector()
                world.get_spectator().set_transform(carla.Transform(
                    loc - forward * 10 + carla.Location(z=5),
                    carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)))
                offset = loc - wp.transform.location
                right = wp.transform.get_right_vector()
                lateral = offset.x * right.x + offset.y * right.y
                corner_departure = False
                for x in (-ego.bounding_box.extent.x, ego.bounding_box.extent.x):
                    for y in (-ego.bounding_box.extent.y, ego.bounding_box.extent.y):
                        corner = transform.transform(ego.bounding_box.location + carla.Location(x=x, y=y))
                        cw = m.get_waypoint(corner, project_to_road=False, lane_type=carla.LaneType.Driving)
                        if cw is None or cw.road_id != 1157 or cw.lane_id != -3:
                            corner_departure = True
                if not braking:
                    if collisions:
                        reason = 'collision'
                    elif corner_departure:
                        reason = 'footprint_left_planned_lane'
                    elif wp.road_id != 1157 or wp.s >= 1620:
                        reason = 'route_end_margin'
                    elif step >= args.seconds * 20:
                        reason = 'time_limit'
                    else:
                        reason = None
                    if reason:
                        braking = True
                        brake_step = step
                        ego.set_autopilot(False, 8100)
                if braking:
                    # Retain lane steering using a geometric lookahead; bounded
                    # brake command is solely for this probe's final stop.
                    next_wps = wp.next(max(5, v * .6))
                    target = next_wps[0].transform.location if next_wps else wp.transform.location
                    dx, dy = target.x - loc.x, target.y - loc.y
                    yaw = math.radians(transform.rotation.yaw)
                    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
                    steer = max(-.4, min(.4, math.atan2(2 * 2.8 * local_y, max(dx*dx+dy*dy, 1)) / .6))
                    ego.apply_control(carla.VehicleControl(brake=.4, steer=steer))
                control = ego.get_control()
                row = {'frame': frame, 'simulation_s': (step + 1) * .05,
                       'speed_kmh': v * 3.6, 'x': loc.x, 'y': loc.y, 'z': loc.z,
                       'road': wp.road_id, 'lane': wp.lane_id, 'road_s': wp.s,
                       'lateral_error_m': lateral, 'footprint_outside_planned_lane': corner_departure,
                       'throttle': control.throttle, 'brake': control.brake, 'steer': control.steer,
                       'phase': 'braking' if braking else 'accelerate_cruise'}
                rows.append(row)
                log.write(json.dumps(row) + '\n')
                while not images.empty():
                    packet = images.get_nowait()
                    packet.save_to_disk(str(out / 'frames' / f'{packet.frame:08d}.png'))
                    frame_count += 1
                if step % 200 == 0:
                    log.flush()
                    print(json.dumps(row), flush=True)
                if braking and step - brake_step > 20 and v < .1:
                    break
                time.sleep(max(0, start + (step+1)*.05 - time.monotonic()))
        # Allow asynchronous sensor callbacks for the final tick to finish.
        time.sleep(.2)
        speeds = [r['speed_kmh'] for r in rows]
        report.update(status='completed', stop_reason=reason, wall_seconds=time.monotonic()-start,
                      simulated_seconds=rows[-1]['simulation_s'], maximum_speed_kmh=max(speeds),
                      seconds_in_95_105_kmh=sum(95 <= v <= 105 for v in speeds)*.05,
                      final_speed_kmh=speeds[-1], collision_events=collisions,
                      lane_invasion_events=invasions,
                      footprint_departure_frames=sum(r['footprint_outside_planned_lane'] for r in rows),
                      maximum_lateral_error_m=max(abs(r['lateral_error_m']) for r in rows),
                      distance_travelled_m=sum(math.dist((a['x'],a['y'],a['z']),(b['x'],b['y'],b['z']))
                                               for a,b in zip(rows,rows[1:])),
                      saved_camera_frames=frame_count)
        report['bounded_demo_passed'] = (report['seconds_in_95_105_kmh'] >= 10 and
            not collisions and not invasions and not report['footprint_departure_frames'] and speeds[-1] < 1)
    except Exception as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        cleanup_errors = []
        for sensor in sensors:
            try:
                sensor.stop()
            except Exception as error:
                cleanup_errors.append(repr(error))
        for actor in reversed(actors):
            try:
                actor.destroy()
            except Exception as error:
                cleanup_errors.append(repr(error))
        if tm:
            tm.set_synchronous_mode(False)
        if world and original:
            world.apply_settings(original)
        report['cleanup_errors'] = cleanup_errors
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
