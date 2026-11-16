"""Read-only live health check; never ticks or changes the simulator."""
import argparse
import json
import time
from pathlib import Path
import carla

parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=int, default=120)
args = parser.parse_args()
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(5)
world = client.get_world()
output = Path(__file__).resolve().parents[1] / 'logs/preview-health.jsonl'
started = time.monotonic()
previous_frame = None
with output.open('a', encoding='utf-8') as log:
    while True:
        snapshot = world.get_snapshot()
        vehicles = world.get_actors().filter('vehicle.*')
        row = {'wall_seconds': round(time.monotonic()-started, 2),
               'frame': snapshot.frame,
               'simulation_seconds': snapshot.timestamp.elapsed_seconds,
               'vehicles': [{'id': v.id,
                             'speed_kmh': round(v.get_velocity().length()*3.6, 2),
                             'x': round(v.get_location().x, 2),
                             'y': round(v.get_location().y, 2)} for v in vehicles]}
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row)+'\n')
        log.flush()
        if previous_frame is not None and snapshot.frame <= previous_frame:
            raise RuntimeError('Simulator frames stopped advancing')
        previous_frame = snapshot.frame
        if time.monotonic()-started >= args.seconds:
            break
        time.sleep(10)
