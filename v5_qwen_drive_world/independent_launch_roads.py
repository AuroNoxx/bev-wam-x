"""Frozen-planner evaluation on road IDs disjoint from earlier v5 low-speed runs."""
import hashlib
import json
import re
from pathlib import Path
from .mine_launch_pairs import DATASET, PAIR, load

OUT = PAIR / 'independent-roads'
RAW = DATASET.parent / 'independent-launch-roads'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock():
    OUT.mkdir(exist_ok=True)
    # Include rejected collection attempts as well as accepted geometry and planner windows.
    seen = set()
    pattern = re.compile(r'(Town[\w]+?)-(?:road|r)(\d+)-')
    for name in ('lead-pairs','lead-pairs-aligned','lead-readout-refit','paired-launch'):
        for item in (DATASET.parent / name).rglob('*'):
            match = pattern.match(item.name)
            if match and item.is_dir():
                seen.add((match[1],int(match[2])))
    protocol = dict(candidate_sha=sha(PAIR/'best.ckpt'),step=162,
                    reference_sha=sha(PAIR.parent/'best.ckpt'),towns=['Town03','Town10HD_Opt','Town02'],
                    excluded=sorted([list(p) for p in seen]),target_roads=12,per_map=6,
                    seeds='65001 + road ordinal, same noise within pair',
                    scoring='All usable pairs, including missing visual estimates. Five-second max displacement <0.5 m on holds, >=3 m on launches. Report per-second displacement, first crossing and matched pair success.')
    path=OUT/'protocol.json'
    if path.exists() and load(OUT,path.name)!=protocol:raise ValueError('Frozen protocol changed')
    path.write_text(json.dumps(protocol,indent=2))
    return protocol


def collect():
    import carla
    from .collect_launch_pairs import branch
    protocol=lock()
    excluded=set(map(tuple,protocol['excluded']))
    client=carla.Client('127.0.0.1',2175);client.set_timeout(45)
    rows=[]
    for town in protocol['towns']:
        if len(rows)//2>=protocol['target_roads']:break
        world=client.get_world()
        if world.get_map().name.split('/')[-1]!=town:world=client.load_world(town)
        original=world.get_settings(); settings=world.get_settings()
        settings.synchronous_mode=True;settings.fixed_delta_seconds=.1;world.apply_settings(settings)
        try:
            road_map=world.get_map()
            used=set()
            for wp in sorted(road_map.generate_waypoints(24),key=lambda w:(w.road_id,w.s,w.lane_id)):
                k=len(rows)//2
                if len(used)>=protocol['per_map'] or k>=protocol['target_roads']:break
                road=wp.road_id
                if road in used or (town,road) in excluded or wp.is_junction or abs(wp.transform.rotation.pitch)>4:continue
                ahead=wp.next([8,10,12,14][k%4]); tail=wp.next(42)
                if any(len(v)!=1 or v[0].road_id!=road or v[0].is_junction for v in (ahead,tail)):continue
                if abs((wp.transform.rotation.yaw-tail[0].transform.rotation.yaw+180)%360-180)>6:continue
                folder=RAW/f'{town}-r{road}-l{wp.lane_id}-s{wp.s:.0f}'
                folder.mkdir(parents=True,exist_ok=True);path=folder/'pair.json'
                if not path.exists():
                    world.set_weather([carla.WeatherParameters.CloudyNoon,carla.WeatherParameters.ClearNoon][k%2])
                    try:
                        pair=[branch(world,road_map,wp,ahead[0],folder/b,b=='launch',k,k%2==1) for b in ('hold','launch')]
                    except ValueError as e:
                        print('collection rejected',folder.name,str(e),flush=True)
                        continue
                    path.write_text(json.dumps(pair))
                pair=load(folder,'pair.json')
                assert len(pair)==2 and pair[0]['route']==pair[1]['route']==folder.name
                used.add(road);rows.extend(pair)
                print(town,road,'pairs',len(rows)//2,flush=True)
        finally:world.apply_settings(original)
    if not rows:raise RuntimeError('No new road pair collected')
    (OUT/'cohort.json').write_text(json.dumps(rows,allow_nan=False))


if __name__=='__main__':collect()
