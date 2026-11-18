"""Route-isolated matched triplets; keep the original pair evaluation read-only."""
import json
import numpy as np
from .collect_lead_pairs import ROOT,CFG,branch,sites,carla

CONFIG=json.loads((ROOT/'configs/v5-qwen-drive-world/lead-readout-refit.json').read_text())
OUT=ROOT/'data/v5-qwen-drive-world/lead-readout-refit'


def main():
    OUT.mkdir(exist_ok=True)
    lock=OUT/'protocol.json'
    if lock.exists() and json.loads(lock.read_text())!=CONFIG:raise ValueError('Collection protocol changed')
    lock.write_text(json.dumps(CONFIG,indent=2))
    client=carla.Client('127.0.0.1',2175)
    client.set_timeout(90)
    for split in ('fit','acceptance'):
        collected=[]
        for town,n in CONFIG[split].items():
            folder=OUT/split/town
            folder.mkdir(parents=True,exist_ok=True)
            world=client.get_world()
            if world.get_map().name.split('/')[-1]!=town:world=client.load_world(town)
            original=world.get_settings()
            settings=world.get_settings()
            settings.synchronous_mode=True
            settings.fixed_delta_seconds=CFG['dt_s']
            world.apply_settings(settings)
            accepted=set()
            positions=[]
            try:
                roadmap=world.get_map()
                for reverse in (False,True):
                    for wp,_ in sites(roadmap,reverse):
                        if wp.road_id in accepted or any(wp.transform.location.distance(loc)<60 for loc in positions):continue
                        k=len(accepted)
                        gap=CONFIG['gap_m'][k%3]
                        ahead=wp.next(gap)
                        if len(ahead)!=1 or ahead[0].road_id!=wp.road_id or ahead[0].is_junction:continue
                        name=f'{town}-r{wp.road_id}-l{wp.lane_id}-s{wp.s:.0f}'
                        path=folder/name
                        path.mkdir(exist_ok=True)
                        result=path/'triplet.json'
                        if not result.exists():
                            ego_speed=CONFIG['ego_cruise_mps'][k%2]
                            world.set_weather([carla.WeatherParameters.ClearNoon,carla.WeatherParameters.CloudyNoon][k%2])
                            rows=[]
                            attempt=path/f'attempt-{len(list(path.glob("attempt-*")))+1}'
                            attempt.mkdir()
                            try:
                                for label,(speed,tick) in CONFIG['branches'].items():
                                    row=branch(world,(wp,ahead[0]),name,bool(speed),attempt/label,
                                               ego_cruise=ego_speed,brake_rate=(ego_speed-.4)/8,
                                               lead_cruise=speed,release_tick=tick)
                                    row['branch']=label
                                    rows.append(row)
                                h,s,f=rows
                                valid=(abs(h['target'])<=.1 and .3<s['target']<1 and 1<=f['target']<=3
                                       and all(v['speed_kmh']<=3 and 3<v['gap']<35 and abs(v['lateral'])<2
                                               and v['ego_approach_m']>.15 for v in rows))
                                record=dict(rows=rows,valid=valid)
                            except ValueError as exc:
                                record=dict(rows=rows,valid=False,error=str(exc))
                            result.write_text(json.dumps(record,indent=2))
                        record=json.loads(result.read_text())
                        if record['valid']:
                            accepted.add(wp.road_id)
                            positions.append(wp.transform.location)
                            collected.extend(record['rows'])
                        print(split,name,record['valid'],len(accepted),'/',n,flush=True)
                        if len(accepted)==n:break
                    if len(accepted)==n:break
                if len(accepted)!=n:raise RuntimeError(f'{town}: only {len(accepted)} complete triplets')
            finally:
                world.apply_settings(original)
        (OUT/f'{split}.json').write_text(json.dumps(collected,indent=2))


if __name__=='__main__':
    main()
