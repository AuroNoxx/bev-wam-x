"""Reconstruct the seeded collection schedule and measure actual recovery / change coverage."""
import argparse
from collections import Counter
import json
from pathlib import Path

import carla
import numpy as np
from .bulk import Mission, expert_route
from .clean_driver import CollectionDriver
from .teacher_route import route as search_teacher


def audit(raw, released, spec, road, candidates, bounds):
    recording=json.loads((raw/'episode.json').read_text())
    meta=json.loads((released/'episode.json').read_text())
    # Use exactly the original teacher mission, even if its independent SD
    # cartography was subsequently revised for publication.
    spec=dict(spec,xy=recording['plan_world_xy'],laps=1)
    mission=Mission(spec)
    finder=search_teacher if spec.get('expert_search')=='both_directions' else expert_route
    route=finder(road,mission,bounds,candidates)
    driver=CollectionDriver(route,mission.speed,meta['seed'],mission.split=='train')
    if 'scheduled_recoveries' in recording and not np.allclose(driver.recovery_plans,recording['scheduled_recoveries']):
        raise ValueError(f'Cannot reproduce the recorded recovery schedule at {raw.name}')
    previous=0.
    geometry=Counter()
    events={}
    recoveries=0
    previous_recovery=False
    for i,row in enumerate(meta['frames']):
        side=row.get('lane_change')
        if row['recovery'] and not previous_recovery:recoveries+=1
        previous_recovery=row['recovery']
        if i%10 and i!=len(meta['frames'])-1 and not side:continue
        with np.load(released/f"{row['frame']:07d}"/'index.npz') as idx:
            pose=idx['world_from_ego']
        previous,error=driver.route.match(pose,previous)
        if error>8.:raise ValueError(f'Original teacher route mismatch at {raw.name}/{i}: {error}')
        if not side:continue
        wp=road.get_waypoint(carla.Location(*pose[:3,3]))
        back=wp.previous(4.);ahead=wp.next(4.)
        if len(back)!=1 or len(ahead)!=1:kind='ambiguous'
        else:
            dv=(ahead[0].transform.rotation.yaw-back[0].transform.rotation.yaw+180)%360-180
            k=abs(np.deg2rad(dv))/8.
            kind='straight' if k<.001 else 'gentle' if k<.018 else 'higher_curvature'
        geometry[side+'_'+kind]+=1
        events[row['lane_change_event']]=side
    return dict(eligible_windows_started=sum(s<=previous for s in driver.cruise_windows),
                recovery_windows_started=sum(s<=previous for s in driver.recovery_plans),
                recovery_windows_fully_traversed=sum(s+50<=previous for s in driver.recovery_plans),
                observed_recovery_events=recoveries,change_frames_by_road_curvature=geometry,
                distinct_lane_changes=Counter(events.values()))


def main(root,dest):
    plan=json.loads((root/'missions.json').read_text())
    maps={}
    for folder in sorted((dest/'episodes').glob('ep_*')):
        if not folder.name[3:].isdigit():continue
        output=root/(folder.name+'-maneuvers.json')
        if output.is_file():continue
        meta=json.loads((folder/'episode.json').read_text())
        town=meta['frames'][0]['map']
        if town not in maps:
            road=carla.Map(town,(root/(town+'.xodr')).read_text())
            maps[town]=(road,road.generate_waypoints(10))
        spec=next(m for m in plan[town]['missions'] if m['name']==meta['mission'])
        bounds=np.asarray(plan[town]['validation_bounds_xy']) if town=='Town12' else None
        result=audit(root/'episodes'/folder.name,folder,spec,*maps[town],bounds)
        print(folder.name,result,flush=True)
        output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('raw',type=Path);parser.add_argument('released',type=Path)
    args=parser.parse_args();main(args.raw,args.released)
