"""Conservative, offline behavior candidates from train/validation actor tracks.

These are review aids, not ground-truth maneuver labels or model inputs.
No test data, training, simulation, or changes to dataset membership.
"""
import csv, gzip, json, math, time
from pathlib import Path
import numpy as np
from dataset_v1 import ROOT, digest
from build_aligned_cache import atomic_json

OUT = ROOT / 'artifacts/diagnostics/behavior-screen'
DT = 0.1
BINS = [(0,20),(20,40),(40,60),(60,80),(80,100),(100,120)]


def runs(values, ids):
    start = 0
    for j in range(1, len(values)+1):
        if j == len(values) or values[j] != values[start] or ids[j] != ids[j-1]+1:
            yield start, j, values[start]
            start = j


def scan_route(route, clip=None):
    clip = clip or ROOT / 'data/public/bench2drive/clips' / route['route'] / 'anno'
    frames = []
    previous_position = None
    previous_id = None
    previous_actors = {}
    for path in sorted(clip.glob('*.json.gz')):
        fid = int(path.name.split('.')[0])
        with gzip.open(path, 'rt') as stream:
            a = json.load(stream)
        ego = next(b for b in a['bounding_boxes'] if b['class']=='ego_vehicle')
        w2e = np.asarray(ego['world2ego'])
        position = np.linalg.inv(w2e)[:3,3]
        motion_ok = previous_position is None or (fid == previous_id+1 and np.linalg.norm(position-previous_position)<=8)
        previous_position, previous_id = position, fid
        lane = (ego['road_id'],ego['section_id'],ego['lane_id'])
        actors = {}
        for b in a['bounding_boxes']:
            if b['class']!='vehicle':
                continue
            # A teleported target must not look like a successful pass or a continuous lead.
            prior=previous_actors.get(str(b['id']))
            if prior is not None and np.linalg.norm(np.asarray(b['center'])-prior)>8:
                continue
            rel = w2e @ np.r_[b['center'],1]
            angle = math.radians(b['rotation'][2]-ego['rotation'][2])
            heading_ok = abs(math.atan2(math.sin(angle),math.cos(angle))) <= math.radians(20)
            if not heading_ok or abs(rel[2])>2 or abs(rel[1])>10 or abs(rel[0])>80:
                continue
            actors[str(b['id'])] = {'x':float(rel[0]),'y_right':float(rel[1]),'speed':b['speed'],
                                  'lane':(b['road_id'],b['section_id'],b['lane_id'])}
        previous_actors={str(b['id']):np.asarray(b['center']) for b in a['bounding_boxes'] if b['class']=='vehicle'}
        leads = [(b['x'], aid) for aid,b in actors.items() if b['lane']==lane and 2<b['x']<=60]
        lead = min(leads)[1] if leads else None
        queue_n = sum(0<b['x']<50 and b['lane'][0:2]==lane[0:2]
                      and b['lane'][2]*lane[2]>0 and b['speed']*3.6<10 for b in actors.values())
        frames.append({'frame':fid,'speed_kmh':a['speed']*3.6,'desired_speed_raw':a.get('desired_speed'),
                       'lane':lane,'lead':lead,'lead_x_m':actors[lead]['x'] if lead else None,
                       'lead_speed_kmh':actors[lead]['speed']*3.6 if lead else None,
                       'slow_vehicles_ahead_50m':queue_n,'motion_ok':motion_ok,'actors':actors,
                       'command_near':a['command_near'],'command_far':a['command_far']})
    ids = [f['frame'] for f in frames]
    speed = np.array([f['speed_kmh'] for f in frames])
    events = []

    def event(kind,s,e,**extra):
        chunk=frames[s:e]
        events.append({'kind':kind,'start_frame':ids[s],'end_frame':ids[e-1],
                       'samples':e-s,'elapsed_s':round((ids[e-1]-ids[s])*DT,3),
                       'speed_min_kmh':float(speed[s:e].min()),'speed_max_kmh':float(speed[s:e].max()),
                       'status':'heuristic_candidate_requires_review',**extra})

    # Keep actor identity; an interrupted lead is a different candidate segment.
    follow=[f['lead'] if f['motion_ok'] and f['speed_kmh']>=1 else None for f in frames]
    for s,e,aid in runs(follow,ids):
        if aid is not None and (ids[e-1]-ids[s])*DT>=3:
            event('following',s,e,actor_id=aid,center_distance_min_m=min(f['lead_x_m'] for f in frames[s:e]),
                  center_distance_max_m=max(f['lead_x_m'] for f in frames[s:e]))

    jam=[bool(f['motion_ok'] and 0<=f['speed_kmh']<10 and f['lead'] is not None
              and f['lead_speed_kmh']<10 and f['slow_vehicles_ahead_50m']>=2) for f in frames]
    for s,e,yes in runs(jam,ids):
        if yes and (ids[e-1]-ids[s])*DT>=3:
            event('slow_queue',s,e)

    stops=[bool(f['motion_ok'] and abs(f['speed_kmh'])<1) for f in frames]
    for s,e,yes in runs(stops,ids):
        if not yes or (ids[e-1]-ids[s])*DT<1:
            continue
        before=[k for k in range(max(0,s-100),s) if speed[k]>10]
        after=[k for k in range(e,min(len(frames),e+100)) if speed[k]>10]
        if before and after:
            left,right=before[-1],after[0]+1
            if ids[right-1]-ids[left]==right-left-1 and all(f['motion_ok'] for f in frames[left:right]):
                event('stop_then_restart',left,right,stop_start_frame=ids[s],stop_end_frame=ids[e-1],
                      has_slow_queue_evidence=any(jam[left:right]))

    # Require 1 s stable lane assignments on both sides; exclude road/section transitions.
    lane_runs=list(runs([f['lane'] for f in frames],ids))
    for (s,e,old),(u,v,new) in zip(lane_runs,lane_runs[1:]):
        if e!=u or ids[u]!=ids[e-1]+1 or e-s<11 or v-u<11:
            continue
        if old[:2]==new[:2] and old[2]*new[2]>0 and abs(old[2]-new[2])==1:
            if all(f['motion_ok'] for f in frames[e-11:u+11]):
                event('lane_change',e-11,u+11,transition_frame=ids[u],from_lane=list(old),to_lane=list(new))

    # Passing is deliberately separate from completed overtaking.
    # Target must remain on one lane, same road/section, heading-compatible, continuously tracked.
    actor_ids=sorted({aid for f in frames for aid in f['actors']})
    for aid in actor_ids:
        visible=[aid in f['actors'] and f['motion_ok'] and
                 f['actors'][aid]['lane'][:2]==f['lane'][:2] and
                 f['actors'][aid]['lane'][2]*f['lane'][2]>0 for f in frames]
        for s,e,yes in runs(visible,ids):
            if not yes or e-s<21:
                continue
            for k in range(s+1,e):
                if not (frames[k-1]['actors'][aid]['x']>=-5 and frames[k]['actors'][aid]['x']<-5):
                    continue
                candidates=[j for j in range(max(s,k-300),k) if frames[j]['actors'][aid]['x']>5]
                if not candidates:
                    continue
                left=candidates[0]
                target_lanes={tuple(f['actors'][aid]['lane']) for f in frames[left:k+1]}
                target_speeds=[f['actors'][aid]['speed'] for f in frames[left:k+1]]
                if len(target_lanes)!=1 or min(target_speeds)<2:
                    continue
                if float(speed[left:k+1].mean())/3.6-np.mean(target_speeds)<.5:
                    continue
                event('passing_target',left,k+1,actor_id=aid)
                lane=next(iter(target_lanes))
                initial=all(f['lane']==lane for f in frames[left:min(k+1,left+11)]) and k-left>=11
                changed=any(ev['kind']=='lane_change' and ids[left]<=ev['transition_frame']<=ids[k] for ev in events)
                # Require return to target's lane, ahead of the same moving target, stable for 1 s.
                returned=None
                for j in range(k,min(e-10,k+101)):
                    if all(f['lane']==lane and f['actors'][aid]['x']<-5 for f in frames[j:j+11]):
                        returned=j+11
                        break
                if initial and changed and returned is not None:
                    event('completed_overtake',left,returned,actor_id=aid)

    metrics=[]
    for lo,hi in BINS:
        inside=[bool(lo<=v<hi or hi==120 and v==120) for v in speed]
        longest=max(((ids[e-1]-ids[s])*DT for s,e,yes in runs(inside,ids) if yes),default=0)
        metrics.append({'range_kmh':[lo,hi],'frames':sum(inside),'nominal_sample_seconds':round(sum(inside)*DT,2),
                        'longest_contiguous_elapsed_s':round(longest,2)})
    timeline=OUT/'timelines'/(route['route']+'.csv')
    timeline.parent.mkdir(parents=True,exist_ok=True)
    fields=['frame','speed_kmh','desired_speed_raw','lane','lead','lead_x_m','lead_speed_kmh',
            'slow_vehicles_ahead_50m','motion_ok','command_near','command_far']
    with timeline.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:f[k] for k in fields} for f in frames)
    return {'route':route['route'],'split':route['split'],'family':route['family'],
            'frames':len(frames),'nominal_sample_seconds':len(frames)*DT,
            'source_annotation_file_count':len(frames),'speed_bins':metrics,'events':events,
            'timeline':str(timeline.relative_to(ROOT)),
            'unique_road_section_pairs':len({f['lane'][:2] for f in frames})}


def main():
    started=time.monotonic()
    path=ROOT/'data/occ_v1/manifest.json'
    manifest=json.loads(path.read_text())
    rows=[]
    for r in manifest['routes']:
        if r['split'] not in ['train','validation']:
            continue
        rows.append(scan_route(r))
        print(r['route'],len(rows[-1]['events']),'candidates',flush=True)
    summary={}
    for split in ['train','validation']:
        chosen=[r for r in rows if r['split']==split]
        summary[split]={}
        for kind in ['following','slow_queue','stop_then_restart','lane_change','passing_target','completed_overtake']:
            events=[e for r in chosen for e in r['events'] if e['kind']==kind]
            summary[split][kind]={'candidate_events':len(events),
                                 'routes_with_candidates':sum(any(e['kind']==kind for e in r['events']) for r in chosen),
                                 'sum_event_elapsed_seconds':round(sum(e['elapsed_s'] for e in events),2)}
    report={'status':'heuristic_screen_not_ground_truth','source_manifest_sha256':digest(path),
            'nominal_dt_s':DT,'test_read':False,'routes':rows,'summary':summary,'seconds':time.monotonic()-started,
            'thresholds':{'following':'Same actor, same road/section/lane, center ahead 2..60 m, heading within 20 deg, ego >=1 km/h, continuous >=3 s',
                          'slow_queue':'ego and lead <10 km/h, >=2 slow same-direction vehicles ahead within50m, >=3s',
                          'stop_then_restart':'stop <1km/h >=1s, >10km/h within10s before/after, not necessarily traffic congestion',
                          'lane_change':'adjacent same-sign lane on same road/section; 1s stable before/after',
                          'passing_target':'same moving actor from >5m ahead to <-5m behind, same target lane, same road/section, relative progress',
                          'completed_overtake':'passing plus initial same lane, ego lane change, and return ahead to same lane for>=1s'},
            'limitations':['Threshold-dependent candidates, not human-reviewed labels',
                           'Strict road/section identity misses behavior across segment boundaries',
                           'Actor visibility/truncation, short clips and map lane assignment can miss or split events',
                           'Nominal timing from file index; no original timestamps',
                           'Overlapping candidate durations are not unique independent driving time',
                           'Annotations contain privileged actors; this screen must not become model observations',
                           'Zero detected candidates does not prove behavior is absent']}
    atomic_json(OUT/'report.json',report)
    # Deduplicate frames within a behavior/speed cell. Different behavior cells can overlap.
    cells={}
    for row in rows:
        with (ROOT/row['timeline']).open() as stream: timeline=list(csv.DictReader(stream))
        for kind in summary[row['split']]:
            selected={i for e in row['events'] if e['kind']==kind for i in range(e['start_frame'],e['end_frame']+1)}
            for lo,hi in BINS:
                chosen=[f for f in timeline if int(f['frame']) in selected and
                        (lo<=float(f['speed_kmh'])<hi or hi==120 and float(f['speed_kmh'])==120)]
                key=(row['split'],kind,lo,hi)
                cell=cells.setdefault(key,{'frames':0,'routes':set(),'road_sections':set()})
                cell['frames']+=len(chosen)
                if chosen:cell['routes'].add(row['route'])
                town=row['route'].split('_')[1]
                cell['road_sections'].update(town+':'+f['lane'].rsplit(',',1)[0] for f in chosen)
    matrix=[{'split':k[0],'behavior':k[1],'speed_kmh':[k[2],k[3]],'unique_frames':v['frames'],
             'nominal_sample_seconds':round(v['frames']*DT,2),'routes':len(v['routes']),
             'road_sections':len(v['road_sections'])} for k,v in cells.items()]
    atomic_json(OUT/'speed-behavior-matrix.json',{'status':'candidate_intervals_not_ground_truth',
                'note':'Frames deduplicated within each cell; cells can overlap. Lane-change cells include the stable before/after review context.',
                'cells':matrix})
    print(json.dumps({'summary':summary,'seconds':report['seconds']}),flush=True)


if __name__=='__main__':
    main()
