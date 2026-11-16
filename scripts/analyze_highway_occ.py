"""Post-run evaluation. Privileged truth is consumed ONLY here, never by planning."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def stats(values):
    return dict(zip(['min','p50','p95','max'],np.percentile(values,[0,50,95,100]).tolist())) if len(values) else None


def analyze(folder):
    report=json.loads((folder/'report.json').read_text());rows=lines(folder/'telemetry.jsonl');truth=lines(folder/'evaluation-only.jsonl')
    out={'folder':str(folder),'status':report['status'],'error':report.get('error'),'config':report['config'],
         'seconds':report['simulated_seconds'],'distance_m':report['distance_m'],'max_speed_kmh':report['max_speed_kmh'],
         'collision_events':len(report['collisions']),'solid_marking_events':sum(any('Solid' in t for t in e['types']) for e in report['lane_markings']),
         'events':report['events'],'realtime_factor':report.get('realtime_factor'),
         'planning_ms_p50_p95_max':report['planning_ms_p50_p95_max']}
    if not rows:return out
    speeds=np.array([r['speed_kmh'] for r in rows]);dt=.05
    out['speed_exposure_s']={f'{lo}-{hi}':float(np.sum((speeds>=lo)&(speeds<hi))*dt) for lo,hi in [(0,5),(5,30),(30,60),(60,90),(90,110),(110,120),(120,200)]}
    out['reasons_s']={k:v*dt for k,v in Counter(r['reason'] for r in rows).items()}
    out['acceleration_mps2']=stats(np.diff(speeds/3.6)/dt)
    out['jerk_mps3_raw_20Hz']=stats(np.diff(speeds/3.6,n=2)/dt**2)
    # 0.5-second finite differences reduce physics/sampling noise; retain raw too.
    if len(speeds)>20:
        accel=(speeds[10:]-speeds[:-10])/3.6/.5
        out['acceleration_mps2_0p5s']=stats(accel);out['jerk_mps3_0p5s']=stats((accel[10:]-accel[:-10])/.5)
    out['lateral_error_m']=stats(np.abs([r['lateral_error_m'] for r in rows]))
    by_frame={r['frame']-1:r for r in rows} # logged control row describes pre-tick observation
    histories=defaultdict(list);ttcs=[];gaps=[];headways=[];errors=[];cutins=[]
    for snapshot in truth:
        ego=by_frame.get(snapshot['frame'])
        if ego is None:continue
        for n in snapshot['npcs']:
            histories[n['id']].append((snapshot['timestamp'],n,ego))
            if n['road']==report['config']['road'] and n['lane']==ego['lane'] and n['local'][0]>0:
                gap=n['local'][0]-5.;gaps.append(gap);v=ego['speed_kmh']/3.6
                if v>1:headways.append(gap/v)
                closing=v-n['velocity'][0]
                if closing>1:ttcs.append(max(0,gap)/closing)
        errors.extend(m['velocity_error_mps'] for m in snapshot['matches'] if m['hits']>=5)
    npcchanges=[];passed=[];npc_accels=[];npc_speeds=[]
    for aid,h in histories.items():
        stable=h[0][1]['lane'];pending=None;since=None;previous=h[0];ahead=False
        for t,n,e in h:
            npc_speeds.append(n['speed_mps']*3.6)
            delta=t-previous[0]
            if delta>0:npc_accels.append((n['speed_mps']-previous[1]['speed_mps'])/delta)
            if n['lane']!=stable:
                if pending!=n['lane']:pending=n['lane'];since=t
                elif t-since>=.4:
                    change={'actor_id':aid,'from':stable,'to':n['lane'],'timestamp':t,'ego_speed_kmh':e['speed_kmh'],'relative_x_m':n['local'][0]}
                    npcchanges.append(change)
                    recent=[ee for tt,nn,ee in h if t-1.2<=tt<=t]
                    if n['lane']==e['lane'] and all(ee['lane']==e['lane'] and ee['target_lane']==ee['lane'] for ee in recent) and 0<n['local'][0]<max(30.,e['speed_kmh']/3.6*2.):cutins.append(change)
                    stable=n['lane'];pending=None
            else:pending=None
            if n['local'][0]>15:ahead=True
            if ahead and n['local'][0]<-15:
                passed.append({'actor_id':aid,'timestamp':t});ahead=False
            previous=(t,n,e)
    out.update(NPC_lane_changes=npcchanges,NPC_cutin_events=cutins,NPC_acceleration_mps2=stats(npc_accels),NPC_speed_kmh=stats(npc_speeds),
               NPC_braking_exposure_s=sum(a<-1 for a in npc_accels)*.2,NPC_acceleration_exposure_s=sum(a>1 for a in npc_accels)*.2,
               passed_actors=passed,minimum_same_lane_center_gap_minus5_m=min(gaps) if gaps else None,
               same_lane_TTC_s=stats(ttcs),same_lane_headway_s=stats(headways),tracked_velocity_error_mps=stats(errors),tracked_velocity_matches=len(errors))
    out['interpretation']=['Passed actors is geometric relative progress, not proof of a complete overtake-and-return.',
        'Cut-in requires NPC lane transition into a stable ego lane; approximate center-lane classification.',
        'Same-lane gap/TTC use evaluation truth, fixed 5m combined lengths; these are screening proxies.',
        'Velocity matches are nearest visible-centroid assignments, not independent detection recall.',
        'Zero events can mean no exposure; duration completion alone is not capability acceptance.']
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folders',nargs='+',type=Path);args=p.parse_args()
    for folder in args.folders:
        result=analyze(folder);(folder/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({k:result.get(k) for k in ['folder','status','seconds','distance_m','max_speed_kmh','collision_events','events','NPC_cutin_events','tracked_velocity_error_mps']},ensure_ascii=False))
