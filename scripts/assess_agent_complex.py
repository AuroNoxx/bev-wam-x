"""Offline factual metrics/replay for bounded agent-decision traffic cases."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def assess(folder):
    report=json.loads((folder/'report.json').read_text())
    scenario=json.loads((folder/'scenario.json').read_text())
    rows=[json.loads(line) for line in (folder/'telemetry.jsonl').read_text().splitlines()]
    decisions=json.loads((folder/'decisions.json').read_text())
    gaps=[];ttcs=[];headways=[];rear_ttcs=[]
    for row in rows:
        e=row['actors']['ego']
        for name,o in row['actors'].items():
            if name=='ego' or o['road']!=e['road'] or o['lane']!=e['lane']:continue
            delta=o['s']-e['s'];gap=abs(delta)-e['half_length']-o['half_length']
            if delta>0:
                gaps.append(gap)
                if e['speed']>10/3.6:
                    headways.append(max(0,gap)/e['speed'])
                    closing=e['long_speed']-o['long_speed']
                    if closing>.1:ttcs.append(max(0,gap)/closing)
            else:
                closing=o['long_speed']-e['long_speed']
                if closing>.1 and o['speed']>10/3.6:rear_ttcs.append(max(0,gap)/closing)
    speeds=np.array([r['actors']['ego']['speed'] for r in rows]);a=np.diff(speeds)/.05
    smoothed=np.convolve(a,np.ones(11)/11,mode='valid');j=np.diff(smoothed)/.05
    marks=report.get('lane_marking_events',[])
    solid=[event for event in marks if any('Solid' in v for v in event['markings'])]
    summary={'scenario':report['scenario'],'status':report['status'],'simulated_seconds':report['simulated_seconds'],
             'decisions':len(decisions),'vehicles':len(rows[0]['actors']),'collision_events':len(report.get('collisions',[])),
             'guard_frames':len(report.get('guard_interventions',[])),'solid_line_events':len(solid),'all_line_events':len(marks),
             'max_speed_kmh':float(max(speeds)*3.6),'min_same_lane_front_gap_m':min(gaps) if gaps else None,
             'min_same_lane_front_ttc_s':min(ttcs) if ttcs else None,
             'min_same_lane_headway_s':min(headways) if headways else None,
             'min_same_lane_rear_ttc_s':min(rear_ttcs) if rear_ttcs else None,
             'distance_m':sum(math.dist([x['actors']['ego'][k] for k in ('x','y','z')],[y['actors']['ego'][k] for k in ('x','y','z')]) for x,y in zip(rows,rows[1:])),
             'max_abs_lane_error_outside_transition_m':max(abs(r['actors']['ego']['lateral_error']) for r in rows if not r['lane_change_active']),
             'acceleration_abs_p95_mps2':float(np.percentile(abs(a),95)),
             'acceleration_abs_max_mps2':float(max(abs(a))),
             'jerk_smoothed_0p55s_abs_p95_mps3':float(np.percentile(abs(j),95)),
             'lanes_visited':sorted(set(r['actors']['ego']['lane'] for r in rows)),
             'initial_screen_only':True,'expert_qualified':False,
             'metric_limits':['TTC uses longitudinal road-s and box half lengths, not swept collision geometry',
                              'Same lane is center-based; during a lane change inspect both lanes separately',
                              ('Traffic distribution known; sampled NPC targets/seed withheld during decisions; logical isolation, not a security sandbox' if scenario.get('independent_traffic') else 'Scenario author knew setup; not blind held-out evaluation'),
                              'Simulation pauses between agent decisions; not real-time latency validation']}
    gates=scenario['screening']
    summary['predeclared_numeric_screen_passed']=(report['status']=='completed' and not summary['collision_events'] and not summary['guard_frames'] and not solid and
        (summary['min_same_lane_front_ttc_s'] is None or summary['min_same_lane_front_ttc_s']>=gates['min_same_lane_front_ttc_s']) and
        (summary['min_same_lane_headway_s'] is None or summary['min_same_lane_headway_s']>=gates['min_same_lane_headway_above_10kmh_s']))
    (folder/'assessment.json').write_text(json.dumps(summary,indent=2))
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',15);lookup={r['frame']:r for r in rows};frames=[]
    for file in sorted((folder/'frames').glob('*.png')):
        im=Image.open(file).convert('RGB');d=ImageDraw.Draw(im);d.rectangle((0,0,640,68),fill='black')
        d.text((8,3),report['scenario']+' | AGENT | 4x replay',fill='white',font=font)
        r=lookup.get(int(file.stem))
        if r:
            e=r['actors']['ego'];d.text((8,24),f"t={r['simulation_s']:.1f}s | {e['speed']*3.6:.1f}km/h | lane {e['lane']} -> {r['target_lane']}",fill='white',font=font)
            d.text((8,45),f"decision {r['command_id']} | ego autopilot OFF | ground-truth input",fill='yellow',font=font)
        frames.append(im.quantize(colors=128))
    frames[0].save(folder/'replay-4x.gif',save_all=True,append_images=frames[1:],duration=125,loop=0,optimize=False)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('folders',nargs='+',type=Path);args=p.parse_args()
    print(json.dumps([assess(folder) for folder in args.folders],indent=2))
