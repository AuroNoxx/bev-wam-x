"""Render transparent review examples; longest candidate of each type, no inference labels."""
import csv,json,html,math
from pathlib import Path
from dataset_v1 import ROOT

base=ROOT/'artifacts/diagnostics/behavior-screen'
report=json.loads((base/'report.json').read_text())
panels=[]
selection=[]
for kind in ['following','slow_queue','stop_then_restart','lane_change']:
    options=[(r,e) for r in report['routes'] for e in r['events'] if e['kind']==kind]
    route,event=max(options,key=lambda x:x[1]['elapsed_s'])
    with (ROOT/route['timeline']).open() as stream: rows=list(csv.DictReader(stream))
    t=[int(r['frame'])*.1 for r in rows]
    def x(v):return 50+900*(v-t[0])/max(.1,t[-1]-t[0])
    def y(v):return 220-180*v/80
    def path_for(key):
        commands=[];start=True
        for r,tm in zip(rows,t):
            if not r[key]:start=True;continue
            commands.append(f'{"M" if start else "L"}{x(tm):.2f},{y(float(r[key])):.2f}');start=False
        return ' '.join(commands)
    start=x(event['start_frame']*.1);end=x(event['end_frame']*.1)
    svg=f'<rect x="{start}" y="25" width="{end-start}" height="200" fill="#ffdf91"/>'
    for v in range(0,81,20):
        svg+=f'<line x1="50" y1="{y(v)}" x2="950" y2="{y(v)}" stroke="#ddd"/><text x="6" y="{y(v)}">{v}</text>'
    svg+=f'<path d="{path_for("speed_kmh")}" fill="none" stroke="#1760bd" stroke-width="2"/>'
    svg+=f'<path d="{path_for("lead_speed_kmh")}" fill="none" stroke="#bd5117" stroke-width="2"/>'
    svg+=f'<text x="50" y="250">{t[0]:.1f} s</text><text x="890" y="250">{t[-1]:.1f} s</text>'
    table='<tr>'+''.join(f'<th>{html.escape(k)}</th>' for k in rows[0] if k!='desired_speed_raw')+'</tr>'
    table+=''.join('<tr>'+''.join(f'<td>{html.escape(v)}</td>' for k,v in r.items() if k!='desired_speed_raw')+'</tr>' for r in rows)
    panels.append(f'<section><h2>{kind}</h2><p>{html.escape(route["route"])}</p><svg viewBox="0 0 1000 270">{svg}</svg><p>Speed: blue ego, orange nearest same-lane lead (km/h). Yellow: candidate interval. Time from frame index.</p><details><summary>Inspect frame, lane IDs, lead identity and distance</summary><table>{table}</table></details></section>')
    selection.append({'route':route['route'],'event':event})
page='<!doctype html><meta charset="utf-8"><title>Behavior review</title><style>body{font:15px system-ui;max-width:1100px;margin:30px auto}section{margin:24px 0;padding:20px;border:1px solid #ddd}svg{width:100%}td,th{padding:5px;border-bottom:1px solid #ddd}details{overflow:auto;max-height:500px}</style><h1>Logged behavior candidates</h1><p>Longest candidate per kind. These are heuristic review examples, not ground truth. Lead identity can change outside the highlighted segment. Road boundaries, visibility and thresholds can omit events.</p>'+''.join(panels)
(base/'review-examples.html').write_text(page,encoding='utf-8')
(base/'review-selection.json').write_text(json.dumps(selection,indent=2))
print(base/'review-examples.html')
