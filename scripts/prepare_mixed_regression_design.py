import json,numpy as np,collections
from pathlib import Path
R=Path(__file__).resolve().parents[1];old=json.loads((R/'artifacts/structural-prototypes-light512/selection.json').read_text());extra=[x['file'] for x in json.loads((R/'artifacts/scene-balance-light512/design.json').read_text())['validation']];protected=sorted(set(old['validation']+extra))
def meta(n):
 with np.load(R/'data/occ_v3'/n) as f:
  return {'file':n,'route':Path(n).parent.name,'scene':Path(n).parent.name.split('_Town')[0],'start':int(np.min(f['history_pose_ids'])),'end':max(int(np.max(f['future_pose_ids'])),int(f['frame'])+50),'frame':int(f['frame']),'v0':float(f['state'][0])*30,'v1':float(f['trajectory'][-1,4])*30,'lateral_max':float(abs(f['trajectory'][:,1]).max())*10}
held=[meta(n) for n in protected];eligible=[]
for p in sorted((R/'data/occ_v3/cache/train').rglob('*.npz')):
 n=str(p.relative_to(R/'data/occ_v3'))
 x=meta(n)
 if not any(x['route']==y['route'] and max(x['start'],y['start'])<=min(x['end'],y['end']) for y in held):eligible.append(x)
by=collections.defaultdict(list)
for x in eligible:by[x['scene']].append(x)
chosen=[]
for scene,items in sorted(by.items()):
 moving=[x for x in items if x['v0']>2];assert len(moving)>1
 first=max(moving,key=lambda x:abs(x['v1']-x['v0'])+.2*x['lateral_max']);other=[x for x in moving if x['route']!=first['route']];assert other;second=max(other,key=lambda x:x['v0']);chosen.extend([first,second])
assert len(chosen)==12 and len({x['scene'] for x in chosen})==6
out=R/'artifacts/mixed-regression-flow512';out.mkdir(exist_ok=True);design={'fit':chosen,'held':old['validation'],'all_protected_windows':protected,'eligible_counts':{k:len(v) for k,v in by.items()},'no_support_overlap_same_recording':True,'limits':'Train-only learnability,not route-independent generalization. Fit chosen from train-only support-disjoint moving windows, no protected42/38 windows or support intervals reused. Some routes overlap held; same scene route/weather may share geometry. No officialtest.'};(out/'design.json').write_text(json.dumps(design,indent=2));print(json.dumps({'fit':chosen,'eligible_counts':design['eligible_counts']},indent=2))

