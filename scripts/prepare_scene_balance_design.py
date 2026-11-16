import json,numpy as np,re,collections,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'artifacts/scene-balance-light512';out.mkdir(exist_ok=True)
old=json.loads((ROOT/'artifacts/structural-prototypes-light512/selection.json').read_text());diag=json.loads((ROOT/'artifacts/implementation-plan-v2/stop_deceleration_train_only_selection.json').read_text())['selection']
def route(n):return re.match(r'(.+?_Town\d+_Route\d+)',Path(n).parent.name).group(1)
hold={route(x['file']) for x in diag};exclude=set(old['validation']);pool=[];held=[]
for p in sorted((ROOT/'data/occ_v3/cache/train').rglob('*.npz')):
 name=str(p.relative_to(ROOT/'data/occ_v3'))
 with np.load(p,allow_pickle=False) as d:v0=float(d['state'][0])*30;v1=float(d['trajectory'][-1,4])*30
 kind='steady_stop' if abs(v0)<.5 and abs(v1)<.5 else 'moving_stop' if v0>2 and abs(v1)<.5 else 'slowing' if v0-v1>3 and v1>.5 else 'other'
 item={'file':name,'route_group':route(name),'scenario':Path(name).parent.name.split('_Town')[0],'stratum':kind,'v0':v0,'v1':v1}
 if route(name) in hold:held.append(item)
 elif name not in exclude:pool.append(item)
assert pool and not ({x['route_group'] for x in pool}&hold)
# Preserve24 diagnostic windows exactly, add moving/approximately steady windows on held routes.
seen={x['file'] for x in diag};extra=[]
for g in sorted(hold):
 choices=[x for x in held if x['route_group']==g and x['file'] not in seen and x['v0']>2 and x['v1']>2 and abs(x['v1']-x['v0'])<2]
 if choices:
  x=choices[len(choices)//2];extra.append({**x,'stratum':'normal_moving'});seen.add(x['file'])
validation=diag+extra
scenario_routes=collections.defaultdict(set);route_counts=collections.Counter(x['route_group'] for x in pool)
for x in pool:scenario_routes[x['scenario']].add(x['route_group'])
p=np.array([1/len(scenario_routes)/len(scenario_routes[x['scenario']])/route_counts[x['route_group']] for x in pool]);p/=p.sum()
strata=['steady_stop','moving_stop','slowing','other'];mass={k:float(sum(p[i] for i,x in enumerate(pool) if x['stratum']==k)) for k in strata};target=dict(zip(strata,[0.,.2,.2,.6]));print('PREFLIGHT',dict(collections.Counter(x['stratum'] for x in pool)),mass,flush=True);assert all(mass[k]>0 for k in strata if target[k]>0)
q=p*np.array([target[x['stratum']]/mass[x['stratum']] for x in pool]);q/=q.sum();streams={};stream_reports={}
for seed in [2020,2021]:
 streams[str(seed)]={};stream_reports[str(seed)]={}
 u=np.random.default_rng(seed).random(4096)
 for arm,w in [('representative',p),('balanced',q)]:
  ids=np.searchsorted(np.cumsum(w),u).clip(max=len(pool)-1);items=[pool[i] for i in ids];streams[str(seed)][arm]=[x['file'] for x in items];stream_reports[str(seed)][arm]={'counts':dict(collections.Counter(x['stratum'] for x in items)),'scenarios':dict(collections.Counter(x['scenario'] for x in items)),'unique_windows':len(set(x['file'] for x in items))}
result={'held_route_groups':sorted(hold),'train_route_groups':sorted(set(x['route_group'] for x in pool)),'pool_count':len(pool),'pool_strata':dict(collections.Counter(x['stratum'] for x in pool)),'pool_scenarios':dict(collections.Counter(x['scenario'] for x in pool)),'representative_mass':mass,'balanced_target':target,'validation':validation,'streams':streams,'stream_reports':stream_reports,'limits':'Train split only, whole scenario/Town/Route group excluded including repeated weather logs. Existing24 diagnostic preserved. Development selection previously inspected, not final holdout. No HardBreak training if all its routes are held; known scenario shift. Sampling is sole intervention; effective route marginals change with behavior reweighting. Same4096 samples budget, repeats differ.'}
(out/'design.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['streams','validation','held_route_groups','train_route_groups']}));print('route_groups',len(hold),len(result['train_route_groups']),'validation',len(validation))


