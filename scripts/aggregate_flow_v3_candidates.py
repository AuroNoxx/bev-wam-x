import numpy as np, json
from pathlib import Path
m=json.load(open('data/occ_v3/manifest.json',encoding='utf-8'))
names=np.asarray(m['splits']['validation'])
target=[];state=[]
for n in names:
    with np.load(Path('data/occ_v3')/n) as a: target.append(a['trajectory']); state.append(a['state'])
target=np.stack(target); state=np.stack(state)
p=np.load('artifacts/diagnostics/flow-v3-assessment/flow_v3.npz')['prediction']  # seeds,w,c,25,5
cv=np.load('artifacts/diagnostics/flow-v3-assessment/constant_velocity.npz')['prediction'][0,:,0]  # w,25,5
truth=target[...,:2]*[50,10]
cand=p[...,:2]*[50,10]
speed0=state[:,0]*30.0; speed_end=target[:,-1,4]*30.0; dv=speed_end-speed0
lateral=np.abs(truth[:,-1,1]-truth[:,0,1])

def ade_xy(x):  # x: w,25,2 -> w
    return np.linalg.norm(x-truth,axis=-1).mean(-1)

cv_ade=ade_xy(cv[...,:2]*[50,10])

def per_seed_stats(fn, mask):
    ades=[];fdes=[];worse=[]
    for s in range(cand.shape[0]):
        out=fn(cand[s])
        e=np.linalg.norm(out[mask]-truth[mask],axis=-1).mean(-1)
        f=np.linalg.norm(out[mask,-1]-truth[mask,-1],axis=-1)
        ades.append(e.mean());fdes.append(f.mean())
        worse.append(float((e>=cv_ade[mask]).mean()))
    return {'ADE_m':float(np.mean(ades)),'ADE_m_seed_std':float(np.std(ades)),'FDE_m':float(np.mean(fdes)),
            'fraction_not_better_than_constant_velocity':float(np.mean(worse))}

fns={
 'first_candidate': lambda c: c[:,0],
 'candidate_mean': lambda c: c.mean(1),
 'candidate_median': lambda c: np.median(c,axis=1),
}
buckets={'all':np.ones(len(state),bool),
 'cruise_steady':(np.abs(dv)<1.0)&(lateral<1.0),
 'decel_gt2':dv<-2.0,
 'accel_gt2':dv>2.0,
 'lane_change_lateral_gt2m':lateral>2.0}
out={'windows':int(len(state)),'seeds':int(cand.shape[0]),'candidates':int(cand.shape[2]),
     'aggregation':'per sampling seed, then averaged over the 3 seeds (deployable single-run statistic)',
     'constant_velocity_ADE_by_bucket':{k:float(cv_ade[v].mean()) for k,v in buckets.items()},
     'methods':{k:{} for k in fns},'windows_by_bucket':{k:int(v.sum()) for k,v in buckets.items()}}
for label,mask in buckets.items():
    for name,fn in fns.items():
        out['methods'][name][label]=per_seed_stats(fn,mask)
Path('artifacts/diagnostics/v3-limits').mkdir(parents=True,exist_ok=True)
Path('artifacts/diagnostics/v3-limits/candidate-aggregation.json').write_text(json.dumps(out,indent=2))
d=out['methods']
print('median all',d['candidate_median']['all'])
print('first all',d['first_candidate']['all'])
print('mean all',d['candidate_mean']['all'])
print('median cruise',d['candidate_median']['cruise_steady'])
print('cv',out['constant_velocity_ADE_by_bucket'])
