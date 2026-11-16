"""Route-balanced train-only frozen JEPA audit; no training and no test enumeration."""
from audit_jepa_space_boundary_v2 import *
from collections import defaultdict

def main():
 idle();configure_runtime();started=time.perf_counter();src=ROOT/'artifacts/masked-teacher-155m-continue2048';out=ROOT/'artifacts/jepa-route-balanced-train-audit';out.mkdir(exist_ok=False);allsel=json.loads((src/'selection.json').read_text());routes=defaultdict(list)
 for name in allsel['train']:
  assert name.replace('\\','/').split('/')[1]=='train';routes[name.replace('\\','/').split('/')[2]].append(name)
 names=[];metadata=[]
 for route,items in sorted(routes.items()):
  items=sorted(items)
  for slot,i in enumerate(np.linspace(0,len(items)-1,3,dtype=int)):
   names.append(items[i]);metadata.append({'route':route,'scenario':route.split('_Town')[0],'slot':slot,'file':items[i]})
 (out/'selection.json').write_text(json.dumps(metadata,indent=2));ck=torch.load(src/'best.ckpt',map_location='cpu',weights_only=False);cfg=json.loads((src/'config.json').read_text());task=Task(cfg,ck['task_state']['manifest_sha'],ck['task_state']['settings']['selection_sha']);task.load_state_dict(ck['state_dict']);del ck;m=task.model.cuda().eval();versions=[p._version for p in m.parameters()];ds=DiskSamples(names);col={k:[] for k in ['pred','target','persistence','valid','dynamic','changed']};causal=[];coverage=[]
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for start in range(0,len(ds),4):
   assert time.perf_counter()-started<300,'budget exceeded';stop=min(start+4,len(ds));b=batch_to([ds[i] for i in range(start,stop)],'cuda');actor={k:b[k] for k in ACTOR_KEYS};h,pred=m.forecast_batch(actor);hp,pp=m.forecast_batch({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)});assert torch.equal(h,hp) and torch.equal(pred,pp);future,valid,_=m.target_data(b);cur=m.canonical_history(b)[:,-1:];target=m.teacher(future).float();pers=m.teacher(cur).float().expand_as(target);raw=b['future'];known=(raw>0)&(raw<8);support=(cur>0)&(cur<8);dynamic=pool(((raw==4)|(raw==5)|(cur==4)|(cur==5))&known&support);changed=pool((raw!=cur)&known&support)
   for k,v in [('pred',pred.float()),('target',target),('persistence',pers),('valid',valid),('dynamic',dynamic),('changed',changed)]:col[k].append(v.cpu())
   for j,ix in enumerate(range(start,stop)):
    speed=float(b['state'][j,0])*30;metadata[ix]['speed_mps']=speed;metadata[ix]['speed_bin']='0-2' if speed<2 else '2-5' if speed<5 else '5-10' if speed<10 else '10+'
    with np.load(ROOT/'data/occ_v3'/names[ix]) as f:
     hi=f['history_pose_ids'];fi=f['future_pose_ids'][:2];assert np.all(hi<=hi[-1]) and np.all(fi>hi[-1]);causal.append({'index':ix,'history_pose_ids':hi.tolist(),'future_pose_ids':fi.tolist(),'actor_future_poison_exact':True})
    for horizon in range(2):
     fk=known[j,horizon];dyn=(raw[j,horizon]==4)|(raw[j,horizon]==5);sp=support[j,0];coverage.append({'index':ix,'horizon_s':horizon+1,'known_voxels':int(fk.sum()),'known_excluded':int((fk&~sp).sum()),'dynamic_voxels':int(dyn.sum()),'dynamic_excluded':int((dyn&~sp).sum()),'valid_tokens':int(valid[j,horizon].sum())})
 z={k:torch.cat(v) for k,v in col.items()};p=z['pred'];t=z['target'];c=z['persistence'];windowrows=[]
 for ix,md in enumerate(metadata):
  for h in range(2):
   for kind,mask in [('all',z['valid'][ix,h]),('dynamic_proxy',z['valid'][ix,h]&z['dynamic'][ix,h]),('changed_known_class',z['valid'][ix,h]&z['changed'][ix,h])]:
    n=int(mask.sum());windowrows.append({**md,'index':ix,'horizon_s':h+1,'group':kind,'tokens':n,'jepa_mse':float((p[ix,h]-t[ix,h]).square().mean(-1)[mask].mean()) if n else None,'persistence_mse':float((c[ix,h]-t[ix,h]).square().mean(-1)[mask].mean()) if n else None})
 def summarize(rows):
  rr=[r for r in rows if r['tokens']];return {'windows':len(rr),'tokens':sum(r['tokens'] for r in rr),'jepa_macro':float(np.mean([r['jepa_mse'] for r in rr])) if rr else None,'persistence_macro':float(np.mean([r['persistence_mse'] for r in rr])) if rr else None,'jepa_wins_windows':sum(r['jepa_mse']<r['persistence_mse'] for r in rr)}
 summary=[]
 for field in ['all','scenario','speed_bin','route']:
  values=['all'] if field=='all' else sorted(set(md[field] for md in metadata))
  for value in values:
   for h in [1,2]:
    for kind in ['all','dynamic_proxy','changed_known_class']:summary.append({'stratum':field,'value':value,'horizon_s':h,'group':kind,**summarize([r for r in windowrows if r['horizon_s']==h and r['group']==kind and (field=='all' or r[field]==value)])})
 perms=[];scenarios=sorted(set(x['scenario'] for x in metadata))
 for seed in [1921,1922,1923,1924,1925]:
  rng=np.random.default_rng(seed);perm=np.zeros(len(metadata),dtype=int)
  for scenario in scenarios:
   rr=sorted(set(x['route'] for x in metadata if x['scenario']==scenario));order=list(rng.permutation(rr));shift=int(rng.integers(1,len(order)))
   for ri,route in enumerate(order):
    srcids=[i for i,x in enumerate(metadata) if x['route']==route];dstids=[i for i,x in enumerate(metadata) if x['route']==order[(ri+shift)%len(order)]];dstids=list(rng.permutation(dstids))
    for a,b in zip(srcids,dstids):perm[a]=b
  assert len(set(perm.tolist()))==len(metadata) and all(metadata[i]['route']!=metadata[j]['route'] and metadata[i]['scenario']==metadata[j]['scenario'] for i,j in enumerate(perm));pt=t[perm];pv=z['valid'][perm];samples=[]
  for ix,md in enumerate(metadata):
   for h in range(2):
    for kind,subset in [('all',z['valid'][ix,h]),('dynamic_proxy',z['valid'][ix,h]&z['dynamic'][ix,h]),('changed_known_class',z['valid'][ix,h]&z['changed'][ix,h])]:
     mask=subset&pv[ix,h];n=int(mask.sum());samples.append({'index':ix,'scenario':md['scenario'],'route':md['route'],'horizon_s':h+1,'group':kind,'tokens':n,'true_mse':float((p[ix,h]-t[ix,h]).square().mean(-1)[mask].mean()) if n else None,'shuffled_mse':float((p[ix,h]-pt[ix,h]).square().mean(-1)[mask].mean()) if n else None})
  ag=[]
  for h in [1,2]:
   for kind in ['all','dynamic_proxy','changed_known_class']:
    ss=[r for r in samples if r['horizon_s']==h and r['group']==kind and r['tokens']];ag.append({'horizon_s':h,'group':kind,'windows':len(ss),'true_macro':float(np.mean([r['true_mse'] for r in ss])),'shuffled_macro':float(np.mean([r['shuffled_mse'] for r in ss])),'true_wins_windows':sum(r['true_mse']<r['shuffled_mse'] for r in ss)})
  perms.append({'seed':seed,'permutation':perm.tolist(),'summary':ag,'samples':samples})
 assert versions==[p._version for p in m.parameters()]
 report={'checkpoint':'original2048 best','routes':len(routes),'windows':len(names),'scenario_count':len(scenarios),'summary':summary,'window_rows':windowrows,'permutations':perms,'causal_checks':causal,'coverage':coverage,'updates':0,'test_read':False,'parameters_unchanged':True,'wall_s':time.perf_counter()-started,'limits':'Seen TRAIN only; equal3windows/route; macros omit groups with zero eligible tokens and report counts. Dynamic classes/label changes are proxies, no free-space truth or source OCC future-leak certificate. Cross-route target mismatch is a sensitivity diagnostic, not a physically possible paired future.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['summary','window_rows','permutations','causal_checks','coverage']},indent=2));print('SUMMARY '+json.dumps([x for x in summary if x['stratum']=='all']));print('DONE')
if __name__=='__main__':main()
