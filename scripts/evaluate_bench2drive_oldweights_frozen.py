"""Frozen old Bench2Drive checkpoints vs exact public-data baseline windows."""
from train_mixed_regression_flow512 import *
from torch.nn.attention import sdpa_kernel,SDPBackend
from pathlib import Path
O=ROOT/'artifacts/bench2drive-frozen-oldweights-v1';D=ROOT/'artifacts/route-regression-flow1024';B=ROOT/'artifacts/bench2drive-official-contract-baselines-v3'
@torch.no_grad()
def generate(m,b,noise):
 c=m.context(b);bs=len(c)
 if m.method=='regression':
  q=m.velocity(c,torch.zeros(bs,25,2,device='cuda'),torch.zeros(bs,1,device='cuda'));return m.from_flow_space(b,q)[:,None].repeat(1,4,1,1)
 x=noise[:,:,:,:2].flatten(0,1).clone();c=c.repeat_interleave(4,0)
 for j in range(16):x+=m.velocity(c,x,torch.full((len(c),1),j/16,device='cuda'))/16
 p=m.from_flow_space({'state':b['state'].repeat_interleave(4,0)},x);p[...,2:4]=torch.nn.functional.normalize(p[...,2:4],dim=-1,eps=1e-6);p[...,4]=p[...,4].clamp_min(0);return p.reshape(bs,4,25,5)
def summary(rows):
 if not rows:return {'n':0,'metrics':None}
 a=np.array([r['position_endpoint_m'] for r in rows]);s=np.array([r['speed_MAE_mps'] for r in rows]);ade=np.array([r['prefix_ADE_m'] for r in rows]);return {'n':len(rows),'routes':len(set(r['route'] for r in rows)),'position_endpoint_mean_m':a.mean(0).tolist(),'position_endpoint_p95_m':np.quantile(a,.95,axis=0).tolist(),'prefix_ADE_mean_m':ade.mean(0).tolist(),'speed_magnitude_MAE_mps':s.mean(0).tolist()}
def main():
 idle();configure_runtime();O.mkdir(exist_ok=False);mft=json.loads((ROOT/'data/occ_v3/manifest.json').read_text());names=mft['splits']['validation'];assert len(names)==1124;d=json.loads((D/'design.json').read_text());labels=json.loads((B/'rows.json').read_text());assert [r['file'] for r in labels]==names
 plan={'models':[f'{s}-{m}' for s in [2050,2051] for m in ['fm','regression']],'primary':'latest1024 fixed, candidate0 perwindowseed1940+index, neverbestofK','noise':'4iidGaussian perwindow CPUGenerator, sharedbetweenmodels;K1fixedfirst','precision':'FP32MATH differsoriginalBF16','data':'same1124val/rawtruth/horizons1,2,3,5','budget_s':300,'no_training':True,'no_test':True,'bootstrap':'1000resamplesof9routes,windowweightedmeans;descriptive notindependentwindowtest','no_closedloop':True}
 (O/'plan.json').write_text(json.dumps(plan,indent=2));start=time.perf_counter();overall=[];noise=torch.stack([torch.randn(4,25,5,generator=torch.Generator().manual_seed(1940+i)) for i in range(len(names))]);torch.save(noise,O/'fixed_noise.pt');idx=np.array([4,9,14,24])
 for seed in [2050,2051]:
  assert not set(d['streams'][str(seed)])&set(names)
  for method in ['fm','regression']:
   assert time.perf_counter()-start<300;path=D/f'{seed}-{method}'/'latest.ckpt';digest=hashlib.sha256(path.read_bytes()).hexdigest();ck=torch.load(path,map_location='cpu',weights_only=False);model=CompareModel(method);model.set_scales(ck['task_state']['settings']['residual_scale']);model.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')},strict=True);model.cuda().eval().requires_grad_(False);folder=O/f'{seed}-{method}';folder.mkdir();checks=[]
   with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
    for i in [0,len(names)//2,len(names)-1]:
     b=batch_to([DiskSamples(names)[i]],'cuda');x=noise[i:i+1,0,:,:2].cuda();tt=torch.full((1,1),.3,device='cuda');out=model(b,x,tt);poison=model({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)},x,tt);assert torch.equal(out,poison);q=model.to_flow_space(b,b['trajectory']);rec=model.from_flow_space(b,q);rv=rec[...,4:5]*30*rec[...,[3,2]];tv=b['trajectory'][...,4:5]*30*b['trajectory'][...,[3,2]];ve=float(abs(rv-tv).max());assert ve<1e-4;checks.append({'window':i,'future_poison_exact':True,'velocity_roundtrip_max_mps':ve,'integrated_label_position_mean_m':float(((rec[...,:2]-b['trajectory'][...,:2])*rec.new_tensor([50,10])).norm(dim=-1).mean()),'history_shape':list(b['history'].shape),'state_dim':b['state'].shape[-1],'history_ages':b['history_age_seconds'].tolist()})
    if method=='fm':
     b=batch_to([DiskSamples(names)[0]],'cuda');c=model.context(b);x=noise[0:1,0,:,:2].cuda().clone()
     for j in range(16):x+=model.velocity(c,x,torch.full((1,1),j/16,device='cuda'))/16
     single=model.from_flow_space(b,x);multi=generate(model,b,noise[:1].cuda())[:,0];diff=float(abs(single-multi).max());assert diff<1e-4
    else:diff=0.
   (folder/'preflight.json').write_text(json.dumps({'checkpoint_sha':digest,'strict_load':True,'checks':checks,'K1_vs_K4_first_normalized_max_abs':diff,'position_label_roundtrip_not_assumed_exact':True}));preds=[];rows=[];peakstart=time.perf_counter();torch.cuda.reset_peak_memory_stats();timings=[]
   with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
    for st in range(0,len(names),4):
     assert time.perf_counter()-start<300;b=batch_to([DiskSamples(names)[j] for j in range(st,min(st+4,len(names)))],'cuda');torch.cuda.synchronize();t=time.perf_counter();p=generate(model,b,noise[st:st+len(b['state'])].cuda());torch.cuda.synchronize();timings.append(time.perf_counter()-t);assert torch.isfinite(p).all();p=p.cpu().numpy();preds.append(p);y=b['trajectory'].cpu().numpy()
     for j in range(len(p)):
      for k in range(4):
       pos=np.linalg.norm((p[j,k,:,:2]-y[j,:,:2])*[50,10],axis=1);sp=abs(p[j,k,:,4]*30-abs(y[j,:,4]*30));rows.append({**{z:labels[st+j][z] for z in ['file','route','family','static_current','hold_full5s','deceleration','acceleration']},'candidate':k,'position_endpoint_m':pos[idx].tolist(),'prefix_ADE_m':[float(pos[:q+1].mean()) for q in idx],'speed_MAE_mps':sp[idx].tolist()})
   pred=np.concatenate(preds);np.save(folder/'predictions.npy',pred);primary=[r for r in rows if r['candidate']==0];groups={'all':primary,**{g:[r for r in primary if r[g]] for g in ['static_current','hold_full5s','deceleration','acceleration']}};groups.update({'route/'+r:[x for x in primary if x['route']==r] for r in sorted(set(x['route'] for x in primary))});groups.update({'family/'+r:[x for x in primary if x['family']==r] for r in sorted(set(x['family'] for x in primary))});routes=sorted(set(r['route'] for r in primary));sums=np.array([np.array([r['position_endpoint_m'] for r in primary if r['route']==n]).sum(0) for n in routes]);counts=np.array([sum(r['route']==n for r in primary) for n in routes]);rng=np.random.default_rng(8100);boots=[]
   for _ in range(1000):ids=rng.integers(0,len(routes),len(routes));boots.append(sums[ids].sum(0)/counts[ids].sum())
   xy=pred[...,:2]*[50,10];report={'seed':seed,'method':method,'checkpoint_sha256':digest,'groups':{k:summary(v) for k,v in groups.items()},'K4_all':summary(rows),'K4_spread_rms_m':float(np.sqrt(((xy-xy.mean(1,keepdims=True))**2).sum(-1).mean())),'route_bootstrap_position_95interval_m':np.quantile(boots,[.025,.975],axis=0).tolist(),'inference_wall_s_including_IO':time.perf_counter()-peakstart,'batch_forward_median_s':float(np.median(timings)),'batch_forward_p95_s':float(np.quantile(timings,.95)),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'scope':'not officialmodel,repeateddev,oracleOCC/pose/nav;bootstrapdescriptive,no safety,nobestofK'}
   (folder/'report.json').write_text(json.dumps(report,indent=2));(folder/'rows.json').write_text(json.dumps(rows));overall.append(report);assert hashlib.sha256(path.read_bytes()).hexdigest()==digest;print('DONE',seed,method,json.dumps(report['groups']['all']),report['inference_wall_s_including_IO'],flush=True);del model,ck,b,p;gc.collect();torch.cuda.empty_cache()
 (O/'summary.json').write_text(json.dumps({'results':overall,'wall_s':time.perf_counter()-start,'no_updates':True,'test_read':False},indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
