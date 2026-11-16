from train_route_regression_flow1024 import *
from audit_general_trajectory_frozen import sample_context,numerical_test
@torch.no_grad()
def main():
 idle();configure_runtime();base=ROOT/'artifacts/route-regression-flow1024';out=ROOT/'artifacts/route-flow-frozen-full';out.mkdir(exist_ok=False);d=json.loads((base/'design.json').read_text());routes={g for s in d['routes'].values() for g in s['held']};import re
 names=[str(p.relative_to(ROOT/'data/occ_v3')) for p in sorted((ROOT/'data/occ_v3/cache/train').rglob('*.npz')) if re.search(r'Town\d+_Route\d+',p.parent.name).group() in routes];assert len(names)==828
 (out/'selection.json').write_text(json.dumps({'held':names,'limits':'All828 held-route windows overlap in physical time; six independent route IDs, not828 independent scenarios. Previously inspected train-development data, not test.'},indent=2));(out/'analytic.json').write_text(json.dumps(numerical_test()));start=time.perf_counter()
 for seed in [2050,2051]:
  for method in ['fm','regression']:
   idle();assert time.perf_counter()-start<600;src=base/f'{seed}-{method}/latest.ckpt';sha=hashlib.sha256(src.read_bytes()).hexdigest();ck=torch.load(src,map_location='cpu',weights_only=False);m=CompareModel(method);m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});del ck;m.cuda().eval();ref=copy.deepcopy(m.teacher).cuda().eval();ev=evaluate(m,names,ref,True);(out/f'{seed}-{method}-full.json').write_text(json.dumps(ev));rows=[]
   if method=='fm':
    ds=DiskSamples([x['file'] for x in d['held']])
    for st in range(0,len(ds),4):
     b=batch_to([ds[i] for i in range(st,min(st+4,len(ds)))],'cuda');n=len(b['state']);torch.manual_seed(2250+st);noise=torch.randn(n*4,25,5,device='cuda')[...,:2]
     with torch.autocast('cuda',dtype=torch.bfloat16):
      c=m.context(b);assert torch.equal(c,m.context({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)}));c=c.repeat_interleave(4,0);target=m.to_flow_space(b,b['trajectory']).repeat_interleave(4,0);truth=b['trajectory'].repeat_interleave(4,0)
      for kind in ['roll1','roll4','roll16','roll32','bridge0','bridge0.5','bridge0.9']:
       if kind.startswith('roll'):q,p=sample_context(m,b,noise,int(kind[4:]),'none');q=q.reshape(n*4,25,2);p=p.reshape(n*4,25,5)
       else:
        t=float(kind[6:]);x=(1-t)*noise+t*target;q=x+(1-t)*m.velocity(c,x,torch.full((n*4,1),t,device='cuda'));p=m.from_flow_space({'state':b['state'].repeat_interleave(4,0)},q)
       for j in range(n):
        qs=q[j*4:j*4+4].float();tr=target[j*4].float();bias=(qs.mean(0)-tr).square().mean()*100;spread=(qs-qs.mean(0)).square().mean()*100;phy=p[j*4:j*4+4].float();v=phy[...,4:5]*30*phy[...,[3,2]];v0=m.initial_velocity({'state':b['state'][j:j+1]}).expand(4,-1,-1);a=torch.diff(torch.cat([v0,v],1),dim=1)/.2;jerk=torch.diff(a,dim=1)/.2
        rows.append({'kind':kind,'window':st+j,'bias_rmse_mps':float(bias.sqrt()),'noise_spread_mps':float(spread.sqrt()),'position_fit_m':float(((phy[...,:2]-truth[j*4,:,:2])*phy.new_tensor([50,10])).norm(dim=-1).mean()),'forward_m':float(phy[:,-1,0].mean()*50),'vector_peak_jerk_mps3':float(jerk.norm(dim=-1).max(-1).values.mean())})
   result={'seed':seed,'method':method,'hash_unchanged':sha==hashlib.sha256(src.read_bytes()).hexdigest(),'full_summary':ev['summary'],'world':ev['world_summary'],'path_rows':rows};(out/f'{seed}-{method}-audit.json').write_text(json.dumps(result,indent=2));print('DONE',seed,method,flush=True);del m,ref;gc.collect();torch.cuda.empty_cache()
 (out/'complete.json').write_text(json.dumps({'wall_s':time.perf_counter()-start,'no_updates':True}));print('COMPLETE',flush=True)
if __name__=='__main__':main()
