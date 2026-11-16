"""Frozen velocity2 stopping/noise diagnostic; no optimizers or updates."""
from train_scene_balance_light512 import *
@torch.no_grad()
def rollout(m,b,noise,steps,swap=False):
 z=dict(b)
 if swap:
  ix=torch.arange(len(b['state'])-1,-1,-1,device='cuda');z['history']=b['history'][ix];z['geometry_valid']=b['geometry_valid'][ix]
 c=m.context(z).repeat_interleave(4,0);x=noise.clone()
 for i in range(steps):x=x+m.velocity(c,x,torch.full((len(x),1),i/steps,device=x.device))/steps
 return m.from_flow_space({'state':b['state'].repeat_interleave(4,0)},x).reshape(len(b['state']),4,25,5),x.reshape(len(b['state']),4,25,2)
def main():
 idle();configure_runtime();start=time.perf_counter();base=ROOT/'artifacts/scene-balance-light512';out=ROOT/'artifacts/scene-balance-frozen-stop-noise';out.mkdir(exist_ok=False);d=json.loads((base/'design.json').read_text());train=[]
 for n in sorted(set(sum([v for arms in d['streams'].values() for v in arms.values()],[]))):
  with np.load(ROOT/'data/occ_v3'/n) as f:
   if float(f['state'][0])*30>2 and abs(float(f['trajectory'][-1,4])*30)<.5:train.append(n)
 held=[r['file'] for r in d['validation'] if r['stratum']=='moving_to_stopped'];assert not set(train)&set(held);sel={'train_stop':train,'held_stop':held};(out/'selection.json').write_text(json.dumps(sel,indent=2));cfg=json.loads((base/'config.json').read_text());scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m'];allresults=[]
 for seed in [2020,2021]:
  for arm in ['representative','balanced']:
   idle();assert time.perf_counter()-start<600;src=base/f'{seed}-{arm}'/'latest.ckpt';before=hashlib.sha256(src.read_bytes()).hexdigest();ck=torch.load(src,map_location='cpu',weights_only=False);task=ProtoTask(cfg,seed,'velocity_only',scales,'readonly');task.load_state_dict(ck['state_dict'],strict=True);del ck;m=task.model.cuda().eval();rows=[];bridge=[]
   for split,names in sel.items():
    ds=DiskSamples(names)
    for st in range(0,len(ds),4):
     b=batch_to([ds[i] for i in range(st,min(st+4,len(ds)))],'cuda');torch.manual_seed(2030+st);noise=torch.randn(len(b['state'])*4,25,5,device='cuda')[...,:2];original=None
     with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
      c=m.context(b);assert torch.equal(c,m.context({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)}));target=m.to_flow_space(b,b['trajectory']);single=noise.reshape(len(b['state']),4,25,2)[:,0]
      for t in [0.,.5,.9]:
       x=(1-t)*single+t*target;pred=m.velocity(c,x,torch.full((len(x),1),t,device='cuda')).float();end=x+(1-t)*pred;err=(end-target).square().mean((1,2)).sqrt()*10
       for j,e in enumerate(err):bridge.append({'split':split,'file':names[st+j],'t':t,'oracle_bridge_endpoint_velocity_rmse_mps':float(e)})
      for scale,steps,swap in [(1.,16,False),(1.,32,False),(0.,16,False),(0.,32,False),(1.,16,True)]:
       p,q=rollout(m,b,noise*scale,steps,swap);physical=(p.float()*p.new_tensor([50,10,1,1,30])).cpu().numpy()
       if scale==1 and steps==16 and not swap:original=physical.copy()
       for j,pp in enumerate(physical):
        initial=float(b['state'][j,0])*30;truth=(b['trajectory'][j].float()*p.new_tensor([50,10,1,1,30])).cpu().numpy();vel=pp[...,4:5]*pp[...,[3,2]];acc=np.diff(np.concatenate([np.broadcast_to(np.array([initial,0]),(4,1,2)),vel],axis=1),axis=1)/.2;jerk=np.diff(acc,axis=1)/.2
        rows.append({'split':split,'file':names[st+j],'noise_scale':scale,'euler_steps':steps,'history_swapped':swap,'swap_effective':bool(swap and len(b['state'])>1),'initial_speed_mps':initial,'end_speed_mps':float(pp[:,-1,4].mean()),'end_stop_fraction':float((pp[:,-1,4]<.5).mean()),'forward_proxy_m':float(pp[:,-1,0].mean()),'velocity_spread_mps':float(np.std(vel,axis=0).mean()),'mean_peak_vector_accel':float(np.linalg.norm(acc,axis=-1).max(-1).mean()),'mean_peak_vector_jerk':float(np.linalg.norm(jerk,axis=-1).max(-1).mean()),'position_fit_m':float(np.linalg.norm(pp[...,:2]-truth[None,:,:2],axis=-1).mean()),'q_final_change_from_prior':float(q[j,:,-1,0].float().mean()),'label_q_final_x':float(target[j,-1,0]),'swap_trajectory_rms_difference_m':float(np.sqrt(((physical[j,:,:,:2]-original[j,:,:,:2])**2).mean())) if swap else None})
   groups=[]
   for split in sel:
    for scale,steps,swap in [(1.,16,False),(1.,32,False),(0.,16,False),(0.,32,False),(1.,16,True)]:
     subset=[r for r in rows if r['split']==split and r['noise_scale']==scale and r['euler_steps']==steps and r['history_swapped']==swap];keys=['initial_speed_mps','end_speed_mps','end_stop_fraction','forward_proxy_m','velocity_spread_mps','mean_peak_vector_accel','mean_peak_vector_jerk','position_fit_m','q_final_change_from_prior','label_q_final_x'];groups.append({'split':split,'windows':len(subset),'noise_scale':scale,'steps':steps,'history_swapped':swap,**{k:float(np.mean([r[k] for r in subset])) for k in keys},'swap_rms_m':float(np.mean([r['swap_trajectory_rms_difference_m'] for r in subset if r['swap_effective']])) if swap else None})
   br={split:{str(t):float(np.mean([r['oracle_bridge_endpoint_velocity_rmse_mps'] for r in bridge if r['split']==split and r['t']==t])) for t in [0.,.5,.9]} for split in sel};r={'seed':seed,'arm':arm,'groups':groups,'oracle_bridge':br,'checkpoint_sha_unchanged':before==hashlib.sha256(src.read_bytes()).hexdigest(),'rows':rows,'bridge':bridge};(out/f'{seed}-{arm}.json').write_text(json.dumps(r,indent=2));allresults.append({k:v for k,v in r.items() if k not in ['rows','bridge']});print('DONE_ARM',seed,arm,flush=True);del task,m;gc.collect();torch.cuda.empty_cache()
 (out/'summary.json').write_text(json.dumps({'results':allresults,'wall_s':time.perf_counter()-start,'limits':'Frozen no updates; source OCC causality not certified; train and held routes differ and speeds must be reported; scale0 is off-training-distribution diagnostic, not deployable sampler fix; bridge t>0 uses target only in diagnostic and is not online performance; history swaps can be unrealistic and singleton batches ineffective; no collision/safety score.'},indent=2));print('COMPLETE')
if __name__=='__main__':main()
