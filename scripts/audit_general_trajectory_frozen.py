"""Analytic sampler/decoder checks and all-six-scenario frozen conditional diagnostics."""
from train_scene_balance_light512 import *
from pathlib import Path
@torch.no_grad()
def numerical_test():
 torch.manual_seed(2200);m=PrototypeJoint(32,1,4,'velocity2','current_support');t=torch.arange(1,26)*.2;cases={};v0s={}
 cases['constant']=torch.stack([t*0+8,t*0],-1);v0s['constant']=8
 cases['brake']=torch.stack([(8-1.6*t).clamp_min(0),t*0],-1);v0s['brake']=8
 cases['accelerate']=torch.stack([4+.8*t,t*0],-1);v0s['accelerate']=4
 cases['lane_change']=torch.stack([t*0+10,torch.sin(t/5*2*torch.pi)],-1);v0s['lane_change']=10
 cases['turn']=torch.stack([8*torch.cos(t*.1),8*torch.sin(t*.1)],-1);v0s['turn']=8
 cases['merge']=torch.stack([6+.8*t,.5*torch.sin(t/5*torch.pi)],-1);v0s['merge']=6
 results=[]
 for name,v in cases.items():
  state=torch.zeros(1,20);state[0,0]=v0s[name]/30;b={'state':state};q=(v[None]-m.initial_velocity(b))/10;noise=torch.randn_like(q);expected_xy=((torch.cat([torch.tensor([[[v0s[name],0.]]]),v[None,:-1]],1)+v[None])*.1).cumsum(1)
  for steps in [4,16,32]:
   x=noise.clone()
   for i in range(steps):x=x+((q-x)/(1-i/steps))/steps
   p=m.from_flow_space(b,x);decoded_v=p[...,4:5]*30*p[...,[3,2]];xy=p[...,:2]*torch.tensor([50,10]);r={'case':name,'steps':steps,'q_endpoint_max':float((x-q).abs().max()),'velocity_max_error_mps':float((decoded_v-v[None]).abs().max()),'position_max_error_m':float((xy-expected_xy).abs().max()),'finite':bool(torch.isfinite(p).all())};assert r['finite'] and r['velocity_max_error_mps']<1e-4 and r['position_max_error_m']<1e-4;results.append(r)
 return {'results':results,'purpose':'Oracle q target is used ONLY by synthetic analytic field, never policy input. Checks flow-time integration and physical-time decoder units/axes, not learned quality. v0 lateral assumed zero; motion direction not independent body heading.'}
@torch.no_grad()
def sample_context(m,b,noise,steps,intervention):
 z=dict(b);ix=torch.arange(len(b['state']),device='cuda').roll(1)
 if intervention=='history_swap':z['history']=b['history'][ix];z['geometry_valid']=b['geometry_valid'][ix]
 if intervention=='non_speed_state_swap':z['state']=b['state'][ix].clone();z['state'][:,0]=b['state'][:,0]
 c=m.context(z).repeat_interleave(4,0);x=noise.clone()
 for i in range(steps):x=x+m.velocity(c,x,torch.full((len(x),1),i/steps,device=x.device))/steps
 return x.reshape(len(b['state']),4,25,2),m.from_flow_space({'state':b['state'].repeat_interleave(4,0)},x).reshape(len(b['state']),4,25,5)
def main():
 idle();configure_runtime();out=ROOT/'artifacts/general-trajectory-frozen-audit';out.mkdir(exist_ok=False);(out/'analytic.json').write_text(json.dumps(numerical_test(),indent=2));names=json.loads((ROOT/'artifacts/structural-prototypes-light512/selection.json').read_text())['validation'];by={}
 for n in names:by.setdefault(Path(n).parent.name.split('_Town')[0],[]).append(n)
 assert len(by)==6 and all(len(v)==7 for v in by.values());ordered=[by[k][i] for i in range(7) for k in sorted(by)];(out/'selection.json').write_text(json.dumps({'files':ordered,'batching':'one window per scenario per batch; seven batches; previously reused internal diagnostic, not independent validation'},indent=2));scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m'];sources=[('balanced2020',ROOT/'artifacts/scene-balance-light512/2020-balanced/latest.ckpt',2020),('balanced2021',ROOT/'artifacts/scene-balance-light512/2021-balanced/latest.ckpt',2021),('tiny2020',ROOT/'artifacts/tiny-stop-overfit512-v2/2020/latest.ckpt',2020)];allresults=[];start=time.perf_counter()
 for label,src,seed in sources:
  idle();assert time.perf_counter()-start<300;sha=hashlib.sha256(src.read_bytes()).hexdigest();ck=torch.load(src,map_location='cpu',weights_only=False);cfg=ck['task_state']['settings'];m=PrototypeJoint(128,2,4,'velocity2','current_support');m.set_scales(scales);weights={k[len('model.'):]:v for k,v in ck['state_dict'].items() if k.startswith('model.')};m.load_state_dict(weights,strict=True);del ck,weights;m.cuda().eval();rows=[];ds=DiskSamples(ordered)
  for st in range(0,len(ds),6):
   b=batch_to([ds[i] for i in range(st,st+6)],'cuda');torch.manual_seed(2210+st);noise=torch.randn(24,25,5,device='cuda')[...,:2]
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    context=m.context(b);assert torch.equal(context,m.context({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)}));target=m.to_flow_space(b,b['trajectory']).float();c=context.repeat_interleave(4,0);targ=target[:,None].expand(-1,4,-1,-1);one=(noise+m.velocity(c,noise,torch.zeros(24,1,device='cuda'))).reshape(6,4,25,2).float();q,p=sample_context(m,b,noise,16,'none');qh,ph=sample_context(m,b,noise,16,'history_swap');qs,ps=sample_context(m,b,noise,16,'non_speed_state_swap')
   for j in range(6):
    phy=(p[j].float()*p.new_tensor([50,10,1,1,30])).cpu().numpy();v=phy[...,4:5]*phy[...,[3,2]];v0=float(b['state'][j,0])*30;a=np.diff(np.concatenate([np.broadcast_to(np.array([v0,0]),(4,1,2)),v],1),axis=1)/.2;jerk=np.diff(a,axis=1)/.2;truth=(b['trajectory'][j].float()*p.new_tensor([50,10,1,1,30])).cpu().numpy();r={'file':ordered[st+j],'scenario':Path(ordered[st+j]).parent.name.split('_Town')[0],'v0_mps':v0,'label_final_speed_mps':float(truth[-1,4]),'final_speed_mps':float(phy[:,-1,4].mean()),'forward_proxy_m':float(phy[:,-1,0].mean()),'label_forward_m':float(truth[-1,0]),'position_label_fit_m':float(np.linalg.norm(phy[...,:2]-truth[None,:,:2],axis=-1).mean()),'mean_peak_vector_accel_mps2':float(np.linalg.norm(a,axis=-1).max(-1).mean()),'mean_peak_vector_jerk_mps3':float(np.linalg.norm(jerk,axis=-1).max(-1).mean()),'history_swap_velocity_rms_mps':float(((qh[j]-q[j]).float().square().mean()).sqrt()*10),'non_speed_state_swap_velocity_rms_mps':float(((qs[j]-q[j]).float().square().mean()).sqrt()*10),'collision':None,'actual_event_outcome':None}
    for prefix,x in [('one_step_t0',one[j]),('rollout16',q[j].float())]:
     mean=x.mean(0);bias=(mean-target[j]).square().mean()*100;var=(x-mean).square().mean()*100;total=(x-target[j]).square().mean()*100;assert torch.allclose(bias+var,total,rtol=1e-4,atol=1e-4);r[prefix+'_mean_bias_rmse_mps']=float(bias.sqrt());r[prefix+'_candidate_spread_rms_mps']=float(var.sqrt());r[prefix+'_total_label_rmse_mps']=float(total.sqrt())
    rows.append(r)
  groups={}
  for scenario in by:
   subset=[r for r in rows if r['scenario']==scenario];groups[scenario]={'windows':len(subset),'mean':{k:float(np.mean([r[k] for r in subset])) for k,v in subset[0].items() if isinstance(v,(float,int))}}
  r={'model':label,'checkpoint_sha_unchanged':sha==hashlib.sha256(src.read_bytes()).hexdigest(),'groups':groups,'rows':rows};(out/f'{label}.json').write_text(json.dumps(r,indent=2));allresults.append({k:v for k,v in r.items() if k!='rows'});print('DONE_MODEL',label,flush=True);del m;gc.collect();torch.cuda.empty_cache()
 (out/'summary.json').write_text(json.dumps({'results':allresults,'wall_s':time.perf_counter()-start,'limits':'Directory scenario labels, not confirmed triggered event or success. Reused42windows, some training routes but exact windows excluded in latest scene experiment. Conditional-label mean/spread decomposition is not irreducible uncertainty or optimal action. Mismatched context swaps are diagnostic OOD and preserve initial-speed scalar. Unknown collision/free/TTC and source causal certification. No optimizer/training.'},indent=2));print('COMPLETE')
if __name__=='__main__':main()
