from train_matched6_fresh_methods2048 import *
@torch.no_grad()
def extra(task,names):
 b=batch_to([ReactiveData(names)[i] for i in np.linspace(0,len(names)-1,12,dtype=int)],'cuda');m=task.model;target=m.to_flow_space(b,b['trajectory']);out={}
 if task.mode=='fm':
  context=m.context(b)
  with torch.random.fork_rng(devices=[0]):torch.manual_seed(9900);noise=torch.randn(len(target),25,5,device='cuda')[...,:2]
  for t0 in [.05,.25,.5,.75,.95]:
   t=torch.full((len(target),1),t0,device='cuda');p=m.velocity(context,(1-t[:,:,None])*noise+t[:,:,None]*target,t);err=p-(target-noise);out[str(t0)]={'mse':float(err.square().mean()),'channels_mse':err.square().mean((0,1)).tolist(),'endpoint_channels_mse':err[:,-1].square().mean(0).tolist()}
 return {'teacher_forced_FM_t_buckets_train_only':out,'scope':'diagnostic vectorfield error uses future labels to construct bridge, NOT deployable prediction or online actor input'}
def main():
 safe_idle();configure_runtime();assert (OUT/'complete.json').exists();ReactiveData.acceleration=True;d=json.loads((ROOT/'data/carla_matched6_cache_v1/manifest.json').read_text());strata=json.loads((OUT/'strata_label_only.json').read_text());cfg=json.loads((OUT/'config.json').read_text());scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m'];summary=[]
 for seed in [7301,7302]:
  for method in ['fm','regression']:
   folder=OUT/f'{seed}-{method}';ck=torch.load(folder/'latest.ckpt',map_location='cpu',weights_only=False);task=FreshTask(cfg,seed,scales,'posteval',method);task.load_state_dict(ck['state_dict']);task.cuda().eval()
   with sdpa_kernel(SDPBackend.MATH):fit,p=assess(task.model,d['train'][::8],strata);buckets=extra(task,d['train'])
   (folder/'train_fixed_stride8_eval.json').write_text(json.dumps(fit));(folder/'FM_bridge_diagnostic.json').write_text(json.dumps(buckets));del task,ck;gc.collect();torch.cuda.empty_cache()
   pred=np.load(folder/'predictions2048.npy')[:,0];ev=json.loads((folder/'eval2048.json').read_text());info=json.loads((folder/'complete.json').read_text());gates={}
   for ep in ['episode4','episode5']:
    ids=[i for i,n in enumerate(d['dev']) if Path(n).parent.name==ep];hold=[i for i in ids if d['dev'][i] in strata['hold1s']];start=[i for i in ids if d['dev'][i] in strata['start_detectable_motion']];decel=[i for i in ids if d['dev'][i] in strata['deceleration']];metrics=ev['candidate0'][ep]
    hold_drift=float(np.linalg.norm(pred[hold,4,:2]*[50,10],axis=-1).mean()) if hold else None
    def speedgate(ix,baseline):
     if not ix:return None
     ys=[];v=[]
     for i in ix:
      with np.load(d['dev'][i]) as z:ys.append(z['trajectory'][-1,4]*30);v.append(z['state'][0]*30)
     ys=np.array(ys);err=float(abs(pred[ix,-1,4]*30-ys).mean());base=float(abs((np.array(v) if baseline=='constant' else 0)-ys).mean());return {'n':len(ix),'endpoint_speed_MAE':err,'baseline_MAE':base,'pass':err<=.8*base}
    gates[ep]={'detectable_start':speedgate(start,'stationary'),'decel':speedgate(decel,'constant'),'hold_n':len(hold),'hold1s_drift_m':hold_drift,'hold_pass':None if hold_drift is None else hold_drift<=.3,'accel_pass':metrics['peak_vector_accel_with_t0_mean']<=6,'jerk_pass':metrics['peak_vector_jerk_mean']<=15,'negative_vx_pass':metrics['negative_vx_fraction']<=.01,'start_positive_endpoint':None if not start else bool((pred[start,-1,0]>0).all())}
   r={'seed':seed,'method':method,'wall_s':info['wall_s'],'peak_gib':info['peak_gib'],'train_stride8':fit['candidate0']['all'],'dev':ev['candidate0']['all'],'world':ev['world_same_ema'],'spread':ev['candidate_spread_rms_m'],'gates':gates,'main_gate_pass':False,'note':'coverage missing inepisode4 means unknown, notpass; any additional dynamic failure separately shown; noCARLA'};summary.append(r);(folder/'admission_evaluation.json').write_text(json.dumps(r,indent=2));print('SUMMARY',seed,method,json.dumps(r),flush=True)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
