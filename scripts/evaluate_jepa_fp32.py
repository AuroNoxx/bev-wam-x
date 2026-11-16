from train_jepa_nojepa_paired import *
@torch.no_grad()
def evaluate_fp32(m,names,reference,enabled):
 m.cuda().eval();ds=DiskSamples(names);rows=[];world=[];div=[]
 for start in range(0,len(ds),4):
  b=batch_to([ds[i] for i in range(start,min(start+4,len(ds)))],'cuda')
  with torch.random.fork_rng(devices=[0]),torch.autocast('cuda',enabled=False):
   torch.manual_seed(1940+start);p=m.sample(b,k=4,steps=16).float();physical=(p*p.new_tensor([50.,10.,1,1,30.])).cpu().numpy();_,z=m.forecast_batch(b);future,valid,_=m.target_data(b);target=reference(future).float();persistent=reference(m.canonical_history(b)[:,-1:]).float().expand_as(target);own=m.teacher(future).float();current=m.teacher(m.canonical_history(b)[:,-1:]).float().expand_as(own)
   # Future poison must not change actor context, even in fusion arm.
   assert torch.equal(m.context(b),m.context({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)}))
  for j,q in enumerate(physical):
   div.append(float(np.sqrt(((q[:,:,:2]-q[:,:,:2].mean(0))**2).sum(-1).mean())))
   for k,trajectory in enumerate(q):
    a=audit(trajectory,float(b['state'][j,0])*30);derived=np.linalg.norm(np.diff(np.vstack(([0.,0.],trajectory[:,:2])),axis=0),axis=1)/.2;rows.append({'window':start+j,'candidate':k,'first_residual_mps':a['first_segment']['kinematic_residual_norm_mps'],'first_accel_mps2':a['first_segment']['signed_speed_accel_mps2'],'kinematic_mps':a['full']['mean_kinematic_residual_mps'],'max_accel_mps2':a['full']['max_abs_scalar_accel_mps2'],'max_jerk_mps3':a['full']['max_abs_scalar_jerk_mps3'],'forward_proxy_m':float(trajectory[-1,0]),'negative_forward_segments':int((np.diff(np.r_[0,trajectory[:,0]])<0).sum()),'nearzero':float((trajectory[:,4]<.5).mean()),'speed_disagreement':float(abs(derived-trajectory[:,4]).mean()),'points':trajectory.tolist(),'collision':None,'TTC':None,'route_progress':None})
   mask=valid[j];n=int(mask.sum());world.append({'window':start+j,'valid_tokens':n,'fixed_reference_mse':float((z[j].float()-target[j]).square().mean(-1)[mask].mean()) if n else None,'fixed_persistence_mse':float((persistent[j]-target[j]).square().mean(-1)[mask].mean()) if n else None,'own_ema_mse':float((z[j].float()-own[j]).square().mean(-1)[mask].mean()) if n else None,'own_persistence_mse':float((current[j]-own[j]).square().mean(-1)[mask].mean()) if n else None})
 keys=['first_residual_mps','kinematic_mps','max_accel_mps2','max_jerk_mps3','forward_proxy_m','negative_forward_segments','nearzero','speed_disagreement'];return {'summary':{**{k:float(np.mean([r[k] for r in rows])) for k in keys},'spread_rms_m':float(np.mean(div)),'global_first_accel':max(abs(r['first_accel_mps2']) for r in rows),'global_jerk':max(r['max_jerk_mps3'] for r in rows)},'world_summary':{k:float(np.mean([r[k] for r in world if r[k] is not None])) for k in ['fixed_reference_mse','fixed_persistence_mse','own_ema_mse','own_persistence_mse']},'world_head_active':enabled,'world_rows':world,'records':rows,'actor_future_poison_exact':True}
