"""Frozen JEPA target/persistence and boundary mechanism audit, bounded development only."""
from diagnose_frozen_rollout import *

def pool(mask):
 b,t=mask.shape[:2];return F.adaptive_avg_pool2d(mask.float().mean(2).flatten(0,1)[:,None],(8,20)).flatten(1).reshape(b,t,160)>0

def main():
 idle();configure_runtime();started=time.perf_counter();out=ROOT/'artifacts/jepa-space-boundary-audit';out.mkdir(exist_ok=False);base=ROOT/'artifacts/masked-teacher-155m-continue2048';paired=ROOT/'artifacts/teacher-rollout-steps-paired';sel=json.loads((paired/'selection.json').read_text());cfg=json.loads((paired/'config.json').read_text());ck=torch.load(base/'best.ckpt',map_location='cpu',weights_only=False);task=Task(cfg,ck['task_state']['manifest_sha'],ck['task_state']['settings']['selection_sha']);del ck;m=task.model.cuda().eval();ds=DiskSamples(sel['validation']);paths=[('original2048',base/'best.ckpt'),('1810-rollout16',paired/'1810-rollout16/latest.ckpt'),('1811-rollout16',paired/'1811-rollout16/latest.ckpt')];reports=[];reference=None
 for label,path in paths:
  assert time.perf_counter()-started<180,'budget exceeded';ck=torch.load(path,map_location='cpu',weights_only=False);task.load_state_dict(ck['state_dict']);del ck;collect={k:[] for k in ['pred','target','persistent','valid','dynamic','changed','target_unknown','online']};coverage=[];horizon_ids=[]
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   for start in range(0,16,4):
    b=batch_to([ds[i] for i in range(start,start+4)],'cuda');cur=m.canonical_history(b)[:,-1:];future,valid,frac=m.target_data(b);_,pred=m.forecast_batch(b);target=m.teacher(future).float();persistent=m.teacher(cur).float().expand_as(target);online=m.encoder(future).float();fu=torch.where((future>0)&(future<8),future,255);target_unknown=m.teacher(fu).float();raw=b['future'];knownraw=(raw>0)&(raw<8);support=(cur>0)&(cur<8);dynamic=pool(((raw==4)|(raw==5)|(cur==4)|(cur==5)) & knownraw & support);changed=pool((raw!=cur)&knownraw&support)
    for k,x in [('pred',pred.float()),('target',target),('persistent',persistent),('valid',valid),('dynamic',dynamic),('changed',changed),('target_unknown',target_unknown),('online',online)]:collect[k].append(x.cpu())
    for j in range(4):
     for h in range(2):
      fk=knownraw[j,h];dyn=(raw[j,h]==4)|(raw[j,h]==5);sp=support[j,0];coverage.append({'window':start+j,'horizon_s':h+1,'initial_speed_mps':float(b['state'][j,0])*30,'raw_known_voxels':int(fk.sum()),'known_outside_current_support_fraction':float((fk&~sp).sum()/fk.sum().clamp_min(1)),'dynamic_voxels':int(dyn.sum()),'dynamic_outside_current_support_fraction':float((dyn&~sp).sum()/dyn.sum().clamp_min(1)) if dyn.any() else None,'future_zero_inside_support_voxels':int(((raw[j,h]==0)&sp).sum()),'valid_tokens':int(valid[j,h].sum()),'dynamic_proxy_tokens':int((valid[j,h]&dynamic[j,h]).sum()),'known_class_change_tokens':int((valid[j,h]&changed[j,h]).sum())})
    for name in sel['validation'][start:start+4]:
     with np.load(ROOT/'data/occ_v3'/name) as f:horizon_ids.append((f['future_pose_ids'][:2]-f['history_pose_ids'][-1]).tolist())
   # Reproduce exact prior evaluation noise for the persistent worst boundary candidate.
   b=batch_to([ds[i] for i in range(12,16)],'cuda');context=m.context({k:b[k] for k in ACTOR_KEYS}).repeat_interleave(4,0);torch.manual_seed(8193);x=torch.randn(16,25,5,device='cuda')
   for i in range(16):x=x+m.velocity(context,x,torch.full((16,1),i/16,device='cuda'))/16
   x=x.float();rawv=float(x[6,0,4])*30;v0=float(b['state'][1,0])*30;boundary={'window':13,'candidate':2,'observed_speed_mps':v0,'raw_first_speed_mps':rawv,'clamped_first_speed_mps':max(rawv,0),'first_accel_mps2':(max(rawv,0)-v0)/.2,'local_clamp_derivative_for_strictly_negative_raw':0 if rawv<0 else 1,'hard_initial_boundary_enforced':False,'noise_seed':8193}
  z={k:torch.cat(v) for k,v in collect.items()};groups=[]
  for horizon in [0,1]:
   for kind in ['all_valid','dynamic_class_proxy','changed_known_class','other_valid']:
    mask=z['valid'][:,horizon].clone()
    if kind=='dynamic_class_proxy':mask&=z['dynamic'][:,horizon]
    if kind=='changed_known_class':mask&=z['changed'][:,horizon]
    if kind=='other_valid':mask&=~z['dynamic'][:,horizon]
    n=int(mask.sum());p=z['pred'][:,horizon];t=z['target'][:,horizon];c=z['persistent'][:,horizon];tu=z['target_unknown'][:,horizon]
    def mse(a,b):return float((a-b).square().mean(-1)[mask].mean()) if n else None
    groups.append({'horizon_s':horizon+1,'group':kind,'tokens':n,'jepa_mse':mse(p,t),'persistence_mse':mse(c,t),'jepa_unitnorm_mse':mse(F.normalize(p,dim=-1),F.normalize(t,dim=-1)),'persistence_unitnorm_mse':mse(F.normalize(c,dim=-1),F.normalize(t,dim=-1)),'target_zero_to_unknown_drift_mse':mse(t,tu),'jepa_to_unknown_canonical_target_mse':mse(p,tu),'persistence_to_unknown_canonical_target_mse':mse(c,tu),'online_ema_drift_mse':mse(z['online'][:,horizon],t),'pred_rms':float(p.square().mean(-1)[mask].sqrt().mean()) if n else None,'target_rms':float(t.square().mean(-1)[mask].sqrt().mean()) if n else None})
  mask=z['valid'];common=mask&mask.roll(1,0);shuffled=float((z['pred']-z['target'].roll(1,0)).square().mean(-1)[common].mean());targetstd=float(z['target'].std(0,unbiased=False).mean());predstd=float(z['pred'].std(0,unbiased=False).mean());targetdrift=None if reference is None else float((z['target']-reference).square().mean());reference=z['target'].clone() if reference is None else reference
  report={'checkpoint':label,'groups':groups,'coverage':coverage,'future_pose_offsets':horizon_ids,'shuffled_target_mse_on_common_valid':shuffled,'target_cross_example_std_unmasked':targetstd,'pred_cross_example_std_unmasked':predstd,'target_drift_from_original2048_mse_unmasked':targetdrift,'boundary':boundary};reports.append(report);(out/'report.json').write_text(json.dumps(reports,indent=2));print('RESULT '+label+' '+json.dumps({'groups':groups,'boundary':boundary,'target_std':targetstd,'pred_std':predstd,'target_drift':targetdrift}),flush=True)
 (out/'completion.json').write_text(json.dumps({'updates':0,'test_read':False,'wall_seconds':time.perf_counter()-started,'checkpoints':3,'development_windows':16,'dynamic_labels_are_proxies_not_actor_tracks':True,'boundary_only_local_clamp_derivative_not_full_gradient':True},indent=2));print('DONE')
if __name__=='__main__':main()
