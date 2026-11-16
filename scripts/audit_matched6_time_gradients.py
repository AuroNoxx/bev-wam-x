"""Frozen physical-time path, weighted-gradient audit. No optimizer/EMA updates."""
from train_matched6_regression512 import *
from torch.nn import functional as F
O=ROOT/'artifacts/matched6-time-gradient-audit'
def vec(grads,params):return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for g,p in zip(grads,params)])
def main():
 safe_idle();configure_runtime();O.mkdir(exist_ok=False);ReactiveData.acceleration=True;cfg=json.loads((ROOT/'artifacts/structural-prototypes-light512/config.json').read_text());scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m'];d=json.loads((ROOT/'data/carla_matched6_cache_v1/manifest.json').read_text());audit=json.loads((ROOT/'artifacts/matched6-start-frozen-audit/rows.json').read_text());groups={}
 groups['start']=[r['file'] for r in audit if r['split']=='train' and r['future_start_label'] and r['lead_current_speed_eval_only']>.2]
 groups['decel']=[r['file'] for r in audit if r['split']=='train' and r['decel_label'] and r['noid_motion_above_OLD_train_threshold']]
 groups['hold']=[r['file'] for r in audit if r['split']=='train' and r['hold_1s_label']]
 groups['steady']=[]
 for name in d['train']:
  with np.load(name) as z:v=z['state'][0]*30;vf=z['trajectory'][-1,4]*30
  if v>2 and abs(vf-v)<.5:groups['steady'].append(name)
 report={'scope':'2 frozen512checkpoints,2 fixed train-only balanced batches of12; no update/EMA/test; zeroxt/t regression inference, same objective as training; eval mode dropout0','groups_available':{k:len(v) for k,v in groups.items()},'seeds':{}}
 for seed in [7201,7202]:
  pl.seed_everything(seed);task=RegressionTask(cfg,seed,scales,'audit');initpos=task.model.position.detach().clone();initout=task.model.output.weight.detach().clone();source=ROOT/f'artifacts/matched6-regression512/{seed}/latest.ckpt';sha=hashlib.sha256(source.read_bytes()).hexdigest();ck=torch.load(source,map_location='cpu',weights_only=False);task.load_state_dict(ck['state_dict']);m=task.model.cuda().eval();params=[p for p in m.parameters() if p.requires_grad];names=[n for n,p in m.named_parameters() if p.requires_grad];result={'position_initial_std':float(initpos.std()),'position_current_std':float(m.position.std()),'position_delta_rms':float((m.position.cpu()-initpos).square().mean().sqrt()),'output_weight_delta_rms':float((m.output.weight.cpu()-initout).square().mean().sqrt()),'batches':[]}
  for bi in [0,1]:
   selected=[]
   for k,g in groups.items():selected.extend([g[j] for j in np.linspace(bi,len(g)-2+bi,3,dtype=int)])
   b=batch_to([ReactiveData(selected)[i] for i in range(12)],'cuda');original=m.position.detach().clone();zeros=torch.zeros(12,25,2,device='cuda');t=torch.zeros(12,1,device='cuda');rng=torch.Generator(device='cuda').manual_seed(9900+bi)
   with sdpa_kernel(SDPBackend.MATH):
    context=m.context(b);q=m.velocity(context,zeros,t).float();target=m.to_flow_space(b,b['trajectory']);physical=m.from_flow_space(b,q);err=(physical[...,:2]-b['trajectory'][...,:2])*q.new_tensor([50,10]);gate=(torch.rand(12,device='cuda',generator=rng)>=.5).float()*2
    losses={'action':F.mse_loss(q,target),'weighted_position':.2*(F.smooth_l1_loss(err/5,torch.zeros_like(err),reduction='none').flatten(1).mean(1)*gate).mean(),'weighted_world':sum(.1*v for v in m.auxiliary_loss(b))};gv={};per={}
    for label,loss in losses.items():
     grad=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);gv[label]=vec(grad,params);per[label]={'loss':float(loss),'global_norm':float(gv[label].norm()),'position_norm':float(grad[names.index('position')].norm()) if grad[names.index('position')] is not None else 0,'output_weight_norm':float(grad[names.index('output.weight')].norm()) if grad[names.index('output.weight')] is not None else 0}
     pg=grad[names.index('position')]
     if pg is not None:per[label]['position_temporal_contrast_gradient_norm']=float((pg-pg.mean(1,keepdim=True)).norm());per[label]['position_each_time_grad_norm']=pg.norm(dim=-1).flatten().tolist()
    combined=sum(gv.values());cos={a+'__'+bb:float(F.cosine_similarity(gv[a],gv[bb],dim=0)) for a in gv for bb in gv if a<bb};cos['sum_clip1_coefficient']=min(1.,1./float(combined.norm()))
    stats={}
    for gi,k in enumerate(groups):
     sl=slice(gi*3,gi*3+3);stats[k]={'files':selected[sl],'pred_vx_each_time_mean':(m.initial_velocity(b)+q*q.new_tensor([10,.5]))[sl,:,0].mean(0).detach().tolist(),'label_vx_each_time_mean':(m.initial_velocity(b)+target*q.new_tensor([10,.5]))[sl,:,0].mean(0).detach().tolist(),'normalized_channel_mse_each_time':(q-target)[sl].square().mean(0).detach().tolist()}
    with torch.no_grad():
     permutation=torch.arange(24,-1,-1,device='cuda');m.position.copy_(original[:,permutation]);reversedq=m.velocity(context.detach(),zeros,t);permerr=float(abs(reversedq-q.detach()[:,permutation]).max());m.position.zero_();nopos=m.velocity(context.detach(),zeros,t);noposvariation=float((nopos-nopos.mean(1,keepdim=True)).abs().max());m.position.copy_(original)
    result['batches'].append({'selection':selected,'fixed_position_gate':gate.tolist(),'gradient_terms':per,'cosines':cos,'combined_norm':float(combined.norm()),'time_permutation_max_error':permerr,'zero_position_time_variation_max':noposvariation,'baseline_q_time_variation_std':float(q.std(1).mean()),'groups':stats})
   del context,q,target,losses,gv,combined,b
  assert all(torch.equal(v.detach().cpu(),ck['state_dict']['model.'+k]) for k,v in m.state_dict().items());assert hashlib.sha256(source.read_bytes()).hexdigest()==sha;report['seeds'][str(seed)]=result;del task,m,ck;torch.cuda.empty_cache()
 (O/'report.json').write_text(json.dumps(report,indent=2));print('COMPLETE')
if __name__=='__main__':main()
