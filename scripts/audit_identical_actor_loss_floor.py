"""Exact duplicate actors: analytic action-MSE floor versus frozen prediction. No fitting."""
from train_matched6_regression512 import *
O=ROOT/'artifacts/matched6-ambiguity-shape-v2'
def main():
 safe_idle();configure_runtime();ReactiveData.acceleration=True;groups=json.loads((O/'report.json').read_text())['exact_duplicate_actor_groups'];cfg=json.loads((ROOT/'artifacts/structural-prototypes-light512/config.json').read_text());cfg.update(batch_size=12,max_steps=512);scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m'];report={'definition':'Foridenticalactors, normalized2Dvelocity actionMSE=irreducible empirical variance aroundgroupmean+prediction excess. Thisis not theoptimumofcombinedposition/worldloss, not safety. Groupsselectedtrain-only.','groups':[]}
 for group in groups:
  names=[r['file'] for r in group];b=batch_to([ReactiveData(names)[i] for i in range(len(names))],'cuda');keys=['history','state','history_valid','history_age_seconds','geometry_valid'];assert all(torch.equal(b[k],b[k][0:1].expand_as(b[k])) for k in keys);entry={'n':len(names),'first':names[0],'last':names[-1],'full_actor_tensors_exact':True,'models':{}}
  for seed in [7201,7202]:
   source=ROOT/f'artifacts/matched6-regression-resume1024-v3/{seed}/latest.ckpt';sha=hashlib.sha256(source.read_bytes()).hexdigest();ck=torch.load(source,map_location='cpu',weights_only=False);task=RegressionTask(cfg,seed,scales,'audit');task.load_state_dict(ck['state_dict']);m=task.model.cuda().eval()
   with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
    target=m.to_flow_space(b,b['trajectory']);mean=target.mean(0);first={k:v[:1] for k,v in b.items()};q=m(first,torch.zeros(1,25,2,device='cuda'),torch.zeros(1,1,device='cuda'))[0];floor=float((target-mean).square().mean());excess=float((q-mean).square().mean());total=float((q[None]-target).square().mean());assert abs(total-floor-excess)<1e-6
    entry['action_variance_floor']=floor;entry['conditional_mean_vx25']=(mean[:,0]*10+b['state'][0,0]*30).tolist();entry['models'][str(seed)]={'action_mse':total,'excess_mse':excess,'excess_fraction':excess/total,'pred_vx25':(q[:,0]*10+b['state'][0,0]*30).tolist()}
   assert hashlib.sha256(source.read_bytes()).hexdigest()==sha;del m,task,ck;torch.cuda.empty_cache()
  report['groups'].append(entry)
 (O/'identical_actor_loss_floor.json').write_text(json.dumps(report,indent=2));print('COMPLETE')
if __name__=='__main__':main()
