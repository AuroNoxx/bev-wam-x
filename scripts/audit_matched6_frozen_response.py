"""No-update response probes on actual cached inputs; ablations are diagnostics, not valid driving inputs."""
from train_matched6_regression512 import RegressionTask,ReactiveData,batch_to,configure_runtime,safe_idle,ROOT,sdpa_kernel,SDPBackend,torch,json,np
from pathlib import Path
import hashlib,time
O=ROOT/'artifacts/matched6-start-frozen-audit';D=json.loads((ROOT/'data/carla_matched6_cache_v1/manifest.json').read_text());rows=json.loads((O/'rows.json').read_text())
def main():
 safe_idle();configure_runtime();ReactiveData.acceleration=True;cfg=json.loads((ROOT/'artifacts/structural-prototypes-light512/config.json').read_text());scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m'];groups={}
 for split in ['train','dev']:
  rs=[r for r in rows if r['split']==split]
  for label,select in [('early_start',lambda r:r['future_start_label'] and r['lead_current_speed_eval_only']>.2),('hold1s',lambda r:r['hold_1s_label']),('decel_observed_motion',lambda r:r['decel_label'] and r['noid_motion_above_OLD_train_threshold'])]:
   found=[r for r in rs if select(r)];ids=np.linspace(0,len(found)-1,min(12,len(found)),dtype=int);groups[split+'_'+label]=[found[i]['file'] for i in ids]
 result={'selection':groups,'limits':'Frozen diagnostic interventions: repeated-current history andunknown-OCC are out of distribution, not deployable controls; zeroaccel onlyablation. GTonlyselectsreportstrata, neverforward. No optimization.','seeds':{}}
 for seed in [7201,7202]:
  path=ROOT/f'artifacts/matched6-regression512/{seed}/latest.ckpt';sha=hashlib.sha256(path.read_bytes()).hexdigest();ck=torch.load(path,map_location='cpu',weights_only=False);model=RegressionTask(cfg,seed,scales,'frozen');model.load_state_dict(ck['state_dict']);model=model.model.cuda().eval();result['seeds'][str(seed)]={}
  with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
   for name,names in groups.items():
    b=batch_to([ReactiveData(names)[i] for i in range(len(names))],'cuda');p=model.sample(b,k=1)[:,0];pv=p[...,4:5]*30*p[...,[3,2]];labels=b['trajectory'][...,4]*30;entry={'n':len(names),'base_speed_02_06_1_5':(p[...,4]*30).mean(0)[[0,2,4,24]].tolist(),'label_speed_02_06_1_5':labels.mean(0)[[0,2,4,24]].tolist(),'withintrajectory_vx_std_mean':float(pv[...,0].std(1).mean()),'label_speed_std_mean':float(labels.std(1).mean())}
    for arm in ['repeat_current_occ','unknown_occ','zero_history_accel']:
     q={k:v.clone() for k,v in b.items()}
     if arm=='repeat_current_occ':q['history']=q['history'][:,-1:].expand_as(q['history']).clone();q['geometry_valid']=q['history']!=255
     elif arm=='unknown_occ':q['history'].fill_(255);q['geometry_valid'].fill_(False)
     else:q['state'][:,3]=0
     pp=model.sample(q,k=1)[:,0];vv=pp[...,4:5]*30*pp[...,[3,2]];entry[arm]={'vector_velocity_change_mean_mps':float((vv-pv).norm(dim=-1).mean()),'vx_signed_change_02_06_1_5':(vv[...,0]-pv[...,0]).mean(0)[[0,2,4,24]].tolist(),'speed_02_06_1_5':(pp[...,4]*30).mean(0)[[0,2,4,24]].tolist()}
    result['seeds'][str(seed)][name]=entry
  assert hashlib.sha256(path.read_bytes()).hexdigest()==sha;del model,ck;torch.cuda.empty_cache()
 (O/'frozen_response.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['seeds'],indent=2))
if __name__=='__main__':main()
