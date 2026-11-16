from train_carla_nav_recovery_factorial512 import *
from audit_general_trajectory_frozen import sample_context
@torch.no_grad()
def main():
 safe_idle();configure_runtime();base=ROOT/'artifacts/carla-nav-recovery-factorial512';out=ROOT/'artifacts/carla-recovery-exposure-nav-frozen';out.mkdir(exist_ok=False);d=json.loads((base/'design.json').read_text());exposure={}
 for seqname in ['original_sequence','mixed_sequence']:
  seq=d[seqname];counts={'draws':len(seq),'unique':len(set(seq)),'recovery_draws':0,'offcenter_draws':0,'offcenter_moving_draws':0,'start_draws':0};unique={'offcenter':set(),'moving':set()}
  for n in seq:
   with np.load(n) as z:
    v=float(z['state'][0])*30;end=float(z['trajectory'][-1,4])*30;e=float(z['lane_error_right_m']) if 'lane_error_right_m' in z else 0
    counts['recovery_draws']+=int('lane_error_right_m' in z);counts['start_draws']+=int(v<.2 and end>2)
    if abs(e)>.1:
     counts['offcenter_draws']+=1;unique['offcenter'].add(n)
     if v>.5:counts['offcenter_moving_draws']+=1;unique['moving'].add(n)
  counts.update(offcenter_unique=len(unique['offcenter']),offcenter_moving_unique=len(unique['moving']));exposure[seqname]=counts
 rec=json.loads((ROOT/'data/carla_recovery_combined_v2/manifest.json').read_text());names=[n for n in rec['train'] if int(Path(n).stem) in [50,80]]+[n for n in rec['dev'] if int(Path(n).stem) in [50,80,100]];NavData.enabled=True;ds=NavData(names);b=batch_to([ds[i] for i in range(len(ds))],'cuda');torch.manual_seed(2500);noise=torch.randn(len(ds)*4,25,5,device='cuda')[...,:2];results=[]
 for arm in ['on_original','on_recovery']:
  path=base/arm/'latest.ckpt';sha=hashlib.sha256(path.read_bytes()).hexdigest();ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['global_step']==512;mode=ck['task_state']['settings']['state_mode'];assert mode=='speed_near';m=CompareModel('fm');m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});del ck;m.cuda().eval();outputs={}
  with torch.autocast('cuda',dtype=torch.bfloat16):
   for mode in ['actual','zero_nav','flip_y','y_times10_ood']:
    z={**b,'state':b['state'].clone()}
    if mode=='zero_nav':z['state'][:,4:6]=0
    if mode=='flip_y':z['state'][:,5]*=-1
    if mode=='y_times10_ood':z['state'][:,5]*=10
    q,p=sample_context(m,z,noise,16,'none');outputs[mode]=(q.float(),p.float())
  rows=[]
  for i,n in enumerate(names):
   for mode,(q,p) in outputs.items():
    phy=p[i]*p.new_tensor([50,10,1,1,30]);baseq=outputs['actual'][0][i];rows.append({'file':n,'mode':mode,'source':'train' if n in rec['train'] else 'dev','near_flu_m':(b['state'][i,4:6]*50).cpu().tolist(),'actual_target_y06_m':float(b['trajectory'][i,2,1]*10),'mean_pred_y06_m':float(phy[:,2,1].mean()),'mean_pred_y5_m':float(phy[:,-1,1].mean()),'velocity_change_rms_mps':float((q[i]-baseq).square().mean().sqrt()*10),'mean_forward5_m':float(phy[:,-1,0].mean())})
  results.append({'arm':arm,'rows':rows,'sha_unchanged':sha==hashlib.sha256(path.read_bytes()).hexdigest()});del m;gc.collect();torch.cuda.empty_cache()
 (out/'report.json').write_text(json.dumps({'exposure':exposure,'results':results,'limits':'14fixedactualtrain/devinputs,shared4Gaussian perinput, navigationcounterfactualonly. yx10OODnotcorrectionpolicy. Nooptimizer/test/closedloop.'},indent=2));print(exposure)
if __name__=='__main__':main()
