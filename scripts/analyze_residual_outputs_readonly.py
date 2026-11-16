import json,numpy as np
from pathlib import Path
root=Path('E:/ai/projects/highway-occ-flow');src=root/'artifacts/residual-representation-paired128';out=root/'artifacts/residual-output-readonly-audit';out.mkdir(exist_ok=False);sel=json.load(open(src/'selection.json'));scales=np.array(json.load(open(src/'scale_calibration.json'))['scales_m']);states=[];truth=[]
for p in sel['validation']:
 with np.load(root/'data/occ_v3'/p,allow_pickle=False) as z:states.append(z['state']);truth.append(z['trajectory']*np.array([50,10,1,1,30]))
states=np.array(states);truth=np.array(truth);v0=states[:,0]*30;t=np.arange(1,26)*.2;prior=v0[:,None]*t[None,:];reports=[]
for seed in [1980,1981]:
 for arm in ['original','residual']:
  records=json.load(open(src/f'{seed}-{arm}/report.json'))['eval']['records'];rows=[]
  for r in records:
   i=r['window'];p=np.array(r['points']);xy=p[:,:2];chord=np.diff(np.vstack(([0,0],xy)),axis=0)/.2;vel=p[:,4,None]*p[:,[3,2]];position_acc=np.diff(np.vstack(([v0[i],0],chord)),axis=0)/.2;position_jerk=np.diff(position_acc,axis=0)/.2;headacc=np.diff(np.r_[v0[i],p[:,4]])/.2;headjerk=np.diff(headacc)/.2;label=truth[i];rows.append({'window':i,'candidate':r['candidate'],'v0':float(v0[i]),'label_end_x':float(label[-1,0]),'pred_end_x':float(xy[-1,0]),'prior_end_x':float(prior[i,-1]),'decoded_residual_end_x':float(xy[-1,0]-prior[i,-1]),'flow_residual_end_x_if_residual':float((xy[-1,0]-prior[i,-1])/scales[-1,0]),'negative_segment_fraction':float((chord[:,0]<0).mean()),'heading_backward_fraction':float((p[:,3]<0).mean()),'negative_position_with_forward_heading_fraction':float(((chord[:,0]<0)&(p[:,3]>=0)).mean()),'position_derived_acc_peak':float(np.linalg.norm(position_acc,axis=-1).max()),'position_derived_jerk_peak':float(np.linalg.norm(position_jerk,axis=-1).max()),'speed_head_acc_peak':float(abs(headacc).max()),'speed_head_jerk_peak':float(abs(headjerk).max()),'position_velocity_mean':float(np.linalg.norm(chord,axis=-1).mean()),'head_velocity_mean':float(p[:,4].mean()),'x_second_difference_rms_m':float(np.sqrt(np.mean(np.diff(np.r_[0,xy[:,0]],n=2)**2)))})
  keys=[k for k in rows[0] if k not in ['window','candidate']];means={k:float(np.mean([r[k] for r in rows])) for k in keys};groups={}
  for name,lo,hi in [('near_stop',-1,.5),('slow',.5,5),('moving',5,100)]:
   rr=[r for r in rows if lo<=r['v0']<hi];groups[name]={'candidates':len(rr),**({k:float(np.mean([x[k] for x in rr])) for k in keys} if rr else {})}
  reports.append({'seed':seed,'arm':arm,'means':means,'v0_groups':groups,'worst_backward':sorted(rows,key=lambda r:r['pred_end_x'])[:5]})
report={'input_v0_min_max_mps':[float(v0.min()),float(v0.max())],'label_end_x_min_max_m':[float(truth[:,-1,0].min()),float(truth[:,-1,0].max())],'label_backward_endpoint_count':int((truth[:,-1,0]<0).sum()),'results':reports,'limits':'read only existing42window4candidate evidence; positions and speed-head trajectories are predictions, not actual control; repeated route-correlated train diagnostic; derivative t0initial v assumedforward; no training, no test'};(out/'report.json').write_text(json.dumps(report,indent=2))
for r in reports:
 print(r['seed'],r['arm'],json.dumps(r['means']));print('MOVING',r['v0_groups']['moving']['pred_end_x'],r['v0_groups']['moving']['label_end_x'])
print(report['input_v0_min_max_mps'],report['label_end_x_min_max_m'],report['label_backward_endpoint_count'])
