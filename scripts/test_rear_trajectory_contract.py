import json,numpy as np,torch
from pathlib import Path
from rear_trajectory_contract import RearTrajectoryDecoder,supervision
from audit_signed_curvature_expressivity import decode
from audit_rear_reference_sensitivity import correction
O=Path('artifacts/rear-trajectory-contract-v1');O.mkdir(exist_ok=True);m=RearTrajectoryDecoder();torch.manual_seed(2701);c=(torch.randn(3,6,2,dtype=torch.float64)*.03).requires_grad_();v=torch.tensor([0.,3.,-2.],dtype=torch.float64);p=m(v,c);checks={}
for i in range(3):
 sv,y,xy=decode(float(v[i]),c[i,:,0].detach().numpy(),c[i,:,1].detach().numpy());expected=xy+correction(y,m.geometry.rear_distance_m);assert np.allclose(p['actor_xy'][i].detach(),expected[20::20],atol=1e-11)
checks['numpy_label_decoder_equivalence']=True
label={k:p[k].detach().clone() for k in ['actor_xy','signed_speed','body_yaw']};label.update(valid=torch.ones(3,25,dtype=torch.bool),yaw_observable=torch.ones(3,25,dtype=torch.bool),interval_valid=torch.ones(3,25,dtype=torch.bool),curvature_identifiable=torch.ones(3,25,dtype=torch.bool));yy=torch.cat([torch.zeros(3,1),label['body_yaw']],1);label['interval_yaw_rate']=torch.diff(yy,dim=1)/.2
assert max(float(x[0].detach()) for x in supervision(p,label).values())<1e-20;checks['label_roundtrip_zero_loss']=True
label['curvature_identifiable'][:,:8]=False;a=supervision(p,label)['interval_yaw_rate_rad2ps2'][0];label['interval_yaw_rate'][:,:8]=1e20;b=supervision(p,label)['interval_yaw_rate_rad2ps2'][0];assert torch.equal(a,b);checks['low_speed_label_poison_exact']=True
label['actor_xy']+=.01;loss=sum(x[0] for x in supervision(p,label).values());loss.backward();assert torch.isfinite(c.grad).all();checks['finite_cpu_backward_no_optimizer']=True
rest=torch.zeros(1,6,2,dtype=torch.float64,requires_grad=True);rest.data[:,:,1]=.1;q=m(torch.zeros(1),rest);assert torch.equal(q['actor_xy'],torch.zeros_like(q['actor_xy'])) and torch.equal(q['body_yaw'],torch.zeros_like(q['body_yaw']));q['actor_xy'].sum().backward();assert torch.equal(rest.grad[:,:,1],torch.zeros_like(rest.grad[:,:,1]));checks['rest_no_motion_curvature_unobservable']=True
checks['future_labels_not_decoder_signature']=True;checks['geometry_not_episode_certified']=not m.geometry.episode_certified
(O/'tests.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

