from train_carla_nav_recovery_factorial512 import *
from scaled_velocity_flow import ScaledVelocityFlow
safe_idle();configure_runtime();out=ROOT/'artifacts/carla-lateral-scale512';out.mkdir(exist_ok=False);d=json.loads((ROOT/'artifacts/carla-nav-recovery-factorial512/design.json').read_text());v=[]
for n in sorted(set(d['mixed_sequence'])):
 with np.load(n) as z:v.append(z['trajectory'][:,4]*30*z['trajectory'][:,2])
v=np.concatenate(v);cal={'train_unique':len(set(d['mixed_sequence'])),'lateral_rms':float(np.sqrt((v*v).mean())),'abs_p95':float(np.quantile(abs(v),.95)),'abs_max':float(abs(v).max()),'chosen_scale_mps':.5,'selection':'train-only magnitude, rounded above observedmax, notclip orphysicalbound'};assert cal['abs_max']<.5
m=ScaledVelocityFlow().cuda().eval();original=ScaledVelocityFlow(10).cuda().eval();state=torch.zeros(3,20,device='cuda');state[:,0]=torch.tensor([0,5,10],device='cuda')/30;b={'state':state};torch.manual_seed(2600);q=torch.randn(3,25,2,device='cuda');p=m.from_flow_space(b,q);qr=m.to_flow_space(b,p);assert (q-qr).abs().max()<1e-5;oldq=q*q.new_tensor([1,.05]);pold=original.from_flow_space(b,oldq);assert (p-pold).abs().max()<1e-5
qpos=torch.zeros_like(q);qpos[:,:,1]=2;pp=m.from_flow_space(b,qpos);assert (pp[:,:,1]>0).all();qbig=qpos*100;pb=m.from_flow_space(b,qbig);assert pb[...,4].max()*30>99
NavData.enabled=True;batch=batch_to([NavData([d['mixed_sequence'][0]])[0]],'cuda')
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
 c=m.context(batch);other={**batch,'trajectory':torch.full_like(batch['trajectory'],999),'future':torch.full_like(batch['future'],255)};assert torch.equal(c,m.context(other))
cal.update(q_roundtrip_max=float((q-qr).abs().max()),physical_equivalence_max=float((p-pold).abs().max()),future_poison_actor_exact=True,axis_test=True,no_clipping_test=True);(out/'preflight.json').write_text(json.dumps(cal,indent=2));print(cal)
