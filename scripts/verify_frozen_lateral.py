from train_carla_nav_recovery_factorial512 import *
from direct_state_velocity import DirectStateVelocity
from frozen_lateral_residual import FrozenLateralResidual
@torch.no_grad()
def main():
 safe_idle();configure_runtime();root=ROOT/'artifacts/carla-frozen-lateral512';d=json.loads((root/'design.json').read_text());names=sum(d['devs'].values(),[]);NavData.enabled=True;ds=NavData(names);res={}
 for seed in [2060,2061]:
  old=torch.load(ROOT/f'artifacts/carla-directstate-paired512/{seed}-direct/latest.ckpt',map_location='cpu',weights_only=False);new=torch.load(root/f'{seed}-lateral/latest.ckpt',map_location='cpu',weights_only=False);ow={k[6:]:v for k,v in old['state_dict'].items() if k.startswith('model.')};nw={k[6:]:v for k,v in new['state_dict'].items() if k.startswith('model.')};assert all(torch.equal(v,nw[k]) for k,v in ow.items());a=DirectStateVelocity().cuda().eval().requires_grad_(False);a.load_state_dict(ow);c=FrozenLateralResidual(new['task_state']['settings']['interaction_rms']).cuda().eval().requires_grad_(False);c.load_state_dict(nw);delta=0.
  for st in range(0,len(ds),4):
   b=batch_to([ds[i] for i in range(st,min(st+4,len(ds)))],'cuda');z=torch.zeros(len(b['state']),25,2,device='cuda');t=torch.zeros(len(z),1,device='cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):
    qa=a(b,z,t);qc=c(b,z,t);pa=a.sample(b,1);pc=c.sample(b,1)
   assert torch.equal(qa[:,:,0],qc[:,:,0]);assert torch.equal(pa[:,:,:,0],pc[:,:,:,0]);delta=max(delta,float((pa[:,:,:,1]-pc[:,:,:,1]).abs().max()*10))
  res[str(seed)]={'all_original_state_entries_equal':len(ow),'dev_windows_x_qx_exact':len(ds),'max_lateral_position_change_m':delta};del a,c,old,new;gc.collect();torch.cuda.empty_cache()
 (root/'longitudinal_verification.json').write_text(json.dumps(res,indent=2));print(res)
if __name__=='__main__':main()
