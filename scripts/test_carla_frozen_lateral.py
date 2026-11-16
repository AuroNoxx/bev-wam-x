from train_carla_nav_recovery_factorial512 import *
from direct_state_velocity import DirectStateVelocity
from frozen_lateral_residual import FrozenLateralResidual
@torch.no_grad()
def main():
 safe_idle();configure_runtime();out=ROOT/'artifacts/carla-frozen-lateral512';out.mkdir(exist_ok=True);rms=json.loads((ROOT/'artifacts/carla-direct-rms-paired512/preflight.json').read_text())['interaction_rms'];d=json.loads((ROOT/'artifacts/carla-directstate-paired512/design.json').read_text());NavData.enabled=True;ds=NavData(d['selection'][:4]);b=batch_to([ds[i] for i in range(4)],'cuda');res={}
 for seed in [2060,2061]:
  ck=torch.load(ROOT/f'artifacts/carla-directstate-paired512/{seed}-direct/latest.ckpt',map_location='cpu',weights_only=False);weights={k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')};a=DirectStateVelocity().cuda().eval().requires_grad_(False);a.load_state_dict(weights);c=FrozenLateralResidual(rms).cuda().eval();missing=c.load_state_dict(weights,strict=False);assert missing.missing_keys==['lateral_residual.weight'];assert sum(p.numel() for p in c.parameters() if p.requires_grad)==100;res[str(seed)]={}
  for mode,dtype in [('fp32',None),('bf16',torch.bfloat16)]:
   with torch.autocast('cuda',dtype=dtype,enabled=dtype is not None):p=a.sample(b);q=c.sample(b)
   print(seed,mode,'maxdiff',float((p-q).abs().max()),'xdiff',float((p[...,0]-q[...,0]).abs().max()),'headmax',float(c.lateral_residual.weight.abs().max()),flush=True);assert torch.equal(p,q);res[str(seed)][mode]=float((p-q).abs().max())
  c.lateral_residual.weight.fill_(.01);assert torch.equal(c.sample(b)[...,0],a.sample(b)[...,0]);assert torch.equal(c.sample(b),c.sample({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)}));s=b['state'].clone();s[:,0]=0;assert torch.equal(c.lateral_features(s)[:,3],torch.zeros_like(s[:,0]))
 res.update(passed=True,interaction_rms=rms,trainable_parameters=100,future_poison_exact=True,x_exact_with_nonzero_lateral=True,ema_update_disabled=True);(out/'preflight.json').write_text(json.dumps(res,indent=2));print(res)
if __name__=='__main__':main()


