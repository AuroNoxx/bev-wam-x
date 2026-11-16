from train_carla_nav_recovery_factorial512 import *
from deterministic_velocity_model import DeterministicVelocityFlow
from direct_state_velocity import DirectStateVelocity
@torch.no_grad()
def main():
 safe_idle();configure_runtime();out=ROOT/'artifacts/carla-directstate-paired512';out.mkdir(exist_ok=False);res={};d=json.loads((ROOT/'artifacts/carla-lateral-scale512/design.json').read_text());NavData.enabled=True;ds=NavData(d['selection'][:4]);batch=batch_to([ds[i] for i in range(4)],'cuda')
 for seed in [2060,2061]:
  pl.seed_everything(seed);a=DeterministicVelocityFlow().cuda().eval();c=DirectStateVelocity().cuda().eval();missing=c.load_state_dict(a.state_dict(),strict=False);assert missing.missing_keys==['direct_state.weight'] and not missing.unexpected_keys
  res[str(seed)]={}
  for mode,dtype in [('fp32',None),('bf16',torch.bfloat16)]:
   with torch.autocast('cuda',dtype=dtype,enabled=dtype is not None):p=a.sample(batch,4,16);q=c.sample(batch,4,16)
   assert torch.equal(p,q);res[str(seed)][mode]=float((p-q).abs().max())
  poison={**batch,'future':torch.full_like(batch['future'],255),'trajectory':torch.full_like(batch['trajectory'],999)};assert torch.equal(c.sample(batch),c.sample(poison));s=torch.zeros(1,20,device='cuda');s[0,[0,4,5]]=s.new_tensor([10/30,5/50,.2/50]);assert torch.allclose(c.direct_features(s),s.new_tensor([[1,.5,.2,.2]]));changed=s.clone();changed[:,[1,2,3,6,7,8,19]]=999;assert torch.equal(c.direct_features(s),c.direct_features(changed))
 res.update(passed=True,actor_future_poison_exact=True,feature_contract='speed/10mps, nearX/10m, nearY/1m, product(speed/10,nearY/1)',new_parameters=200);(out/'preflight.json').write_text(json.dumps(res,indent=2));print(res)
if __name__=='__main__':main()
