from train_carla_nav_recovery_factorial512 import *
from direct_state_velocity import DirectStateVelocity
from normalized_direct_state import NormalizedDirectState
from carla_nav_contract import cache_near
@torch.no_grad()
def main():
 safe_idle();configure_runtime();out=ROOT/'artifacts/carla-direct-rms-paired512';out.mkdir(exist_ok=False);d=json.loads((ROOT/'artifacts/carla-directstate-paired512/design.json').read_text());seq=d['selection'];cache={}
 for p in set(seq):
  with np.load(p) as z:cache[p]=float(z['state'][0])*3*float(cache_near(p)[1])*50
 rms=float(np.sqrt(np.mean([cache[p]**2 for p in seq])));assert 0<rms<1;NavData.enabled=True;ds=NavData(seq[:4]);b=batch_to([ds[i] for i in range(4)],'cuda');res={}
 for seed in [2060,2061]:
  pl.seed_everything(seed);a=DirectStateVelocity().cuda().eval();c=NormalizedDirectState(rms).cuda().eval();c.load_state_dict(a.state_dict());res[str(seed)]={}
  for mode,dtype in [('fp32',None),('bf16',torch.bfloat16)]:
   with torch.autocast('cuda',dtype=dtype,enabled=dtype is not None):p=a.sample(b);q=c.sample(b)
   assert torch.equal(p,q);res[str(seed)][mode]=float((p-q).abs().max())
  assert torch.equal(c.sample(b),c.sample({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)}));s=torch.zeros(1,20,device='cuda');s[:,5]=.4/50;assert float(c.direct_features(s)[0,3])==0;s[:,0]=10/30;assert abs(float(c.direct_features(s)[0,3])-.4/rms)<1e-5;assert torch.equal(c.direct_features(s)[:,:3],a.direct_features(s)[:,:3])
 res.update(passed=True,train_draws=len(seq),train_unique=len(cache),selection_sha=hashlib.sha256(json.dumps(seq).encode()).hexdigest(),interaction_rms=rms,static_exact_zero=True,no_centering=True,future_poison_exact=True);(out/'preflight.json').write_text(json.dumps(res,indent=2));print(res)
if __name__=='__main__':main()
