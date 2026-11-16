from train_carla_nav_recovery_factorial512 import *
from navigation_scaled_flow import NavigationScaledFlow
from carla_nav_contract import near_goal,cache_near,raw_rows
@torch.no_grad()
def main():
 safe_idle();configure_runtime();out=ROOT/'artifacts/carla-navscale-paired512';out.mkdir(exist_ok=True);d=json.loads((ROOT/'artifacts/carla-lateral-scale512/design.json').read_text());names=d['selection'][::1000];NavData.enabled=True;ds=NavData(names);b=batch_to([ds[i] for i in range(len(ds))],'cuda');pl.seed_everything(2060);a=NavigationScaledFlow(1).cuda().eval();c=NavigationScaledFlow(50).cuda().eval();c.load_state_dict(a.state_dict());c.state[0].weight[:,5]/=50
 encoded=c.encoded_state(b['state']);assert torch.equal(encoded[:,:5],b['state'][:,:5]) and torch.equal(encoded[:,6:],b['state'][:,6:]);assert torch.allclose(encoded[:,5]/50,b['state'][:,5]);assert torch.equal(c.context(b),c.context({**b,'trajectory':torch.full_like(b['trajectory'],999),'future':torch.full_like(b['future'],255)}));res={}
 for name,dtype in [('fp32',None),('bf16',torch.bfloat16)]:
  with torch.autocast('cuda',dtype=dtype,enabled=dtype is not None):
   ctx0=a.context(b);ctx1=c.context(b);torch.manual_seed(2500);p0=a.sample(b,4,16);torch.manual_seed(2500);p1=c.sample(b,4,16)
  physical=(p1-p0)*p0.new_tensor([50,10,1,1,30]);res[name]={'context_max':float((ctx1-ctx0).abs().max()),'physical_max':float(physical.abs().max())};assert res[name]['context_max']<(1e-5 if dtype is None else .02);assert res[name]['physical_max']<(1e-3 if dtype is None else .2)
 parity=[]
 for n in d['selection']:
  p=Path(n)
  if p.parent.parent.name!='carla_miniloop_cache_v1':continue
  row=raw_rows(str(ROOT/'data/carla_causal_miniloop_20261116_town12_r2'),p.parent.name)[int(p.stem)];online=near_goal(row);cached=cache_near(n);assert np.allclose(online,cached,atol=1e-7);parity.append(float(np.max(abs(online-cached))))
  if len(parity)==3:break
 assert len(parity)==3;res.update(passed=True,online_cache_parity_max=max(parity),future_poison_exact=True,unchanged_other19_slots=True,canonical_input='always nearFLU/50; internal model y gain50; do not double scale worker',initial_compensation='first state Linear column5 divided50');(out/'preflight.json').write_text(json.dumps(res,indent=2));print(res)
if __name__=='__main__':main()

