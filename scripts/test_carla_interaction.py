from train_carla_nav_recovery_factorial512 import *
from interaction_flow import InteractionFlow
@torch.no_grad()
def main():
 safe_idle();configure_runtime();out=ROOT/'artifacts/carla-interaction-paired512';out.mkdir(exist_ok=False);d=json.loads((ROOT/'artifacts/carla-navscale-paired512/design.json').read_text());names=sorted(set(d['selection']));NavData.enabled=True;ds=NavData(names);assert all(float(ds[i]['state'][19])==0 for i in range(len(ds)));b=batch_to([ds[i] for i in range(0,len(ds),300)],'cuda');pl.seed_everything(2060);a=InteractionFlow(False).cuda().eval();c=InteractionFlow(True).cuda().eval();c.load_state_dict(a.state_dict());c.state[0].weight[:,19]=0;res={}
 for mode,dtype in [('fp32',None),('bf16',torch.bfloat16)]:
  with torch.autocast('cuda',dtype=dtype,enabled=dtype is not None):
   x=a.context(b);y=c.context(b);torch.manual_seed(2700);p=a.sample(b,4,16);torch.manual_seed(2700);q=c.sample(b,4,16)
  res[mode]={'context_max':float((x-y).abs().max()),'output_max':float((p-q).abs().max())};assert torch.equal(x,y) and torch.equal(p,q)
 assert torch.equal(c.context(b),c.context({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)}));s=torch.zeros(1,20,device='cuda');s[0,0]=10/30;s[0,5]=.2/50;assert abs(float(c.encoded_state(s)[0,19])-.2)<1e-6;res.update(passed=True,train_unused_slot19_verified=len(ds),formula='(speed_mps/10)*(near_goal_flu_y_m/1)',future_poison_exact=True);(out/'preflight.json').write_text(json.dumps(res,indent=2));print(res)
if __name__=='__main__':main()
