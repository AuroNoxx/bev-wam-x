from train_carla_nav_recovery_factorial512 import *
from interaction_flow import InteractionFlow
from deterministic_velocity_model import DeterministicVelocityFlow
@torch.no_grad()
def main():
 safe_idle();configure_runtime();out=ROOT/'artifacts/carla-deterministic25-paired512';out.mkdir(exist_ok=False);res={}
 for seed in [2060,2061]:
  pl.seed_everything(seed);a=InteractionFlow(True);pl.seed_everything(seed);b=DeterministicVelocityFlow();assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items());res[str(seed)]={'initial_parameters_identical':True,'parameter_count':sum(p.numel() for p in b.parameters())}
 NavData.enabled=True;d=json.loads((ROOT/'artifacts/carla-lateral-scale512/design.json').read_text());ds=NavData(d['selection'][:2]);batch=batch_to([ds[0],ds[1]],'cuda');b.cuda().eval();p=b.sample(batch,4,16);assert torch.isfinite(p).all() and torch.equal(p[:,0],p[:,1]);q=b.to_flow_space(batch,batch['trajectory']);decoded=b.from_flow_space(batch,q);err=float((decoded[:,:,:2]-batch['trajectory'][:,:,:2]).abs().max());assert torch.equal(b.context(batch),b.context({**batch,'future':torch.full_like(batch['future'],255),'trajectory':torch.full_like(batch['trajectory'],999)}));res.update(passed=True,deterministic_candidates_equal=True,decode_label_xy_difference_normalized=err,decode_note='original label position and trapezoidal velocity integration need not exactly coincide',actor_future_poison_exact=True);(out/'preflight.json').write_text(json.dumps(res,indent=2));print(res)
if __name__=='__main__':main()
