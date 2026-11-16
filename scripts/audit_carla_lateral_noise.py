from train_mixed_regression_flow512 import *
from carla_miniloop_adapter import actor_inputs
from train_carla_miniloop_flow512 import safe_idle
@torch.no_grad()
def main():
 safe_idle();configure_runtime();base=ROOT/'artifacts/carla-startstop-mixture-decel25';out=ROOT/'artifacts/carla-lateral-noise-frozen';out.mkdir(exist_ok=False);rows=json.loads((base/'frames.json').read_text());src=ROOT/'artifacts/carla-startstop-mixture512/latest.ckpt';sha=hashlib.sha256(src.read_bytes()).hexdigest();ck=torch.load(src,map_location='cpu',weights_only=False);m=CompareModel('fm');m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});del ck;m.cuda().eval();results=[]
 for i in [36,66,86]:
  ids=[i-6,i-4,i-2,i];raw=[]
  for j in ids:
   with np.load(base/'live'/f'{j:05d}.npz') as z:raw.append({k:z[k].copy() for k in z.files})
  a=actor_inputs(raw,[rows[j] for j in ids]);a['geometry_valid']=a['history']!=255;b={k:torch.from_numpy(v)[None].cuda() for k,v in a.items()};nav=np.linalg.inv(np.array(rows[i]['ego_world']))@np.r_[rows[i]['navigation_world'],1]
  with torch.autocast('cuda',dtype=torch.bfloat16):
   c=m.context(b)
   for seed in [2301]+list(range(2400,2416))+['zero']:
    if seed=='zero':x=torch.zeros(1,25,2,device='cuda')
    else:torch.manual_seed(seed);x=torch.randn(1,25,5,device='cuda')[...,:2] # same sample() noise shape
    for k in range(16):x=x+m.velocity(c,x,torch.full((1,1),k/16,device='cuda'))/16
    p=(m.from_flow_space(b,x)[0].float()*x.new_tensor([50,10,1,1,30])).cpu().numpy();speed=float(a['state'][0])*30;ix=int(np.argmin(abs(np.linalg.norm(p[:,:2],axis=1)-max(4,speed*.8))));results.append({'input_index':i,'sim_elapsed':(i-36)*.1,'seed':seed,'v0':speed,'navigation_local_flu':[float(nav[0]),float(-nav[1])],'state_navigation_nonzero':int(np.count_nonzero(a['state'][4:])),'y02':float(p[0,1]),'y06':float(p[2,1]),'y5':float(p[-1,1]),'look_y':float(p[ix,1]),'look_x':float(p[ix,0]),'forward5':float(p[-1,0]),'velocity_y02':float(p[0,4]*p[0,2])})
 assert sha==hashlib.sha256(src.read_bytes()).hexdigest();r={'rows':results,'checkpoint_unchanged':True,'limits':'3recordedcausalinputs,16declaredseeds+production2301+zeroOODdiagnostic. Nooptim/control. Navigationmeasurednotinjected. Recordedinputdriftdependsontheoriginalclosedloop.'};(out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
if __name__=='__main__':main()

