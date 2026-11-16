from train_carla_nav_recovery_factorial512 import *
from navigation_scaled_flow import NavigationScaledFlow
from audit_general_trajectory_frozen import sample_context
from carla_nav_contract import cache_near

def main():
 safe_idle();configure_runtime();out=ROOT/'artifacts/carla-identifiability-frozen';out.mkdir(exist_ok=False);rec=json.loads((ROOT/'data/carla_recovery_combined_v2/manifest.json').read_text());pairs=[];stats=[]
 for split in ['train','dev']:
  roots={Path(p).parent.name:Path(p).parent for p in rec[split]}
  for ep in range(0,len(roots),2):
   for idx in range(36,350):
    paths=[str(roots['episode'+str(ep+j)]/f'{idx:05d}.npz') for j in [0,1]]
    with np.load(paths[0]) as aa,np.load(paths[1]) as bb:
     nav=[cache_near(p)*50 for p in paths];traj=[z['trajectory']*np.array([50,10,1,1,30]) for z in [aa,bb]];speed=[float(z['state'][0])*30 for z in [aa,bb]];h0=aa['history'];h1=bb['history'];union=(h0!=255)|(h1!=255);diff=float(((h0!=h1)&union).sum()/max(1,union.sum()));row={'split':split,'pair':ep,'idx':idx,'speed':speed,'nav_y':[float(x[1]) for x in nav],'y06':[float(x[2,1]) for x in traj],'y5':[float(x[-1,1]) for x in traj],'future_speed':[float(x[-1,4]) for x in traj],'occ_union_difference_fraction':diff};stats.append(row)
    if idx in [50,66,80,90,100,110,330] and ep==0:pairs.append((row,paths))
 (out/'data.json').write_text(json.dumps(stats));NavData.enabled=True;ckpath=ROOT/'artifacts/carla-navscale-paired512/gain50/latest.ckpt';sha=hashlib.sha256(ckpath.read_bytes()).hexdigest();ck=torch.load(ckpath,map_location='cpu',weights_only=False);m=NavigationScaledFlow(50);m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});m.cuda().eval().requires_grad_(False);del ck;rows=[]
 for meta,paths in pairs:
  ds=NavData(paths);b=batch_to([ds[0],ds[1]],'cuda');torch.manual_seed(2600);noise=torch.randn(4,25,5,device='cuda')[...,:2].repeat(2,1,1);outputs={}
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   for mode in ['actual','nav_swap','occ_swap','both_swap']:
    z={**b,'state':b['state'].clone()}
    if mode in ['nav_swap','both_swap']:z['state'][:,4:6]=b['state'].flip(0)[:,4:6]
    if mode in ['occ_swap','both_swap']:
     for key in ['history','geometry_valid']:z[key]=b[key].flip(0)
    _,p=sample_context(m,z,noise,16,'none');outputs[mode]={'y06':(p[:,:,2,1]*10).mean(1).cpu().tolist(),'y5':(p[:,:,-1,1]*10).mean(1).cpu().tolist(),'x5':(p[:,:,-1,0]*50).mean(1).cpu().tolist()}
  state=b['state'].detach().clone().requires_grad_(True);z={**b,'state':state};single=noise[[0,4]].float();pred=m(z,single,torch.zeros(2,1,device='cuda')).float();grads=[]
  for i in range(2):
   value=pred[i,:3,1].mean()*.5;g=torch.autograd.grad(value,state,retain_graph=i==0)[0][i];grads.append({'single_flow_vy_derivative_per_nav_meter':float(g[5]/50),'single_flow_vy_derivative_per_speed_mps':float(g[0]/30),'single_flow_vy':float(value)})
  rows.append({'meta':meta,'outputs':outputs,'single_step_fp32_gradients':grads})
 assert sha==hashlib.sha256(ckpath.read_bytes()).hexdigest();(out/'model.json').write_text(json.dumps(rows,indent=2));print('DONE')
if __name__=='__main__':main()
