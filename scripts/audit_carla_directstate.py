from train_carla_nav_recovery_factorial512 import *
from direct_state_velocity import DirectStateVelocity
from deterministic_velocity_model import DeterministicVelocityFlow
@torch.no_grad()
def main():
 safe_idle();configure_runtime();out=ROOT/'artifacts/carla-directstate-perturb-frozen';out.mkdir(exist_ok=False);rec=json.loads((ROOT/'data/carla_recovery_combined_v2/manifest.json').read_text());names=[p for ep in [0,1] for idx in [50,80,100,200,330] for p in rec['dev'] if Path(p).parent.name==f'episode{ep}' and int(Path(p).stem)==idx];NavData.enabled=True;ds=NavData(names);b=batch_to([ds[i] for i in range(len(ds))],'cuda');rows=[]
 for seed in [2060,2061]:
  for direct in [False,True]:
   path=ROOT/f'artifacts/carla-directstate-paired512/{seed}-direct/latest.ckpt' if direct else ROOT/f'artifacts/carla-deterministic25-paired512/{seed}-regression/latest.ckpt';ck=torch.load(path,map_location='cpu',weights_only=False);m=DirectStateVelocity() if direct else DeterministicVelocityFlow();m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});del ck;m.cuda().eval();outputs={}
   with torch.autocast('cuda',dtype=torch.bfloat16):
    for mode in ['actual','flip_nav','swap_occ','unknown_occ']:
     z={**b,'state':b['state'].clone()}
     if mode=='flip_nav':z['state'][:,5]*=-1
     if mode=='swap_occ':
      for key in ['history','geometry_valid']:z[key]=b[key].roll(5,0)
     if mode=='unknown_occ':z['history']=torch.full_like(b['history'],255);z['geometry_valid']=torch.zeros_like(b['geometry_valid'])
     p=m.sample(z,1,16)[:,0].float()*b['state'].new_tensor([50,10,1,1,30]);outputs[mode]=p
   rows.append({'seed':seed,'direct':direct,'effects':{mode:{'xy_delta_rms_m':float((p[:,:,:2]-outputs['actual'][:,:,:2]).square().mean().sqrt()),'y06':p[:,2,1].cpu().tolist(),'x5':p[:,-1,0].cpu().tolist()} for mode,p in outputs.items()}});del m;gc.collect();torch.cuda.empty_cache()
 (out/'report.json').write_text(json.dumps({'names':names,'rows':rows,'limits':'10 dev states, OODoccswap/unknown sensitivity only, not evidence obstacle response/safety'},indent=2));print('DONE')
if __name__=='__main__':main()
