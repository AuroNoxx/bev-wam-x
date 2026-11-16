from train_carla_nav_recovery_factorial512 import *
from navigation_scaled_flow import NavigationScaledFlow
from audit_general_trajectory_frozen import sample_context
@torch.no_grad()
def main():
 safe_idle();configure_runtime();out=ROOT/'artifacts/carla-navscale-response-frozen';out.mkdir(exist_ok=False);rec=json.loads((ROOT/'data/carla_recovery_combined_v2/manifest.json').read_text());names=[n for n in rec['dev'] if int(Path(n).stem) in [50,80,90,100,110,330]];NavData.enabled=True;ds=NavData(names);b=batch_to([ds[i] for i in range(len(ds))],'cuda');torch.manual_seed(2500);noise=torch.randn(len(ds)*4,25,5,device='cuda')[...,:2];result=[]
 for arm,path in [('gain1',ROOT/'artifacts/carla-navscale-paired512/gain1/latest.ckpt'),('gain50',ROOT/'artifacts/carla-navscale-paired512/gain50/latest.ckpt')]:
  ck=torch.load(path,map_location='cpu',weights_only=False);m=NavigationScaledFlow(1 if arm=='gain1' else 50);m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});m.cuda().eval();del ck
  with torch.autocast('cuda',dtype=torch.bfloat16):
   for mode in ['actual','flip_y','zero_nav']:
    z={**b,'state':b['state'].clone()}
    if mode=='flip_y':z['state'][:,5]*=-1
    if mode=='zero_nav':z['state'][:,4:6]=0
    q,p=sample_context(m,z,noise,16,'none');p=p*p.new_tensor([50,10,1,1,30]);truth=b['trajectory']*p.new_tensor([50,10,1,1,30])
    for i,n in enumerate(names):result.append({'arm':arm,'mode':mode,'file':n,'v0':float(b['state'][i,0]*30),'nav_y_m':float(b['state'][i,5]*50),'label_y06':float(truth[i,2,1]),'label_y5':float(truth[i,-1,1]),'pred_y06':p[i,:,2,1].cpu().tolist(),'pred_y5':p[i,:,-1,1].cpu().tolist(),'pred_x5':p[i,:,-1,0].cpu().tolist()})
  del m;gc.collect();torch.cuda.empty_cache()
 (out/'report.json').write_text(json.dumps(result,indent=2));print('DONE')
if __name__=='__main__':main()

