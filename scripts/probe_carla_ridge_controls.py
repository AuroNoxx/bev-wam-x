from train_carla_nav_recovery_factorial512 import *
from navigation_scaled_flow import NavigationScaledFlow
@torch.no_grad()
def main():
 safe_idle();configure_runtime();out=ROOT/'artifacts/carla-frozen-ridge-controls';out.mkdir(exist_ok=False);d=json.loads((ROOT/'artifacts/carla-navscale-paired512/design.json').read_text());train=sorted(set(d['selection']));dev=d['devs'];spec={'alpha_state':.01,'alpha_context':1.0,'preset':'exactly these values before evaluation; no dev tuning','objective':'mean squared signed y at0.6sec + alpha L2 standardized coefficients; intercept unpenalized','train_windows':len(train),'fit_labels':'train only','context':'pool spatial160tokens per current/+1/+2 block, concatenate384 features; limitation: pooling may lose spatial details','state':'canonical current speed and near-goal x/y only, train standardized','no_model_updates':True};(out/'design.json').write_text(json.dumps(spec,indent=2));ckpath=ROOT/'artifacts/carla-navscale-paired512/gain50/latest.ckpt';sha=hashlib.sha256(ckpath.read_bytes()).hexdigest();ck=torch.load(ckpath,map_location='cpu',weights_only=False);m=NavigationScaledFlow(50);m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});del ck;m.cuda().eval().requires_grad_(False);NavData.enabled=True
 def extract(names):
  ds=NavData(names);states=[];features=[];targets=[];groups=[]
  for start in range(0,len(ds),8):
   rows=[ds[i] for i in range(start,min(start+8,len(ds)))];b=batch_to(rows,'cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):c=m.context(b).float().reshape(len(rows),3,160,128).mean(2).flatten(1)
   states.extend(b['state'][:,[0,4,5]].cpu().numpy());features.extend(c.cpu().numpy());targets.extend((b['trajectory'][:,2,1]*10).cpu().numpy())
   for i in range(len(rows)):
    v=float(b['state'][i,0]*30);ny=float(b['state'][i,5]*50);idx=int(Path(names[start+i]).stem);groups.append('static' if v<.2 else 'early_corrective' if idx<=120 and abs(ny)>.01 else 'near_center_moving' if abs(ny)<=.01 else 'other_moving')
  state=np.asarray(states,dtype=np.float64);interaction=np.column_stack([state,state[:,0]*state[:,2]]);return {'state':state,'state_interaction':interaction,'context':np.asarray(features,dtype=np.float64),'y':np.asarray(targets,dtype=np.float64),'groups':groups}
 start=time.perf_counter();fit=extract(train);readouts={}
 for name in ['state','state_interaction','context']:
  x=fit[name];mean=x.mean(0);std=x.std(0).clip(1e-5);z=(x-mean)/std;ym=fit['y'].mean();w=np.linalg.solve(z.T@z/len(z)+(1. if name=='context' else .01)*np.eye(z.shape[1]),z.T@(fit['y']-ym)/len(z));readouts[name]=(mean,std,w,ym)
 result={}
 for split,names in [('fit',train)]+list(dev.items()):
  data=fit if split=='fit' else extract(names);pred={'zero':np.zeros(len(names))}
  for name,(mean,std,w,ym) in readouts.items():pred[name]=(data[name]-mean)/std@w+ym
  if split!='fit':
   ev=json.loads((ROOT/'artifacts/carla-navscale-paired512/gain50-eval.json').read_text())[split];rr={i:[] for i in range(len(names))}
   for row in ev['records']:rr[row['window']].append(row['points'][2][1])
   pred['FM_mean4']=np.array([np.mean(rr[i]) for i in range(len(names))])
  result[split]={}
  for group in ['ALL','static','early_corrective','early_corrective_positive','early_corrective_negative','near_center_moving','other_moving']:
   ix=np.array([group=='ALL' or g==group or (g=='early_corrective' and group=='early_corrective_positive' and int(Path(names[i]).parent.name.replace('episode',''))%2==0) or (g=='early_corrective' and group=='early_corrective_negative' and int(Path(names[i]).parent.name.replace('episode',''))%2==1) for i,g in enumerate(data['groups'])]);y=data['y'][ix]
   if not len(y):continue
   result[split][group]={'n':len(y),'label_abs_mean':float(abs(y).mean()),'models':{}}
   for name,p in pred.items():
    p=p[ix];sig=abs(y)>.005;result[split][group]['models'][name]={'MAE_m':float(abs(p-y).mean()),'RMSE_m':float(np.sqrt(((p-y)**2).mean())),'pred_abs_mean':float(abs(p).mean()),'sign_accuracy_nontrivial':float(np.mean(p[sig]*y[sig]>0)) if sig.any() else None,'sign_n':int(sig.sum())}
 (out/'report.json').write_text(json.dumps({'metrics':result,'wall_s':time.perf_counter()-start,'checkpoint_unchanged':sha==hashlib.sha256(ckpath.read_bytes()).hexdigest()},indent=2));print('DONE',time.perf_counter()-start)
if __name__=='__main__':main()
