from train_carla_nav_recovery_factorial512 import *
from direct_state_velocity import DirectStateVelocity
@torch.no_grad()
def main():
 safe_idle();configure_runtime();out=ROOT/'artifacts/carla-directstate-amplitude-frozen';out.mkdir(exist_ok=False);d=json.loads((ROOT/'artifacts/carla-directstate-paired512/design.json').read_text());seq=d['selection'];names=sorted(set(seq));NavData.enabled=True;ds=NavData(names);rows=[]
 for seed in [2060,2061]:
  path=ROOT/f'artifacts/carla-directstate-paired512/{seed}-direct/latest.ckpt';sha=hashlib.sha256(path.read_bytes()).hexdigest();ck=torch.load(path,map_location='cpu',weights_only=False);m=DirectStateVelocity();m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});del ck;m.cuda().eval();xx=[];yy=[];bb=[];gg=[]
  for st in range(0,len(ds),8):
   b=batch_to([ds[i] for i in range(st,min(st+8,len(ds)))],'cuda');x=m.direct_features(b['state']);target=m.to_flow_space(b,b['trajectory']);zero=torch.zeros(len(x),25,2,device='cuda');t=torch.zeros(len(x),1,device='cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):backbone=m.velocity(m.context(b),zero,t).float()
   xx.extend(x.cpu().numpy());yy.extend(target.flatten(1).cpu().numpy());bb.extend(backbone.flatten(1).cpu().numpy())
   for i in range(len(x)):
    v=float(b['state'][i,0]*30);ny=float(b['state'][i,5]*50);idx=int(Path(names[st+i]).stem);gg.append('static' if v<.2 else 'early_corrective' if idx<=120 and abs(ny)>.01 else 'near_center_moving' if abs(ny)<=.01 else 'other_moving')
  x=np.asarray(xx,np.float64);target=np.asarray(yy,np.float64);back=np.asarray(bb,np.float64);w=m.direct_state.weight.detach().cpu().numpy().T.astype(np.float64);residual=target-back;alpha=1e-6;ref=np.linalg.solve(x.T@x/len(x)+alpha*np.eye(4),x.T@residual/len(x));pred=back+x@w;refpred=back+x@ref;grads={};counts={}
  for group in ['ALL','static','early_corrective','near_center_moving']:
   ix=np.array([group=='ALL' or g==group for g in gg]);er=pred[ix]-target[ix];gradient=(2/(len(x)*50))*(x[ix].T@er);counts[group]={'unique':int(ix.sum()),'draws':sum(1 for p in seq if p in {names[j] for j in np.where(ix)[0]})};grads[group]={'weighted_action_gradient_norm':float(np.linalg.norm(gradient)),'first3_lateral_columns_norm':float(np.linalg.norm(gradient[:,[1,3,5]])),'actual_q_rmse':float(np.sqrt((er**2).mean())),'reference_q_rmse':float(np.sqrt(((refpred[ix]-target[ix])**2).mean())),'learned_direct_physical_y_first3_rms':float(np.sqrt(((x[ix]@w)[:,[1,3,5]]*.5)**2 .mean())) if False else float(np.sqrt(np.mean(((x[ix]@w)[:,[1,3,5]]*.5)**2))),'reference_direct_physical_y_first3_rms':float(np.sqrt(np.mean(((x[ix]@ref)[:,[1,3,5]]*.5)**2)))}
  rows.append({'seed':seed,'counts':counts,'feature_rms':np.sqrt((x*x).mean(0)).tolist(),'learned_first3_lateral_coeff':w[:,[1,3,5]].T.tolist(),'reference_first3_lateral_coeff':ref[:,[1,3,5]].T.tolist(),'learned_max_abs':float(abs(w).max()),'reference_max_abs':float(abs(ref).max()),'gradient_groups':grads,'all_target_vy_abs_gt002_fraction':float(np.mean(abs(target.reshape(-1,25,2)[:,:,1]*.5)>.02)),'reference_regularization':alpha,'gram_condition_number':float(np.linalg.cond(x.T@x/len(x)+alpha*np.eye(4))),'sha_unchanged':sha==hashlib.sha256(path.read_bytes()).hexdigest()})
 (out/'report.json').write_text(json.dumps({'rows':rows,'limits':'reference is train-only closed-form residual qMSE with fixed backbone,not full objective optimum/not deployed/not dev tuned; actualgradient exact linear action contribution aggregated unique windows not drawweighted; position andworld contributions not included'},indent=2));print('DONE')
if __name__=='__main__':main()
