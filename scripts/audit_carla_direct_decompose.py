from train_carla_nav_recovery_factorial512 import *
from direct_state_velocity import DirectStateVelocity
from normalized_direct_state import NormalizedDirectState

def main():
 safe_idle();configure_runtime();out=ROOT/'artifacts/carla-direct-decompose-frozen';out.mkdir(exist_ok=False);d=json.loads((ROOT/'artifacts/carla-direct-rms-paired512/design.json').read_text());names=d['devs']['recovery'];NavData.enabled=True;ds=NavData(names);result=[]
 for seed in [2060,2061]:
  for normalized in [False,True]:
   path=ROOT/f'artifacts/carla-direct-rms-paired512/{seed}-normalized/latest.ckpt' if normalized else ROOT/f'artifacts/carla-directstate-paired512/{seed}-direct/latest.ckpt';sha=hashlib.sha256(path.read_bytes()).hexdigest();ck=torch.load(path,map_location='cpu',weights_only=False);m=NormalizedDirectState(ck['task_state']['settings']['interaction_rms']) if normalized else DirectStateVelocity();m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});cfg=ck['task_state']['settings']['objective'];del ck;m.cuda().eval();rr=[]
   with torch.no_grad():
    for st in range(0,len(ds),8):
     b=batch_to([ds[i] for i in range(st,min(st+8,len(ds)))],'cuda');zero=torch.zeros(len(b['state']),25,2,device='cuda');t=torch.zeros(len(zero),1,device='cuda')
     with torch.autocast('cuda',dtype=torch.bfloat16):
      back=m.velocity(m.context(b),zero,t).float();branch=m.direct_state(m.direct_features(b['state'])).reshape(-1,25,2).float();total=back+branch
     pp=m.from_flow_space(b,total)*total.new_tensor([50,10,1,1,30]);pb=m.from_flow_space(b,back)*total.new_tensor([50,10,1,1,30]);
     for j in range(len(zero)):
      v=float(b['state'][j,0]*30);end=float(b['trajectory'][j,-1,4]*30);kind='start' if v<.2 and end>2 else 'stopped' if v<.5 and end<.5 else 'deceleration' if v-end>2 else 'moving';rr.append({'group':kind,'endx':float(pp[j,-1,0]),'backbone_endx':float(pb[j,-1,0]),'direct_endx_delta':float(pp[j,-1,0]-pb[j,-1,0]),'direct_vx_rms':float((branch[j,:,0]*10).square().mean().sqrt()),'direct_vy_rms':float((branch[j,:,1]*.5).square().mean().sqrt())})
   summary={g:{'n':sum(x['group']==g for x in rr),**{k:float(np.mean([x[k] for x in rr if x['group']==g])) for k in ['endx','backbone_endx','direct_endx_delta','direct_vx_rms','direct_vy_rms']}} for g in ['start','stopped','deceleration','moving']}
   trainnames=[p for ep in [0,1] for idx in [50,66,80,100] for p in d['selection'] if Path(p).parent.name==f'episode{ep}' and int(Path(p).stem)==idx and 'recovery_cache' in p];trainnames=list(dict.fromkeys(trainnames));td=NavData(trainnames);b=batch_to([td[i] for i in range(len(td))],'cuda');q=m(b,torch.zeros(len(td),25,2,device='cuda'),torch.zeros(len(td),1,device='cuda'));target=m.to_flow_space(b,b['trajectory']);err=q-target;phys=m.from_flow_space(b,q);e=(phys[:,:,:2]-b['trajectory'][:,:,:2])*q.new_tensor([50,10]);terms={'action_x':.5*err[:,:,0].square().mean(),'action_y':.5*err[:,:,1].square().mean(),'position_expected_weight':.2*F.smooth_l1_loss(e/5,torch.zeros_like(e)),'world':.1*sum(m.auxiliary_loss(b))};params=[(n,p) for n,p in m.named_parameters() if p.requires_grad];gr={};flat={}
   for k,loss in terms.items():
    grads=torch.autograd.grad(loss,[p for n,p in params],retain_graph=True,allow_unused=True);branchg=next(g for (n,p),g in zip(params,grads) if n=='direct_state.weight');backg=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for (n,p),g in zip(params,grads) if n!='direct_state.weight']);flat[k]=backg;gr[k]={'backbone_norm':float(backg.norm()),'branch_x_norm':float(branchg.reshape(25,2,4)[:,0].norm()) if branchg is not None else 0.,'branch_y_norm':float(branchg.reshape(25,2,4)[:,1].norm()) if branchg is not None else 0.}
   w=m.direct_state.weight.detach().reshape(25,2,4);gr['action_xy_backbone_cosine']=float((flat['action_x']*flat['action_y']).sum()/(flat['action_x'].norm()*flat['action_y'].norm()).clamp_min(1e-12));result.append({'seed':seed,'normalized':normalized,'summary':summary,'gradients':gr,'gradient_train_n':len(td),'interaction_coeff_x_norm':float(w[:,0,3].norm()),'interaction_coeff_y_norm':float(w[:,1,3].norm()),'checkpoint_unchanged':sha==hashlib.sha256(path.read_bytes()).hexdigest()});del m;gc.collect();torch.cuda.empty_cache()
 (out/'report.json').write_text(json.dumps(result,indent=2));print('DONE')
if __name__=='__main__':main()
