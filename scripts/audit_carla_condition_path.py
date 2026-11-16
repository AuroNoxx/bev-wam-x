from train_carla_nav_recovery_factorial512 import *
from interaction_flow import InteractionFlow
from deterministic_velocity_model import DeterministicVelocityFlow

def main():
 safe_idle();configure_runtime();out=ROOT/'artifacts/carla-condition-path-frozen';out.mkdir(exist_ok=True);rec=json.loads((ROOT/'data/carla_recovery_combined_v2/manifest.json').read_text());names=[p for ep in [0,1] for idx in [50,66,80,100] for p in rec['train'] if Path(p).parent.name==f'episode{ep}' and int(Path(p).stem)==idx];NavData.enabled=True;ds=NavData(names);b=batch_to([ds[i] for i in range(len(ds))],'cuda');swap=torch.arange(8,device='cuda').roll(4);rows=[]
 for arm,path in [('regression',ROOT/'artifacts/carla-deterministic25-paired512/2060-regression/latest.ckpt'),('fm',ROOT/'artifacts/carla-interaction-paired512/2060-interaction/latest.ckpt')]:
  sha=hashlib.sha256(path.read_bytes()).hexdigest();ck=torch.load(path,map_location='cpu',weights_only=False);m=DeterministicVelocityFlow() if arm=='regression' else InteractionFlow(True);m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});del ck;m.cuda().eval().requires_grad_(False);original_encoded=m.encoded_state(b['state']);m.encoded_state=lambda s:torch.cat([s[:,:5],s[:,5:6]*50,s[:,6:19],(s[:,0:1]*3)*(s[:,5:6]*50)],1);assert torch.equal(original_encoded,m.encoded_state(b['state']));state=b['state'].clone().requires_grad_(True);bb={**b,'state':state};emb=[];hook=m.state.register_forward_hook(lambda mod,args,output:emb.append(output));context=m.context(bb);hook.remove();t=torch.zeros(8,1,device='cuda');torch.manual_seed(2900);xt=torch.zeros(8,25,2,device='cuda') if arm=='regression' else torch.randn(4,25,2,device='cuda').repeat(2,1,1);saved=[];hooks=[]
  for layer in m.transformer.layers:hooks.append(layer.multihead_attn.register_forward_pre_hook(lambda mod,args:saved.append((mod,tuple(a.detach() for a in args[:3])))))
  pred=m.velocity(context,xt,t);[h.remove() for h in hooks];scalar=pred[:,:3,1].mean()*.5;gs,ge,gc=torch.autograd.grad(scalar,[state,emb[0],context]);grad={'state_cols':[float(gs[:,i].norm()) for i in [0,5,19]],'state_embedding_norm':float(ge.norm()),'context_blocks':[float(gc[:,j*160:(j+1)*160].norm()) for j in range(3)]};att=[]
  with torch.no_grad():
   for mod,args in saved:
    _,w=mod(*args,need_weights=True,average_attn_weights=False);att.append({'mass_by_current_future1_future2':[float(w[:,:,:,j*160:(j+1)*160].sum(-1).mean()) for j in range(3)],'entropy_over_log480':float((-(w*w.clamp_min(1e-10).log()).sum(-1)).mean()/np.log(480))})
   z={**b,'state':b['state'].clone()};z['state'][:,4:6]=b['state'][swap,4:6];se=m.state(m.encoded_state(b['state']));se2=m.state(m.encoded_state(z['state']));ctx2=m.context(z);q2=m.velocity(ctx2,xt,t);effects={}
   for mode,c in [('actual',context),('zero_context',torch.zeros_like(context)),('spatial_mean_per_block',context.reshape(8,3,160,128).mean(2,keepdim=True).expand(-1,-1,160,-1).reshape_as(context))]:
    q=m.velocity(c,xt,t);effects[mode]={'q_diff_rms_from_actual':float((q-pred).square().mean().sqrt()),'qy_first3':q[:,:3,1].mean(1).cpu().tolist()}
   rows.append({'arm':arm,'norms':{'state_emb_rms':float(se.square().mean().sqrt()),'context_rms':float(context.square().mean().sqrt()),'nav_swap_emb_rms_delta':float((se2-se).square().mean().sqrt()),'nav_swap_context_rms_delta':float((ctx2-context).square().mean().sqrt()),'nav_swap_field_q_rms_delta':float((q2-pred).square().mean().sqrt())},'gradient':grad,'attention':att,'effects':effects,'checkpoint_unchanged':sha==hashlib.sha256(path.read_bytes()).hexdigest()})
  del m;gc.collect() if False else None;torch.cuda.empty_cache()
 (out/'report.json').write_text(json.dumps({'names':names,'rows':rows,'limits':'8 train states, FP32 eval frozen; t0 fixed-noise FM field vs deterministic head, not fullFMrollout; synthetic zero/meancontext is OOD ablation; attentionweights not causal attribution.'},indent=2));print('DONE')
if __name__=='__main__':main()


