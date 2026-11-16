from continue_residual_pair512_v2 import *
def main():
 idle();configure_runtime();src=ROOT/'artifacts/residual-representation-resume512-v2';base=ROOT/'artifacts/residual-representation-paired128';out=ROOT/'artifacts/residual512-weighted-gradient-audit';out.mkdir(exist_ok=False);cfg=json.loads((ROOT/'artifacts/fresh-jepa-paired-first1024/config.json').read_text());sel=json.loads((base/'selection.json').read_text());scales=json.loads((base/'scale_calibration.json').read_text())['scales_m'];b=batch_to([DiskSamples(sel['train'])[i] for i in range(4)],'cuda');results=[]
 for seed in [1980,1981]:
  ck=torch.load(src/f'{seed}-residual/step512.ckpt',map_location='cpu',weights_only=False);m=ResidualJEPA(768,15,12);m.set_scales(scales);m.load_state_dict({k.removeprefix('model.'):v for k,v in ck['state_dict'].items()},strict=True);m.cuda().eval();params=[p for p in m.parameters() if p.requires_grad]
  with torch.autocast('cuda',dtype=torch.bfloat16):
   torch.manual_seed(1990);total,terms=consistent_flow_loss(m,b,cfg['objective']);aux=m.auxiliary_loss(b);groups={'fm':terms['fm']*cfg['objective']['weights']['fm'],'physical':sum(v*cfg['objective']['weights'][k] for k,v in terms.items() if k!='fm'),'jepa':sum(.1*v for v in aux)}
  vectors={};values={}
  for name,v in groups.items():
   gs=torch.autograd.grad(v,params,retain_graph=True,allow_unused=True);vectors[name]=torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten() for p,g in zip(params,gs)]);values[name]=float(v.detach());del gs
  combined=sum(vectors.values());norm=float(combined.norm());norms={k:float(v.norm()) for k,v in vectors.items()};cos={a+'__'+z:float(torch.dot(vectors[a],vectors[z])/(vectors[a].norm()*vectors[z].norm()).clamp_min(1e-20)) for a,z in [('fm','physical'),('fm','jepa'),('physical','jepa')]};gcpu=combined.cpu();opt=ck['optimizer_states'][0];pg=opt['param_groups'][0];assert len(pg['params'])==len(params);lr=pg['lr'];b1,b2=pg['betas'];eps=pg['eps'];offset=0;u1sq=u5sq=dot=0.;weightnorm=0.
  # Compute hypothetical NEXT Adam updates without applying them. Same saved moments.
  for p,pid in zip(params,pg['params']):
   n=p.numel();g=gcpu[offset:offset+n].reshape(p.shape);offset+=n;s=opt['state'].get(pid);assert s is not None;step=int(s['step'])+1;u=[]
   for limit in [1.,5.]:
    gg=g*min(1.,limit/(norm+1e-6));mm=s['exp_avg']*b1+gg*(1-b1);vv=s['exp_avg_sq']*b2+gg.square()*(1-b2);update=-lr*(mm/(1-b1**step))/(torch.sqrt(vv/(1-b2**step))+eps);u.append(update)
   u1sq+=float(u[0].square().sum());u5sq+=float(u[1].square().sum());dot+=float((u[0]*u[1]).sum());weightnorm+=float(p.detach().float().square().sum())
  r={'seed':seed,'weighted_losses':values,'gradient_norms':norms,'pairwise_cosines':cos,'combined_norm':norm,'clip_coefficient_at1':min(1,1/(norm+1e-6)),'hypothetical_adam_without_weight_decay':{'update_norm_clip1':u1sq**.5,'update_norm_clip5':u5sq**.5,'norm_ratio':(u5sq/u1sq)**.5,'direction_cosine':dot/(u1sq*u5sq)**.5,'clip1_update_to_parameter_norm':(u1sq/weightnorm)**.5,'lr':lr},'note':'same saved Adam moments; no optimizer step; excludes decoupled decay which is equal acrossvariants; one fixedbatch4,not originaltrainingbatch'};results.append(r);print(json.dumps(r),flush=True);del m,ck,params,total,terms,aux,groups,vectors,combined,gcpu,opt;gc.collect();torch.cuda.empty_cache()
 (out/'report.json').write_text(json.dumps({'results':results,'limits':'2frozenresidual512 models,4same trainwindows1noise; localnextupdate illustration cannot establishtraining effect; no parameters updated/test/weight overwrite'},indent=2));print('DONE',flush=True)
if __name__=='__main__':main()
