from diagnose_frozen_rollout import *
idle();configure_runtime();src=ROOT/'artifacts/masked-teacher-155m-continue2048';out=ROOT/'artifacts/frozen-rollout-convergence-gradient';cfg=json.loads((ROOT/'artifacts/teacher-rollout-dynamics-paired/config.json').read_text());sel=json.loads((ROOT/'artifacts/teacher-rollout-dynamics-paired/selection.json').read_text());ck=torch.load(src/'best.ckpt',map_location='cpu',weights_only=False);task=Task(cfg,ck['task_state']['manifest_sha'],ck['task_state']['settings']['selection_sha']);task.load_state_dict(ck['state_dict']);del ck;m=task.model.cuda().train();params=[p for p in m.parameters() if p.requires_grad];rows=[]
def dot(a,b):return sum(float((x.float()*y.float()).sum()) for x,y in zip(a,b) if x is not None and y is not None)
for seed in [1810,1811]:
 b={k:v.cuda() for k,v in next(iter(SeedData(sel,seed).loader('train'))).items()};small={k:v[:2] for k,v in b.items()}
 with torch.autocast('cuda',dtype=torch.bfloat16):
  torch.manual_seed(seed*100000);_,terms=consistent_flow_loss(m,b,cfg['objective']);aux=m.auxiliary_loss(b);losses={k:v*cfg['objective']['weights'][k] for k,v in terms.items()};losses.update({k:v*cfg['aux_weights'][k] for k,v in zip(['latent','variance','semantic'],aux)});torch.manual_seed(seed*100000+50000);p=rollout(m,small,4);_,rt=dynamics_loss(p,small['state']);losses.update({k:.01*v for k,v in rt.items()});total=sum(losses.values())
 gt=torch.autograd.grad(total,params,retain_graph=True,allow_unused=True);gf=torch.autograd.grad(losses['fm'],params,retain_graph=True,allow_unused=True);nt=dot(gt,gt)**.5;nf=dot(gf,gf)**.5;components={}
 for name,value in losses.items():
  g=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True);n=dot(g,g)**.5;components[name]={'weighted_loss':float(value.detach()),'weighted_grad_norm':n,'projection_onto_total':dot(g,gt)/nt,'cosine_to_total':dot(g,gt)/max(n*nt,1e-20),'cosine_to_fm':dot(g,gf)/max(n*nf,1e-20)};del g
 rows.append({'seed':seed,'combined_gradient_norm':nt,'clip_coefficient':min(1,1/(nt+1e-6)),'components':components});print(json.dumps(rows[-1]),flush=True);del gt,gf,total,losses,terms,aux,p,rt;gc.collect();torch.cuda.empty_cache()
(out/'detailed-gradient.json').write_text(json.dumps(rows,indent=2));print('DONE')
