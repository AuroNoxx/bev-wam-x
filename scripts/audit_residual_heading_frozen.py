from probe_residual_representation import *

def main():
 idle();configure_runtime();src=ROOT/'artifacts/residual-representation-paired128';out=ROOT/'artifacts/residual-heading-frozen-audit';out.mkdir(exist_ok=False);cfg=json.loads((ROOT/'artifacts/fresh-jepa-paired-first1024/config.json').read_text());sel=json.loads((src/'selection.json').read_text());scales=json.loads((src/'scale_calibration.json').read_text())['scales_m'];b=batch_to([DiskSamples(sel['train'])[i] for i in range(8)],'cuda');results=[]
 for seed in [1980,1981]:
  pl.seed_everything(seed);base=MaskedHistoryJoint(768,15,12);initial={k:v.detach().clone() for k,v in base.state_dict().items()};del base
  for arm,stage in [('residual','initial'),('residual','trained128'),('original','trained128')]:
   m=ResidualJEPA(768,15,12) if arm=='residual' else MaskedHistoryJoint(768,15,12)
   if arm=='residual':m.set_scales(scales)
   if stage=='initial':m.load_state_dict(initial,strict=True)
   else:
    ck=torch.load(src/f'{seed}-{arm}/step128.ckpt',map_location='cpu',weights_only=False);m.load_state_dict({k.removeprefix('model.'):v for k,v in ck['state_dict'].items()},strict=True);del ck
   m.cuda().eval();raw=[]
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    context=m.context(b);torch.manual_seed(1970);x=torch.randn(8,25,5,device='cuda')
    for i in range(17):
     if i in [0,1,4,8,12,16]:raw.append({'flow_t':i/16,'raw_sin_mean':float(x[...,2].mean()),'raw_cos_mean':float(x[...,3].mean()),'raw_cos_negative_fraction':float((x[...,3]<0).float().mean()),'raw_heading_norm_mean':float(x[...,2:4].norm(dim=-1).mean())})
     if i<16:x=x+m.velocity(context,x,torch.full((8,1),i/16,device='cuda'))/16
   with torch.autocast('cuda',dtype=torch.bfloat16):
    torch.manual_seed(seed*100000+127);total,terms=consistent_flow_loss(m,b,cfg['objective']);aux=m.auxiliary_loss(b)
   head=list(m.output.parameters());grads={};combined=[torch.zeros_like(p,dtype=torch.float32) for p in head]
   for name,term in terms.items():
    weight=cfg['objective']['weights'][name];g=torch.autograd.grad(term*weight,head,retain_graph=True,allow_unused=True);g=[torch.zeros_like(p) if v is None else v for p,v in zip(head,g)];norm=float(sum(v.float().square().sum() for v in g).sqrt());combined=[a+v.float() for a,v in zip(combined,g)];grads[name]={'raw_loss':float(term.detach()),'weight':weight,'weighted_loss':float(term.detach())*weight,'output_head_weighted_grad_norm':norm,'cos_output_bias_gradient':float(g[1][3]),'sin_output_bias_gradient':float(g[1][2])}
   for name,v in zip(['latent','variance','semantic'],aux):grads[name]={'raw_loss':float(v.detach()),'weight':.1,'weighted_loss':float(v.detach())*.1,'output_head_weighted_grad_norm':0.,'reason':'auxiliary graph does not use trajectory output head'}
   results.append({'seed':seed,'arm':arm,'stage':stage,'raw_rollout':raw,'loss_and_head_gradients':grads,'combined_output_bias_gradient':combined[1].tolist(),'note':'output-head gradients only; not full-model gradient dominance or Adam update attribution; fixed8train batch, not original last batch'});del m,context,x,total,terms,aux,g,head,combined;gc.collect();torch.cuda.empty_cache()
  del initial;gc.collect()
 target=b['trajectory'];report={'target_heading':{'cos_min':float(target[...,3].min()),'cos_mean':float(target[...,3].mean()),'sin_min':float(target[...,2].min()),'sin_max':float(target[...,2].max())},'results':results,'limits':'frozen initial/replayed128 models; no updates/test;8trainwindows,one noise; partial gradients explicitly output head'};(out/'report.json').write_text(json.dumps(report,indent=2));print('DONE',flush=True)
if __name__=='__main__':main()
