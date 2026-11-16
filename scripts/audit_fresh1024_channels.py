from audit_fresh1024_sampling_path import *

def main():
 idle();configure_runtime();src=ROOT/'artifacts/fresh-jepa-paired-first1024';out=ROOT/'artifacts/fresh1024-channel-audit';out.mkdir(exist_ok=False);cfg=json.loads((src/'config.json').read_text());sel=json.loads((src/'selection.json').read_text());names=[sel['validation'][i] for i in np.linspace(0,26,8,dtype=int)];b=batch_to([DiskSamples(names)[i] for i in range(8)],'cuda');target=b['trajectory'].float();results=[]
 for seed,mode in [(1960,'jepa'),(1960,'no_jepa'),(1961,'jepa'),(1961,'no_jepa')]:
  task=FreshTask(cfg,'fresh-no-checkpoint','diagnostic',seed,mode);ck=torch.load(src/f'{seed}-{mode}/step1024.ckpt',map_location='cpu',weights_only=False);task.load_state_dict(ck['state_dict'],strict=True);del ck;m=task.model.cuda().eval();m.requires_grad_(False);rows=[]
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   context=m.context(b);torch.manual_seed(1970);noise=torch.randn_like(target)
   for tval in [.25,.5,.75,.95]:
    t=torch.full((len(target),1),tval,device='cuda');xt=(1-tval)*noise+tval*target;pred=m.velocity(context,xt,t).float();error=pred-(target-noise);endpoint=xt+(1-tval)*pred;ep=endpoint-target;phys=ep*ep.new_tensor([50,10,1,1,30]);pdelta=torch.cat([phys[:,:1,:2],phys[:,1:,:2]-phys[:,:-1,:2]],1)/.2
    rows.append({'t':tval,'fm_channel_mse':error.square().mean((0,1)).tolist(),'fm_time_channel_mse':error.square().mean(0).tolist(),'endpoint_physical_rmse':phys.square().mean((0,1)).sqrt().tolist(),'first_endpoint_physical_rmse':phys[:,0].square().mean(0).sqrt().tolist(),'position_error_derivative_rmse_xy_mps':pdelta.square().mean((0,1)).sqrt().tolist(),'position_error_derivative_time_rmse':pdelta.square().mean(0).sqrt().tolist(),'endpoint_identity_max_abs':float((ep-(1-tval)*error).abs().max())})
   # Same noise as public sample(k=1), independent reproducibility check.
   x=noise.clone()
   for i in range(16):x=x+m.velocity(context,x,torch.full((len(x),1),i/16,device='cuda'))/16
   x[...,2:4]=torch.nn.functional.normalize(x[...,2:4],dim=-1,eps=1e-6);x[...,4]=x[...,4].clamp_min(0);torch.manual_seed(1970);public=m.sample(b,k=1,steps=16)[:,0];sampling_diff=float((public-x).abs().max())
  results.append({'seed':seed,'mode':mode,'rows':rows,'public_vs_manual_sampler_max_abs':sampling_diff});del task,m,context;gc.collect();torch.cuda.empty_cache()
 # Read only trajectory fields in train and already-used dev. No test.
 negatives=[];mins=[];count=0;negcount=0;severe=0
 for name in sel['train']:
  with np.load(ROOT/'data/occ_v3'/name,allow_pickle=False) as a:v=a['trajectory'][:,4]*30
  count+=v.size;negcount+=int((v<0).sum());severe+=int((v<-.1).sum());mins.append(float(v.min()))
  if (v<0).any():negatives.append({'file':name,'min_mps':float(v.min()),'negative_points':int((v<0).sum())})
 report={'channels':['x/50','y/10','sin_yaw','cos_yaw','speed/30'],'windows':names,'results':results,'train_negative_speed':{'windows':len(sel['train']),'total_points':count,'negative_points':negcount,'below_minus_point1_mps':severe,'min_mps':min(mins),'negative_windows':len(negatives),'worst_windows':sorted(negatives,key=lambda r:r['min_mps'])[:10]},'limits':'Frozen4checkpoints8reuseddevwindows1noise; alltrain labels read for negative speed only; no test or updates; physical RMSE by channel not driving quality'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['train_negative_speed']),flush=True);print('DONE',flush=True)
if __name__=='__main__':main()
