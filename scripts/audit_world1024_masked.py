from continue_ar_pair1024 import *
def main():
 idle();configure_runtime();out=ROOT/'artifacts/world1024-frozen-mask-audit';out.mkdir(exist_ok=False);src=ROOT/'artifacts/ar-noise-paired1024';base=ROOT/'artifacts/residual-representation-paired128';sel=json.loads((base/'selection.json').read_text());scales=json.loads((base/'scale_calibration.json').read_text())['scales_m'];ds=DiskSamples(sel['validation']);allresults=[]
 for seed in [1980,1981]:
  m=NoiseResidualJEPA(768,15,12);m.set_scales(scales);m.set_noise(0);ck=torch.load(src/f'{seed}-iid/step1024.ckpt',map_location='cpu',weights_only=False);m.load_state_dict({k.removeprefix('model.'):v for k,v in ck['state_dict'].items()},strict=True);del ck;m.cuda().eval();rows=[]
  for start in range(0,len(ds),4):
   b=batch_to([ds[i] for i in range(start,min(start+4,len(ds)))],'cuda')
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    _,pred=m.forecast_batch(b);future,valid,frac=m.target_data(b);target=m.teacher(future).float();history=m.canonical_history(b);current=m.teacher(history[:,-1:]).float();previous=m.teacher(history[:,-2:-1]).float();step=b['history_age_seconds'][:,-2]-b['history_age_seconds'][:,-1];trend=current+(current-previous)*(torch.tensor([1.,2.],device='cuda')[None,:,None,None]/step[:,None,None,None]);persist=current.expand_as(target);dynamic=valid&(frac[...,3:5].sum(-1)>.05)
    for j in range(len(target)):
     for h in range(2):
      for kind,mask in [('all',valid[j,h]),('dynamic_class_proxy',dynamic[j,h])]:
       if not mask.any():continue
       tar=target[j,h];p=pred[j,h].float();cur=persist[j,h];tr=trend[j,h];rows.append({'window':start+j,'horizon_s':h+1,'kind':kind,'tokens':int(mask.sum()),'prediction_mse':float((p-tar).square().mean(-1)[mask].mean()),'persistence_mse':float((cur-tar).square().mean(-1)[mask].mean()),'trend_mse':float((tr-tar).square().mean(-1)[mask].mean()),'prediction_rms':float(p[mask].square().mean().sqrt()),'target_rms':float(tar[mask].square().mean().sqrt()),'prediction_to_current_mse':float((p-cur).square().mean(-1)[mask].mean())})
  summaries=[]
  for h in [1,2]:
   for kind in ['all','dynamic_class_proxy']:
    rr=[r for r in rows if r['horizon_s']==h and r['kind']==kind];summaries.append({'horizon':h,'kind':kind,'windows':len(rr),'wins':sum(r['prediction_mse']<r['persistence_mse'] for r in rr),**{k:float(np.mean([r[k] for r in rr])) for k in ['prediction_mse','persistence_mse','trend_mse','prediction_rms','target_rms','prediction_to_current_mse']}})
  allresults.append({'seed':seed,'summaries':summaries,'rows':rows});print(json.dumps({'seed':seed,'summaries':summaries}),flush=True);del m;gc.collect();torch.cuda.empty_cache()
 (out/'report.json').write_text(json.dumps({'results':allresults,'limits':'42same-routeinternaldiagnostic; fixedcheckpointEMAspacecommonmask;dynamicclassnotactualmotion;lineartrendnottrained andnotphysicallywarpedlatentmotion;futuretargetsupportcurrentoccupiedonly,notfree;no actorfutureinputs/updates/test'},indent=2));print('DONE')
if __name__=='__main__':main()
