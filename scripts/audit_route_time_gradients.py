from train_route_regression_flow1024 import *
def flatgrad(loss,params):
 g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);return torch.cat([(torch.zeros_like(p) if v is None else v).detach().float().reshape(-1) for p,v in zip(params,g)])
def cosine(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-20))
def main():
 idle();configure_runtime();base=ROOT/'artifacts/route-regression-flow1024';out=ROOT/'artifacts/route-time-gradient-audit';out.mkdir(exist_ok=False);d=json.loads((base/'design.json').read_text());rows=[];sampling={};start=time.perf_counter()
 for seed in [2050,2051]:
  src=base/f'{seed}-fm/latest.ckpt';sha=hashlib.sha256(src.read_bytes()).hexdigest();ck=torch.load(src,map_location='cpu',weights_only=False);m=CompareModel('fm');m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});del ck;m.cuda().train();params=[p for p in m.parameters() if p.requires_grad];initial={k:v.detach().clone() for k,v in m.state_dict().items()};hist=np.zeros(10,dtype=int)
  for step in range(1024):
   torch.manual_seed(seed*100000+step);torch.randn(12,25,5,device='cuda');ts=torch.rand(12,1,device='cuda').cpu().numpy();hist+=np.histogram(ts,bins=np.linspace(0,1,11))[0]
  sampling[str(seed)]={'per_decile':hist.tolist(),'active_position_samples':int(hist[5:].sum()),'total':int(hist.sum())}
  for batchid,offset in enumerate([0,6144]):
   names=d['streams'][str(seed)][offset:offset+12];ds=DiskSamples(names);b=batch_to([ds[i] for i in range(12)],'cuda');torch.manual_seed(seed*100000+offset//12);noise=torch.randn(12,25,5,device='cuda')[...,:2];actualt=torch.rand(12,1,device='cuda');target=m.to_flow_space(b,b['trajectory'])
   for tv in [0.,.1,.3,.49,.5,.7,.9,.99,'replay']:
    assert time.perf_counter()-start<180;t=actualt if tv=='replay' else torch.full((12,1),tv,device='cuda');torch.manual_seed(2300+batchid)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     x=(1-t[:,:,None])*noise+t[:,:,None]*target;pred=m(b,x,t).float();fm=F.mse_loss(pred,target-noise);q=x+(1-t[:,:,None])*pred;p=m.from_flow_space(b,q);err=(p[...,:2]-b['trajectory'][...,:2])*q.new_tensor([50,10]);per=F.smooth_l1_loss(err/5,torch.zeros_like(err),reduction='none').flatten(1).mean(1);gate=(t[:,0]>=.5).float()*2;pos=.2*(per*gate).mean();allpos=.2*per.mean();aux=m.auxiliary_loss(b);world=.1*sum(aux)
    gf=flatgrad(fm,params);gp=flatgrad(pos,params);ga=flatgrad(allpos,params);gw=flatgrad(world,params);old=gf+gp+gw;new=gf+ga+gw
    rows.append({'seed':seed,'batch':batchid,'t':tv,'fm_loss':float(fm),'field_rmse_q':float(fm.sqrt()),'endpoint_q_rmse':float((q-target).square().mean().sqrt()),'weighted_position_loss':float(pos),'alltime_position_loss':float(allpos),'world_weighted_loss':float(world),'fm_grad':float(gf.norm()),'position_grad':float(gp.norm()),'alltime_position_grad':float(ga.norm()),'world_grad':float(gw.norm()),'fm_position_cos':cosine(gf,gp),'fm_alltime_position_cos':cosine(gf,ga),'fm_world_cos':cosine(gf,gw),'total_norm':float(old.norm()),'alltime_total_norm':float(new.norm()),'clip_coefficient':min(1,1/float(old.norm())),'alltime_clip_coefficient':min(1,1/float(new.norm())),'old_vs_alltime_cos':cosine(old,new)})
    del gf,gp,ga,gw,old,new,fm,pos,allpos,world,pred,q,p,aux
  assert all(torch.equal(v,initial[k]) for k,v in m.state_dict().items());assert sha==hashlib.sha256(src.read_bytes()).hexdigest();del m,params,initial;gc.collect();torch.cuda.empty_cache()
 result={'rows':rows,'sampling':sampling,'wall_s':time.perf_counter()-start,'no_updates':True,'weights_buffers_unchanged':True,'limits':'Two actual train batches per seed, fixed t sweeps not full training distribution. Same sampled bridge/noise as selected replay steps, frozen trained checkpoint not historic weights. BF16 training mode. World combined actually weighted0.1each; alltime0.2counterfactual only, not applied. No test or optimizer.'};(out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
