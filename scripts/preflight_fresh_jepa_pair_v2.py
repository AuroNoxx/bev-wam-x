"""Disposable fresh paired initialization resource smoke, not learning acceptance."""
from train_jepa_nojepa_paired import *

def main():
 idle();configure_runtime();out=ROOT/'artifacts/fresh-jepa-initialization-preflight-v2';out.mkdir(exist_ok=False);src=ROOT/'artifacts/masked-teacher-155m-continue2048';cfg=json.loads((src/'config.json').read_text());sel=json.loads((src/'selection.json').read_text());cfg.update(max_steps=8,learning_rate=1e-4);seed=1950;pl.seed_everything(seed);initial=MaskedHistoryJoint(768,15,12).cpu();state={k:v.detach().clone() for k,v in initial.state_dict().items()};ema_equal=all(torch.equal(a,b) for a,b in zip(initial.encoder.parameters(),initial.teacher.parameters()));del initial;results=[];first_batches=[]
 for mode in ['jepa','no_jepa']:
  idle();task=AblationTask(cfg,'fresh-no-source-checkpoint','fresh-preflight',seed,mode);task.model.load_state_dict(state,strict=True);assert all(torch.equal(v,state[k]) for k,v in task.model.state_dict().items());m=task.model.cuda().train();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01);it=iter(SeedData(sel,seed).loader('train'));torch.cuda.reset_peak_memory_stats();rows=[];fingerprints=[]
  for step in range(8):
   torch.cuda.synchronize();start=time.perf_counter();b={k:v.cuda() for k,v in next(it).items()};fingerprints.append(hashlib.sha256(b['state'].cpu().numpy().tobytes()).hexdigest());opt.zero_grad(set_to_none=True)
   with torch.random.fork_rng(devices=[0]),torch.autocast('cuda',dtype=torch.bfloat16):
    torch.manual_seed(seed*100000+step);loss,terms=consistent_flow_loss(m,b,cfg['objective'])
    if mode=='jepa':
     for name,v in zip(['latent','variance','semantic'],m.auxiliary_loss(b)):loss=loss+.1*v
   loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step();m.update_teacher(.99);torch.cuda.synchronize();rows.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(gn),'wall_s_with_data':time.perf_counter()-start})
  result={'mode':mode,'updates_discarded':8,'active_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'rows':rows,'warm_mean_s':float(np.mean([r['wall_s_with_data'] for r in rows[2:]])),'warm_max_s':max(r['wall_s_with_data'] for r in rows[2:]),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30};results.append(result);first_batches.append(fingerprints);print('RESULT '+json.dumps(result),flush=True);del task,m,opt,it,b,loss,terms;gc.collect();torch.cuda.empty_cache()
 assert first_batches[0]==first_batches[1];(out/'report.json').write_text(json.dumps({'seed':seed,'fresh_shared_initial_state_exact':True,'initial_EMA_equals_encoder':ema_equal,'paired_first8_batch_state_hashes_equal':True,'source_checkpoint_loaded':False,'checkpoint_saved':False,'data_train_windows':len(sel['train']),'results':results,'limits':'1seed8updates per arm discarded; finite cost smoke only, not convergence or quality; no test or CARLA'},indent=2));print('DONE')
if __name__=='__main__':main()

