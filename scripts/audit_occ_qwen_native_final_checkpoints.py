"""Read-only CPU integrity audit of completed native checkpoints."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['HIP_VISIBLE_DEVICES']='-1';os.environ['ROCR_VISIBLE_DEVICES']='-1'
import gc,json,hashlib,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def file_sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def tensor_audit(mapping):
 h=hashlib.sha256();nonfinite=[];count=0;numel=0
 def walk(x,path):
  nonlocal count,numel
  if isinstance(x,torch.Tensor):
   x=x.detach().cpu().contiguous();count+=1;numel+=x.numel();h.update((path+str(x.dtype)+str(tuple(x.shape))).encode())
   flat=x.reshape(-1)
   for chunk in flat.split(1048576):
    if (chunk.is_floating_point() or chunk.is_complex()) and not torch.isfinite(chunk).all():nonfinite.append(path)
    h.update(chunk.view(torch.uint8).numpy().tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=str):walk(x[k],path+'/'+str(k))
  elif isinstance(x,(list,tuple)):
   for i,v in enumerate(x):walk(v,path+'/'+str(i))
 walk(mapping,'');return {'tensor_count':count,'numel':numel,'all_finite':not nonfinite,'nonfinite_paths':sorted(set(nonfinite)),'tensor_content_sha256':h.hexdigest()}

def main():
 torch.set_num_threads(1);start=time.perf_counter();folder=ROOT/'artifacts/occ-qwen-native-training';out=ROOT/'artifacts/occ-qwen-native-final-audit';out.mkdir(exist_ok=True);result={'cpu_only':True,'read_only_checkpoint_audit':True,'checkpoints':{}}
 for name in ['best.ckpt','latest.ckpt']:
  p=folder/name;before=p.stat();cp=torch.load(p,map_location='cpu',weights_only=False);task=cp['task_state'];model=tensor_audit(cp['state_dict']);opt=tensor_audit(cp['optimizer_states']);states=[v for o in cp['optimizer_states'] for v in o['state'].values()];steps=sorted(set(int(v['step']) for v in states if 'step' in v));scheduler=[s['last_epoch'] for s in cp['lr_schedulers']];loader=cp['loops']['fit_loop']['state_dict']['combined_loader'];rng=task.get('rng',{});teacher={k:v for k,v in cp['state_dict'].items() if '.teacher.' in k};encoder={k:v for k,v in cp['state_dict'].items() if '.encoder.' in k}
  callbacks=[]
  for k,v in cp['callbacks'].items():
   callbacks.append({'callback_key':k,'monitor':v.get('monitor'),'best_model_score':float(v['best_model_score']) if v.get('best_model_score') is not None else None,'best_model_path':v.get('best_model_path'),'last_model_path':v.get('last_model_path')})
  row={'bytes':p.stat().st_size,'sha256':file_sha(p),'epoch_zero_based':cp['epoch'],'epoch_one_based':cp['epoch']+1,'global_step':cp['global_step'],'optimizer_steps':steps,'scheduler_last_epochs':scheduler,'optimizer_scheduler_step_match':steps==[cp['global_step']] and scheduler==[cp['global_step']],'model':model,'optimizer':opt,'ema_tensor_count':len(teacher),'ema_matches_encoder_key_structure':{k.replace('.teacher.','.encoder.') for k in teacher}==set(encoder),'rng_keys':sorted(rng),'rng_complete':set(rng)=={'python','numpy','torch','cuda'},'cpu_rng_bytes':rng['torch'].numel(),'cuda_rng_states_saved':len(rng['cuda']),'exact_loader_generator_present':all('_exact_generator_v1' in l for l in loader),'loader_generator_bytes':[l['_exact_generator_v1'].numel() if '_exact_generator_v1' in l else None for l in loader],'loader_dataset_states':[l.get('dataset_state') for l in loader],'manifest_sha256':task['manifest_sha'],'performance':task.get('performance'),'contract_test':task['contract_test'],'callbacks':callbacks}
  assert p.stat().st_size==before.st_size and p.stat().st_mtime_ns==before.st_mtime_ns
  result['checkpoints'][name]=row;del cp,teacher,encoder,states,loader,rng,task;gc.collect();print('AUDITED '+name,flush=True)
 logs=[]
 for filename in ['stdout.log','optimized-stdout.log']:
  for line in (folder/filename).read_text(encoding='utf-8-sig').splitlines():
   if line.startswith('STATUS '):
    x=json.loads(line[7:])
    if x['phase']=='validated':logs.append({'epoch':x['epoch'],'global_step':x['global_step'],'ade_m':x['metrics']['val/action_ade_m'],'val_loss':x['metrics']['val/loss'],'log':filename})
 by_epoch={}
 for x in logs:by_epoch.setdefault(x['epoch'],[]).append(x)
 unique=[v[-1] for k,v in sorted(by_epoch.items())];best=min(unique,key=lambda x:x['ade_m'])
 b=result['checkpoints']['best.ckpt'];l=result['checkpoints']['latest.ckpt']
 result.update(validation_epochs=unique,validation_record_count=len(logs),unique_validation_epochs=len(by_epoch),exactly_twenty_epochs=sorted(by_epoch)==list(range(1,21)),duplicate_validation_epochs={k:len(v) for k,v in by_epoch.items() if len(v)>1},best_actual_validation=best,best_checkpoint_matches_logged_best=b['global_step']==best['global_step'] and b['epoch_one_based']==best['epoch'],best_latest_model_tensor_equivalent=b['model']['tensor_content_sha256']==l['model']['tensor_content_sha256'],best_latest_optimizer_tensor_equivalent=b['optimizer']['tensor_content_sha256']==l['optimizer']['tensor_content_sha256'],seconds=time.perf_counter()-start)
 result['passed']=result['exactly_twenty_epochs'] and result['best_checkpoint_matches_logged_best'] and all(x['optimizer_scheduler_step_match'] and x['model']['all_finite'] and x['optimizer']['all_finite'] and x['ema_matches_encoder_key_structure'] and x['rng_complete'] and x['exact_loader_generator_present'] and not x['contract_test'] for x in result['checkpoints'].values())
 (out/'checkpoint-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['checkpoints','validation_epochs']},indent=2))
if __name__=='__main__':main()
