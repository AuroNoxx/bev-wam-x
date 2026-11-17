"""Bounded pretrained GPU v2 full resume gate; test weights are never production."""
import copy,hashlib,json,time,psutil
import torch
import pytorch_lightning as pl
from torchdata.stateful_dataloader import StatefulDataLoader
from native_occ_exact_loader import ExactResumeDataLoader
from native_occ_cached_data import NativeCachedDataset
from train_occ_qwen_native_v2 import NativeV2Task,config,make_model
from lightning_task import VerifiedCheckpointIO
from b2d_native_occ import ROOT
class Data(pl.LightningDataModule):
 def __init__(self,rows,cfg):super().__init__();self.rows=rows;self.cfg=cfg
 def train_dataloader(self):return ExactResumeDataLoader(self.rows,batch_size=8,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(self.cfg['seed']))
def main():
 for p in psutil.process_iter(['name','cmdline']):
  if p.info['name'] and p.info['name'].lower()=='python.exe' and any('scripts/train_occ_qwen_native' in c for c in p.info['cmdline'] or []):raise RuntimeError('trainer alive; gate cannot compete')
 cfg=config();torch.set_num_threads(4);folder=ROOT/'artifacts/occ-qwen-native-v2-gpu-gate';folder.mkdir(exist_ok=True)
 ds=NativeCachedDataset(ROOT/cfg['cache_index'],'train');rows=[ds[i] for i in range(16)];sha=hashlib.sha256((ROOT/cfg['manifest']).read_bytes()).hexdigest()
 class Save(pl.Callback):
  def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
   if trainer.global_step==2:trainer.save_checkpoint(folder/'split.ckpt')
 def run(resume=None):
  pl.seed_everything(cfg['seed'],workers=True,verbose=False);model=make_model(cfg);before=model.encoder.frame.embedding.weight.detach().clone();task=NativeV2Task(model,cfg,sha,contract_test=True)
  trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_steps=4,max_epochs=3,plugins=[VerifiedCheckpointIO()],callbacks=[] if resume else [Save()],logger=False,gradient_clip_val=1.,limit_val_batches=0,num_sanity_val_steps=0,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False)
  started=time.monotonic();trainer.fit(task,datamodule=Data(rows,cfg),ckpt_path=str(resume) if resume else None,weights_only=False)
  after=model.encoder.frame.embedding.weight.detach().cpu();assert not torch.equal(before,after)
  result={'state':{k:v.detach().cpu().clone() for k,v in task.state_dict().items()},'optimizer':copy.deepcopy(trainer.optimizers[0].state_dict()),'scheduler':copy.deepcopy(trainer.lr_scheduler_configs[0].scheduler.state_dict()),'step':trainer.global_step,'seconds':time.monotonic()-started,'allocated_gb':torch.cuda.max_memory_allocated()/1e9,'reserved_gb':torch.cuda.max_memory_reserved()/1e9}
  return result
 full=run();resumed=run(folder/'split.ckpt');maxdiff=max(float((v.float()-resumed['state'][k].float()).abs().max()) for k,v in full['state'].items());model_ok=all(torch.allclose(v,resumed['state'][k],rtol=1e-5,atol=1e-7) for k,v in full['state'].items())
 optdiff=0.;optok=True
 for i,s in full['optimizer']['state'].items():
  for k,v in s.items():
   other=resumed['optimizer']['state'][i][k]
   if isinstance(v,torch.Tensor):
    optdiff=max(optdiff,float((v-other).abs().max()));optok=optok and (torch.equal(v,other) if k=='step' else torch.allclose(v,other,rtol=1e-5,atol=1e-7))
 report={'passed':model_ok and optok and full['scheduler']==resumed['scheduler'] and full['step']==resumed['step']==4,'test_only':True,'production_weights_saved':False,'pretrained_backbone_layers':12,'batch_size':8,'updates_executed':6,'model_max_difference':maxdiff,'model_allclose':model_ok,'optimizer_max_difference':optdiff,'optimizer_allclose':optok,'scheduler_exact':full['scheduler']==resumed['scheduler'],'full_4step_seconds':full['seconds'],'peak_allocated_gb':full['allocated_gb'],'peak_reserved_gb':full['reserved_gb'],'tolerance':{'rtol':1e-5,'atol':1e-7}}
 (folder/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));assert report['passed']
if __name__=='__main__':main()
