"""Two real-data SFT phases, orchestrated entirely by Lightning."""
from .runtime import ROOT,configure,math_attention
configure()
import argparse,json,time,traceback,hashlib,math
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,Callback
from pytorch_lightning.loggers import TensorBoardLogger
from .real_dataset import RealData
from .sft_task import SupervisedTask
from .checkpoint_io import VerifiedCheckpointIO


class RollingCheckpoint(ModelCheckpoint):
    def on_train_epoch_end(self,trainer,pl_module):
        super().on_train_epoch_end(trainer,pl_module)
        if self._last_global_step_saved!=trainer.global_step:
            self._save_topk_checkpoint(trainer,self._monitor_candidates(trainer))


class Report(Callback):
    @staticmethod
    def expected_updates(task):
        return math.ceil(task.cfg['samples']/task.cfg['effective_batch_size'])*task.cfg['epochs']
    def __init__(self,path,phase):
        self.path=path;self.started=time.perf_counter();self.phase=phase
    def write(self,trainer,task,state):
        row={'phase':self.phase,'state':state,'epoch':trainer.current_epoch+1,'step':trainer.global_step,
             'expected_updates':self.expected_updates(task),
             'completed_effective_epochs':trainer.global_step/math.ceil(task.cfg['samples']/task.cfg['effective_batch_size']),
             'seconds':time.perf_counter()-self.started,'microbatch':1,'accumulation':8,'effective_batch':8,
             'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
             'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
             'prefix_cache_GiB':task.prefix.bytes/2**30,'cache_hits':task.prefix.hits,'cache_misses':task.prefix.misses,
             'recent_seconds_per_sample':float(np.mean(task.step_times[-16:])) if task.step_times else None,
             'seen_tail':task.seen[-16:]}
        self.path.write_text(json.dumps(row,indent=2),encoding='utf-8')
    def on_train_batch_end(self,trainer,task,outputs,batch,batch_idx):
        torch.cuda.synchronize()
        r=batch['record']
        sample={'epoch':trainer.current_epoch+1,'step':trainer.global_step,'batch_index':batch_idx,
                'id':task.seen[-1],'source':r['source'],'speed_kmh':r['speed_kmh'],
                'seconds':time.perf_counter()-task.started,'losses':task.epoch_terms[-1]}
        with self.path.with_name('samples.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(sample)+'\n')
        self.write(trainer,task,'training')
        if (batch_idx+1)%8==0:print(json.dumps({'epoch':trainer.current_epoch+1,'step':trainer.global_step,'sample':batch_idx+1,'loss':task.epoch_terms[-1]}),flush=True)
    def on_train_epoch_end(self,trainer,task):
        row={'epoch':trainer.current_epoch+1,'step':trainer.global_step,'samples':len(task.epoch_terms),
             'unique_samples_in_fragment':len(set(task.seen[-len(task.epoch_terms):])) if task.epoch_terms else 0,
             'complete_epoch_metrics':len(task.epoch_terms)==task.cfg['samples'],
             'train':{k:float(np.mean([r[k] for r in task.epoch_terms])) for k in task.epoch_terms[0]} if task.epoch_terms else {}}
        with self.path.with_name('epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        task.epoch_terms=[];self.write(trainer,task,'epoch_complete')
    def on_validation_epoch_end(self,trainer,task):
        rows=task.validation_rows;metrics={}
        for name,group in [('all',rows),('80_120',[r for r in rows if r['speed_kmh']>=80])]:
            metrics[name]={'n':len(group),**{k:float(np.mean([r[k] for r in group])) if group else None for k in ('ade_m','fde_m','world_brier','persistence_brier','world_focal','ay_abs_max','jerk_abs_max')}}
        row={'epoch':trainer.current_epoch+1,'step':trainer.global_step,'scope':'same training64 overfit' if self.phase=='overfit64' else 'independent route validation','metrics':metrics,'samples':rows}
        with self.path.with_name('validation.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({'validation':metrics,'epoch':trainer.current_epoch+1}),flush=True)
    def on_fit_end(self,trainer,task):
        self.write(trainer,task,'finished' if trainer.global_step>=self.expected_updates(task) else 'incomplete')


def cache_contract(task,data,path):
    """Actual six-camera miss/hit value and LoRA-gradient equivalence."""
    record=data.records[0];task.train();results=[]
    for _ in range(2):
        task.zero_grad(set_to_none=True);torch.manual_seed(71);torch.cuda.manual_seed_all(71)
        with torch.autocast('cuda',dtype=torch.bfloat16),math_attention():
            native,inputs=task.encode_path(record['actor_path'])
            value=native['hidden'].float().square().mean()+sum(k.float().square().mean()+v.float().square().mean() for k,v in native['scene_cache'][6:])
        with math_attention():value.backward()
        results.append({'hidden':native['hidden'].detach().cpu(),'kv':[(k.detach().cpu(),v.detach().cpu()) for k,v in native['scene_cache']],
                        'grad':{n:p.grad.detach().cpu().clone() for n,p in task.model.vlm.named_parameters() if p.grad is not None}})
        del native,value
    a,b=results
    torch.testing.assert_close(a['hidden'],b['hidden'],rtol=0,atol=0)
    for x,y in zip(a['kv'],b['kv']):
        for xx,yy in zip(x,y):torch.testing.assert_close(xx,yy,rtol=0,atol=0)
    assert a['grad'].keys()==b['grad'].keys()
    for n in a['grad']:torch.testing.assert_close(a['grad'][n],b['grad'][n],rtol=1e-5,atol=1e-7)
    path.write_text(json.dumps({'passed':True,'gradient_tensors':len(a['grad']),'real_kv_groups':8,'input_audit':inputs['input_audit']},indent=2))
    task.zero_grad(set_to_none=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['overfit64','subset2000'],required=True)
    parser.add_argument('--max-steps',type=int,default=-1);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--initialize-from');parser.add_argument('--verify-cache',action='store_true')
    args=parser.parse_args();data=RealData(args.phase)
    cfg=json.loads((ROOT/'configs/v5-qwen-drive-world/lightweight.json').read_text())
    cfg.update(samples=len(data.records),epochs=30 if args.phase=='overfit64' else 3,batch_size=1,accumulate_grad_batches=8,effective_batch_size=8)
    folder=ROOT/'artifacts/v5-qwen-drive-world'/args.phase;folder.mkdir(parents=True,exist_ok=True)
    (folder/'config.json').write_text(json.dumps(cfg,indent=2))
    pl.seed_everything(71,workers=True);torch.set_num_threads(6);torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.8)
    task=SupervisedTask(cfg,data.identity,args.phase).cuda()
    if args.verify_cache:cache_contract(task,data,folder/'cache-contract.json');return
    if args.initialize_from:
        checkpoint=torch.load(args.initialize_from,map_location='cpu',weights_only=False)
        if checkpoint['task_state']['manifest_identity']!=data.identity:raise ValueError('Initialization dataset identity changed')
        task.load_trainable(checkpoint);task.load_state_dict(checkpoint['state_dict']);del checkpoint
    # Two fixed filenames only. Each replacement is validated atomically.
    latest=RollingCheckpoint(dirpath=folder,filename='latest',save_top_k=1,every_n_train_steps=1 if args.max_steps>0 else 8,
                           save_on_train_epoch_end=True,enable_version_counter=False)
    best=ModelCheckpoint(dirpath=folder,filename='best',monitor='validation_ade',mode='min',save_top_k=1,
                         save_on_train_epoch_end=False,enable_version_counter=False)
    report=Report(folder/'status.json',args.phase)
    trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',accumulate_grad_batches=8,
                       max_epochs=cfg['epochs'],max_steps=args.max_steps,gradient_clip_val=1.,
                       callbacks=[report,latest,best],plugins=[VerifiedCheckpointIO()],
                       logger=TensorBoardLogger(str(ROOT/'logs/v5-qwen-drive-world'),name=args.phase,version='main'),
                       enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,
                       check_val_every_n_epoch=5 if args.phase=='overfit64' else 1,inference_mode=False)
    try:
        trainer.fit(task,datamodule=data,ckpt_path=str(folder/'latest.ckpt') if args.resume else None,weights_only=False)
    except Exception:
        report.write(trainer,task,'failed')
        (folder/'error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
