"""Windows entry point using Lightning Trainer, without a custom training loop."""
import argparse,json,os,time,statistics
from pathlib import Path
import torch,psutil
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_task import ROOT,configure_runtime,OccData,DrivingTask,VerifiedCheckpointIO,callbacks

class SpeedProbe(pl.Callback):
    def __init__(self,out):self.out=out;self.times=[];self.last=None
    def on_train_batch_start(self,trainer,module,batch,batch_idx):
        if self.last is None:self.last=time.perf_counter()
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        torch.cuda.synchronize();now=time.perf_counter()
        if trainer.global_step>10:self.times.append(now-self.last)
        self.last=now
    def on_train_end(self,trainer,module):
        report={'settings':module.settings,'steps':len(self.times),'samples_s':8/statistics.mean(self.times),'mean_step_seconds':statistics.mean(self.times),
                'allocated_peak_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_peak_gib':torch.cuda.max_memory_reserved()/2**30}
        self.out.parent.mkdir(parents=True,exist_ok=True);self.out.write_text(json.dumps(report,indent=2));print('SPEED_RESULT '+json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['flow','world'],default='flow');p.add_argument('--resume',action='store_true')
    p.add_argument('--benchmark');p.add_argument('--settings',default='configs/lightning.json');p.add_argument('--checkpoint');p.add_argument('--max-steps',type=int,default=-1)
    p.add_argument('--run-dir');p.add_argument('--init-weights');p.add_argument('--validate-only',action='store_true')
    p.add_argument('--finish-max-epochs',action='store_true',help='Explicit resume override: disable early stopping and finish the configured epoch limit')
    a=p.parse_args()
    if a.finish_max_epochs and (not a.resume or a.validate_only or a.benchmark):
        raise ValueError('--finish-max-epochs requires a normal --resume training run')
    pause=ROOT/'configs/TRAINING_PAUSED.json'
    if not a.validate_only and pause.exists() and json.loads(pause.read_text()).get('paused'):
        raise RuntimeError('Training paused for contract audit; see configs/TRAINING_PAUSED.json')
    configure_runtime();settings=json.loads((ROOT/a.settings).read_text())
    if a.init_weights and (a.resume or a.checkpoint):raise ValueError('Weights initialization and exact resume are separate operations')
    if psutil.virtual_memory().available<4*2**30:raise RuntimeError('Insufficient host RAM')
    if any(x.info['name']=='CarlaUnreal-Win64-Shipping.exe' for x in psutil.process_iter(['name'])):raise RuntimeError('Stop CARLA first')
    pl.seed_everything(20261114,workers=True);data=OccData(a.mode,settings);module=DrivingTask(a.mode,settings,data.sha,data.manifest['counts']['train'])
    if a.init_weights:
        source=torch.load(ROOT/a.init_weights,map_location='cpu',weights_only=False)
        if source['task_state']['manifest_sha']!=data.sha:raise RuntimeError('Initialization data split mismatch')
        module.load_state_dict(source['state_dict'],strict=True)
    folder=ROOT/(a.run_dir or f'artifacts/lightning/{a.mode}');folder.mkdir(parents=True,exist_ok=True)
    cfg={**data.config[a.mode],**settings.get('training',{})}
    version=a.benchmark or settings.get('run_name','formal')
    if a.validate_only:version+='-assessment'
    logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name=f'lightning-{a.mode}',version=version,default_hp_metric=False)
    cb=[SpeedProbe(ROOT/f'artifacts/lightning/benchmarks/{a.benchmark}.json')] if a.benchmark else callbacks(folder,a.mode,settings)
    if a.finish_max_epochs:
        cb=[callback for callback in cb if not isinstance(callback,pl.callbacks.EarlyStopping)]
    trainer=pl.Trainer(accelerator='gpu',devices=1,precision=settings['precision'],logger=logger,callbacks=cb,plugins=[VerifiedCheckpointIO()],
        max_epochs=cfg['max_epochs'],min_epochs=0 if a.benchmark or a.max_steps>0 else cfg['min_epochs'],max_steps=110 if a.benchmark else a.max_steps,
        gradient_clip_val=1.,gradient_clip_algorithm='norm',log_every_n_steps=settings['log_every_n_steps'],num_sanity_val_steps=0,
        enable_progress_bar=False,enable_model_summary=False,enable_checkpointing=not bool(a.benchmark),
        limit_val_batches=0 if a.benchmark else 1.0,benchmark=settings.get('conv_autotune',False),default_root_dir=str(folder))
    ckpt=ROOT/a.checkpoint if a.checkpoint else (folder/'latest.ckpt' if a.resume else None)
    if not a.benchmark and ckpt is None and (folder/'latest.ckpt').exists():raise RuntimeError('Use --resume for the existing run')
    runtime={'framework':pl.__version__,'torch':torch.__version__,'mode':a.mode,'settings':settings,'manifest_sha':data.sha,'resume_from':str(ckpt) if ckpt else None,'initialize_weights_from':a.init_weights,'pid':os.getpid(),
             'early_stopping_disabled_by_explicit_resume_override':a.finish_max_epochs}
    (folder/'runtime.json').write_text(json.dumps(runtime,indent=2));print('START '+json.dumps(runtime),flush=True)
    if a.validate_only:
        started=time.perf_counter()
        result=trainer.validate(module,datamodule=data,ckpt_path=str(ckpt) if ckpt else None,weights_only=False,verbose=False)
        report={'metrics':result,'seconds':time.perf_counter()-started,'initialization':a.init_weights,'checkpoint':str(ckpt) if ckpt else None}
        (folder/'validation.json').write_text(json.dumps(report,indent=2));print('VALIDATION '+json.dumps(report),flush=True)
    else:
        trainer.fit(module,datamodule=data,ckpt_path=str(ckpt) if ckpt else None,weights_only=False)

if __name__=='__main__':main()
