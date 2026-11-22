"""Single-device BF16 Lightning training for the 18k/2k Phase 2 split."""
import argparse
from collections import Counter
import hashlib
import json
import math
import msvcrt
import os
from pathlib import Path
import time

# TunableOp reads environment flags on initialization.
os.environ['PYTORCH_TUNABLEOP_ENABLED'] = '1'

import psutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader

from v1_dual_system.checkpoint_io import VerifiedCheckpointIO
from .phase2_data import Corpus, bucket, prepare
from .phase2_model import GAMMA, Phase2, configure_sdpa, make_grid_axes
from .phase2_objectives import make_batch, metrics_from_totals, multitask_loss, validation_totals
from .render_probe import LOCKED_CONFIG, locked_tier


class Learner(pl.LightningModule):
    def __init__(self, config, sampler=None, compile_backend=None):
        super().__init__()
        self.model = Phase2(pretrained=config.get('imagenet_pretrained',False)).to(memory_format=torch.channels_last)
        self.config = config
        self.sampler = sampler
        self.compiled_forward = (torch.compile(self.model.forward,backend=compile_backend,dynamic=False)
                                 if compile_backend else None)
        xx, yy = make_grid_axes()
        self.register_buffer('xx', xx, persistent=False)
        self.register_buffer('yy', yy, persistent=False)
        self.totals = None
        self.batch_peak = 0
        self.start = time.perf_counter()

    def compute(self, batch, augmented):
        args, targets = make_batch(batch, augmented, (self.xx,self.yy))
        forward = self.compiled_forward or self.model
        outputs = forward(*args)
        return multitask_loss(outputs, targets,self.config.get('phase25',False)), outputs, targets

    def on_train_epoch_start(self):
        if self.sampler is not None:
            self.sampler.generator.manual_seed(self.config['seed']+self.current_epoch)

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.compute(batch, True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=self.config['batch'])
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (batch_idx+1) % 50 == 0 or batch_idx == 0:
            value = float(outputs['loss'])
            if not torch.isfinite(outputs['loss']):
                raise FloatingPointError(f'Nonfinite loss at update {self.global_step}')
            print(f'epoch {self.current_epoch+1} batch {batch_idx+1} update {self.global_step} '
                  f'loss {value:.4f} elapsed {time.perf_counter()-self.start:.1f}s', flush=True)

    def on_validation_epoch_start(self):
        self.totals = torch.zeros(36, device=self.device)

    def validation_step(self, batch, batch_idx):
        loss, out, target = self.compute(batch, False)
        self.totals += validation_totals(out, target)
        self.log('val_loss', loss, batch_size=self.config['batch'])

    def on_validation_epoch_end(self):
        result = metrics_from_totals(self.totals)
        if self.config.get('phase25',False):
            score = .25*result['lateral_near_m']-result['map_miou']
        else:
            score = 2*result['lateral_near_m']+result['lateral_far_m']
        self.log('score', score)
        if not self.trainer.sanity_checking:
            result.update(epoch=self.current_epoch+1,updates=self.global_step,score=score,
                          lr=self.optimizers().param_groups[0]['lr'],
                          peak_alloc_gib=torch.cuda.max_memory_allocated()/2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                          elapsed_s=time.perf_counter()-self.start)
            with (Path(self.config['out'])/'validation.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(result)+'\n')
            print(json.dumps(result), flush=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'], weight_decay=.01, foreach=True)
        if self.config.get('phase25',False):
            n = self.config['updates_per_epoch']
            warm = torch.optim.lr_scheduler.LinearLR(opt,start_factor=.05,end_factor=1.,total_iters=n)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=(self.config['epochs']-1)*n,eta_min=1e-5)
            schedule = torch.optim.lr_scheduler.SequentialLR(opt,[warm,cosine],milestones=[n])
            return {'optimizer':opt,'lr_scheduler':{'scheduler':schedule,'interval':'step'}}
        lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.config['epochs'], eta_min=self.config['lr']*.13)
        return {'optimizer': opt, 'lr_scheduler': lr}

    def on_save_checkpoint(self, checkpoint):
        checkpoint['task_state'] = self.config

    def on_load_checkpoint(self, checkpoint):
        original = checkpoint['task_state']
        for name in ('manifest_sha256','batch','accumulate','seed','renderer_sha256','gamma','sdpa'):
            if original[name] != self.config[name]:
                raise ValueError(f'Checkpoint resume changes {name}')
        if self.config.get('phase25') and any(original.get(name)!=self.config[name] for name in (
                'phase25','epochs','lr','updates_per_epoch','imagenet_pretrained')):
            raise ValueError('Phase 2.5 resume changes the frozen schedule or initialization contract')
        self.config['resumed_from_update'] = checkpoint['global_step']


class Phase25Best(ModelCheckpoint):
    def on_validation_end(self,trainer,pl_module):
        if trainer.current_epoch+1>12:
            super().on_validation_end(trainer,pl_module)


def check_running():
    own = {os.getpid(), os.getppid()}
    for process in psutil.process_iter(['pid','name','cmdline']):
        if process.info['pid'] in own:
            continue
        name = (process.info['name'] or '').lower()
        cmd = ' '.join(process.info['cmdline'] or []).lower()
        if name.startswith(('carlaunreal','carlaue4')) or (
                name.startswith('python') and ('train_' in cmd or 'phase2_train' in cmd or 'pytorch_lightning' in cmd)):
            raise RuntimeError(f'Another simulator or training process is active (PID {process.pid}): {cmd}')


def freeze(best, destination, config):
    checkpoint = torch.load(best, map_location='cpu', weights_only=False)
    state = {k.removeprefix('model.'):t for k,t in checkpoint['state_dict'].items() if k.startswith('model.')}
    Phase2().load_state_dict(state, strict=True)
    if not state or not all(torch.isfinite(v).all() for v in state.values()):
        raise RuntimeError('Invalid frozen state')
    destination.mkdir(parents=True, exist_ok=True)
    final = destination/'stage2.pt'
    pending = destination/'stage2.pending'
    if final.exists():
        raise FileExistsError(final)
    torch.save({'model':state,'config':config,'updates':checkpoint['global_step'],
                'epoch':checkpoint['epoch']+1}, pending)
    check = torch.load(pending,map_location='cpu',weights_only=True)
    for key,tensor in state.items():
        if not torch.equal(tensor,check['model'][key]):
            raise RuntimeError(f'Frozen serialization mismatch: {key}')
    with pending.open('rb+') as stream:
        os.fsync(stream.fileno())
    os.replace(pending,final)
    with final.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    record = {'frozen':str(final.resolve()),'selected_epoch':checkpoint['epoch']+1,
              'selected_updates':checkpoint['global_step'],'sha256':digest}
    (destination/'metadata.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record


def main(args):
    check_running()
    if args.phase25 and (args.epochs!=25 or args.lr!=3e-4):
        raise ValueError('Phase 2.5 requires 25 epochs and base LR 3e-4')
    if not torch.cuda.tunable.is_enabled():
        raise RuntimeError('TunableOp did not enable')
    configure_sdpa(args.sdpa)
    torch.set_num_threads(4)
    pl.seed_everything(args.seed,workers=True)
    tier = locked_tier()
    prepare(args.dataset,args.cache)
    train, val = (Corpus(args.dataset,args.cache,split) for split in ('train','val'))
    if len(train)%args.batch or len(val)%args.batch:
        raise ValueError('Physical batch must divide train and validation sizes without padding or dropping frames')
    sampler = train.sampler(args.seed)
    draw = list(sampler)
    sampling = dict(Counter(bucket(train.rows[i]) for i in draw))
    sampling['straight_tag'] = sum(train.rows[i]['straight'] for i in draw)
    train_loader = DataLoader(train,batch_size=args.batch,sampler=sampler,pin_memory=True,
                              num_workers=args.workers,persistent_workers=bool(args.workers))
    val_loader = DataLoader(val,batch_size=args.batch,pin_memory=True,num_workers=args.workers,
                            persistent_workers=bool(args.workers))
    config = dict(architecture='Phase2',out=str(args.out.resolve()),manifest_sha256=train.digest,
                  renderer=tier,renderer_sha256=hashlib.sha256(LOCKED_CONFIG.read_bytes()).hexdigest(),
                  gamma=GAMMA,seed=args.seed,sdpa=args.sdpa,precision='bf16-mixed',epochs=args.epochs,
                  batch=args.batch,accumulate=args.accumulate,lr=args.lr,foreach=True,tunable=True,
                  compile=args.compile,compile_backend=args.compile_backend if args.compile else None,
                  first_epoch_draw=sampling)
    if args.phase25:
        config.update(phase25=True,imagenet_pretrained=True,
                      updates_per_epoch=math.ceil(len(train_loader)/args.accumulate),
                      schedule='1 epoch linear warmup (5% to base LR), then cosine to 1e-5',
                      map_loss_scale=2., occupancy_loss_scale=3.,
                      best_score='minimize 0.25 * lateral_near_m - map_miou, epochs 13–25')
    learner = Learner(config,sampler,config['compile_backend'])
    config['parameters'] = sum(p.numel() for p in learner.model.parameters())
    print(json.dumps(config),flush=True)
    (args.out/'training-config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    checkpoint_class = Phase25Best if args.phase25 else ModelCheckpoint
    checkpoints = checkpoint_class(dirpath=args.out/'checkpoints',monitor='score',mode='min',
                                  filename='best',save_top_k=1,save_last=False,enable_version_counter=False,
                                  save_on_train_epoch_end=False)
    latest = ModelCheckpoint(dirpath=args.out/'checkpoints',monitor=None,
                             filename='last',save_top_k=1,save_last=False,enable_version_counter=False,
                             save_on_train_epoch_end=False)
    trainer = pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_epochs=args.epochs,
                         accumulate_grad_batches=args.accumulate,gradient_clip_val=2.,
                         callbacks=[checkpoints,latest],plugins=[VerifiedCheckpointIO()],
                         logger=CSVLogger(args.out,name='lightning'),enable_progress_bar=False,
                         enable_model_summary=False,num_sanity_val_steps=0,log_every_n_steps=50)
    trainer.fit(learner,train_loader,val_loader,ckpt_path=str(args.resume) if args.resume else None)
    saved_last = torch.load(args.out/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
    if saved_last['global_step']!=trainer.global_step:
        raise RuntimeError('Latest verified checkpoint does not contain the last optimizer update')
    del saved_last
    if not checkpoints.best_model_path:
        raise RuntimeError('No validated best checkpoint')
    record = freeze(checkpoints.best_model_path,args.freeze,config)
    if args.phase25 and record['selected_epoch']<=12:
        raise RuntimeError('Phase 2.5 selection ignored the 12-epoch exclusion')
    record.update(actual_updates=trainer.global_step,physical_batch=args.batch,
                  gradient_accumulation=args.accumulate, wall_s=time.perf_counter()-learner.start)
    (args.out/'training-result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','cache','out','freeze'):
        parser.add_argument(f'--{name}',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=6)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--accumulate',type=int,default=1)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--seed',type=int,default=20261120)
    parser.add_argument('--lr',type=float,default=4e-4)
    parser.add_argument('--compile',action='store_true')
    parser.add_argument('--compile-backend',choices=('aot_eager','inductor'),default='aot_eager')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--phase25',action='store_true')
    parser.add_argument('--sdpa',choices=('aotriton','math'),default='aotriton')
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.phase2-training.lock').open('a+b') as lock:
        if lock.tell()==0:
            lock.write(b'\0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main(args)
