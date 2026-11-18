"""Supervised RGB/intent/Flow/World task. Lightning owns all optimization loops."""
from .runtime import ROOT,configure,math_attention
configure()
import copy,gc,json,math,random,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from .components import load_planning,load_perception
from .planning import ControlFlowPlanner
from .world2d import ActionConditionedWorld2D
from .real_dataset import inputs_for
from .frozen_prefix import FrozenPrefixCache
from .data import intent_labels
from .kinematics import normalize_controls
from .objectives import sft_loss,focal_probability,masked_mean


def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])


class SupervisedTask(pl.LightningModule):
    def __init__(self,cfg,manifest_identity,phase):
        super().__init__();self.cfg=cfg;self.manifest_identity=manifest_identity;self.phase=phase
        self.model=load_planning(cfg)
        tokenizer=self.model.processor.tokenizer
        initial=[self.model.vlm.get_input_embeddings().weight[tokenizer.encode(w,add_special_tokens=False)].detach().mean(0) for w in ('reasoning','planning')]
        tokenizer.add_special_tokens({'additional_special_tokens':['<REASON_TOKEN>','<PLAN_TOKEN>']})
        self.model.vlm.resize_token_embeddings(len(tokenizer),mean_resizing=False)
        embedding=self.model.vlm.get_input_embeddings();embedding.requires_grad_(False)
        self.token_ids=tokenizer.convert_tokens_to_ids(['<REASON_TOKEN>','<PLAN_TOKEN>'])
        with torch.no_grad():
            for index,value in zip(self.token_ids,initial):embedding.weight[index].copy_(value)
        expert=self.model.planning_expert;del self.model.planning_expert
        self.planner=ControlFlowPlanner(expert,self.model.config.trajectory_scale).cuda()
        perception,_=load_perception(cfg,device='cpu');self.world=ActionConditionedWorld2D(perception).cuda();del perception;gc.collect()
        hidden=embedding.weight.shape[1];self.reason_head=nn.Linear(hidden,12,dtype=torch.float32).cuda()
        self.ema_world=copy.deepcopy(self.world).requires_grad_(False).eval()
        self.lora_names=[n for n,p in self.model.vlm.named_parameters() if p.requires_grad]
        parameters=dict(self.model.vlm.named_parameters())
        self.ema_lora=nn.ParameterList([nn.Parameter(parameters[n].detach().clone(),requires_grad=False) for n in self.lora_names])
        self.prefix=FrozenPrefixCache(max_bytes=3*2**30);self.pending_rng=None;self.last_ema_step=0
        self.validation_rows=[];self.epoch_terms=[];self.seen=[];self.step_times=[]
        self.base_identity=json.loads((ROOT/'artifacts/v5-qwen-drive-world/download.json').read_text())['files']
        assert all(p.dtype==torch.float32 for p in self.parameters() if p.requires_grad)
        self.train()

    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'ema_world'):self.ema_world.eval()
        return self

    def encode_path(self,path,planner=True):
        entry=self.prefix.entries.get((path,planner))
        if entry is not None:inputs={k:v.to(self.device) if isinstance(v,torch.Tensor) else v for k,v in entry['inputs'].items()}
        else:inputs=inputs_for(self.model.processor,path,self.device,self.token_ids)
        return self.prefix.encode(self.model,inputs,path,planner),inputs

    @torch.no_grad()
    def teacher_targets(self,record,labels):
        parameters=dict(self.model.vlm.named_parameters());active=[parameters[n] for n in self.lora_names]
        saved=[p.detach().clone() for p in active];was_training=self.model.vlm.training
        targets=[];valid=[];self.model.vlm.eval()
        try:
            for p,e in zip(active,self.ema_lora):p.copy_(e)
            yy,xx=torch.meshgrid(torch.arange(8,device=self.device),torch.arange(8,device=self.device),indexing='ij')
            x=-16+(xx+.5)*16;y=-19.2+(yy+.5)*4.8
            for h,path in enumerate(record['future_actor_paths']):
                native,_=self.encode_path(path,planner=False)
                z=self.ema_world.encode_current(native['hidden'])
                pose=labels['future_ego_poses'][:,h].float();dx=x[None]-pose[:,0,None,None];dy=y[None]-pose[:,1,None,None]
                c=pose[:,2,None,None].cos();s=pose[:,2,None,None].sin()
                fx=c*dx+s*dy;fy=-s*dx+c*dy
                grid=torch.stack((2*(fx+16)/128-1,2*(fy+19.2)/38.4-1),-1)
                with torch.autocast('cuda',enabled=False):
                    targets.append(F.grid_sample(z.float(),grid.float(),align_corners=False))
                valid.append((fx>=-16)&(fx<112)&(fy>=-19.2)&(fy<19.2))
                del native,z
        finally:
            for p,v in zip(active,saved):p.copy_(v)
            self.model.vlm.train(was_training)
        support=F.adaptive_avg_pool2d(labels['support'].float().flatten(0,1)[:,None],8).reshape(-1,3,8,8)>=.999
        return torch.stack(targets,1),torch.stack(valid,1)&support

    def configure_optimizers(self):
        groups=[{'params':[p for p in self.model.parameters() if p.requires_grad]+list(self.reason_head.parameters()),'lr':self.cfg['lr']['lora']},
                {'params':[p for p in self.planner.parameters() if p.requires_grad],'lr':self.cfg['lr']['planner']},
                {'params':self.world.parameters(),'lr':self.cfg['lr']['world']}]
        optimizer=torch.optim.AdamW(groups,weight_decay=.01,foreach=True)
        total=math.ceil(self.cfg['samples']/8)*self.cfg['epochs'];warm=max(1,round(.1*total))
        def schedule(step):
            if step<warm:return (step+1)/warm
            return .5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,total-warm))))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
        return {'optimizer':optimizer,'lr_scheduler':{'scheduler':scheduler,'interval':'step'}}

    def training_step(self,batch,batch_idx):
        if self.pending_rng is not None:restore_rng(self.pending_rng);self.pending_rng=None
        torch.cuda.synchronize();self.started=time.perf_counter();record=batch['record'];label=batch['labels']
        with math_attention():
            target,latent_support=self.teacher_targets(record,label)
            native,inputs=self.encode_path(record['actor_path'])
            context=self.planner.context(native,inputs)
            clean=normalize_controls(label['controls']);noise=torch.randn_like(clean);t=torch.rand(len(clean),device=self.device)
            mixed=(1-t[:,None,None])*noise+t[:,None,None]*clean
            velocity=self.planner.velocity(mixed,t,context)
            reason=self.reason_head(native['hidden'][:,-2])
            current=self.world.encode_current(native['hidden'])
            # Observed future occupancy is conditioned on the actually executed
            # expert action. It is not supervision for an unexecuted candidate.
            future=self.world(current,label['trajectory'],label['v0'])
            intent=intent_labels(label['controls'],label['trajectory'])
            loss,terms=sft_loss(reason,intent,velocity,noise,clean,mixed,t,label['v0'],label['trajectory'],
                                future['occupancy'],label['occupancy'],label['support'],
                                future['delta_latent'],current,target,latent_support)
            current_prob=self.world.decoder(current).squeeze(1).float().sigmoid()
            current_loss=focal_probability(current_prob,label['current_occupancy'],label['current_support'])
            loss=loss+.25*current_loss
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite supervised loss')
        values={k:float(v.detach()) for k,v in terms.items()};values.update(loss=float(loss.detach()),current_world=float(current_loss.detach()))
        self.epoch_terms.append(values);self.seen.append(f"{record['source']}:{record['route']}:{record['index']}")
        self.log_dict({f'train_{k}':v for k,v in values.items()},on_step=True,on_epoch=True,batch_size=1)
        return loss

    def backward(self,loss,*args,**kwargs):
        with math_attention():super().backward(loss,*args,**kwargs)

    def on_before_optimizer_step(self,optimizer):
        for module in (self.model,self.planner,self.world,self.reason_head):
            grads=[p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not all(g.dtype==torch.float32 and torch.isfinite(g).all() for g in grads):
                raise RuntimeError('Missing, non-FP32 or nonfinite training gradients')

    def optimizer_step(self,*args,**kwargs):
        super().optimizer_step(*args,**kwargs)
        # Run before checkpoint callbacks at batch end, so EMA and optimizer
        # always describe the same completed update in a saved checkpoint.
        with torch.no_grad():
            active=dict(self.model.vlm.named_parameters())
            for n,e in zip(self.lora_names,self.ema_lora):e.lerp_(active[n],.01)
            for e,p in zip(self.ema_world.parameters(),self.world.parameters()):e.lerp_(p,.01)
        self.last_ema_step=self.global_step+1

    def on_train_batch_end(self,outputs,batch,batch_idx):
        torch.cuda.synchronize();self.step_times.append(time.perf_counter()-self.started)

    def on_validation_epoch_start(self):self.validation_rows=[]

    def validation_step(self,batch,batch_idx):
        label=batch['labels'];record=batch['record']
        with math_attention():
            native,inputs=self.encode_path(record['actor_path']);context=self.planner.context(native,inputs)
            with torch.random.fork_rng(devices=[self.device.index]):
                torch.manual_seed(1907+batch_idx)
                pred=self.planner.sample(context,label['v0'],torch.randn(1,50,2,device=self.device),steps=10)
            trajectory=pred['trajectory'];current=self.world.encode_current(native['hidden'])
            probability=self.world(current,label['trajectory'],label['v0'])['occupancy']
            persistence=self.world.decoder(current).squeeze(1).float().sigmoid()[:,None].expand(-1,3,-1,-1)
            mask=label['support']&label['current_support'][:,None]
            error=(trajectory[...,:2]-label['trajectory']).norm(dim=-1)
            brier=masked_mean((probability-label['occupancy']).square(),mask)
            static=masked_mean((persistence-label['occupancy']).square(),mask)
            world_loss=sum(focal_probability(probability[:,h],label['occupancy'][:,h],label['support'][:,h]) for h in range(3))
            row={'ade_m':float(error.mean()),'fde_m':float(error[:,-1].mean()),'world_brier':float(brier),
                 'persistence_brier':float(static),'world_focal':float(world_loss),
                 'ay_abs_max':float(trajectory[...,5].abs().max()),
                 'jerk_abs_max':float((trajectory[...,4:6].diff(dim=1)/.1).norm(dim=-1).max()),
                 'speed_kmh':record['speed_kmh'],'source':record['source'],'route':record['route']}
        self.validation_rows.append(row)
        self.log('validation_ade',error.mean(),on_epoch=True,batch_size=1)

    def on_save_checkpoint(self,checkpoint):
        trainable={n for n,p in self.named_parameters() if p.requires_grad}
        checkpoint['state_dict']={k:v for k,v in checkpoint['state_dict'].items() if k in trainable or k.startswith(('ema_world.','ema_lora.'))}
        checkpoint['task_state']={'config':self.cfg,'manifest_identity':self.manifest_identity,'phase':self.phase,
                                  'base_identity':self.base_identity,'token_ids':self.token_ids,
                                  'token_rows':self.model.vlm.get_input_embeddings().weight[self.token_ids].detach().cpu(),
                                  'rng':rng_state(),'ema_step':self.last_ema_step,'seen_tail':self.seen[-16:],
                                  'resume_granularity':'optimizer-update boundary; Lightning loops and combined StatefulDataLoader state'}

    def on_load_checkpoint(self,checkpoint):
        task=checkpoint['task_state']
        if task['config']!=self.cfg or task['manifest_identity']!=self.manifest_identity or task['base_identity']!=self.base_identity:
            raise ValueError('Resume model/data/config identity differs')
        self.pending_rng=task['rng'];self.last_ema_step=task['ema_step'];self.seen=task.get('seen_tail',[])
        self.load_trainable(checkpoint)

    def load_trainable(self,checkpoint):
        saved=checkpoint['state_dict'];current=self.state_dict()
        expected={n for n,p in self.named_parameters() if p.requires_grad}
        if not expected<=saved.keys():raise ValueError('Missing trainable state')
        current.update(saved);checkpoint['state_dict']=current
        with torch.no_grad():self.model.vlm.get_input_embeddings().weight[self.token_ids]=checkpoint['task_state']['token_rows'].to(self.device)
