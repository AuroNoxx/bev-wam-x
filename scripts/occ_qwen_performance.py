"""Opt-in performance plumbing; does not change native input/model/loss contracts."""
import copy,json
import torch
from torch import nn
from train_occ_qwen_native import NativeTask

def channels_last_input(module,args):
    return (args[0].contiguous(memory_format=torch.channels_last),)

def enable_channels_last(model):
    for encoder in [model.encoder,model.teacher]:
        torch.nn.utils.convert_conv2d_weight_memory_format(encoder,torch.channels_last)
        # Apply at each CNN boundary, including target teacher, without a state key change.
        encoder.frame.conv.register_forward_pre_hook(channels_last_input)

class OptimizedNativeTask(NativeTask):
    def __init__(self,*args,performance=None,**kwargs):
        super().__init__(*args,**kwargs);self.performance=performance or {}
        if self.performance.get('channels_last'):enable_channels_last(self.model)
    def configure_gradient_clipping(self,optimizer,gradient_clip_val=None,gradient_clip_algorithm=None):
        if getattr(self.trainer.precision_plugin,'scaler',None) is not None:raise RuntimeError('Validated fused clipping requires BF16/no GradScaler')
        torch.nn.utils.clip_grad_norm_([p for p in self.parameters() if p.requires_grad],gradient_clip_val,error_if_nonfinite=True)
    def on_train_start(self):
        if self.performance.get('fused_adamw'):
            for optimizer in self.trainer.optimizers:
                # Loaded old checkpoint flags must not silently override requested execution mode.
                optimizer.defaults.update(fused=True,foreach=False)
                optimizer._step_supports_amp_scaling=True
                for group in optimizer.param_groups:
                    group.update(fused=True,foreach=False)
                    for p in group['params']:
                        state=optimizer.state.get(p,{})
                        if isinstance(state.get('step'),torch.Tensor):state['step']=state['step'].to(p.device)
                        for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                            if key in state:state[key]=torch.empty_like(p,memory_format=torch.preserve_format).copy_(state[key])
        counters=sorted({int(state['step'].item()) for optimizer in self.trainer.optimizers for state in optimizer.state.values() if 'step' in state})
        schedulers=[sc.scheduler.last_epoch for sc in self.trainer.lr_scheduler_configs]
        if (counters!=[self.trainer.global_step] and not (self.trainer.global_step==0 and counters==[])) or schedulers!=[self.trainer.global_step]:
            raise RuntimeError('Optimizer/scheduler step mismatch after restore: '+str((counters,schedulers,self.trainer.global_step)))
        print('RESUME_STATE '+json.dumps({'global_step':self.trainer.global_step,'optimizer_steps':counters,'scheduler_last_epoch':schedulers,'learning_rates':[g['lr'] for o in self.trainer.optimizers for g in o.param_groups],'performance':self.performance}),flush=True)
    def on_save_checkpoint(self,checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['task_state']['performance']=self.performance
    def on_load_checkpoint(self,checkpoint):
        # NativeTask still enforces exact architecture, data and optimizer hyperparameter settings.
        super().on_load_checkpoint(checkpoint)
        self.previous_performance=checkpoint['task_state'].get('performance',{})

