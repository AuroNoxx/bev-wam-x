"""Complete the missing subset pass without changing the 750-update schedule."""
from .runtime import ROOT,configure
configure()
import json,math,sys
import torch
from . import train_supervised as training
from . import resume_disk_cached as resumed
from .epoch_resume import boundary_sampler_state,restore_boundary_sampler


def main():
    if '--resume' not in sys.argv or '--phase' not in sys.argv or sys.argv[sys.argv.index('--phase')+1]!='subset2000':
        raise ValueError('This entry only completes the existing subset2000 run with --resume')
    folder=ROOT/'artifacts/v5-qwen-drive-world/subset2000'
    checkpoint=torch.load(folder/'latest.ckpt',map_location='cpu',weights_only=False,mmap=True)
    cfg=checkpoint['task_state']['config'];step=checkpoint['global_step']
    assert cfg['samples']==2000 and cfg['epochs']==3 and cfg['effective_batch_size']==8
    assert checkpoint['task_state']['ema_step']==step
    target=750
    if step>=target:raise ValueError('750 updates already complete; do not restart training')
    state=boundary_sampler_state(checkpoint)
    saved=checkpoint['loops']['fit_loop']['state_dict']['combined_loader'][0]['_index_sampler_state']
    assert saved['total_drawn']==step*8,'Only committed optimizer boundaries may resume'
    # Framework epoch 2 was empty. Allow that historical epoch to remain in
    # the audit, while stopping at exactly the original number of updates.
    epoch_limit=max(4,checkpoint['epoch']+1+math.ceil((target-step)/250))
    (folder/'completion-repair.json').write_text(json.dumps({
        'resume_step':step,'target_updates':target,'framework_max_epochs':epoch_limit,
        'original_schedule_updates':750,'completed_sampler_draws':saved['total_drawn'],
        'boundary_sampler_restored':state is not None,'config_unchanged':True},indent=2))
    del checkpoint
    base_data=training.RealData;original_trainer=training.pl.Trainer

    class ContinuedData(base_data):
        def train_dataloader(self):
            return restore_boundary_sampler(super().train_dataloader(),state)

    class CompletionTrainer(original_trainer):
        def __init__(self,*args,**kwargs):
            # Keep epoch-based Lightning restart semantics of the source run.
            # Switching max_steps on at resume changes its epoch-end recovery.
            kwargs.update(max_epochs=epoch_limit,max_steps=-1)
            kwargs['callbacks'].append(StopAtTarget())
            super().__init__(*args,**kwargs)

    class StopAtTarget(training.pl.Callback):
        def on_train_batch_end(self,trainer,task,outputs,batch,batch_idx):
            if trainer.global_step>=target:trainer.should_stop=True

    training.RealData=ContinuedData
    training.pl.Trainer=CompletionTrainer
    resumed.main()


if __name__=='__main__':main()
