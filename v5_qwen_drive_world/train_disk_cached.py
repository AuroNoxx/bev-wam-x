"""Resume the same Lightning SFT with a persistent frozen-prefix cache."""
from .runtime import ROOT,configure
configure()
import json,sys,time,os,traceback
import torch
from . import train_supervised as training
from .sft_task import SupervisedTask
from .real_dataset import inputs_for
from .disk_prefix import DiskPrefixCache,identity_for


class DiskCachedTask(SupervisedTask):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.prefix=DiskPrefixCache(identity_for(self))

    def on_load_checkpoint(self,checkpoint):
        super().on_load_checkpoint(checkpoint)
        if identity_for(self)!=self.prefix.identity:raise ValueError('Restored frozen embedding identity changed')

    def encode_path(self,path,planner=True):
        entry=self.prefix.lookup(path,planner)
        if entry is None:inputs=inputs_for(self.model.processor,path,self.device,self.token_ids)
        else:inputs={k:v.to(self.device) if isinstance(v,torch.Tensor) else v for k,v in entry['inputs'].items()}
        return self.prefix.encode(self.model,inputs,path,planner),inputs


class DiskReport(training.Report):
    def write(self,trainer,task,state):
        super().write(trainer,task,state)
        row=json.loads(self.path.read_text());row['disk_cache']=task.prefix.statistics()
        self.path.write_text(json.dumps(row,indent=2),encoding='utf-8')
    def on_train_start(self,trainer,task):
        states=trainer.optimizers[0].state.values()
        assert states and all(int(s['step'])==trainer.global_step and s['exp_avg'].dtype==torch.float32 for s in states)
        self.resume_step=trainer.global_step;self.resume_seen=len(task.seen)
        row={'step':trainer.global_step,'ema_step':task.last_ema_step,'epoch':trainer.current_epoch,
             'optimizer_fp32':True,'config_unchanged':True,'namespace':task.prefix.namespace}
        (self.path.parent/'disk-resume.json').write_text(json.dumps(row,indent=2))
        path=ROOT/'artifacts/v5-qwen-drive-world/two-stage-run.json'
        run=json.loads(path.read_text(encoding='utf-8-sig'))
        run.update(state='running',active_phase=task.phase,pid=os.getpid(),child_pid=os.getpid(),
                   entrypoint='v5_qwen_drive_world.train_disk_cached',disk_cache_enabled=True)
        path.write_text(json.dumps(run,indent=2))
    def on_train_batch_end(self,trainer,task,outputs,batch,batch_idx):
        super().on_train_batch_end(trainer,task,outputs,batch,batch_idx)
        proof=self.path.parent/'disk-resume-expected.json'
        if proof.exists() and len(task.seen)-self.resume_seen==8:
            expected=json.loads(proof.read_text())
            assert self.resume_step==expected['step'] and task.seen[-8:]==expected['next_sample_ids']
            result=json.loads((self.path.parent/'disk-resume.json').read_text())
            result.update(next_eight_samples_verified=True,continued_step=trainer.global_step)
            (self.path.parent/'disk-resume.json').write_text(json.dumps(result,indent=2))
    def on_fit_end(self,trainer,task):
        super().on_fit_end(trainer,task)
        if trainer.global_step>=self.expected_updates(task):
            path=ROOT/'artifacts/v5-qwen-drive-world/two-stage-run.json'
            run=json.loads(path.read_text());run['state']='completed';run['active_phase']=None
            run['stages'].append(json.loads(self.path.read_text()));run['finished_unix']=time.time()
            path.write_text(json.dumps(run,indent=2))
        else:
            path=ROOT/'artifacts/v5-qwen-drive-world/two-stage-run.json'
            run=json.loads(path.read_text());run.update(state='incomplete',active_phase=task.phase)
            path.write_text(json.dumps(run,indent=2))


def main():
    # Reuse the exact existing Trainer, loss, optimizer, sampler and checkpoint
    # callbacks; replace only the task's frozen-feature storage implementation.
    training.SupervisedTask=DiskCachedTask;training.Report=DiskReport
    try:training.main()
    except Exception:
        path=ROOT/'artifacts/v5-qwen-drive-world/two-stage-run.json'
        run=json.loads(path.read_text(encoding='utf-8-sig'));run.update(state='failed',error=traceback.format_exc())
        path.write_text(json.dumps(run,indent=2));raise


if __name__=='__main__':main()
