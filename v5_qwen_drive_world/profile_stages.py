"""Ten bounded diagnostic microsteps; no weight/checkpoint files are changed.

The native KV boundary is detached ONLY for measurement. Its exact gradients
are then passed back to the original tensors, preserving the chain rule while
separating Flow and VLM backward wall times. Not a production training loop.
"""
from .runtime import ROOT,configure,math_attention
configure()
import json,time,torch
import pytorch_lightning as pl
from torchdata.stateful_dataloader import StatefulDataLoader
from .verify_accumulation import AccumulationProbe
from .native_scene import encode_native_scene


class ProfileFinished(Exception): pass


def stamp():
    torch.cuda.synchronize()
    return time.perf_counter()


class StageProfile(AccumulationProbe):
    def __init__(self,report,save):
        cfg=json.loads((ROOT/'artifacts/v5-qwen-drive-world/config-before-lightweight.json').read_text())
        super().__init__(report,save,0,cfg=cfg)
        self.timing=None
        def before(label):
            def hook(*args):
                if self.timing is not None: self.starts[label]=stamp()
            return hook
        def after(label):
            def hook(*args):
                if self.timing is not None: self.timing[label]=stamp()-self.starts[label]
            return hook
        for label,module in [('vit_forward_s',self.model.vlm.model.visual),
                             ('llm_forward_s',self.model.vlm.model.language_model)]:
            module.register_forward_pre_hook(before(label))
            module.register_forward_hook(after(label))

    def training_step(self,batch,batch_idx):
        self.micro_started=stamp();self.starts={};self.timing={'microbatch':batch_idx+1}
        with math_attention():
            native=encode_native_scene(self.model,self.inputs)
            self.originals=[t for pair in native['scene_cache'] for t in pair]
            self.leaves=[t.detach().requires_grad_() for t in self.originals]
            context_native={**native,'scene_cache':list(zip(self.leaves[::2],self.leaves[1::2]))}
            start=stamp()
            context=self.planner.context(context_native,self.inputs)
            velocity=self.planner.velocity(torch.randn(1,50,2,device='cuda'),
                                           torch.tensor([.5],device='cuda'),context)
            loss=velocity.float().square().mean()
            self.timing['flow_forward_s']=stamp()-start
        return loss

    def backward(self,loss,*args,**kwargs):
        with math_attention():
            start=stamp();loss.backward()
            self.timing['flow_backward_s']=stamp()-start
            pairs=[(x,y.grad) for x,y in zip(self.originals,self.leaves) if y.grad is not None]
            start=stamp()
            torch.autograd.backward([x for x,g in pairs],[g for x,g in pairs])
            self.timing['llm_backward_recompute_s']=stamp()-start
        self.originals=self.leaves=None

    def on_before_optimizer_step(self,optimizer):
        super().on_before_optimizer_step(optimizer)
        self.optimizer_started=stamp()

    def on_train_batch_end(self,outputs,batch,batch_idx):
        if (batch_idx+1)%8==0:
            self.timing['optimizer_and_clip_s']=stamp()-self.optimizer_started
        self.timing['total_micro_s']=stamp()-self.micro_started
        self.report['timings'].append(self.timing)
        self.timing=None
        self.report['optimizer_updates']=self.global_step
        self.save()
        if batch_idx==9: raise ProfileFinished()


def main():
    started=time.perf_counter()
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.85)
    pl.seed_everything(71,workers=True)
    report={'phase':'loading','formal_training':False,'timings':[],
            'method':'synchronized wall time; exact KV gradient handoff separates backward branches; discarded in-memory optimizer update'}
    path=ROOT/'artifacts/v5-qwen-drive-world/profile-before-lightweight.json'
    def save():path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    task=StageProfile(report,save)
    trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',accumulate_grad_batches=8,
                       max_epochs=1,gradient_clip_val=1.,logger=False,enable_checkpointing=False,
                       enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0)
    try:trainer.fit(task,train_dataloaders=StatefulDataLoader(list(range(16)),batch_size=1,num_workers=0))
    except ProfileFinished:pass
    report.update(phase='passed',seconds_including_loading=time.perf_counter()-started,
                  peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    rows=report['timings'][1:]
    keys=['vit_forward_s','llm_forward_s','flow_forward_s','flow_backward_s','llm_backward_recompute_s','total_micro_s']
    report['warm_mean_s']={key:sum(r[key] for r in rows)/len(rows) for key in keys}
    save();print(json.dumps(report,indent=2))


if __name__=='__main__':main()
