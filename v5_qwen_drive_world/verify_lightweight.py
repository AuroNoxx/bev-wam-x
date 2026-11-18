"""Actual pretrained native-resolution RGB -> compressed VLM -> Flow + 2D World.

Lightning owns accumulation and updates. Repeated-scene diagnostic loss only;
this is not supervised training or a driving-quality evaluation.
"""
from .runtime import ROOT,configure,math_attention
configure()
import argparse,gc,json,time,traceback
import torch
import pytorch_lightning as pl
from torchdata.stateful_dataloader import StatefulDataLoader
from qwen_drive.benchmarks import read_scene_file
from .components import load_planning,load_perception
from .lightweight_input import planning_inputs,repeat_scene
from .native_scene import encode_native_scene
from .planning import ControlFlowPlanner
from .world2d import ActionConditionedWorld2D
from .kinematics import integrate,denormalize_controls


class LightweightProbe(pl.LightningModule):
    def __init__(self,cfg,batch,report,save):
        super().__init__();self.cfg,self.batch,self.report,self.save=cfg,batch,report,save
        self.model=load_planning(cfg)
        expert=self.model.planning_expert;del self.model.planning_expert
        self.planner=ControlFlowPlanner(expert,self.model.config.trajectory_scale).cuda()
        perception,_=load_perception(cfg,device='cpu')
        self.world=ActionConditionedWorld2D(perception).cuda();del perception;gc.collect()
        demo=ROOT/'runtime/qwen-drive-official/data/demo'
        scene=next(read_scene_file(demo/'planning_scenes.jsonl',image_root=demo)).scene
        inputs=planning_inputs(self.model.processor,scene,limit=cfg['visual_tokens']['limit'])
        self.inputs=repeat_scene(inputs,batch)
        self.lora=next(p for n,p in self.model.vlm.named_parameters() if 'lora_B' in n)
        self.targets={'lora':self.lora,'planner':self.planner.expert.out_proj.weight,'world':self.world.delta[-1].weight}
        self.snapshots={k:v.detach().clone() for k,v in self.targets.items()}
        def check_prefix(module,args,result):
            assert not result.requires_grad,'Frozen prefix unexpectedly retains a gradient graph'
            self.report['prefix_output_requires_grad']=False
        self.model.vlm.model.language_model.layers[23].register_forward_hook(check_prefix)
        layers=self.model.vlm.model.language_model.layers
        assert not any(p.requires_grad for layer in layers[:24] for p in layer.parameters())
        assert all(not layer.gradient_checkpointing for layer in layers[:24])
        assert all(layer.gradient_checkpointing for layer in layers[24:])
        assert all(any('lora_' in n and p.requires_grad for n,p in layer.named_parameters()) for layer in layers[24:])
        assert not any(isinstance(m,torch.nn.Conv3d) for m in self.world.modules())
        assert all(p.dtype==torch.float32 for p in self.parameters() if p.requires_grad)
        report.update(input_audit=inputs['input_audit'],world_migration=self.world.migration,
                      layers=32,frozen_prefix_layers=24,lora_layers=list(range(24,32)),
                      trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                      world_parameters=sum(p.numel() for p in self.world.parameters()),
                      conv3d_in_world=0,benchmark=torch.backends.cudnn.benchmark)
        self.train();save()

    def configure_optimizers(self):
        groups=[{'params':[p for p in module.parameters() if p.requires_grad],'lr':self.cfg['lr'][name]}
                for name,module in [('lora',self.model),('planner',self.planner),('world',self.world)]]
        return torch.optim.AdamW(groups,weight_decay=.01,foreach=True,fused=False)

    def training_step(self,batch,batch_idx):
        torch.cuda.synchronize();self.started=time.perf_counter()
        self.report.update(phase='native_forward',current_microbatch=batch_idx+1);self.save()
        with math_attention():
            native=encode_native_scene(self.model,self.inputs)
            assert len(native['scene_cache'])==8
            assert all(not t.requires_grad for pair in native['scene_cache'][:6] for t in pair)
            assert all(t.requires_grad for pair in native['scene_cache'][6:] for t in pair)
            context=self.planner.context(native,self.inputs)
            noise=torch.randn(self.batch,50,2,device=self.device)
            velocity=self.planner.velocity(noise,torch.full((self.batch,),.5,device=self.device),context)
            v0=self.inputs['ego_status'][:,:2].float().norm(dim=-1)
            trajectory=integrate(denormalize_controls(noise+.5*velocity),v0)
            self.report['phase']='world2d_forward';self.save()
            current=self.world.encode_current(native['hidden'])
            prediction=self.world(current,trajectory[...,:2],v0)
            assert prediction['occupancy'].shape==(self.batch,3,128,128)
            loss=velocity.float().square().mean()+prediction['occupancy'].mean()+1e-4*current.float().square().mean()
        assert torch.isfinite(loss)
        self.report.update(phase='joint_backward',loss=float(loss.detach()));self.save()
        return loss

    def backward(self,loss,*args,**kwargs):
        with math_attention():super().backward(loss,*args,**kwargs)

    def on_before_optimizer_step(self,optimizer):
        norms={}
        for name,module in [('lora',self.model),('planner',self.planner),('world',self.world)]:
            grads=[p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
            assert grads and all(g.dtype==torch.float32 and torch.isfinite(g).all() for g in grads)
            norms[name]=float(torch.stack([g.norm() for g in grads]).norm());assert norms[name]>0
        if self.global_step>=1:
            action_grads=[p.grad for p in self.world.action_condition.parameters() if p.grad is not None]
            assert action_grads and any(g.abs().max()>0 for g in action_grads)
        self.report.update(phase='optimizer_step',gradient_norms=norms);self.save()

    def on_train_batch_end(self,outputs,batch,batch_idx):
        torch.cuda.synchronize()
        delta={k:float((v.detach()-self.snapshots[k]).abs().max()) for k,v in self.targets.items()}
        expected=(batch_idx+1)//(8//self.batch)
        assert self.global_step==expected
        assert all(v==0 for v in delta.values()) if expected==0 else all(v>0 for v in delta.values())
        row={'microbatch':batch_idx+1,'batch':self.batch,'seconds':time.perf_counter()-self.started,
             'optimizer_updates':self.global_step,'max_parameter_changes':delta,
             'allocated_GiB':torch.cuda.memory_allocated()/2**30,'peak_GiB':torch.cuda.max_memory_allocated()/2**30}
        self.report['microbatches'].append(row)
        self.report.update(phase='micro_complete',optimizer_updates=self.global_step);self.save()
        print(json.dumps(row),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--microbatch',type=int,choices=[1,2,4],default=4)
    args=parser.parse_args();cfg=json.loads((ROOT/'configs/v5-qwen-drive-world/lightweight.json').read_text())
    pl.seed_everything(71,workers=True);torch.set_num_threads(6)
    torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.8)
    report={'phase':'loading','formal_training':False,'scope':'official planning demo repeated; all retained pixels processed independently per batch member; diagnostic losses only',
            'microbatch':args.microbatch,'accumulation':8//args.microbatch,'effective_batch':8,
            'microbatches':[],'optimizer_updates':0,'resolution_reduced':False}
    path=ROOT/f'artifacts/v5-qwen-drive-world/lightweight-mb{args.microbatch}.json'
    def save():path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save();started=time.perf_counter()
    try:
        task=LightweightProbe(cfg,args.microbatch,report,save)
        trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',accumulate_grad_batches=8//args.microbatch,
                           max_epochs=1,max_steps=2,gradient_clip_val=1.,logger=False,enable_checkpointing=False,
                           enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0)
        torch.cuda.reset_peak_memory_stats();fit_started=time.perf_counter()
        trainer.fit(task,train_dataloaders=StatefulDataLoader(list(range(16)),batch_size=args.microbatch,num_workers=0))
        report['fit_seconds']=time.perf_counter()-fit_started
        states=list(trainer.optimizers[0].state.values())
        assert states and all(s['exp_avg'].dtype==s['exp_avg_sq'].dtype==torch.float32 and int(s['step'])==2 for s in states)
        assert trainer.global_step==2 and len(report['microbatches'])==16//args.microbatch
        assert torch.cuda.max_memory_allocated()/2**30<16
        rows=report['microbatches'][8//args.microbatch:]
        report.update(phase='passed',fp32_optimizer_states=True,optimizer_state_count=len(states),
                      warm_seconds_per_sample=sum(r['seconds'] for r in rows)/8)
    except Exception:
        report.update(failed_at=report['phase'],phase='failed',error=traceback.format_exc());raise
    finally:
        report.update(seconds_including_loading=time.perf_counter()-started,
                      peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
        save();print(json.dumps({k:v for k,v in report.items() if k!='microbatches'},indent=2),flush=True)


if __name__=='__main__':main()
