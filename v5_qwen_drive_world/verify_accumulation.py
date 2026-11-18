"""Bounded Lightning update on real official RGB; engineering fixture, not SFT.

Repeats one scene for one or two updates to verify accumulation, precision,
gradients and optimizer allocation. No convergence claim or checkpoint save.
"""
from .runtime import ROOT, configure, math_attention
configure()
import argparse
import gc
import json
import time
import traceback
from contextlib import nullcontext
import torch
import pytorch_lightning as pl
from torchdata.stateful_dataloader import StatefulDataLoader
from qwen_drive.benchmarks import read_scene_file
from .components import load_planning, settings
from .native_scene import encode_native_scene
from .planning import ControlFlowPlanner


class AccumulationProbe(pl.LightningModule):
    def __init__(self, report, save, image_size, cfg=None):
        super().__init__()
        self.report, self.save = report, save
        self.cfg = cfg or settings()
        self.model = load_planning(self.cfg)
        self.model.train()
        linear=[m for m in self.model.modules() if m.__class__.__name__ == 'Qwen3_5GatedDeltaNet']
        self.report['acceleration'] = dict(linear_layers=len(linear),
            delta_rule=sorted({m.chunk_gated_delta_rule.__module__ for m in linear}),
            conv=sorted({m.causal_conv1d_fn.__module__ for m in linear}),
            norm=sorted({type(m.norm).__name__ for m in linear}),
            optimizer=self.cfg['optimizer'])
        assert len(linear)==24 and all(m.chunk_gated_delta_rule.__module__.startswith('fla.') for m in linear)
        self.report['phase']='pretrained_causal_gradient';self.save()
        text_model=self.model.vlm.model.language_model
        embeds=text_model.embed_tokens(torch.arange(100,112,device='cuda')[None]).detach().requires_grad_()
        with math_attention(),torch.autocast('cuda',dtype=torch.bfloat16):
            result=text_model(inputs_embeds=embeds,use_cache=False).last_hidden_state
            result[:,:6,:128].float().square().sum().backward()
        suffix=float(embeds.grad[:,6:].abs().max())
        assert suffix==0,suffix
        self.report['pretrained_suffix_gradient_max']=suffix
        self.save()
        self.model.zero_grad(set_to_none=True)
        del embeds,result,text_model
        expert = self.model.planning_expert
        del self.model.planning_expert
        self.planner = ControlFlowPlanner(expert, self.model.config.trajectory_scale).cuda()
        demo = ROOT / "runtime/qwen-drive-official/data/demo"
        scene = next(read_scene_file(demo / "planning_scenes.jsonl", image_root=demo)).scene
        if image_size:
            for frames in scene.views.values():
                for frame in frames:
                    frame.target_size = (image_size, image_size)
        self.inputs = self.model.processor(scene, with_reasoning=True, device="cuda")
        self.report.update(tokens=int(self.inputs["input_ids"].shape[1]),
                           images=int(self.inputs["image_grid_thw"].shape[0]),
                           grid=self.inputs["image_grid_thw"].tolist())
        self.snapshots = {"planner": self.planner.expert.out_proj.weight.detach().clone()}
        self.lora = next(p for n, p in self.model.vlm.named_parameters() if "lora_B" in n)
        self.snapshots["lora"] = self.lora.detach().clone()
        # from_pretrained defaults to eval; Lightning does not recursively reset
        # these freshly attached children before its first fit epoch.
        self.train()
        assert self.model.vlm.model.language_model.training
        assert all(layer.training for layer in self.planner.expert.layers)

    def configure_optimizers(self):
        cfg = self.cfg
        groups = [dict(params=[p for p in self.model.parameters() if p.requires_grad], lr=cfg["lr"]["lora"]),
                  dict(params=[p for p in self.planner.parameters() if p.requires_grad], lr=cfg["lr"]["planner"])]
        return torch.optim.AdamW(groups, weight_decay=cfg["optimizer"]["weight_decay"],
                                foreach=cfg['optimizer']['foreach'],fused=cfg['optimizer'].get('fused',False))

    def training_step(self, batch, batch_idx):
        context = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                   if self.report.get('activation_offload') else nullcontext())
        # PyTorch moves only tensors saved for backward. Model/optimizer FP32
        # masters stay on their original device; this does not alter RGB inputs.
        with context:
            return self._training_step(batch,batch_idx)

    def _training_step(self, batch, batch_idx):
        self.micro_started = time.perf_counter()
        self.report.update(phase="micro_forward", current_microbatch=batch_idx + 1)
        self.save()
        with math_attention():
            native = encode_native_scene(self.model, self.inputs)
            context = self.planner.context(native, self.inputs)
            noise = torch.randn(1, 50, 2, device=self.device)
            velocity = self.planner.velocity(noise, torch.tensor([.5], device=self.device), context)
            # Diagnostic scalar only: no fabricated expert/future supervision.
            loss = velocity.float().square().mean()
        assert torch.isfinite(loss)
        self.report.update(phase="micro_backward", last_loss=float(loss.detach()))
        self.save()
        return loss

    def backward(self, loss, *args, **kwargs):
        from .diagnose_memory import TraceAllocationFailure
        diagnose = TraceAllocationFailure(self.report,self.save) if self.report.get('diagnose_oom') else nullcontext()
        with math_attention(), diagnose:
            super().backward(loss, *args, **kwargs)

    def on_before_optimizer_step(self, optimizer):
        norms = {}
        for label, module in (("lora", self.model), ("planner", self.planner)):
            grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
            assert grads and all(g.dtype == torch.float32 and torch.isfinite(g).all() for g in grads)
            norms[label] = float(torch.stack([g.norm() for g in grads]).norm())
            assert norms[label] > 0
        self.report.update(phase="optimizer_step", gradient_norms=norms)
        self.save()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        delta = {"lora": float((self.lora.detach()-self.snapshots["lora"]).abs().max()),
                 "planner": float((self.planner.expert.out_proj.weight.detach()-self.snapshots["planner"]).abs().max())}
        expected = (batch_idx + 1) // 8
        assert self.global_step == expected
        if expected == 0:
            assert all(value == 0 for value in delta.values())
        else:
            assert all(value > 0 for value in delta.values())
        row = dict(microbatch=batch_idx+1, optimizer_updates=self.global_step,
                   loss=float(outputs["loss"]), parameter_max_change=delta,
                   seconds=time.perf_counter()-self.micro_started,
                   allocated_GiB=torch.cuda.memory_allocated()/2**30,
                   peak_GiB=torch.cuda.max_memory_allocated()/2**30)
        self.report["microbatches"].append(row)
        self.report.update(phase="micro_complete", optimizer_updates=self.global_step)
        self.save()
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=0, help="0 preserves official demo processor sizes; image resizing is not authorized")
    parser.add_argument("--with-world", action="store_true")
    parser.add_argument("--allocator-fraction", type=float, default=.85)
    parser.add_argument("--updates", type=int, default=2, choices=(1,2))
    parser.add_argument("--diagnose-oom",action="store_true")
    parser.add_argument("--activation-offload",action="store_true")
    args = parser.parse_args()
    if args.image_size != 0:
        parser.error("Resolution reduction is not authorized; keep the original processor sizes")
    if not 0 < args.allocator_fraction <= .85:
        parser.error("The bounded verification supports allocator fractions up to .85")
    torch.set_num_threads(6)
    pl.seed_everything(71, workers=True)
    torch.cuda.set_per_process_memory_fraction(args.allocator_fraction, 0)
    cfg = settings()
    assert (cfg["batch_size"], cfg["accumulate_grad_batches"], cfg["effective_batch_size"]) == (1,8,8)
    tag = str(args.image_size) if args.image_size else "native"
    if args.with_world:
        tag += "-joint"
    tag += f"-cap{round(args.allocator_fraction*100)}"
    tag += f"-updates{args.updates}"
    if args.diagnose_oom:
        tag += "-trace"
    if args.activation_offload:
        tag += "-offload"
    tag += '-fla-foreach-tiled'
    if args.with_world:
        tag += '-blocks-convtiles-encoder'
    path = ROOT / f"artifacts/v5-qwen-drive-world/accumulation-{tag}.json"
    report = dict(phase="loading", microbatch_size=1, accumulate_grad_batches=8, effective_batch_size=8,
                  allocator_fraction=args.allocator_fraction, optimizer_updates=0, microbatches=[], image_size=args.image_size,
                  scope="official planning demo repeated 8 times; VLM LoRA + control Flow; diagnostic loss only",
                  formal_training=False, world_integrated=False, checkpoint_resume_tested=False,
                  requested_optimizer_updates=args.updates)
    report['diagnose_oom'] = args.diagnose_oom
    report['activation_offload'] = args.activation_offload
    def save():
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    task = trainer = None
    started = time.perf_counter()
    save()
    try:
        if args.with_world:
            from .joint_probe import JointProbe
            task = JointProbe(report, save, args.image_size)
        else:
            task = AccumulationProbe(report, save, args.image_size)
        report["trainable_parameters"] = sum(p.numel() for p in task.parameters() if p.requires_grad)
        assert all(p.dtype == torch.float32 for p in task.parameters() if p.requires_grad)
        torch.cuda.reset_peak_memory_stats()
        trainer = pl.Trainer(accelerator="gpu", devices=1, precision="bf16-mixed",
                             accumulate_grad_batches=8, max_steps=args.updates, max_epochs=1,
                             gradient_clip_val=1., logger=False, enable_checkpointing=False,
                             enable_progress_bar=False, enable_model_summary=False,
                             num_sanity_val_steps=0, default_root_dir=str(ROOT/"logs/v5-qwen-drive-world/verification"))
        loader = StatefulDataLoader(list(range(8*args.updates)), batch_size=1, num_workers=0)
        fit_started = time.perf_counter()
        trainer.fit(task, train_dataloaders=loader)
        report["fit_seconds"] = time.perf_counter()-fit_started
        optimizer = trainer.optimizers[0]
        states = list(optimizer.state.values())
        assert states and all(s["exp_avg"].dtype == s["exp_avg_sq"].dtype == torch.float32 for s in states)
        assert all(int(s["step"]) == args.updates for s in states)
        assert len(report["microbatches"]) == 8*args.updates and trainer.global_step == args.updates
        report.update(phase="passed", fp32_optimizer_states=True, optimizer_state_count=len(states))
    except Exception:
        report.update(failed_at_phase=report["phase"], phase="failed", error=traceback.format_exc())
        raise
    finally:
        torch.cuda.synchronize()
        report.update(seconds=time.perf_counter()-started,
                      peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
        save()
        print(json.dumps({k:v for k,v in report.items() if k != "microbatches"}, indent=2), flush=True)
        del task, trainer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
