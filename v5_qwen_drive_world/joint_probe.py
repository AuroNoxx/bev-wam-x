"""Shared native RGB features -> Flow and World, for resource/gradient checks.

Official calibrated current-frame images plus explicitly synthetic zero state.
This fixture does not impersonate the missing historical highway SFT dataset.
"""
import torch
import time
from torch.utils.checkpoint import checkpoint
from qwen_drive_perception import QwenDrivePerception
from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor
from .runtime import ROOT, math_attention
from .components import load_perception, settings
from .verify_accumulation import AccumulationProbe
from .native_scene import encode_native_scene
from .world import ActionConditionedWorld
from .kinematics import integrate, denormalize_controls


class JointProbe(AccumulationProbe):
    def __init__(self, report, save, image_size):
        if image_size != 0:
            raise ValueError("Preserve official processor resolution")
        super().__init__(report, save, image_size)
        folder = sorted((ROOT / "runtime/qwen-drive-official/data/demo/perception").iterdir())[0]
        frame = PerceptionFrame(folder)
        processor = PerceptionProcessor(self.model.processor.tokenizer)
        inputs, self.meta = processor(frame, device="cuda")
        for key in ("history", "history_velocity", "history_acceleration", "nav_command", "ego_status"):
            inputs[key] = torch.zeros_like(self.inputs[key])
        self.inputs = inputs
        self.cameras = len(frame.cam_order)
        report.update(phase="loading_world", world_integrated=True, tokens=int(inputs["input_ids"].shape[1]),
                      images=self.cameras, grid=inputs["image_grid_thw"].tolist(), frame=frame.token,
                      scope="shared official calibrated single-frame RGB; synthetic zero ego/history/navigation; diagnostic losses only",
                      historical_input_tested=False, world_encoder_checkpointed=True)
        save()
        perception, _ = load_perception()
        self.world = ActionConditionedWorld(perception, image_size=processor.image_size).cuda()
        report['world_decoder_checkpoint_blocks'] = self.world.decoder_checkpoint_blocks
        report['world_conv3d_layout'] = 'NCDHW_depth_tiles4_complete_halo'
        del perception
        self.world_snapshot = self.world.delta[-1].weight.detach().clone()
        self.train()

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()
        optimizer.add_param_group(dict(params=[p for p in self.world.parameters() if p.requires_grad],
                                       lr=settings()["lr"]["world"]))
        return optimizer

    def _training_step(self, batch, batch_idx):
        self.micro_started = time.perf_counter()
        self.report.update(phase="joint_native_forward", current_microbatch=batch_idx+1)
        self.save()
        captured = {}
        def tap(module, args, output):
            captured["patches"] = args[0]
        handle = self.model.vlm.model.visual.merger.register_forward_hook(tap)
        try:
            with math_attention():
                native = encode_native_scene(self.model, self.inputs)
        finally:
            handle.remove()
        with math_attention():
            context = self.planner.context(native, self.inputs)
            noise = torch.randn(1,50,2,device=self.device)
            velocity = self.planner.velocity(noise, torch.tensor([.5],device=self.device), context)
            trajectory = integrate(denormalize_controls(noise+.5*velocity), torch.zeros(1,device=self.device))
            hidden = self.model.vlm.model.language_model.norm(native["hidden"])
            patches = self.model.vlm.model.visual.merger.norm(captured["patches"])
            vit = torch.stack(QwenDrivePerception._premerge_grids(patches, self.inputs["image_grid_thw"]))
            language = hidden[0][self.inputs["input_ids"][0] == self.model.config.vlm_config.image_token_id]
            grid = self.inputs["image_grid_thw"]
            assert (grid[:,1:] == grid[0,1:]).all()
            language = language.reshape(self.cameras,int(grid[0,1])//2,int(grid[0,2])//2,-1)
            self.report["phase"] = "joint_world_encoder"
            self.save()
            current = checkpoint(self.world.encode_current, vit, language, [self.meta], use_reentrant=False,
                                 context_fn=lambda: (math_attention(), math_attention()))
            self.report["phase"] = "joint_world_future"
            self.save()
            prediction = self.world(current, trajectory[...,:2], torch.zeros(1,device=self.device))
            assert prediction["occupancy"].shape == (1,3,128,128)
            # Current-latent scalar probes encoder gradients at zero-initialized
            # residual startup. These scalars are NOT supervised World targets.
            loss = (velocity.float().square().mean() + prediction["occupancy"].float().mean()
                    + 1e-4*current.float().square().mean())
        assert torch.isfinite(loss)
        self.report.update(phase="joint_backward", last_loss=float(loss.detach()))
        self.save()
        return loss

    def on_before_optimizer_step(self, optimizer):
        super().on_before_optimizer_step(optimizer)
        grads = [p.grad for p in self.world.parameters() if p.grad is not None]
        assert grads and all(g.dtype == torch.float32 and torch.isfinite(g).all() for g in grads)
        value = float(torch.stack([g.norm() for g in grads]).norm())
        assert value > 0
        self.report["gradient_norms"]["world"] = value
        self.save()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        delta = float((self.world.delta[-1].weight.detach()-self.world_snapshot).abs().max())
        assert (delta == 0) if self.global_step == 0 else (delta > 0)
        self.report["microbatches"][-1]["parameter_max_change"]["world"] = delta
        self.save()
