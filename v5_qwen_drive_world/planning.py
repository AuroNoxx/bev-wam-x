"""Control-velocity extension of the loaded official planner-rl expert."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .runtime import math_attention
from .kinematics import denormalize_controls, integrate, euler_flow
from qwen_drive.trajectory import normalize_history


class CheckpointedExpertLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args):
        if not self.training:
            return self.layer(*args)
        return checkpoint(self.layer, *args, use_reentrant=False,
                          context_fn=lambda: (math_attention(), math_attention()))


class ControlFlowPlanner(nn.Module):
    def __init__(self, expert, trajectory_scale, dt=.1):
        super().__init__()
        self.expert = expert
        self.dt = dt
        self.register_buffer("trajectory_scale", torch.tensor(trajectory_scale, dtype=torch.float32))
        # Native 3-channel Fourier and trajectory inputs are retained; the third
        # channel is zero for 2D acceleration controls. Their new meaning must be
        # learned in SFT; no claim of pretrained control semantics is made.
        device = expert.out_proj.weight.device
        self.expert.out_proj = nn.Linear(expert.config.hidden_size, 2, device=device, dtype=torch.float32)
        self.expert.requires_grad_(False)
        for name, parameter in self.expert.named_parameters():
            if "adaln_modulation" in name or name.startswith(("out_proj.", "trajectory_proj.", "fourier_encoder.")):
                parameter.requires_grad_(True)
                parameter.data = parameter.data.float()
        self.expert.layers = nn.ModuleList(CheckpointedExpertLayer(layer) for layer in self.expert.layers)
        # The module's dtype property comes from out_proj; all trainable weights
        # are FP32 and the operation context, not weight casting, selects BF16.

    def context(self, native, inputs):
        history = normalize_history(inputs["history"].float(), self.trajectory_scale)
        queries = self.expert.encode_history(history, inputs["nav_command"],
                                             inputs["history_velocity"].float(), inputs["history_acceleration"].float())
        return dict(history_queries=queries, scene_cache=native["scene_cache"],
                    position_anchor=native["position_anchor"], nav_command=inputs["nav_command"],
                    ego_status=inputs["ego_status"].float())

    def velocity(self, u, s, context):
        padded = torch.cat((u, torch.zeros_like(u[..., :1])), dim=-1)
        with math_attention():
            return self.expert.predict_endpoint(padded, s, **context).float()

    def sample(self, context, v0, noise, steps=10):
        device_type = noise.device.type
        def field(u, s):
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
                return self.velocity(u, s, context)
        u = euler_flow(field, noise, steps)
        return {"controls": denormalize_controls(u), "trajectory": integrate(denormalize_controls(u), v0, self.dt)}
