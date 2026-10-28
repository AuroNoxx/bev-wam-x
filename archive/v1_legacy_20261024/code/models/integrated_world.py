"""Inference modules only: shared frozen encoder plus action-conditioned WAM."""
import torch
from torch import nn
from configs.stage7 import Stage7Config
from models.action_conditioned import JEPAWorldX
from scripts.evaluate_joint import load_joint


class IntegratedWorld(nn.Module):
    def __init__(self, encoder, world):
        super().__init__()
        self.encoder = encoder
        self.world = world

    def forward(self, history, ego, command):
        return self.world(history, ego, command)


def load_integrated_world(device='cuda'):
    cfg = Stage7Config()
    saved = torch.load(cfg.stage6_checkpoint, map_location='cpu', weights_only=False)
    base, _, _ = load_joint(cfg.stage4_checkpoint, 'cpu')
    encoder = base.encoder
    encoder_saved = {k.removeprefix('base.encoder.'): v for k, v in saved['model'].items()
                     if k.startswith('base.encoder.')}
    current = encoder.state_dict()
    if current.keys() != encoder_saved.keys() or any(not torch.equal(current[k], encoder_saved[k]) for k in current):
        raise ValueError('WAM and navigation encoding spaces differ; explicit alignment required')
    world = JEPAWorldX()
    world.load_state_dict({k.removeprefix('world.'): v for k, v in saved['model'].items() if k.startswith('world.')})
    return IntegratedWorld(encoder, world).to(device).eval().requires_grad_(False)
