"""Load only the frozen perception modules used by navigation pilots."""
import torch
from torch import nn


class ResNetNavigationEncoder(nn.Module):
    tokens = 16
    token_dim = 512

    def __init__(self, baseline):
        super().__init__()
        self.stem = baseline.stem
        self.backbone = baseline.backbone
        self.pool = baseline.pool

    def forward(self, bev):
        if bev.dtype == torch.uint8:
            bev = bev.float()/255.
        return self.pool(self.backbone(self.stem(bev))).flatten(2).transpose(1, 2)


def load_navigation_encoder(kind, checkpoint, device):
    if kind == 'joint':
        from scripts.evaluate_joint import load_joint
        base, _, _ = load_joint(checkpoint, device)
        encoder = base.encoder
        shape = (64, 256)
    elif kind == 'resnet':
        from models.baseline import ResNet18Trajectory
        selected = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if selected['data_schema_version'] != 2:
            raise ValueError('Only the project schema2 ResNet checkpoint is permitted')
        cfg = selected['model_config']
        base = ResNet18Trajectory(**{k: cfg[k] for k in ('pooling_size', 'mlp_hidden', 'latent_dim')})
        base.load_state_dict(selected['model'])
        encoder = ResNetNavigationEncoder(base)
        shape = (cfg['pooling_size']**2, 512)
    else:
        raise ValueError('Unknown navigation encoder')
    encoder = encoder.to(device).eval().requires_grad_(False)
    return encoder, shape
