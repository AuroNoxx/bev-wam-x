import torch
from torch import nn
from models.integrated_world import IntegratedWorld


def test_inference_world_does_not_register_unused_heads_or_duplicate_shared_encoder():
    encoder = nn.Linear(2, 4)
    class World(nn.Module):
        def __init__(self):
            super().__init__(); self.proj = nn.Linear(4, 2)
        def forward(self, history, ego, command):
            return self.proj(history)+ego+command
    world = World(); bundle = IntegratedWorld(encoder, world)
    history = encoder(torch.ones(3, 2)); ego = torch.zeros(3, 2); command = torch.ones(3, 2)
    torch.testing.assert_close(bundle(history, ego, command), world(history, ego, command))
    assert sum(p.numel() for p in bundle.parameters()) == sum(p.numel() for p in encoder.parameters())+sum(p.numel() for p in world.parameters())
    assert all(k.startswith(('encoder.', 'world.')) for k in bundle.state_dict())
