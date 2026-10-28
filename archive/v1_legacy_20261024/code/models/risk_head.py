"""Small spatial risk classifier over observed history and predicted consequences."""
import torch
from torch import nn
from torch.nn import functional as F


class ConsequenceRisk(nn.Module):
    def __init__(self, use_world=True, token_dim=256, tokens=64, hidden=576, dropout=.1, proj_dim=None):
        super().__init__()
        self.use_world = use_world
        if proj_dim is None:
            proj_dim = 16 if token_dim >= 64 else min(4, token_dim)
        else:
            proj_dim = min(proj_dim, token_dim)
        self.arch = dict(use_world=use_world, token_dim=token_dim, tokens=tokens,
                         hidden=hidden, dropout=dropout, proj_dim=proj_dim)
        self.projection = nn.Linear(token_dim, proj_dim)
        self.head = nn.Sequential(nn.Linear(6*tokens*proj_dim+48, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.register_buffer('scale', torch.tensor([20., 5., 10., 1.]))

    def forward(self, history, ego, command, predicted):
        if history.shape[1:] != (4, self.arch['tokens'], self.arch['token_dim']):
            raise ValueError('Expected four encoded observed frames')
        if predicted.shape != history.shape or command.shape[1:] != (10, 4):
            raise ValueError('Expected four predicted horizons and a ten-point command')
        current = history[:, -1]
        observed = torch.stack([current, (history[:, :3]-current[:, None]).mean(1)], 1)
        observed = self.projection(observed).flatten(1)
        future = self.projection(predicted-current[:, None]).flatten(1)
        if not self.use_world:
            future = torch.zeros_like(future)
        return self.head(torch.cat([observed, future, ego.flatten(1),
                                   (command/self.scale).flatten(1)], 1))


def masked_risk_loss(logits, target, valid):
    if not valid.any():
        raise ValueError('No observed risk outcome in batch')
    return F.binary_cross_entropy_with_logits(logits[valid], target[valid].float())
