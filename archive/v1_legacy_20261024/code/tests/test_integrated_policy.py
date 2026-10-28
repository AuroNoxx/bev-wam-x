from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import planning.integrated_policy as module


class Flow(nn.Module):
    def sample(self, history, *args, **kwargs):
        value = torch.zeros(1, 8, 10, 4)
        value[..., 0] = torch.arange(1, 11)*.4; value[..., 2] = 2.
        return value


class World(nn.Module):
    def forward(self, history, ego, command):
        assert len(command) <= 4
        return torch.zeros_like(history)


class Risk(nn.Module):
    def forward(self, history, ego, command, predicted):
        return torch.zeros(len(command), 2)


def context():
    nav = SimpleNamespace(points=np.column_stack([np.arange(8)*5., np.zeros(8)]).astype(np.float32),
                          valid=np.ones(8, bool), lane_width_m=3.5)
    return (torch.zeros(1, 4, 64, 256), np.zeros((1, 4, 2), np.float32),
            np.zeros((4, 128, 128), np.uint8), [], [], nav,
            np.array([4.5, 1.8, 2.47, .7], np.float32), np.ones(2, bool), 1)


def test_full_never_selects_unassessed_candidates_even_if_their_cost_is_lower(monkeypatch):
    def score(candidates, *args, **kwargs):
        n = len(candidates); probability = args[6] if len(args)>6 else None
        values = np.arange(n, dtype=float)
        if probability is not None:
            values[:4] += 100  # Unassessed candidates would win without the mask.
        return dict(scores=values, terms=np.zeros((n, 8)), rejection_reasons=[[] for _ in range(n)],
                    eligible=np.ones(n, bool), selected_index=int(values.argmin()))
    monkeypatch.setattr(module, 'strict_score', score)
    policy = module.IntegratedPolicy(nn.Identity(), Flow(), World(), Risk())
    result = policy.decide(*context())
    assert result['selected_index'] == 0
    np.testing.assert_array_equal(result['risk_assessed'], [True]*4+[False]*4)
    assert np.isnan(result['probability'][4:]).all()
    assert all(len(x)==0 for x in result['predicted'][4:])


def test_all_rejections_produce_brake_and_keep_unsafe_fallback_visible(monkeypatch):
    def score(candidates, *args, **kwargs):
        n = len(candidates)
        return dict(scores=np.zeros(n), terms=np.zeros((n, 8)), rejection_reasons=[['dynamic_geometry'] for _ in range(n)],
                    eligible=np.zeros(n, bool), selected_index=-1)
    monkeypatch.setattr(module, 'strict_score', score)
    policy = module.IntegratedPolicy(nn.Identity(), Flow())
    result = policy.decide(*context())
    assert result['fallback'] and result['selected_id']=='STOP'
    assert result['fallback_geometry_safe'] is False
    assert not np.any(result['plan'][:, 2] < 0)
