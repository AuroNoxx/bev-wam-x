import torch
import pytest
from models.navigation_flow import NavigationFlow


def observations():
    return (torch.randn(2, 4, 64, 256), torch.rand(2, 4, 2), torch.randn(2, 8, 2),
            torch.ones(2, 8, dtype=torch.bool), torch.tensor([[4.5, 1.8, 2.47, .7]]).expand(2, -1))


def test_flow_train_sample_share_reference_and_mask_invalid_navigation():
    torch.manual_seed(9)
    model = NavigationFlow(d_model=32, layers=1, heads=2, feedforward=64).eval()
    args = list(observations()); ref = torch.randn(2, 10, 4)
    # Zero-initialized velocity leaves nominal candidate exactly at the reference.
    out = model.sample(*args, ref, candidates=3, steps=2, seed=7)
    torch.testing.assert_close(out[:, 0], ref, atol=0, rtol=0)
    args[3][:, 3:] = False
    a = model.context(*args); args[2][:, 3:] = float('nan')
    torch.testing.assert_close(a, model.context(*args), atol=0, rtol=0)
    loss = model.loss(*args, ref, ref+.1, noise=torch.zeros_like(ref), time=torch.full((2,), .5))
    assert torch.isfinite(loss)
    loss.backward(); assert model.out.weight.grad.abs().sum() > 0
    # Sampling remains deterministic with a fixed seed; nominal is seed-independent.
    x = model.sample(*args, ref, candidates=3, steps=3, seed=2, solver='midpoint')
    y = model.sample(*args, ref, candidates=3, steps=3, seed=2, solver='midpoint')
    torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_permission_profile_never_silently_substitutes_unknown_for_false():
    model = NavigationFlow(d_model=32, layers=1, heads=2, feedforward=64, permission_dim=2).eval()
    args = observations()
    with pytest.raises(ValueError, match='Explicit static'):
        model.context(*args)
    mask = torch.tensor([[True, False], [False, True]])
    first = model.context(*args, maneuver_allowed=mask)
    second = model.context(*args, maneuver_allowed=~mask)
    assert not torch.equal(first[:, -1], second[:, -1])
