"""Stage 3 completion supervision, target isolation and unchanged flow inference."""
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from configs.stage3 import Stage3Config
from models.bev_wa_latent_flow import BEVWALatentFlowModel, MultiHorizonLatentFlowPredictor


def make_model(**kwargs):
    cfg = Stage3Config(encoder_dim=24, encoder_heads=4, encoder_layers=1,
                       encoder_mlp=48, masked_completion_weight=0.5,
                       residual_target=True, residual_horizons=(0, 1), **kwargs)
    model = BEVWALatentFlowModel(cfg)
    # Smaller backbone for logic tests; preserve the actual 4x64 token contract.
    model.predictor = MultiHorizonLatentFlowPredictor(latent_dim=24, d_model=24,
                                                     nhead=4, nlayers=1, dim_feedforward=48)
    return model


def inputs():
    return (torch.randint(0, 256, (2, 4, 4, 128, 128), dtype=torch.uint8),
            torch.randn(2, 4, 2),
            torch.randint(0, 256, (2, 4, 4, 128, 128), dtype=torch.uint8),
            torch.tensor([[True, True, True, True], [True, False, True, False]]))


def test_independent_mask_shape_ratio_and_invalid_horizons():
    torch.manual_seed(9)
    model = make_model(mask_ratio=0.25, masked_horizons=(0, 1))
    valid = torch.ones(1024, 4, dtype=torch.bool)
    valid[::2, 1] = False
    mask, eligible = model.sample_completion_mask(valid)
    assert mask.shape == (1024, 4, 64) and mask.dtype == torch.bool
    assert not mask[:, 2:].any() and not mask[::2, 1].any()
    assert abs(mask.sum().item() / (eligible.sum().item() * 64) - 0.25) < 0.01
    assert not torch.equal(mask[0, 0], mask[1, 0])
    assert not torch.equal(mask[1, 0], mask[1, 1])


def test_forward_mask_values_loss_and_gradients():
    torch.manual_seed(3)
    model = make_model()
    h, ego, f, valid = inputs()
    calls = []
    original = model.predictor.forward_with_features
    def capture(z, s, condition):
        result = original(z, s, condition)
        calls.append((z, s, result))
        return result
    with patch.object(model.predictor, 'forward_with_features', side_effect=capture):
        loss, terms = model.forward_train(h, ego, f, valid)
    assert len(calls) == 2
    z, s, (_, features) = calls[1]
    target = model.target_encoder.encode_future(f)
    mask = (z == model.predictor.mask_token).all(-1)
    assert mask.shape == (2, 4, 64) and mask.any()
    assert not mask[~valid].any()
    torch.testing.assert_close(z[~mask], target[~mask])
    assert torch.equal(s, torch.ones(2))
    assert features.shape == (2, 256, 24)
    completion = model.predictor.completion_head(features).reshape_as(target)
    expected = F.mse_loss(completion[mask], target[mask])
    assert terms['loss_mask'] == pytest.approx(expected.item())
    assert terms['mask_ratio_actual'] == pytest.approx(mask.sum().item() / (valid.sum().item()*64))
    assert loss.item() == pytest.approx(terms['flow_loss'] + 0.5*terms['loss_mask'] + 0.1*terms['var_penalty'])
    loss.backward()
    for parameter in (model.predictor.mask_token, model.predictor.completion_head.weight,
                      model.predictor.in_proj.weight, model.online_encoder.temporal_pos):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.target_encoder.parameters())
    # The masked task itself must drive both shared features and history conditioning.
    model.zero_grad(set_to_none=True)
    _, features = model.predictor.forward_with_features(z.detach(), s, model.online_encoder(h, ego))
    F.mse_loss(model.predictor.completion_head(features).reshape_as(target)[mask], target[mask]).backward()
    assert model.online_encoder.temporal_pos.grad.abs().sum() > 0
    assert model.predictor.in_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('ratio,valid', [(0.0, True), (0.25, False), (1.0, True)])
def test_empty_and_full_masks_are_finite(ratio, valid):
    model = make_model(mask_ratio=ratio)
    h, ego, f, mask = inputs()
    mask[:] = valid
    loss, terms = model.forward_train(h, ego, f, mask)
    assert torch.isfinite(loss)
    if ratio == 0 or not valid:
        assert terms['loss_mask'] == terms['mask_ratio_actual'] == 0
    else:
        assert terms['mask_ratio_actual'] == 1
    loss.backward()


def test_inference_never_calls_completion_head_or_uses_mask_parameters():
    model = make_model().eval()
    h, ego, f, mask = inputs()
    # Nonzero flow head avoids an identity-only trivial comparison.
    torch.nn.init.normal_(model.predictor.out_proj.weight, std=0.02)
    with torch.no_grad():
        condition = model.online_encoder(h, ego)
        noise = torch.randn(2, 4, 64, 24)
        expected = model.predictor.solve_euler(condition, num_steps=2, initial_noise=noise)
        model.predictor.mask_token.fill_(123)
        model.predictor.completion_head.weight.fill_(456)
        with patch.object(model.predictor.completion_head, 'forward', side_effect=AssertionError('inference completion')):
            actual = model.predictor.solve_euler(condition, num_steps=2, initial_noise=noise)
            _, terms = model.forward_train(h, ego, f, mask)
        assert torch.equal(expected, actual)
        assert terms['loss_mask'] == 0


def test_zero_weight_skips_completion_branch():
    from dataclasses import replace
    model = make_model()
    model.cfg = replace(model.cfg, masked_completion_weight=0.0)
    with patch.object(model, 'sample_completion_mask', side_effect=AssertionError('disabled branch')):
        _, terms = model.forward_train(*inputs())
    assert terms['loss_mask'] == terms['mask_ratio_actual'] == 0
