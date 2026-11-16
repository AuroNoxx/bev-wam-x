"""Regression: unchanged legacy path, reversible units, gradients, no future use."""
import importlib.util
import json
from pathlib import Path
import torch
from lightning_task import ROOT, configure_runtime, OccData
from models_v1 import FlowPlanner
from stage1_objective import consistent_flow_loss


def load_old(name):
    path = ROOT / 'artifacts/diagnostics/v3-residual-pilot/code-before' / (name + '.py')
    spec = importlib.util.spec_from_file_location('old_' + name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    configure_runtime()
    settings = json.loads((ROOT / 'configs/lightning-stage1-v3-cv_residual-pilot.json').read_text())
    batch = next(iter(OccData('flow', settings).loader('validation')))
    batch = {k: v[:2].cuda() for k, v in batch.items()}
    old = load_old('models_v1'); old_objective = load_old('stage1_objective')
    torch.manual_seed(11); model = FlowPlanner().cuda().eval()
    torch.manual_seed(11); legacy = old.FlowPlanner().cuda().eval()
    assert all(torch.equal(v, legacy.state_dict()[k]) for k, v in model.state_dict().items())
    torch.manual_seed(12); a = model.sample(batch, k=2, steps=2)
    torch.manual_seed(12); b = legacy.sample(batch, k=2, steps=2)
    assert torch.equal(a, b)
    torch.manual_seed(13); a, terms_a = consistent_flow_loss(model, batch, settings['objective'])
    torch.manual_seed(13); b, terms_b = old_objective.consistent_flow_loss(legacy, batch, settings['objective'])
    assert torch.equal(a, b) and all(torch.equal(v, terms_b[k]) for k, v in terms_a.items())
    model.parameterization = 'cv_residual'
    target = batch['trajectory'].float()
    encoded = model.to_flow_space(batch, target)
    assert torch.allclose(model.from_flow_space(batch, encoded), target, atol=2e-7)
    reference = model.from_flow_space(batch, torch.zeros_like(target))
    expected_x = batch['state'][:, 0, None] * 30 * torch.arange(1, 26, device='cuda')[None] * .2 / 50
    assert torch.allclose(reference[..., 0], expected_x, atol=2e-7)
    assert torch.allclose(reference[..., 4], batch['state'][:, 0, None].expand(-1, 25))
    changed = dict(batch); changed['trajectory'] = target + 99
    assert torch.equal(reference, model.from_flow_space(changed, torch.zeros_like(target)))
    model.train(); torch.manual_seed(14)
    value, _ = consistent_flow_loss(model, batch, settings['objective']); value.backward()
    assert torch.isfinite(value) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    report = {'pass': True, 'legacy_parameters_exact': True, 'legacy_sample_exact': True,
              'legacy_loss_and_terms_exact': True, 'roundtrip_units_pass': True,
              'zero_residual_is_current_speed_cv': True, 'reference_independent_of_truth': True,
              'real_batch_all_parameter_gradients_finite': True, 'residual_smoke_loss': float(value)}
    (ROOT / 'artifacts/diagnostics/v3-residual-pilot/parameterization-tests.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__': main()
