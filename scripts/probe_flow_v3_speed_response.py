"""Probe the frozen v3 flow model's conditional response to ego speed.

Synthetic counterfactual: only state[0] (current speed) is changed, history OCC
is left untouched. This measures whether the model has any calibrated speed
response and whether its 5 s displacement is physically plausible at 80-130 km/h.
It is not a driving test and it cannot create data that does not exist.
"""
from pathlib import Path
import json, sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from models_v1 import FlowPlanner

CKPT = ROOT / 'artifacts/lightning/flow-v3/best.ckpt'
MANIFEST = json.loads((ROOT / 'data/occ_v3/manifest.json').read_text(encoding='utf-8'))
OUT = ROOT / 'artifacts/diagnostics/v3-limits/speed-response.json'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
state_dict = {k[len('model.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
model = FlowPlanner()
model.load_state_dict(state_dict, strict=True)
model = model.to(device).eval()
print('loaded', CKPT.name, 'step', ckpt.get('global_step'), flush=True)

base = ROOT / MANIFEST.get('cache_root', 'data/occ_v3')
files = MANIFEST['splits']['validation']
index = np.linspace(0, len(files) - 1, N).round().astype(int)
hist, state, speed0 = [], [], []
for i in index:
    with np.load(base / files[i], allow_pickle=False) as a:
        hist.append(a['history'])
        state.append(a['state'])
        speed0.append(float(a['state'][0] * 30.0) * 3.6)
history = torch.from_numpy(np.stack(hist)).to(device)
state = torch.from_numpy(np.stack(state)).float().to(device)
speed0 = np.asarray(speed0)

def run(setpoint_kmh, seed=1908):
    torch.manual_seed(seed)
    s = state.clone()
    s[:, 0] = float(setpoint_kmh) / 3.6 / 30.0
    with torch.no_grad():
        pred = model.sample({'history': history, 'state': s}, k=8, steps=16)
    xy = pred[..., :2].float().cpu().numpy() * np.array([50.0, 10.0])
    speed = pred[..., 4].float().cpu().numpy() * 30.0
    dist = np.linalg.norm(xy[:, :, -1], axis=-1)
    return {'setpoint_kmh': setpoint_kmh,
            'mean_endpoint_m': float(dist.mean()), 'p10': float(np.percentile(dist, 10)),
            'p90': float(np.percentile(dist, 90)),
            'mean_end_speed_kmh': float(speed[:, :, -1].mean() * 3.6),
            'mean_speed_change_mps': float((speed[:, :, -1].mean() - setpoint_kmh / 3.6)),
            'implied_cruise_distance_m': float(setpoint_kmh / 3.6 * 5.0)}

results = []
for v in (10, 20, 30, 40, 50, 60, 80, 100, 120, 130):
    r = run(v)
    results.append(r)
    print(json.dumps(r), flush=True)

# candidate sanity: same input, three sampling seeds at the observed speed
spread = []
for seed in (1908, 2911, 3911):
    torch.manual_seed(seed)
    with torch.no_grad():
        pred = model.sample({'history': history, 'state': state}, k=8, steps=16)
    xy = pred[..., :2].float().cpu().numpy() * np.array([50.0, 10.0])
    spread.append(float(np.linalg.norm(xy[:, :, -1], axis=-1).mean()))
report = {'checkpoint': str(CKPT.relative_to(ROOT)), 'global_step': ckpt.get('global_step'),
          'windows': int(N), 'synthetic_state_override_only': True,
          'observed_window_speed_kmh': {'mean': float(speed0.mean()), 'max': float(speed0.max())},
          'speed_sweep': results,
          'endpoint_distance_seed_mean_at_observed_speed_m': spread}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('WROTE', OUT, flush=True)
