"""Evaluate the selected scale-conditioned planner against the recorded step-750 trajectories."""
import argparse
import json

import numpy as np
from .train_scale_profile import CONFIG, DEST, ScaleMotion, extend_observation, fingerprint
from .geometry_repair_features import BASE
from .kinematics import integrate
from .motion_profile import profile_features
from .real_dataset import RealData
from .train_geometry_repair import RepairData
import torch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history-association', action='store_true')
    parser.add_argument('--seeded-replay', action='store_true')
    parser.add_argument('--paired-launch', action='store_true')
    args = parser.parse_args()
    use_history = args.history_association or args.seeded_replay
    folder = DEST / 'history-association' if use_history else DEST
    observation_dir = folder
    if args.paired_launch:
        from .mine_launch_pairs import PAIR, SOURCE as history_run
        folder, observation_dir = PAIR, history_run / 'seeded'
    identity = json.loads((folder / 'identity.json').read_text())
    best = torch.load(folder / 'best.ckpt', map_location='cpu', weights_only=False)
    latest = torch.load(folder / 'latest.ckpt', map_location='cpu', weights_only=False)
    history = [json.loads(x) for x in (folder / 'development.jsonl').read_text().splitlines()]
    assert identity == best['task_state']['identity'] == latest['task_state']['identity']
    assert identity['observations'] == fingerprint(observation_dir / 'observations.json')
    assert latest['global_step'] == identity.get('fit_epoch_windows', len(identity['indices']['fit'])) * CONFIG['epochs'] // 8
    assert best['global_step'] == min(history, key=lambda r: r['score'])['step']
    net = ScaleMotion(identity).cuda().eval()
    net.load_state_dict(best['state_dict'])
    data = RealData('subset2000')
    assert data.identity == identity['manifest']
    cached = RepairData(data.validation, 'validation')
    evaluation = folder / 'seeded' if args.seeded_replay else folder
    inputs = observation_dir if args.paired_launch else evaluation
    observations = json.loads((inputs / 'observations.json').read_text())['splits']['validation']
    prior_folder = folder if args.seeded_replay else DEST
    previous_rows = (json.loads((prior_folder / 'observations.json').read_text())['splits']['validation']
                     if use_history else None)
    audit = BASE / 'final-window-audit'
    baseline = json.loads((audit / 'windows.json').read_text())
    canonical = np.load(audit / 'predictions.npz')
    rows = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for i, (record, row, anchor) in enumerate(zip(data.validation, observations, baseline, strict=True)):
            assert row['actor_path'] == record['actor_path']
            assert record['route'] == anchor['route'] == str(canonical['route'][i])
            assert record['index'] == anchor['index'] == int(canonical['index'][i])
            feat, label, _ = cached[i]
            feat = {k: v[None].cuda() for k, v in feat.items()}
            obs, base = profile_features(feat)
            measured = extend_observation(obs[0], row)[None]
            _, traj, _ = net.profile(measured, base, feat['ctrl'], feat['v0'])
            original = integrate(feat['ctrl'], feat['v0'].flatten())[0]
            reference = torch.as_tensor(canonical['trajectories'][i], device='cuda')
            torch.testing.assert_close(original, reference, rtol=0, atol=1e-5)
            truth = label['trajectory'].cuda()
            old_error = (original[:, :2] - truth).norm(dim=1)
            assert abs(float(old_error.mean()) - anchor['model_ade']) < 1e-4
            err = (traj[0, :, :2] - truth).norm(dim=1)
            dmax = float(truth.norm(dim=1).max())
            low_speed = record['speed_kmh'] <= 3
            launch = low_speed and dmax >= 3
            from .runtime import ROOT
            target = json.loads((ROOT / record['actor_path']).read_text())['target_speed_kmh']
            rows.append(dict(route=record['route'], index=record['index'], speed=record['speed_kmh'],
                             target_speed=target, hold=dmax < .5, launch=launch, zero_launch=launch and target==0,
                             ade=float(err.mean()), old_ade=anchor['model_ade'],
                             fde=[float(err[k]) for k in (9, 29, 49)],
                             jerk=float((traj[0, :, 4:6].diff(dim=0)/.1).norm(dim=1).max()),
                             pred_dist=float(traj[0, :, :2].norm(dim=1).max()),
                             old_dist=float(original[:, :2].norm(dim=1).max()), true_dist=dmax,
                             lead_displacement=row['displacement'], lead_depth=row['depth']))
            if use_history and row['displacement'] is not None and previous_rows[i]['displacement'] is None:
                previous = previous_rows[i]
                assert previous['actor_path'] == row['actor_path'] and previous['displacement'] is None
                # Paired inference at the same weights isolates the effect of the recovered cue.
                prior = extend_observation(obs[0], previous)[None]
                _, without_cue, _ = net.profile(prior, base, feat['ctrl'], feat['v0'])
                rows[-1]['distance_without_recovered_cue'] = float(without_cue[0, :, :2].norm(dim=1).max())
                rows[-1]['ade_without_recovered_cue'] = float((without_cue[0, :, :2]-truth).norm(dim=1).mean())
    groups = dict(all=rows, hold=[r for r in rows if r['hold']], launch=[r for r in rows if r['launch']],
                  zero_launch=[r for r in rows if r['zero_launch']])
    for low, high in ((0, 40), (40, 80), (80, 121)):
        groups[f'{low}-{min(high,120)}'] = [r for r in rows if low <= r['speed'] < high]
    scores = {}
    for name, records in groups.items():
        entries = dict(windows=len(records), routes=len({r['route'] for r in records}),
                       with_track=sum(r['lead_depth'] is not None for r in records))
        if records:
            entries.update(ade=float(np.mean([r['ade'] for r in records])),
                           step750_ade=float(np.mean([r['old_ade'] for r in records])),
                           fde=np.mean([r['fde'] for r in records], axis=0).tolist(),
                           mean_peak_jerk=float(np.mean([r['jerk'] for r in records])),
                           predicted_hold=sum(r['pred_dist'] < .5 for r in records),
                           predicted_launch=sum(r['pred_dist'] >= 3 for r in records))
        scores[name] = entries
    gate = CONFIG['fixed_validation']
    checks = dict(hold=scores['hold']['ade'] < gate['hold_ade_below'],
                  launch=scores['launch']['predicted_launch'] >= gate['launches_at_least'],
                  overall=scores['all']['ade'] <= gate['overall_ade_at_most'])
    output = dict(updates=latest['global_step'], selected_step=best['global_step'], checks=checks, scores=scores, rows=rows)
    output['checkpoint_sha256'] = fingerprint(folder / 'best.ckpt')
    output['evaluation_observations_sha256'] = fingerprint(inputs / 'observations.json')
    (evaluation / 'validation.json').write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in output.items() if k!='rows'}), flush=True)


if __name__ == '__main__':
    main()
