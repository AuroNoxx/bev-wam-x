"""Replay the final validation, retaining paired trajectory and label diagnostics."""
import json
import numpy as np
from .runtime import ROOT, configure, math_attention

configure()
import torch
from .real_dataset import RealData
from .train_disk_cached import DiskCachedTask
from .kinematics import integrate


def ade(xy, truth):
    return float((xy - truth).norm(dim=-1).mean())


def main():
    folder = ROOT / 'artifacts/v5-qwen-drive-world/subset2000'
    out = folder / 'final-window-audit'
    out.mkdir(exist_ok=True)
    history = [json.loads(line) for line in (folder/'validation.jsonl').read_text().splitlines()]
    final = next(row for row in reversed(history) if row['step'] == 750)
    data = RealData('subset2000')
    assert len(final['samples']) == len(data.validation)
    torch.manual_seed(71)
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(.8)
    checkpoint = torch.load(folder/'latest.ckpt', map_location='cpu', weights_only=False, mmap=True)
    assert checkpoint['global_step'] == 750
    task = DiskCachedTask(checkpoint['task_state']['config'], data.identity, 'subset2000').cuda()
    task.on_load_checkpoint(checkpoint)
    task.load_state_dict(checkpoint['state_dict'])
    del checkpoint
    task.eval()
    rows, predictions = [], []
    with torch.no_grad(), math_attention(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i, batch in enumerate(data.val_dataloader()):
            rec = batch['record']
            label = {key: val.cuda() for key, val in batch['labels'].items()}
            truth, speed = label['trajectory'], label['v0']
            log = final['samples'][i]
            assert (log['route'], log['speed_kmh']) == (rec['route'], rec['speed_kmh'])
            native, inputs = task.encode_path(rec['actor_path'])
            ctx = task.planner.context(native, inputs)
            torch.manual_seed(1907+i)
            traj = task.planner.sample(ctx, speed, torch.randn(1,50,2,device='cuda'), steps=10)['trajectory']
            xy = traj[...,:2]
            cv = torch.zeros_like(truth)
            cv[...,0] = speed[:,None] * torch.arange(1,51,device='cuda') * .1
            model_ade = ade(xy, truth)
            ctrl, raw = label['controls'], label['raw_controls']
            clipped = (ctrl-raw).abs() > 1e-5
            gap = model_ade - ade(cv, truth)
            row = dict(route=rec['route'], index=rec['index'], speed_kmh=rec['speed_kmh'],
                       ax_3s=float(ctrl[:,:30,0].mean()), y_3s=float(truth[0,29,1]),
                       max_expert_displacement=float(truth.norm(dim=-1).max()),
                       model_ade=model_ade, cv_ade=ade(cv,truth), excess_ade=gap,
                       clip=float(clipped.float().mean()), clip_ax=float(clipped[...,0].float().mean()),
                       clip_ay=float(clipped[...,1].float().mean()),
                       raw_replay=ade(integrate(raw,speed)[...,:2],truth),
                       bounded_replay=ade(integrate(ctrl,speed)[...,:2],truth),
                       logged_ade=log['ade_m'], logged_fde=log['fde_m'])
            for name,path in [('model',xy),('cv',cv)]:
                err = path - truth
                for horizon in (1,3,5):
                    e = err[0,horizon*10-1]
                    row[f'{name}_fde{horizon}'] = float(e.norm())
                    row[f'{name}_dx{horizon}'] = float(e[0])
                    row[f'{name}_dy{horizon}'] = float(e[1])
            for name, acc in [('model',traj[...,4:6]),('expert',ctrl)]:
                for axis, j in zip('xy', (acc.diff(dim=1)/.1).abs()[0].T):
                    row[f'{name}_jerk_{axis}_p95'] = float(torch.quantile(j.float(),.95))
            rows.append(row)
            predictions.append(traj[0].float().cpu().numpy())
    deltas = [max(abs(r['model_ade']-r['logged_ade']),abs(r['model_fde5']-r['logged_fde'])) for r in rows]
    print('max replay discrepancy (m):',max(deltas),flush=True)
    assert max(deltas) < .02, 'Replay differs from the scored checkpoint'
    (out/'windows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    np.savez_compressed(out/'predictions.npz', trajectories=np.asarray(predictions),
                        route=[r['route'] for r in rows], index=[r['index'] for r in rows])


if __name__ == '__main__':
    main()
