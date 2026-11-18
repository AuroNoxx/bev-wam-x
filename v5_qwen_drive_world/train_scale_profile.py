"""Offline longitudinal readout conditioned on an RGB lead track and its availability."""
import argparse
import hashlib
import json

import numpy as np
from .train_motion_profile import FitMotion, prepare, DEST as SOURCE
from .scale_profile_inputs import DEST, ROOT
from .checkpoint_io import VerifiedCheckpointIO
from .resume_disk_cached import UpdateBoundaryCheckpoint
from .sft_task import restore_rng
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torchdata.stateful_dataloader import StatefulDataLoader

CONFIG = json.loads((ROOT / 'configs/v5-qwen-drive-world/scale-motion-profile.json').read_text())


def extend_observation(obs, row):
    if row['displacement'] is None:
        extra = obs.new_zeros(3)
    else:
        extra = obs.new_tensor([np.arcsinh(row['displacement']), np.arcsinh(row['depth'] / 10), 1.])
    return torch.cat([obs, extra])


def fingerprint(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ScaleMotion(FitMotion):
    def __init__(self, identity, output_dir=DEST):
        super().__init__(identity)
        self.output_dir = output_dir
        pretrained = torch.load(SOURCE / 'best.ckpt', map_location='cpu', weights_only=False)
        self.load_state_dict(pretrained['state_dict'])
        layer = self.profile.mlp[0]
        expanded = torch.nn.Linear(layer.in_features + 3, layer.out_features)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :layer.in_features].copy_(layer.weight)
            expanded.bias.copy_(layer.bias)
        self.profile.mlp[0] = expanded

    def configure_optimizers(self):
        lr = CONFIG['learning_rate']
        steps = self.identity.get('fit_epoch_windows', len(self.identity['indices']['fit'])) * CONFIG['epochs'] // 8
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, foreach=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=lr * .08)
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='step'))

    def training_step(self, batch, _):
        if self.pending_rng:
            restore_rng(self.pending_rng)
            self.pending_rng = None
        obs, base, raw, v0, truth, expert, hold, launch = batch
        control, tr, activity = self.profile(obs, base, raw, v0)
        with torch.autocast('cuda', enabled=False):
            delta = torch.diff(truth, dim=1, prepend=torch.zeros_like(truth[:, :1]))
            speed = delta.norm(dim=-1) / .1
            motion = (speed.reshape(-1, 10, 5).mean(2) / .8).clamp(0, 1)
            span = CONFIG['sustained_samples']
            sustained = (speed > CONFIG['onset_speed_mps']).unfold(1, span, 1).all(-1)
            starts = torch.arange(sustained.shape[1], device=speed.device)
            onset = torch.where(sustained, starts, speed.shape[1]).min(1).values
            ends = torch.arange(4, speed.shape[1], 5, device=speed.device)
            started = (ends[None] >= onset[:, None]).float()
            onset_loss = F.binary_cross_entropy_with_logits(activity, started, reduction='none').mean(1)
            # Only decisive low-speed windows carry the onset label. Treat a missing
            # track as unknown; a motionless lead can still move later.
            eligible = (v0.flatten() <= 3 / 3.6) & (hold | launch)
            loss = (.5 * F.smooth_l1_loss(tr[..., :2], truth)
                    + .7 * F.smooth_l1_loss(tr[..., 3], speed)
                    + 1.5 * F.binary_cross_entropy_with_logits(activity, motion)
                    + .15 * F.smooth_l1_loss(control[..., 0], expert[..., 0])
                    + .08 * F.smooth_l1_loss(control[:, 1:, 0], control[:, :-1, 0])
                    + CONFIG['onset_weight'] * (eligible * onset_loss).mean())
        self.log('train_loss', loss, on_step=True, batch_size=1)
        return loss

    def on_validation_epoch_end(self):
        ade = {k: float(np.mean(v)) for k, v in self.errors.items()}
        score = .4 * ade['hold'] + .3 * ade['launch'] + .3 * ade['all']
        self.log('development_score', score)
        row = dict(epoch=self.current_epoch + 1, step=self.global_step, score=score, ade=ade)
        with (self.output_dir / 'development.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(row, flush=True)


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--resume', action='store_true')
    args.add_argument('--history-association', action='store_true')
    args.add_argument('--paired-launch', action='store_true')
    options = args.parse_args()
    resume = options.resume
    output_dir = DEST / 'history-association' if options.history_association else DEST
    observation_dir = output_dir
    if options.paired_launch:
        from .mine_launch_pairs import PAIR, SOURCE as history_run
        output_dir, observation_dir = PAIR, history_run / 'seeded'
    if (output_dir / 'latest.ckpt').exists() and not resume:
        raise ValueError('Existing scale-conditioned run requires --resume')
    pl.seed_everything(CONFIG['seed'], workers=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    sets, identity = prepare()
    assert identity == json.loads((SOURCE / 'identity.json').read_text())
    source_rows = json.loads((DEST / 'geometry.json').read_text())['splits']['train']
    rows = json.loads((observation_dir / 'observations.json').read_text())['splits']['train']
    fit, dev = [[(extend_observation(s[0], rows[i]), *s[1:])
                 for i, s in zip(identity['indices'][name], samples, strict=True)]
                for name, samples in zip(('fit', 'development'), sets, strict=True)]
    from .geometry_repair_features import split_records
    from .real_dataset import RealData
    paths = [r['actor_path'] for r in split_records(RealData('subset2000'))['train']]
    assert paths == [r['actor_path'] for r in source_rows] == [r['actor_path'] for r in rows]
    identity.update(config=CONFIG, source_profile=fingerprint(SOURCE / 'best.ckpt'),
                    observations=fingerprint(observation_dir / 'observations.json'))
    if options.paired_launch:
        from .paired_launch_training import augment
        fit = augment(fit, identity)
    (output_dir / 'identity.json').write_text(json.dumps(identity, indent=2))
    loader = StatefulDataLoader(fit, batch_size=1, shuffle=True,
                                generator=torch.Generator().manual_seed(CONFIG['seed']))
    best = ModelCheckpoint(dirpath=output_dir, filename='best', monitor='development_score', mode='min',
                           save_on_train_epoch_end=False, enable_version_counter=False)
    latest = UpdateBoundaryCheckpoint(dirpath=output_dir, filename='latest', every_n_train_steps=40,
                                      save_on_train_epoch_end=True, enable_version_counter=False)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=CONFIG['epochs'], precision='bf16-mixed',
                         accumulate_grad_batches=8, gradient_clip_val=2., enable_progress_bar=False,
                         num_sanity_val_steps=0, callbacks=[best, latest], plugins=[VerifiedCheckpointIO()],
                         logger=pl.loggers.TensorBoardLogger(str(ROOT / 'logs/v5-qwen-drive-world'), name='paired-launch' if options.paired_launch else ('scale-motion-history' if options.history_association else 'scale-motion')))
    task = ScaleMotion(identity, output_dir)
    if options.paired_launch:
        base = torch.load(history_run / 'best.ckpt', map_location='cpu', weights_only=False)
        task.load_state_dict(base['state_dict'])
    trainer.fit(task, loader, StatefulDataLoader(dev, batch_size=1),
                ckpt_path=str(output_dir / 'latest.ckpt') if resume else None, weights_only=False)
    assert trainer.global_step == len(fit) * CONFIG['epochs'] // 8


if __name__ == '__main__':
    main()
