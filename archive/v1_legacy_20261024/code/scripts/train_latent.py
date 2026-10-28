"""Train stage 3 only; all outputs are written to a new run directory."""
import argparse
from dataclasses import replace
import faulthandler
from pathlib import Path

from configs.stage3 import Stage3Config
from training.latent import train


def main():
    faulthandler.enable()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('outputs/data/stage2_v2_100k'))
    p.add_argument('--output', type=Path, default=Path('artifacts/restart_stage3/vicreg_001'))
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--validation-pairs', type=int)
    p.add_argument('--max-batches', type=int, help='Smoke only; cannot pass formal full-run acceptance')
    args = p.parse_args()
    changes = {key:getattr(args, key) for key in ('epochs', 'batch_size', 'validation_pairs') if getattr(args, key) is not None}
    train(args.data, args.output, replace(Stage3Config(), **changes), args.max_batches)


if __name__ == '__main__':
    main()
