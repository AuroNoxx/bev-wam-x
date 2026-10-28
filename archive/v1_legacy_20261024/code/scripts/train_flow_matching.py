"""CLI entry point for training Stage 7 Conditional Flow Matching trajectory model."""
import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from configs.stage7 import Stage7Config
from training.flow_matching import train_flow


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='Path to Stage 7 cached features directory')
    p.add_argument('--output', type=Path, required=True, help='Path to output run directory (must not exist)')
    p.add_argument('--seed', type=int, default=20261021, help='Training random seed')
    p.add_argument('--smoke', action='store_true', help='Smoke run: 2 epochs, small batch')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    cfg = Stage7Config(seed=args.seed)
    train_flow(
        cache_dir=args.data,
        output_dir=args.output,
        config=cfg,
        seed=args.seed,
        smoke=args.smoke,
        device=args.device,
    )


if __name__ == '__main__':
    main()
