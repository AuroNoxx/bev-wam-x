"""Train Stage 6 action-conditioned predictor or its matched zero-command control."""
import argparse
from dataclasses import replace
from pathlib import Path
from configs.stage6 import Stage6Config
from training.action_conditioned import train


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--unconditional',action='store_true')
    p.add_argument('--seed',type=int,default=20261019);p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    config=replace(Stage6Config(),seed=args.seed,epochs=args.epochs)
    train(args.data,args.output,config,not args.unconditional,args.smoke)


if __name__=='__main__':main()
