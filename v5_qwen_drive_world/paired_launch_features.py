"""Cache official step-750 actor features for real matched stop/launch pairs."""
import json
from .geometry_repair_features import actor_features, load_base, math_attention, RealData, torch, RUN
from .mine_launch_pairs import PAIR, load, ROOT
from .train_scale_profile import fingerprint
from .geometry_repair_features import BASE


def main(folder=PAIR):
    rows=load(folder,'cohort.json')
    model_identity=load(RUN,'feature-identity.json')
    assert fingerprint(BASE/'latest.ckpt')==model_identity['base_sha256']
    torch.set_num_threads(4);torch.manual_seed(71)
    torch.cuda.set_per_process_memory_fraction(.8)
    torch.backends.cudnn.benchmark=False
    task=load_base(RealData('subset2000'))
    dest=folder/'features'
    dest.mkdir(exist_ok=True)
    with torch.inference_mode(),math_attention(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i,r in enumerate(rows):
            out=dest/f'{i:03}.pt'
            # Same diffusion noise for the two branches of each road.
            seed=(65001 if folder!=PAIR else 55001)+i//2
            identity=dict(actor_path=r['actor_path'],actor_sha256=fingerprint(ROOT/r['actor_path']),
                          model_sha256=model_identity['base_sha256'],seed=seed)
            if out.exists():
                if torch.load(out,map_location='cpu',weights_only=True)['identity']!=identity:
                    raise ValueError('Pair cache identity changed')
                continue
            feat=actor_features(task,r,seed)
            torch.save(dict(features=feat,identity=identity),out.with_suffix('.pending'))
            out.with_suffix('.pending').replace(out)
            if i%4==3:print(i+1,'/',len(rows),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--independent',action='store_true')
    main(PAIR/'independent-roads' if parser.parse_args().independent else PAIR)
