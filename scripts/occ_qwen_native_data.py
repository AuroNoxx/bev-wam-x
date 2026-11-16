"""Native datasets and explicit authorization gate for the isolated Trainer."""
import json,hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from b2d_native_occ import ROOT,load_sample

class NativeDataset(Dataset):
    def __init__(self,split,actor_only=False):
        if split not in ['train','validation']: raise ValueError('held-out test excluded')
        path=ROOT/'data/occ_v3/manifest.json'
        manifest=json.loads(path.read_text(encoding='utf-8-sig'))
        if manifest['preprocess_version']!=3 or manifest['occ_file_offset']!=-1:raise ValueError('unsupported pairing')
        self.names=tuple(manifest['splits'][split]);self.actor_only=actor_only
        self.manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        if len(set(self.names))!=len(self.names):raise ValueError('duplicate sample paths')
        if any(Path(n).parts[1]!=split for n in self.names):raise ValueError('split mismatch')
    def __len__(self):return len(self.names)
    def __getitem__(self,index):
        row,_=load_sample(self.names[index],include_labels=not self.actor_only)
        keys=['history','state'] if self.actor_only else ['history','state','trajectory','future']
        return {k:row[k] for k in keys}

def loader(split='train',actor_only=False):
    cfg=json.loads((ROOT/'configs/occ-qwen-native.json').read_text(encoding='utf-8-sig'))
    return StatefulDataLoader(NativeDataset(split,actor_only),batch_size=cfg['batch_size'],shuffle=split=='train',num_workers=0,generator=torch.Generator().manual_seed(cfg['seed']))

def require_training_authorization():
    cfg=json.loads((ROOT/'configs/occ-qwen-native.json').read_text(encoding='utf-8-sig'))
    if cfg.get('formal_training_paused',True) or not cfg.get('training_authorization'):
        raise RuntimeError('Native OCC-Qwen formal training is paused; old flow-v3 approval is not authorization.')



