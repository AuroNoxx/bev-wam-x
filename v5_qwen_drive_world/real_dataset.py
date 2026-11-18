"""Real actor/label separation and Lightning/TorchData stage input adapters."""
from .runtime import ROOT,configure
configure()
import hashlib,json
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from qwen_drive import CameraFrame,DrivingScene
from .data import SpeedBinBalancedSampler,audit_manifest
from .lightweight_input import planning_inputs


def load_actor(path):
    record=json.loads((ROOT/path).read_text(encoding='utf-8'))
    scene=DrivingScene(views={name:[CameraFrame(ROOT/p) for p in images] for name,images in record['images'].items()},
                       history=np.asarray(record['history']),history_velocity=np.asarray(record['history_velocity']),
                       history_acceleration=np.asarray(record['history_acceleration']),
                       ego_velocity=record['ego_velocity'],ego_acceleration=record['ego_acceleration'],
                       driving_command=record['driving_command'],nav_command=record['nav_command'])
    scene.instruction_text=(scene.instruction()+f"\nCurrent measured speed: {record['speed_kmh']/3.6:.3f} m/s; acceleration FLU: {record['ego_acceleration']}; "
                            f"target cruise: {record['target_speed_kmh']:.2f} km/h; navigation point FLU metres: {record['navigation_flu']}.")
    return scene,record


def inputs_for(processor,path,device,tokens):
    scene,record=load_actor(path)
    inputs=planning_inputs(processor,scene,device=device,limit=850-len(tokens))
    suffix=torch.tensor([tokens],device=device,dtype=torch.long)
    inputs['input_ids']=torch.cat((inputs['input_ids'],suffix),dim=1)
    inputs['input_audit']['total_tokens']+=len(tokens)
    return inputs


class Samples(Dataset):
    def __init__(self,records):self.records=records
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        record=self.records[index]
        with np.load(ROOT/record['label_path'],allow_pickle=False) as f:
            label={k:torch.from_numpy(np.asarray(f[k]).copy())[None] for k in f.files}
        return {'record':record,'labels':label}


def single(items):
    assert len(items)==1
    return items[0]


class RealData(pl.LightningDataModule):
    def __init__(self,phase):
        super().__init__();self.phase=phase
        path=ROOT/'data/v5-qwen-drive-world/manifest.json';self.manifest=json.loads(path.read_text(encoding='utf-8'))
        audit_manifest(self.manifest);self.identity=hashlib.sha256(path.read_bytes()).hexdigest()
        self.records=self.manifest['overfit64'] if phase=='overfit64' else self.manifest['splits']['train']
        self.validation=self.records if phase=='overfit64' else self.manifest['splits']['validation']
    def train_dataloader(self):
        if self.phase=='overfit64':
            return StatefulDataLoader(Samples(self.records),batch_size=1,shuffle=True,num_workers=0,collate_fn=single,
                                      generator=torch.Generator().manual_seed(71))
        sampler=SpeedBinBalancedSampler(self.records,batch_size=1,num_batches=2000,seed=71)
        return StatefulDataLoader(Samples(self.records),batch_sampler=sampler,num_workers=0,collate_fn=single)
    def val_dataloader(self):
        return StatefulDataLoader(Samples(self.validation),batch_size=1,num_workers=0,collate_fn=single)
