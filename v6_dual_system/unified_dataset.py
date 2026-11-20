"""Training windows joined by immutable episode, split and CARLA frame."""
import hashlib
import io
import json
import math

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,Sampler

from .sensors import DATA
from .unified_data import BASE


class UnifiedData(Dataset):
    def __init__(self,split):
        self.episodes=[]
        self.index=[]
        self.groups=[[] for _ in range(4)]
        self.files={}
        digest=hashlib.sha256()
        for source in ("original","supplement"):
            paths=json.loads((BASE/f"{split}-{source}.json").read_bytes())
            for rel in paths:
                folder=BASE/rel
                raw=(folder/"complete.json").read_bytes()
                meta=json.loads(raw)
                if meta["split"]!=split:raise ValueError("Split mismatch: "+rel)
                digest.update(rel.encode());digest.update(raw)
                with np.load(folder/"labels.npz") as f:
                    labels={k:f[k] for k in ("v","cmd","xy","rgb","curve","frame")}
                    if meta["occupancy_valid"]:labels["occ"]=f["occ"]
                digest.update(labels["frame"].tobytes())
                if not (folder/"nav.npy").is_file():raise ValueError(f"Missing prepared navigation: {rel}")
                labels["plane"]=np.load(folder/"plane.npy")
                ep=len(self.episodes)
                self.episodes.append((folder,meta,labels))
                for i,(cmd,k) in enumerate(zip(labels["cmd"],labels["curve"])):
                    group=3 if meta["fork"] else 2 if cmd else 1 if abs(k)>.008 else 0
                    self.groups[group].append(len(self.index))
                    self.index.append((ep,i))
        self.identity=digest.hexdigest()

    def __len__(self):return len(self.index)

    def __getitem__(self,idx):
        ep,i=self.index[idx]
        folder,meta,labels=self.episodes[ep]
        if ep not in self.files:
            self.files[ep]=((DATA/meta["rgb_source"]).open("rb"),np.load(folder/"lidar.npy",mmap_mode="r"),
                            np.load(folder/"nav.npy",mmap_mode="r"))
        stream,lidar,nav=self.files[ep]
        lo,hi=labels["rgb"][i]
        stream.seek(int(lo))
        with Image.open(io.BytesIO(stream.read(int(hi-lo)))) as image:
            rgb=np.asarray(image.convert("RGB")).transpose(2,0,1).copy()
        occ=np.unpackbits(labels["occ"][i],axis=-1) if meta["occupancy_valid"] else np.zeros((5,128,128),np.uint8)
        cmd=np.eye(3,dtype=np.float32)[int(labels["cmd"][i])]
        return rgb,lidar[i].copy(),nav[i,None].copy(),np.float32(labels["v"][i]),cmd,labels["xy"][i].astype(np.float32),occ,meta["occupancy_valid"],labels["plane"][i]


class BalancedBatches(Sampler):
    def __init__(self,dataset,batch_size=40):
        if batch_size%5:raise ValueError("Batch size must be divisible by five")
        self.groups=[np.array(g,dtype=np.int64) for g in dataset.groups]
        self.counts=[batch_size*2//5]+[batch_size//5]*3
        if any(not len(g) for g in self.groups):raise ValueError("Empty training category")
        self.batches=max(math.ceil(len(g)/c) for g,c in zip(self.groups,self.counts))
        self.epoch=0
    def __len__(self):return self.batches
    def set_epoch(self,epoch):self.epoch=epoch
    def __iter__(self):
        rng=np.random.default_rng(641+self.epoch)
        streams=[]
        for g,c in zip(self.groups,self.counts):
            size=self.batches*c
            streams.append(np.concatenate([rng.permutation(g) for _ in range(math.ceil(size/len(g)))])[:size].reshape(-1,c))
        for batch in np.concatenate(streams,axis=1):
            yield rng.permutation(batch).tolist()
