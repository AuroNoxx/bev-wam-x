import io
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .sensors import CAL_ID, DATA


def mirror_lane_sample(sample):
    rgb,bev,state,cmd,xy,occ,lane=sample
    state,cmd,xy=state.copy(),cmd.copy(),xy.copy()
    state[2]*=-1
    cmd[[1,2]]=cmd[[2,1]]
    xy[:,1]*=-1
    # Image width runs left/right. BEV and occupancy store lateral bins as rows.
    return (rgb[:,:,::-1].copy(),bev[:,::-1,:].copy(),state,cmd,xy,
            occ[:,::-1,:].copy(),-lane)


class FastData(Dataset):
    def __init__(self,split,corrective=False,lane_supervision=False):
        suffix="-corrective" if corrective else ""
        index=json.loads((DATA/f"manifest{suffix}.json").read_bytes())
        if index["calibration"]!=CAL_ID or json.loads((DATA/f"audit{suffix}.json").read_bytes())["failures"]:
            raise ValueError("Stage 1 sensor/quality gate failed")
        self.entries=index[split]
        self.identity=index["identity"]
        self.opened={}
        self.lane_truth=None
        self.lane_identity=None
        self.mirror=lane_supervision and split=="train"
        self.flip_rng=None
        if lane_supervision:
            folder=DATA/"lane-supervision"/self.identity
            descriptor=json.loads((folder/"identity.json").read_bytes())
            digest=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest()
            if descriptor["manifest"]!=self.identity or descriptor["calibration"]!=CAL_ID:
                raise ValueError("Lane supervision does not match this manifest/calibration")
            ref=descriptor["splits"][split]
            order=hashlib.sha256(json.dumps(self.entries,separators=(",",":")).encode()).hexdigest()
            raw=(folder/ref["file"]).read_bytes()
            if order!=ref["order_sha256"] or hashlib.sha256(raw).hexdigest()!=ref["sha256"]:
                raise ValueError("Lane supervision data or window ordering changed")
            self.lane_truth=np.load(io.BytesIO(raw),allow_pickle=False)
            if self.lane_truth.shape!=(len(self.entries),2) or not np.isfinite(self.lane_truth).all():
                raise ValueError("Invalid lane supervision")
            self.lane_identity=digest

    def __len__(self):
        return len(self.entries)

    def _mirror_generator(self):
        if self.flip_rng is None:
            # The worker seed is assigned by DataLoader before its first sample.
            self.flip_rng=torch.Generator().manual_seed(torch.initial_seed())
        return self.flip_rng

    def state_dict(self):
        if not self.mirror:return {}
        return {"lane_identity":self.lane_identity,"flip_rng":self._mirror_generator().get_state()}

    def load_state_dict(self,state):
        if not self.mirror:return
        if state["lane_identity"]!=self.lane_identity:
            raise ValueError("Mirrored dataset identity changed on resume")
        self._mirror_generator().set_state(state["flip_rng"])

    def __getitem__(self,idx):
        rel,i=self.entries[idx]
        if rel not in self.opened:
            folder=DATA/rel
            with np.load(folder/"labels.npz") as labels:
                a=dict(labels)
            with np.load(folder/"stream.npz") as stream:
                offsets=stream["rgb"]
            raw=np.memmap(folder/"rgb.bin",dtype=np.uint8,mode="r")
            bev=np.memmap(folder/"bev.bin",dtype=np.uint8,mode="r").reshape(-1,3,128,128)
            self.opened[rel]=(a,offsets,raw,bev)
        a,offsets,raw,bev=self.opened[rel]
        fr=a["frame"][i]
        imgs=[]
        for j in (a["history"][i],fr):
            with Image.open(io.BytesIO(raw[offsets[j]:offsets[j+1]])) as im:
                imgs.append(np.asarray(im).transpose(2,0,1))
        cmd=np.zeros(4,np.float32)
        cmd[a["intent"][i]]=1
        cmd[3]=a["target"][i]
        sample=(np.concatenate(imgs),bev[fr].copy(),a["state"][i].astype(np.float32),
                cmd,a["xy"][i].astype(np.float32),np.unpackbits(a["occ"][i],axis=-1))
        if self.lane_truth is None:return sample
        sample=sample+(self.lane_truth[idx].copy(),)
        if self.mirror and torch.rand((),generator=self._mirror_generator()).item()<.5:
            return mirror_lane_sample(sample)
        return sample


class DaggerMix(FastData):
    """Original training windows plus committed, map-labelled learner observations."""
    def __init__(self):
        super().__init__("train",corrective=True,lane_supervision=True)
        self.base_count=len(self.entries)
        self.episodes=[]
        self.ends=[]
        self.shards={}
        self.max_bytes=6*360*640+3*128*128
        counts={"Town04":0,"Town06":0}
        identities=[]
        for meta_file in sorted((DATA/"dagger-epoch26").glob("*/*/complete.json")):
            meta=json.loads(meta_file.read_bytes())
            n=meta["count"]
            if (meta["calibration"]!=CAL_ID or not meta["empty_dynamic_scene"] or
                meta["checkpoint"]!="93f9184ff76908f77ce89dc016ce4ff6f35064bd68630c53c085ce0ede4a6541" or
                set(meta["protected_roads"]) & (set(meta["teacher_roads"])|set(meta["visited_roads"]))):
                raise ValueError(f"Invalid collection identity or route overlap: {meta_file}")
            root=meta_file.parent
            raw=root/"observations.bin"
            with raw.open("rb") as stream:
                digest=hashlib.file_digest(stream,"sha256").hexdigest()
            if raw.stat().st_size!=n*self.max_bytes or digest!=meta["observations_sha256"]:
                raise ValueError(f"Invalid sensor shard {raw}")
            blob=(root/"labels.npz").read_bytes()
            if hashlib.sha256(blob).hexdigest()!=meta["labels_sha256"]:raise ValueError("Label hash mismatch")
            with np.load(io.BytesIO(blob)) as data:
                labels=dict(data)
            if (labels["xy"].shape!=(n,40,2) or not np.all(np.diff(labels["frame"])>0)
                    or not all(np.isfinite(v).all() for v in labels.values())):raise ValueError("Invalid DAgger labels")
            end=labels["xy"][:,-1]
            if (np.abs(labels["offset"]).max()>7.5 or
                    (np.abs(np.arctan2(end[:,1],end[:,0]))>np.deg2rad(50)).any()):
                raise ValueError(f"Map teacher leaves the local driving corridor: {meta_file}")
            self.episodes.append((raw,labels))
            counts[meta["town"]]+=n
            self.ends.append(sum(counts.values()))
            identities.append(hashlib.sha256(meta_file.read_bytes()).hexdigest())
        if list(counts.values())!=[1000,1000]:raise ValueError(f"Incomplete DAgger collection: {counts}")
        self.training_identity=hashlib.sha256(json.dumps([self.identity,self.lane_identity,identities]).encode()).hexdigest()

    def __len__(self):
        return self.base_count+self.ends[-1]

    def __getitem__(self,idx):
        if idx<self.base_count:return super().__getitem__(idx)
        pos=idx-self.base_count
        shard=int(np.searchsorted(self.ends,pos,side="right"))
        i=pos-(self.ends[shard-1] if shard else 0)
        path,labels=self.episodes[shard]
        if shard not in self.shards:
            self.shards[shard]=np.memmap(path,dtype=np.uint8,mode="r").reshape(-1,self.max_bytes)
        raw=self.shards[shard][i]
        boundary=6*360*640
        sample=(raw[:boundary].reshape(6,360,640).copy(),raw[boundary:].reshape(3,128,128).copy(),
                labels["state"][i].copy(),np.float32([1,0,0,labels["target"][i]]),labels["xy"][i].copy(),
                np.zeros((4,128,128),np.uint8),labels["lane"][i].copy())
        if self.mirror and torch.rand((),generator=self._mirror_generator()).item()<.5:return mirror_lane_sample(sample)
        return sample


class NavigationData(Dataset):
    """Join anchor metadata before applying a single coherent random reflection."""
    def __init__(self, split, spatial=False, film=False):
        self.film=film
        self.spatial=spatial
        self.training = split == "train"
        self.data = DaggerMix() if self.training else FastData(split, True, True)
        self.data.mirror = False
        self.identity = self.data.identity
        self.lane_identity = self.data.lane_identity
        folder = DATA / ("film-navigation" if film else "navigation") / self.identity
        descriptor = (folder / "identity.json").read_bytes()
        desc = json.loads(descriptor)
        if desc["manifest"] != self.identity:
            raise ValueError("Navigation manifest changed")
        def read(ref):
            raw = (folder / ref["file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
                raise ValueError("Navigation checksum mismatch")
            data = np.load(io.BytesIO(raw), allow_pickle=False)
            if len(data) != ref["count"] or not np.isfinite(data).all():
                raise ValueError("Invalid navigation anchor rows")
            return data
        self.focus_descriptor=desc.get("focus",{})
        self.metadata_folder=folder
        ref = desc["splits"][split]
        order = hashlib.sha256(json.dumps(self.data.entries,separators=(",",":")).encode()).hexdigest()
        if order != ref["order"]:
            raise ValueError("Window order changed")
        nav = [read(ref)]
        indices = [np.arange(len(nav[0]))]
        if self.training:
            start = self.data.base_count
            for path,labels in self.data.episodes:
                rows = read(desc["dagger"][path.parent.relative_to(DATA).as_posix()])
                indices.append(start + rows[:,0].astype(np.int64))
                nav.append(rows[:,1:])
                start += len(labels["frame"])
        self.indices = np.concatenate(indices)
        self.nav = np.concatenate(nav)
        if self.nav.shape != (len(self.indices),12 if film else 4):
            raise ValueError("Invalid navigation input shape")
        self.training_identity = hashlib.sha256(descriptor + getattr(self.data,"training_identity","").encode()).hexdigest()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.data[int(self.indices[idx])]
        nav = self.nav[idx].copy()
        return self.augment(sample,nav)

    def augment(self,sample,nav,fork=False):
        straight=self.film and abs(nav[9])<1.2
        signed_delta=2*nav[-1]
        if self.training:
            generator = self.data._mirror_generator()
            if self.film:nav += .4*torch.randn(12,generator=generator).numpy()
            elif self.spatial:nav[1::2] += .8*torch.randn(2,generator=generator).numpy()
            else:nav += torch.randn(4,generator=generator).numpy()
            if torch.rand((),generator=generator).item() < .5:
                sample = mirror_lane_sample(sample)
                nav[1::2] *= -1
                signed_delta *= -1
        rgb,bev,state,cmd,xy,occ,lane = sample
        nav[:4]/=60.
        nav[4:]/=50.
        result=(rgb,bev,np.r_[state[:1],nav],cmd,xy,occ,lane)
        return (*result,straight,np.float32(signed_delta if fork else 0.)) if self.film else result

    def state_dict(self):
        return {"identity":self.training_identity,"rng":self.data._mirror_generator().get_state()}

    def load_state_dict(self, state):
        if state["identity"] != self.training_identity:
            raise ValueError("Navigation data changed on resume")
        self.data._mirror_generator().set_state(state["rng"])


class SpatialData(NavigationData):
    def __init__(self,split,film=False):
        super().__init__(split,spatial=True,film=film)
        self.base_count=len(self.indices)
        self.focus=None
        if not self.training:return
        root=DATA/"spatial-navigation"
        doc=(root/"manifest.json").read_bytes()
        desc=json.loads(doc)
        self.focus=FastData("train",True)
        self.focus.entries=[]
        lane=[];nav=[]
        fork=[]
        digest=hashlib.sha256(self.training_identity.encode()+doc)
        for name in desc["episodes"]:
            folder=root/name
            raw=(folder/"complete.json").read_bytes()
            meta=json.loads(raw)
            if meta["split"]!="train":continue
            if meta["calibration"]!=CAL_ID:raise ValueError("Targeted calibration mismatch")
            with np.load(folder/"labels.npz") as lab:
                n=len(lab["frame"])
                if n!=meta["count"]:raise ValueError("Incomplete targeted episode")
                if film:
                    ref=self.focus_descriptor[name]
                    raw_nav=(self.metadata_folder/ref["file"]).read_bytes()
                    if hashlib.sha256(raw_nav).hexdigest()!=ref["sha256"]:
                        raise ValueError("Targeted route changed")
                    nav.append(np.load(io.BytesIO(raw_nav),allow_pickle=False))
                    fork.extend([meta["case"] in ("left","right")]*n)
                else:
                    nav.append(lab["nav"].astype(np.float32))
                lane.append(lab["lane"].astype(np.float32))
            self.focus.entries.extend([(folder.relative_to(DATA).as_posix(),i) for i in range(n)])
            digest.update(raw)
            digest.update((folder/"labels.npz").read_bytes())
        self.focus.lane_truth=np.concatenate(lane)
        self.focus_nav=np.concatenate(nav)
        self.fork=np.array(fork,dtype=bool)
        self.training_identity=digest.hexdigest()

    def __len__(self):
        return self.base_count+(len(self.focus) if self.focus is not None else 0)

    def __getitem__(self,idx):
        if idx<self.base_count:return super().__getitem__(idx)
        i=idx-self.base_count
        return self.augment(self.focus[i],self.focus_nav[i].copy(),self.film and self.fork[i])


class ForkSampler(torch.utils.data.Sampler):
    def __init__(self,data):
        forks=np.flatnonzero(data.fork)+data.base_count
        if len(forks)!=1200:
            raise ValueError("Expected 1200 eligible fork windows")
        self.indices=np.concatenate((np.arange(len(data)),np.tile(forks,4)))
        self.epoch=0

    def __len__(self):
        return len(self.indices)

    def set_epoch(self,epoch):
        self.epoch=epoch

    def __iter__(self):
        return iter(np.random.default_rng(641+self.epoch).permutation(self.indices).tolist())


class FocusSampler(torch.utils.data.Sampler):
    """Cover the original dataset every epoch and draw one special window per three original."""
    def __init__(self,data):
        self.base=data.base_count
        self.focus=len(data)-self.base
        self.quartets=(self.base+2)//3
        self.epoch=0

    def __len__(self):
        return 4*self.quartets

    def set_epoch(self,epoch):
        self.epoch=epoch

    def __iter__(self):
        rng=np.random.default_rng(641+self.epoch)
        def balanced(size,count):
            return np.concatenate([rng.permutation(size) for _ in range((count+size-1)//size)])[:count]
        batches=np.column_stack((balanced(self.base,3*self.quartets).reshape(-1,3),
                                 self.base+balanced(self.focus,self.quartets)))
        return iter(rng.permuted(batches,axis=1).ravel().tolist())
