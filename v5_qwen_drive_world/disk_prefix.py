"""Lossless disk L2 for the existing frozen-prefix RAM cache only.

Actor and future-teacher entries have separate namespaces. Trainable last-eight
layers, EMA adapters and World outputs are never persisted as features.
"""
import hashlib,json,os,time
from pathlib import Path
import torch
from .runtime import ROOT
from .frozen_prefix import FrozenPrefixCache


def sha(payload):return hashlib.sha256(payload).hexdigest()


class DiskPrefixCache(FrozenPrefixCache):
    def __init__(self,identity,max_bytes=None):
        settings=json.loads((ROOT/'configs/v5-qwen-drive-world/frozen-prefix-disk.json').read_text())
        super().__init__(max_bytes if max_bytes is not None else int(settings['ram_GiB']*2**30))
        self.disk_budget=int(settings['disk_budget_GiB']*2**30)
        self.identity=identity;self.namespace=sha(json.dumps(identity,sort_keys=True).encode())[:24]
        self.folder=ROOT/settings['cache_root']/self.namespace
        if not self.folder.resolve().is_relative_to(ROOT):raise ValueError('Cache must stay in this project')
        self.folder.mkdir(parents=True,exist_ok=True)
        (self.folder/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
        self.disk_hits=0;self.disk_writes=0;self.read_seconds=0.;self.write_seconds=0.
        self.disk_bytes=sum(p.stat().st_size for p in self.folder.rglob('*.pt'))

    def signature(self,path,planner):
        actor=ROOT/path;payload=actor.read_bytes();record=json.loads(payload)
        images=[]
        for view,paths in record['images'].items():
            for p in paths:
                stat=(ROOT/p).stat();images.append((view,p,stat.st_size,stat.st_mtime_ns))
        return sha(json.dumps([path,planner,sha(payload),images],sort_keys=True).encode())

    def location(self,path,planner):
        signature=self.signature(path,planner)
        return self.folder/('actor' if planner else 'future_teacher')/f'{signature}.pt',signature

    def lookup(self,path,planner):
        key=(path,planner)
        if key in self.entries:return self.entries[key]
        file,signature=self.location(path,planner)
        if not file.exists():return None
        started=time.perf_counter()
        saved=torch.load(file,map_location='cpu',weights_only=True)
        if saved['namespace']!=self.namespace or saved['signature']!=signature or saved['planner']!=planner:
            raise ValueError('Frozen cache identity or actor/teacher role differs')
        entry=saved['entry']
        if len(entry['prefix_kv'])!=(6 if planner else 0):raise ValueError('Frozen KV group count differs')
        if entry['hidden'].dtype!=torch.bfloat16:raise ValueError('Cache altered frozen-feature precision')
        while self.entries and self.bytes+entry['bytes']>self.max_bytes:
            _,old=self.entries.popitem(last=False);self.bytes-=old['bytes']
        self.entries[key]=entry;self.bytes+=entry['bytes'];self.disk_hits+=1
        self.read_seconds+=time.perf_counter()-started
        return entry

    def persist(self,path,planner,entry):
        file,signature=self.location(path,planner)
        if file.exists():return
        if self.disk_bytes+entry['bytes']>self.disk_budget:raise RuntimeError('Frozen feature cache exceeds configured disk budget')
        file.parent.mkdir(parents=True,exist_ok=True)
        pending=file.with_suffix('.pending');started=time.perf_counter()
        try:
            torch.save({'namespace':self.namespace,'signature':signature,'planner':planner,'entry':entry},pending)
            with pending.open('rb+') as stream:os.fsync(stream.fileno())
            os.replace(pending,file)
        except Exception:
            if pending.exists():pending.unlink()
            raise
        self.disk_bytes+=file.stat().st_size;self.disk_writes+=1;self.write_seconds+=time.perf_counter()-started

    def encode(self,model,inputs,key,planner=True):
        entry=self.lookup(key,planner)
        native=super().encode(model,inputs,key,planner)
        if entry is None:self.persist(key,planner,self.entries[(key,planner)])
        return native

    def statistics(self):
        return dict(namespace=self.namespace,disk_hits=self.disk_hits,disk_writes=self.disk_writes,
                    disk_GiB=self.disk_bytes/2**30,read_seconds=self.read_seconds,write_seconds=self.write_seconds)


def identity_for(task):
    files=['components.py','native_scene.py','lightweight_input.py','real_dataset.py','frozen_prefix.py','accelerators.py']
    rows=task.model.vlm.get_input_embeddings().weight[task.token_ids].detach().cpu().contiguous()
    return {'schema':1,'model':task.base_identity,'manifest':task.manifest_identity,
            'config':task.cfg,'token_ids':task.token_ids,'token_rows_sha256':sha(rows.view(torch.uint8).numpy().tobytes()),
            'implementation':{f:sha((ROOT/'v5_qwen_drive_world'/f).read_bytes()) for f in files},
            'precision':'original BF16 frozen features; no quantization; FP32 trainable master',
            'torch':str(torch.__version__),'hip':torch.version.hip,'cache_code':sha(Path(__file__).read_bytes())}
