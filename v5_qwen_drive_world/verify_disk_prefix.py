"""Real pretrained joint SFT: cold compute vs disk read, loss + all gradients."""
from .runtime import ROOT,configure,math_attention
configure()
import json,time,statistics,gc
import torch
from .train_disk_cached import DiskCachedTask
from .real_dataset import RealData
from .sft_task import SupervisedTask
from .frozen_prefix import FrozenPrefixCache


def main():
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.8);torch.backends.cudnn.benchmark=False
    data=RealData('subset2000');folder=ROOT/'artifacts/v5-qwen-drive-world/subset2000'
    cfg=json.loads((folder/'config.json').read_text())
    task=DiskCachedTask(cfg,data.identity,'subset2000').cuda()
    checkpoint=torch.load(folder/'latest.ckpt',map_location='cpu',weights_only=False,mmap=True)
    task.on_load_checkpoint(checkpoint);task.load_state_dict(checkpoint['state_dict']);task.pending_rng=None
    loader=data.train_dataloader();loader.load_state_dict(checkpoint['loops']['fit_loop']['state_dict']['combined_loader'][0])
    del checkpoint;gc.collect()
    batch=next(iter(loader));batch['labels']={k:v.cuda() for k,v in batch['labels'].items()}
    results=[];reference=None;seed=1234;task.train();disk=task.prefix
    for mode in ('cold','disk','disk_repeat'):
        if mode=='cold':
            task.prefix=FrozenPrefixCache();task.encode_path=SupervisedTask.encode_path.__get__(task,DiskCachedTask)
        else:
            task.prefix=disk;task.encode_path=DiskCachedTask.encode_path.__get__(task,DiskCachedTask)
        task.zero_grad(set_to_none=True);task.prefix.entries.clear();task.prefix.bytes=0
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.cuda.synchronize();start=time.perf_counter()
        hits=disk.disk_hits
        with torch.autocast('cuda',dtype=torch.bfloat16),math_attention():loss=task.training_step(batch,0)
        with math_attention():loss.backward()
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        gradients={n:p.grad.detach().cpu().clone() for n,p in task.named_parameters() if p.grad is not None}
        if reference is None:
            reference=gradients;reference_loss=float(loss.detach())
            for (path,planner),entry in task.prefix.entries.items():disk.persist(path,planner,entry)
        else:
            assert disk.disk_hits-hits==4,'Must reload actor and all three teacher entries from disk'
            torch.testing.assert_close(loss.detach().cpu(),torch.tensor(reference_loss),rtol=1e-6,atol=1e-6)
            assert gradients.keys()==reference.keys()
            for n in reference:torch.testing.assert_close(gradients[n],reference[n],rtol=1e-5,atol=1e-7,msg=lambda m:f'{n}: {m}')
        results.append({'mode':mode,'seconds':seconds,'loss':float(loss.detach()),'gradient_tensors':len(gradients),'disk_hits':disk.disk_hits-hits})
        del loss,gradients
        print(json.dumps(results[-1]),flush=True)
    report={'passed':True,'real_sample':batch['record'],'comparisons':results,
            'disk_mean_seconds':statistics.mean(r['seconds'] for r in results[1:]),
            'native_resolution_unchanged':True,'optimizer_updates_in_test':0,'cache':task.prefix.statistics(),
            'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
    (folder/'disk-cache-contract.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
