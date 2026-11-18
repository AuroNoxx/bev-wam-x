"""CPU-only sampler replay: remaining unique frozen encodes, not GPU work."""
from .runtime import ROOT,configure
configure()
import json,statistics
from collections import Counter
import torch
from .data import SpeedBinBalancedSampler


def main():
    folder=ROOT/'artifacts/v5-qwen-drive-world/subset2000'
    manifest=json.loads((ROOT/'data/v5-qwen-drive-world/manifest.json').read_text());records=manifest['splits']['train']
    cutover=json.loads((folder/'disk-cutover.json').read_text(encoding='utf-8-sig'))
    step=cutover['checkpoint_step'];epoch=step*8//2000
    sampler=SpeedBinBalancedSampler(records,batch_size=1,num_batches=2000,seed=71)
    for _ in range(epoch):list(sampler)
    iterator=iter(sampler)
    for _ in range(step*8%2000):next(iterator)
    rows=[json.loads(x) for x in (folder/'samples.jsonl').read_text().splitlines()]
    contract=json.loads((folder/'disk-cache-contract.json').read_text());warm=contract['disk_mean_seconds']
    cold={s:statistics.median(r['seconds'] for r in rows if r['step']<=step and r['source']==s and r['seconds']>(15 if s=='bench2drive' else 4)) for s in ('bench2drive','highway')}
    first=contract['real_sample'];cached={(first['actor_path'],True)}|{(p,False) for p in first['future_actor_paths']}
    phases=[]
    for e in range(epoch,3):
        misses=Counter();draws=0
        for batch in sampler:
            r=records[batch[0]];draws+=1
            for key in [(r['actor_path'],True)]+[(p,False) for p in r['future_actor_paths']]:
                if key not in cached:cached.add(key);misses[r['source']]+=1
        seconds=draws*warm+sum(n*max(0,cold[s]-warm)/4 for s,n in misses.items())
        phases.append({'epoch':e+1,'remaining_draws':draws,'new_frozen_scenes':dict(misses),'estimated_compute_hours':seconds/3600})
    ns=contract['cache']['namespace'];cache=ROOT/'data/v5-qwen-drive-world/frozen-prefix-cache'/ns
    actor_sizes=[p.stat().st_size for p in (cache/'actor').glob('*.pt')]
    teacher_sizes=[p.stat().st_size for p in (cache/'future_teacher').glob('*.pt')]
    counts=Counter(role for _,role in cached)
    report={'from_optimizer_step':step,'method':'exact remaining sampler replay with per-source cold median and single-real-sample disk-hit timing; model estimate, not measured total',
            'cold_sample_seconds':cold,'hot_sample_seconds':warm,'phases':phases,
            'estimated_compute_hours':sum(p['estimated_compute_hours'] for p in phases),
            'estimated_cache_GiB':(counts[True]*statistics.mean(actor_sizes)+counts[False]*statistics.mean(teacher_sizes))/2**30,
            'excludes':'validation, checkpoint writes, system load variation; scene-cost divided equally across four branches is approximate'}
    (folder/'disk-time-estimate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
