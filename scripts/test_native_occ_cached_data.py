"""CPU-only exact cache and process-restart StatefulDataLoader checks."""
import argparse,copy,json,subprocess,sys,time
from pathlib import Path
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from b2d_native_occ import ROOT,load_sample
from native_occ_cached_data import NativeCachedDataset,build_cache

def equal(a,b):
    return len(a)==len(b) and all(set(x)==set(y) and all(torch.equal(x[k],y[k]) for k in x) for x,y in zip(a,b))

def make(index,workers):
    return StatefulDataLoader(NativeCachedDataset(index,'train'),batch_size=1,shuffle=True,num_workers=workers,generator=torch.Generator().manual_seed(817),snapshot_every_n_steps=1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--index');p.add_argument('--workers',type=int);p.add_argument('--state');p.add_argument('--result');a=p.parse_args();torch.set_num_threads(1)
    if a.child:
        try:
            loader=make(a.index,a.workers);loader.load_state_dict(torch.load(a.state,weights_only=False));rows=list(loader)
            torch.save({'rows':rows},a.result)
        except Exception as e:torch.save({'error':repr(e)},a.result)
        return
    folder=ROOT/'artifacts/native-occ-cache-pilot';folder.mkdir(exist_ok=True)
    m=json.loads((ROOT/'data/occ_v3/manifest.json').read_text()); names=[n for s in ('train','validation') for n in m['splits'][s][:3]]
    index=build_cache(ROOT/'data/native_occ_aligned_cache_pilot',names)
    results={'index':str(index),'formal_training_changed':False,'gpu_used':False,'equivalence':[],'resume':[]}
    for split in ('train','validation'):
        start=time.perf_counter();ds=NativeCachedDataset(index,split);startup=time.perf_counter()-start; raw_s=0;cached_s=0
        for i,name in enumerate(ds.names):
            start=time.perf_counter();raw,_=load_sample(name);raw_s+=time.perf_counter()-start
            start=time.perf_counter();cached=ds[i];cached_s+=time.perf_counter()-start
            assert all(torch.equal(raw[k],cached[k]) for k in cached)
        results['equivalence'].append({'split':split,'samples':len(ds),'torch_equal':True,'source_hash_validation_seconds':startup,'raw_seconds':raw_s,'cached_seconds':cached_s,'speedup':raw_s/cached_s})
    for source_workers in (0,2):
        loader=make(index,source_workers);it=iter(loader);next(it)
        state=copy.deepcopy(loader.state_dict());statefile=folder/f'state-w{source_workers}.pt';torch.save(state,statefile)
        remaining=list(it)
        for target_workers in (0,2):
            resultfile=folder/f'restored-{source_workers}-{target_workers}.pt'
            cmd=[sys.executable,__file__,'--child','--index',str(index),'--workers',str(target_workers),'--state',str(statefile),'--result',str(resultfile)]
            proc=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
            actual=torch.load(resultfile,weights_only=False) if resultfile.exists() else {'error':proc.stderr}
            match='rows' in actual and equal(remaining,actual['rows'])
            results['resume'].append({'from_workers':source_workers,'to_workers':target_workers,'fresh_process':True,'exact_remaining_batches':match,'error':actual.get('error'),'stderr':proc.stderr[-2000:]})
            if source_workers==target_workers:assert match,results['resume'][-1]
    results['cache_bytes']=sum(p.stat().st_size for p in index.parent.rglob('*.npz'))
    (folder/'report.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
