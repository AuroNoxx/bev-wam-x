"""CPU-only legacy NativeDataset -> cached Dataset restart equivalence evidence.
Not a trainer checkpoint migration tool. Full run integration must verify the
checkpoint task_state manifest/settings and full split order before replacement.
"""
import argparse,copy,json,subprocess,sys
from pathlib import Path
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from b2d_native_occ import ROOT
from occ_qwen_native_data import NativeDataset
from native_occ_cached_data import NativeCachedDataset,sha,digest

def make(ds):return StatefulDataLoader(ds,batch_size=1,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(817))
def main():
    p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    folder=ROOT/'artifacts/native-occ-cache-pilot';report=json.loads((folder/'report.json').read_text());index=report['index'];ds=NativeCachedDataset(index,'train')
    protocol={'manifest_sha256':sha(ROOT/'data/occ_v3/manifest.json'),'names':list(ds.names),'batch_size':1,'shuffle':True,'seed':817,'workers':0}
    statefile=folder/'legacy-state.pt';resultfile=folder/'legacy-restarted.pt'
    if a.child:
        saved=torch.load(statefile,weights_only=False)
        assert saved['protocol']==protocol,'migration protocol changed'
        restored=make(ds);restored.load_state_dict(saved['state']);torch.save(list(restored),resultfile);return
    raw=NativeDataset('train');raw.names=ds.names
    assert raw.manifest_sha256==protocol['manifest_sha256']
    loader=make(raw);it=iter(loader);next(it);state=copy.deepcopy(loader.state_dict());expected=list(it)
    torch.save({'protocol':protocol,'state':state},statefile)
    child=subprocess.run([sys.executable,__file__,'--child'],capture_output=True,text=True,timeout=120);assert child.returncode==0,child.stderr
    actual=torch.load(resultfile,weights_only=False);ok=len(actual)==len(expected) and all(torch.equal(x[k],y[k]) for x,y in zip(expected,actual) for k in x);assert ok
    result={'raw_workers0_to_cached_workers0':ok,'fresh_process':True,'remaining_batches':len(expected),'protocol':protocol,'subset_test_only':True,'full_trainer_resume_not_run':True,'warning':'workers0->2 and 2->0 are incompatible; never discard loader state to change workers'}
    (folder/'legacy-dataset-resume.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
