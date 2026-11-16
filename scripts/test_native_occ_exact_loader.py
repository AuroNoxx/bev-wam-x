"""CPU integer-only production-size loader RNG/epoch-boundary regression."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['HIP_VISIBLE_DEVICES']='-1';os.environ['ROCR_VISIBLE_DEVICES']='-1'
import copy,json
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from native_occ_exact_loader import ExactResumeDataLoader
from b2d_native_occ import ROOT

def make(exact=False):
    cls=ExactResumeDataLoader if exact else StatefulDataLoader
    return cls(list(range(5296)),batch_size=8,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(71),**({'allow_legacy_resume':True} if exact else {}))
def values(it):return [x.tolist() for x in it]
def main():
    torch.set_num_threads(1);loader=make();it=iter(loader);first=[];snap={}
    for i in range(662):
        first.append(next(it).tolist())
        if i+1 in (2,650,662):snap[str(i+1)]=copy.deepcopy(loader.state_dict())
    try:next(it)
    except StopIteration:pass
    else:raise AssertionError('expected boundary')
    snap['finished']=copy.deepcopy(loader.state_dict());second=values(iter(loader));third=values(iter(loader));checks=[]
    for key,state in snap.items():
        resumed=make(True);resumed.load_state_dict(state)
        current=values(iter(resumed))
        if key=='finished':assert current==second;assert values(iter(resumed))==third
        else:assert current==first[int(key):];assert values(iter(resumed))==second;assert values(iter(resumed))==third
        checks.append({'legacy_snapshot':key,'iterator_finished':state['_iterator_finished'],'exact_current_and_next_epochs':True})
    # New-format restore, then checkpoint restored iterator a second time.
    original=make(True);it=iter(original);prefix=[next(it).tolist() for _ in range(650)];state1=copy.deepcopy(original.state_dict());tail=values(it);epoch2=values(iter(original));epoch3=values(iter(original))
    restored=make(True);restored.load_state_dict(state1);rit=iter(restored);seen=[next(rit).tolist() for _ in range(5)];state2=copy.deepcopy(restored.state_dict());assert seen==tail[:5]
    twice=make(True);twice.load_state_dict(state2);assert values(iter(twice))==tail[5:];assert values(iter(twice))==epoch2;assert values(iter(twice))==epoch3
    report={'passed':True,'cpu_integer_only':True,'dataset_length':5296,'batch_size':8,'seed':71,'workers':0,'legacy_cases':checks,'exact_to_exact_second_resume':True,'compared_individual_indices':True,'epochs_checked_after_resume':2}
    (ROOT/'artifacts/native-occ-cache-pilot/integer-loader-regression.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
