"""CPU-only persistence, role separation, invalidation and atomic-write checks."""
from .runtime import ROOT,configure
configure()
import json
from unittest.mock import patch
import torch
from .disk_prefix import DiskPrefixCache


def main():
    folder=ROOT/'artifacts/v5-qwen-drive-world/disk-cache-storage-check';folder.mkdir(parents=True,exist_ok=True)
    actor=folder/'actor.json';actor.write_text(json.dumps({'images':{},'version':1}))
    path=str(actor.relative_to(ROOT));cache=DiskPrefixCache({'schema':'storage-only-unit-check-v1'})
    entry={'hidden':torch.arange(32,dtype=torch.bfloat16).reshape(1,4,8),'positions':torch.arange(4),
           'prefix_kv':[(torch.ones(1,4,2,2,dtype=torch.bfloat16),torch.zeros(1,4,2,2,dtype=torch.bfloat16)) for _ in range(6)],
           'inputs':{'input_ids':torch.arange(4)},'bytes':512}
    cache.persist(path,True,entry)
    reopened=DiskPrefixCache(cache.identity);loaded=reopened.lookup(path,True)
    torch.testing.assert_close(loaded['hidden'],entry['hidden'],rtol=0,atol=0)
    assert reopened.lookup(path,False) is None,'Actor cache cannot be used for future-teacher role'
    actor.write_text(json.dumps({'images':{},'version':2}))
    fresh=DiskPrefixCache(cache.identity);assert fresh.lookup(path,True) is None
    target,_=fresh.location(path,True)
    try:
        with patch('v5_qwen_drive_world.disk_prefix.os.replace',side_effect=OSError('injected atomic replace failure')):
            fresh.persist(path,True,entry)
    except OSError:pass
    else:raise AssertionError('Injected failure did not occur')
    assert not target.exists() and not target.with_suffix('.pending').exists()
    actor.write_text(json.dumps({'images':{},'version':1}))
    assert DiskPrefixCache(cache.identity).lookup(path,True) is not None
    result={'passed':True,'roundtrip_exact':True,'role_separated':True,'input_change_invalidates':True,'failure_preserves_existing':True,
            'scope':'storage only; actual pretrained joint gradient check is separate'}
    (folder/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
