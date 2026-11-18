"""Check resume bookkeeping without saving from an uncommitted microbatch."""
from .runtime import ROOT,configure
configure()
import json
from types import SimpleNamespace
from unittest.mock import patch
from .resume_disk_cached import UpdateBoundaryCheckpoint
from .train_supervised import RollingCheckpoint


def main():
    checkpoint=UpdateBoundaryCheckpoint(every_n_train_steps=8,save_top_k=1)
    trainer=SimpleNamespace(global_step=80,accumulate_grad_batches=8,is_last_batch=False)
    checkpoint.on_train_start(trainer,None)
    assert checkpoint._last_global_step_saved==80
    with patch.object(RollingCheckpoint,'on_train_batch_end') as parent:
        for i in range(640,647):checkpoint.on_train_batch_end(trainer,None,None,None,i)
        assert parent.call_count==0
        trainer.global_step=81;checkpoint.on_train_batch_end(trainer,None,None,None,647)
        assert parent.call_count==1
    assert checkpoint.state_key==RollingCheckpoint(every_n_train_steps=8,save_top_k=1).state_key
    result={'passed':True,'resumed_step_registered':80,'first_seven_uncommitted_micros_do_not_save':True,
            'update_boundary_delegates_to_model_checkpoint':True,'callback_state_identity_compatible':True,
            'scope':'callback bookkeeping unit test; real resume and next checkpoint checked separately'}
    (ROOT/'artifacts/v5-qwen-drive-world/subset2000/checkpoint-boundary-contract.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
