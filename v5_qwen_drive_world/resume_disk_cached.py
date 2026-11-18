"""Disk-cache SFT entry with checkpoint writes only at completed updates."""
from .runtime import ROOT,configure
configure()
import json
from . import train_supervised as training
from . import train_disk_cached as disk_training


class UpdateBoundaryCheckpoint(training.RollingCheckpoint):
    @property
    def state_key(self):
        # Keep the original ModelCheckpoint bookkeeping compatible on resume.
        return super().state_key.replace('UpdateBoundaryCheckpoint{','RollingCheckpoint{',1)

    def on_train_start(self,trainer,pl_module):
        super().on_train_start(trainer,pl_module)
        # Lightning does not restore _last_global_step_saved. Without this,
        # resume at a divisible step can save after its first uncommitted micro.
        self._last_global_step_saved=trainer.global_step

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        if (batch_idx+1)%trainer.accumulate_grad_batches and not trainer.is_last_batch:return
        super().on_train_batch_end(trainer,pl_module,outputs,batch,batch_idx)


class BoundaryReport(disk_training.DiskReport):
    def on_train_start(self,trainer,task):
        super().on_train_start(trainer,task)
        path=ROOT/'artifacts/v5-qwen-drive-world/two-stage-run.json'
        run=json.loads(path.read_text());run['entrypoint']='v5_qwen_drive_world.resume_disk_cached'
        run['checkpoint_boundary_guard']=True;path.write_text(json.dumps(run,indent=2))


def main():
    training.RollingCheckpoint=UpdateBoundaryCheckpoint
    disk_training.DiskReport=BoundaryReport
    disk_training.main()


if __name__=='__main__':main()
