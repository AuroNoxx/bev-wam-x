"""Lightning checkpoint storage: validate a temporary file before replacement."""
import os
from pathlib import Path
import torch
from lightning_fabric.plugins.io import TorchCheckpointIO


class VerifiedCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(self,checkpoint,path,storage_options=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.pending')
        try:
            super().save_checkpoint(checkpoint,temporary,storage_options)
            check=torch.load(temporary,map_location='cpu',weights_only=False)
            required={'state_dict','optimizer_states','lr_schedulers','loops','global_step','task_state'}
            if not required<=check.keys():raise ValueError('Incomplete resumable checkpoint')
            if not all(torch.isfinite(t).all() for t in check['state_dict'].values() if torch.is_floating_point(t)):
                raise ValueError('Nonfinite trainable checkpoint parameter')
            if check['global_step']!=checkpoint['global_step']:raise ValueError('Checkpoint step changed')
            del check
            with temporary.open('rb+') as stream:os.fsync(stream.fileno())
            os.replace(temporary,path)
        except Exception:
            # The previous latest/best is untouched. Only this failed temp is disposable.
            if temporary.exists():temporary.unlink()
            raise
