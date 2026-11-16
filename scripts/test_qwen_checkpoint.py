from train_qwen_visual_pilot import *
from unittest.mock import patch
import copy
F=O/'checkpoint-test';F.mkdir(exist_ok=True)
g=torch.Generator().manual_seed(99);ds=torch.utils.data.TensorDataset(torch.randn(36,16,2048,generator=g),torch.randn(36,20,generator=g),torch.randn(36,25,2,generator=g),torch.randn(36,16,32,generator=g))
class Snap(pl.Callback):
 def on_train_batch_end(self,tr,mo,outputs,batch,batch_idx):
  if tr.global_step==257:tr.save_checkpoint(F/'split.ckpt')
def run(path,snapshot):
 pl.seed_everything(9101);m=Head('fm');d=Data(ds);t=pl.Trainer(accelerator='cpu',max_steps=258,max_epochs=-1,logger=False,enable_checkpointing=False,callbacks=[Snap()] if snapshot else [],plugins=[VerifiedCheckpointIO()],enable_progress_bar=False,enable_model_summary=False,limit_val_batches=0)
 t.fit(m,datamodule=d,ckpt_path=path,weights_only=False)
 return {k:v.detach().clone() for k,v in m.state_dict().items()},copy.deepcopy(t.optimizers[0].state_dict()),t.global_step
torch.set_num_threads(4);full,fo,fs=run(O/'fm/latest.ckpt',True);res,ro,rs=run(F/'split.ckpt',False);delta=max(float((full[k]-res[k]).abs().max()) for k in full);steps=[float(s['step']) for s in ro['state'].values()];assert fs==rs==258 and min(steps)==258
io=VerifiedCheckpointIO();c=torch.load(O/'fm/latest.ckpt',weights_only=False,map_location='cpu');dest=F/'atomic.ckpt';io.save_checkpoint(c,dest);before=dest.read_bytes()
try:
 with patch('lightning_task.os.replace',side_effect=OSError('simulated')):io.save_checkpoint(c,dest)
except OSError:pass
assert dest.read_bytes()==before
report={'full_steps':fs,'resumed_steps':rs,'parameter_max_abs_difference':delta,'optimizer_step_continuity':True,'atomic_failure_retains_previous':True,'synthetic_cpu_resume_test':True,'exact_resume':delta==0}
(F/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
