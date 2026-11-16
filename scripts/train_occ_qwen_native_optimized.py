"""Resume the same authorized run with independently validated execution options."""
import argparse,json,hashlib
from pathlib import Path
import torch
import pytorch_lightning as pl
from torchdata.stateful_dataloader import StatefulDataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from train_occ_qwen_native import settings,make_callbacks,TrainingStatus,NativeData
from occ_qwen_native_data import NativeDataset,require_training_authorization
from native_occ_cached_data import NativeCachedDataset
from native_occ_exact_loader import ExactResumeDataLoader
from occ_qwen_native_flow import NativeOccQwenJointFlow,load_text_backbone
from occ_qwen_performance import OptimizedNativeTask
from lightning_task import VerifiedCheckpointIO
from b2d_native_occ import ROOT

class CachedData(NativeData):
    def __init__(self,cfg,index):
        super().__init__(cfg)
        self.train=NativeCachedDataset(index,'train');self.val=NativeCachedDataset(index,'validation')
        for split,cached in [('train',self.train),('validation',self.val)]:
            if cached.names!=NativeDataset(split).names:raise RuntimeError('Full split order differs: '+split)
    def train_dataloader(self):
        return ExactResumeDataLoader(self.train,allow_legacy_resume=True,batch_size=self.settings['batch_size'],shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(self.settings['seed']))
    def val_dataloader(self):
        return StatefulDataLoader(self.val,batch_size=self.settings['batch_size'],shuffle=False,num_workers=0)

class SafePause(pl.Callback):
    def __init__(self,folder):self.folder=folder
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        marker=self.folder/'pause.request'
        if marker.exists():
            trainer.save_checkpoint(self.folder/'pause.ckpt',weights_only=False)
            marker.replace(self.folder/'pause.consumed')
            trainer.should_stop=True
            print('SAFE_PAUSE '+str(trainer.global_step),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',required=True);p.add_argument('--cache-index',required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    require_training_authorization();cfg=settings();torch.set_num_threads(4)
    folder=ROOT/'artifacts/occ-qwen-native-training';cp=torch.load(a.resume,map_location='cpu',weights_only=False)
    sha=hashlib.sha256((ROOT/cfg['manifest']).read_bytes()).hexdigest();s=cp['task_state']
    if s['settings']!=cfg or s['manifest_sha']!=sha or s['contract_test']:raise RuntimeError('Resume contract mismatch')
    data=CachedData(cfg,a.cache_index)
    if a.preflight:
        print(json.dumps({'preflight':True,'step':cp['global_step'],'train':len(data.train),'validation':len(data.val),'order_exact':True,'settings_exact':True}));return
    gate=json.loads((folder/'performance_gpu_validation.json').read_text())
    if gate['source_step']!=cp['global_step'] or not gate['test_only']:raise RuntimeError('GPU gate does not match resume checkpoint')
    perf={'channels_last':gate['channels_last']['pass'],'fused_adamw':gate['fused_adamw']['pass'],'workers':0,'native_aligned_cache':str(Path(a.cache_index).resolve()),'cache_identity':data.train.identity,'resume_step':cp['global_step']}
    del cp
    pl.seed_everything(cfg['seed'],workers=True,verbose=False)
    model=NativeOccQwenJointFlow(load_text_backbone(ROOT/'models/Qwen3-VL-2B-Instruct'))
    task=OptimizedNativeTask(model,cfg,sha,performance=perf)
    trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_epochs=cfg.get('max_epochs',20),plugins=[VerifiedCheckpointIO()],callbacks=make_callbacks(folder)+[TrainingStatus(folder),SafePause(folder)],logger=TensorBoardLogger(ROOT/'logs/tensorboard',name='occ-qwen-native'),gradient_clip_val=1.,num_sanity_val_steps=0,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=10)
    print('PERFORMANCE '+json.dumps(perf),flush=True)
    trainer.fit(task,datamodule=data,ckpt_path=a.resume,weights_only=False)
if __name__=='__main__':main()

