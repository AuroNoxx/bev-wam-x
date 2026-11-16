"""Serial bounded capacity/batch/history matrix, same 8 train windows."""
import json,gzip,time,gc,hashlib,os
import numpy as np
import torch,psutil
from probe_joint_occ_capacity import CapacityModel
from lightning_task import ROOT,configure_runtime
from train_joint_occ import resource_guard,MemorySamples,batch_to
from dataset_v1 import compact,warp
from stage1_objective import consistent_flow_loss

def idle():
    resource_guard()
    parents={p.pid for p in psutil.Process().parents()}
    for p in psutil.process_iter(['name','cmdline']):
        if p.pid==os.getpid() or p.pid in parents:continue
        cmd=' '.join(p.info['cmdline'] or [])
        if (p.info['name'] or '').lower().startswith('python') and 'tensorboard' not in cmd:
            raise RuntimeError('Other Python task detected; not competing: '+str(p.pid))

def main():
    idle();configure_runtime();out=ROOT/'artifacts/joint-teacher-resource-history-matrix';out.mkdir(exist_ok=False);allstart=time.perf_counter()
    cfg=json.loads((ROOT/'configs/joint-occ-v1.json').read_text());sel=json.loads((ROOT/cfg['selection']).read_text())
    names=[n for n in sel['splits']['train'] if int(Path(n).stem)>=20][:8];assert len(names)==8
    rows=MemorySamples(names).rows;long=[];audit=[]
    for name,row in zip(names,rows):
        route=Path(name).parts[2];i=int(Path(name).stem);clip=ROOT/'data/public/bench2drive/clips'/route
        poses={}
        for j in range(i-20,i+1):
            with gzip.open(clip/'anno'/f'{j:05d}.json.gz','rt') as f:a=json.load(f)
            poses[j]=np.asarray(next(b['world2ego'] for b in a['bounding_boxes'] if b['class']=='ego_vehicle'))
            assert np.isfinite(poses[j]).all()
        motion=np.linalg.norm(np.diff(np.stack([np.linalg.inv(poses[j])[:3,3] for j in range(i-20,i+1)]),axis=0),axis=1)
        assert motion.max()<8
        history=[]
        for j in range(i-20,i+1,2):
            with np.load(clip/'occ'/f'{j-1:05d}.npz') as f:history.append(warp(compact(f['occ'],f['seg']),poses[j],np.linalg.inv(poses[i])))
        history=np.stack(history);assert np.array_equal(history[-4:],row['history'].numpy())
        long.append({**row,'history':torch.from_numpy(history)})
        audit.append({'file':name,'history_pose_indices':list(range(i-20,i+1,2)),'history_occ_indices':list(range(i-21,i,2)),'max_ego_step_m':float(motion.max()),'original_four_frames_exact_match':True,'sha256':hashlib.sha256((ROOT/'data/occ_v3'/name).read_bytes()).hexdigest()})
    results=[];common={'test_read':False,'source_causality_certified':False,'common_train_windows':audit,'effective_batch':8,'optimizer_steps':16,'lr':1e-5,'sampling_evaluation':'same seed, k1 steps8, train only; not quality acceptance','data_rebuild_seconds':time.perf_counter()-allstart}
    specs=[(768,15,12,b,h) for h in [4,11] for b in [2,4,8]]+[(1024,16,16,b,h) for h in [4,11] for b in [4,8]]+[(1024,28,16,4,h) for h in [4,11]]
    for width,layers,heads,b,h in specs:
        idle();start=time.perf_counter();torch.manual_seed(1701);m=CapacityModel(width,layers,heads).cuda();batch=batch_to(rows if h==4 else long,'cuda');opt=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=1e-5)
        def evaluate():
            m.eval()
            with torch.random.fork_rng(devices=[0]),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                torch.manual_seed(8181);pred=m.sample(batch,k=1,steps=8);err=((pred[:,0,:,:2]-batch['trajectory'][...,:2])*pred.new_tensor([50,10])).norm(dim=-1).mean()
            m.train();return float(err)
        before=evaluate();torch.cuda.reset_peak_memory_stats();times=[];losses=[]
        for step in range(16):
            torch.cuda.synchronize();ts=time.perf_counter();opt.zero_grad(set_to_none=True);lossval=0.
            for offset in range(0,8,b):
                sub={k:v[offset:offset+b] for k,v in batch.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    fm,_=consistent_flow_loss(m,sub,cfg['objective']);aux=m.auxiliary_loss(sub);loss=(fm+.1*sum(aux))/(8//b)
                assert torch.isfinite(loss);loss.backward();lossval+=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);assert torch.isfinite(norm);opt.step();m.update_teacher();torch.cuda.synchronize();times.append(time.perf_counter()-ts);losses.append(lossval)
        after=evaluate();torch.cuda.synchronize()
        r={'width':width,'layers':layers,'heads':heads,'non_ema_parameters':sum(p.numel() for n,p in m.named_parameters() if not n.startswith('teacher.')),'physical_batch':b,'history_frames':h,'accumulation':8//b,'optimizer_steps':16,'loss_last':losses[-1],'mean_hot_step_seconds':float(np.mean(times[2:])),'optimizer_wall_seconds':sum(times),'samples_per_second_hot':8/float(np.mean(times[2:])),'end_to_end_config_seconds':time.perf_counter()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'before_train_ADE_k1_steps8':before,'after_train_ADE_k1_steps8':after}
        results.append(r);print(json.dumps(r),flush=True);(out/'report.json').write_text(json.dumps({**common,'results':results,'total_elapsed_seconds':time.perf_counter()-allstart},indent=2))
        del m,opt,batch,sub,fm,aux,loss,norm;gc.collect();torch.cuda.empty_cache()
    print('COMPLETED',flush=True)
from pathlib import Path
if __name__=='__main__':main()
