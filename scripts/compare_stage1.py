"""Paired validation of baseline and continuation; never opens the test split."""
import argparse,json,time
from pathlib import Path
import torch
import pytorch_lightning as pl
from lightning_task import ROOT,configure_runtime,OccData,DrivingTask
from stage1_objective import trajectory_metrics


class Assessment(DrivingTask):
    def predict_step(self,batch,batch_idx):
        results=[]
        for seed in self.assessment_seeds:
            with torch.random.fork_rng(devices=[self.device.index]):
                torch.manual_seed(seed+batch_idx)
                pred=self.model.sample(batch,k=8,steps=16)
            results.append({k:v.cpu() for k,v in trajectory_metrics(pred,batch).items()})
        return results


def assess(path,seeds):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    settings=checkpoint['task_state']['settings'];data=OccData('flow',settings)
    model=Assessment('flow',settings,data.sha,data.manifest['counts']['train']);model.assessment_seeds=seeds
    trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',logger=False,
        enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False)
    started=time.perf_counter()
    outputs=trainer.predict(model,dataloaders=data.loader('validation'),ckpt_path=str(path),weights_only=False)
    values={key:torch.stack([torch.cat([batch[i][key] for batch in outputs]) for i in range(len(seeds))]) for key in outputs[0][0]}
    names=data.manifest['splits']['validation'];routes=[Path(n).parts[-2] for n in names]
    family_lookup={r['route']:r['family'] for r in data.manifest['routes']};families=[family_lookup[r] for r in routes]
    def summarize(indices):
        return {k:float(v[:,indices].mean()) for k,v in values.items()}
    result={'checkpoint':str(path.relative_to(ROOT)),'checkpoint_epoch':checkpoint['epoch']+1,
        'global_step':checkpoint['global_step'],'manifest_sha':data.sha,'windows':len(names),'split':'validation',
        'seeds':seeds,'candidates':8,'ode_steps':16,'seconds':time.perf_counter()-started,
        'mean':summarize(list(range(len(names)))),
        'by_seed':{str(seed):{k:float(v[i].mean()) for k,v in values.items()} for i,seed in enumerate(seeds)},
        'by_family':{f:{'windows':families.count(f),**summarize([i for i,x in enumerate(families) if x==f])} for f in sorted(set(families))},
        'by_route':{r:{'windows':routes.count(r),**summarize([i for i,x in enumerate(routes) if x==r])} for r in sorted(set(routes))}}
    del outputs,values,trainer,model,checkpoint
    import gc;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',default='artifacts/lightning/flow/best.ckpt')
    parser.add_argument('--candidate',default='artifacts/lightning/flow-v2/best.ckpt')
    parser.add_argument('--output',default='artifacts/lightning/flow-v2/comparison.json')
    args=parser.parse_args();configure_runtime();seeds=[1908,2911,3911]
    baseline=assess(ROOT/args.baseline,seeds);candidate=assess(ROOT/args.candidate,seeds)
    assert baseline['manifest_sha']==candidate['manifest_sha']
    result={'baseline':baseline,'candidate':candidate,'delta_candidate_minus_baseline':{k:candidate['mean'][k]-v for k,v in baseline['mean'].items()},
        'notes':['Same full validation windows and paired sampling seeds; test split unused.',
                 'minADE8/minFDE8 are ground-truth oracles; no deployable selection evaluated.',
                 'The validation set selected this checkpoint; this is not an independent test estimate.',
                 'Overlapping windows are correlated. Per-route results are retained; seeds do not create new driving episodes.',
                 'Kinematic consistency does not certify collision avoidance, curvature limits or vehicle controllability.']}
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2))
    print('COMPARISON '+json.dumps({k:v for k,v in result.items() if k not in ['baseline','candidate']}),flush=True)


if __name__=='__main__':main()
