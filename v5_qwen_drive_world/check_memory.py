"""Manual diagnostic, preview by default after the2026-11-17 system freeze.

This is not a training launcher. No model/torch import occurs during preview.
"""
from .runtime import ROOT, configure, math_attention
configure()
import argparse
import json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--batch-size',type=int,default=1)
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'mode':'preview_only','gpu_started':False,
                          'incident':'batch8/native resolution froze the computer during backward; user rebooted',
                          'formal_training_target_batch':8,'diagnostic_batch':args.batch_size,
                          'note':'Original batch8 result is interrupted, not passed. No automatic retry.'},indent=2))
        return
    if args.batch_size != 1:
        raise ValueError('The batch8 stress test is disabled after the system freeze. Only an explicitly requested batch1 diagnostic is supported.')
    import gc
    import traceback
    import torch
    from qwen_drive.benchmarks import read_scene_file
    from .components import load_planning
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(.65,0)
    path=ROOT/'artifacts/v5-qwen-drive-world/manual-batch1-memory.json'
    report={"phase":"loading","training_started":False,"optimizer_updates":0,"requested_batch":args.batch_size,
            "allocator_fraction":.65,"formal_training_batch_unchanged":8,
            "scope":"VLM+loaded planner weights; VLM forward/backward at official demo native image sizes; World/optimizer not allocated"}
    def save():path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    model=None; inputs=None; output=None
    try:
        model=load_planning();model.train()
        demo=ROOT/'runtime/qwen-drive-official/data/demo'
        scene=next(read_scene_file(demo/'planning_scenes.jsonl',image_root=demo)).scene
        single=model.processor(scene,with_reasoning=True,device='cpu')
        inputs={k:v.repeat((args.batch_size,)+(1,)*(v.ndim-1)).cuda() for k,v in single.items()
                if k in ('input_ids','pixel_values','image_grid_thw')}
        report.update(phase='requested_batch_forward',tokens_per_scene=int(inputs['input_ids'].shape[1]),
                      grid_per_scene=single['image_grid_thw'].tolist());save()
        torch.cuda.reset_peak_memory_stats()
        with math_attention(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=model.vlm.model(**inputs,mm_token_type_ids=model._modality_ids(inputs['input_ids']),use_cache=False)
            loss=output.last_hidden_state[:,-1,:128].float().square().mean()
        report['phase']='requested_batch_backward';save()
        with math_attention():loss.backward()
        torch.cuda.synchronize()
        report['phase']='vlm_manual_batch_passed_world_not_tested'
    except torch.OutOfMemoryError:
        report.update(phase='out_of_memory',error=traceback.format_exc())
    except Exception:
        report.update(phase='failed_other',error=traceback.format_exc())
        raise
    finally:
        report['peak_allocated_GiB']=torch.cuda.max_memory_allocated()/2**30
        report['full_system_batch8_verified']=False
        save();print(json.dumps(report,indent=2),flush=True)
        del model,inputs,output
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
