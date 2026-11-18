"""CPU contracts for original pixels, camera retention and token budgets."""
from .runtime import ROOT,configure
configure()
import dataclasses,json
import torch
from transformers import AutoTokenizer
from qwen_drive import QwenDriveConfig,QwenDriveProcessor,CAMERA_VIEWS
from qwen_drive.benchmarks import read_scene_file
from .lightweight_input import planning_inputs,budget_grids,repeat_scene


def main():
    torch.set_num_threads(2)
    path=ROOT/'models/v5-qwen-drive-world/Qwen-Drive-1.0-4B'
    processor=QwenDriveProcessor(AutoTokenizer.from_pretrained(path,local_files_only=True),QwenDriveConfig.from_pretrained(path))
    demo=ROOT/'runtime/qwen-drive-official/data/demo'
    scene=next(read_scene_file(demo/'planning_scenes.jsonl',image_root=demo)).scene
    original=processor(scene,device='cpu')
    inputs=planning_inputs(processor,scene,device='cpu')
    counts=[int(g.prod()) for g in original['image_grid_thw']]
    original_images=original['pixel_values'].split(counts)
    # Original view-major ordering: front0..3, left0..3, right0..3.
    selected=[0,1,2,3,7,11]
    assert torch.equal(inputs['pixel_values'],torch.cat([original_images[i] for i in selected]))
    assert torch.equal(inputs['image_grid_thw'],original['image_grid_thw'][selected])
    for key in ['history','history_velocity','history_acceleration','ego_status','nav_command']:
        assert torch.equal(inputs[key],original[key]),key
    text=processor.tokenizer.decode(inputs['input_ids'][0].tolist())
    assert scene.instruction().split('1. Historical trajectories',1)[1] in text
    assert inputs['input_ids'].shape[1]<=850
    assert sum(f['current'] for f in inputs['input_audit']['frames'])==3
    for batch in [2,4]:
        repeated=repeat_scene(inputs,batch)
        assert repeated['input_ids'].shape==(batch,inputs['input_ids'].shape[1])
        assert repeated['pixel_values'].shape[0]==batch*inputs['pixel_values'].shape[0]
    # Camera-retention contract only: extra views intentionally reuse RGB here.
    extended=dataclasses.replace(scene,views={**scene.views,**{f'<EXTRA CAMERA {i}>':[scene.views[CAMERA_VIEWS[0]][-1]] for i in range(5)}})
    extra=planning_inputs(processor,extended,device='cpu')
    assert sum(f['current'] for f in extra['input_audit']['frames'])==8
    assert extra['input_ids'].shape[1]<=850
    assert all(f['current'] or f['view']==CAMERA_VIEWS[0] for f in extra['input_audit']['frames'])
    native,pooled,final=budget_grids([(1,32,56)]*8,300)
    assert sum(t*h*w for t,h,w in pooled)==896
    assert sum(t*h*w for t,h,w in final)+300<=850
    try:budget_grids([(2,32,56)],300)
    except ValueError:pass
    else:raise AssertionError('Unsupported video grids must be rejected')
    report={'phase':'passed','retained_pixels_bitwise_identical':True,'native_state_fields_identical':True,
            'native_state_navigation_text_preserved':True,'three_camera_tokens':inputs['input_ids'].shape[1],
            'eight_current_camera_contract_tokens':extra['input_ids'].shape[1],
            'eight_camera_scope':'CPU input contract using repeated RGB; not a GPU benchmark or calibrated surround validation',
            'extra_pool_budget_fallback_exercised':True,'unsupported_temporal_grid_rejected':True}
    (ROOT/'artifacts/v5-qwen-drive-world/lightweight-input-contracts.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
