import torch,json
from pathlib import Path
r=Path('artifacts/occ-qwen-native-pretrained-resume-test')
a=torch.load(r/'full.ckpt',map_location='cpu',weights_only=False);b=torch.load(r/'resumed.ckpt',map_location='cpu',weights_only=False)
delta=max(float((v-b['state_dict'][k]).abs().max()) for k,v in a['state_dict'].items());assert delta==0
mx=0.
for i,s in a['optimizer_states'][0]['state'].items():
    for k,v in s.items():
        other=b['optimizer_states'][0]['state'][i][k]
        if isinstance(v,torch.Tensor):
            mx=max(mx,float((v-other).abs().max()))
            if k=='step':assert torch.equal(v,other)
            else:assert torch.allclose(v,other,rtol=1e-5,atol=1e-8)
        else:assert v==other
assert a['lr_schedulers']==b['lr_schedulers'] and a['global_step']==b['global_step']==4
report={'passed':True,'test_only':True,'formal_training':False,'pretrained_frozen_backbone':True,'precision':'bf16-mixed GPU','updates_executed_total':6,'global_step':4,'resumed_from':2,'model_max_abs_difference':delta,'ema_equal':True,'optimizer_max_absolute_difference':mx,'optimizer_allclose':True,'optimizer_tolerance':{'rtol':1e-5,'atol':1e-8},'optimizer_step_exact':True,'scheduler_equal':True,'frozen_backbone_unchanged_checked_in_run':True,'trainable_encoder_updated_checked_in_run':True,'checkpoint_bytes':(r/'full.ckpt').stat().st_size,'atomic_failure_test':'separate prior tiny-model IO test passed; not repeated on full model'}
(r/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
