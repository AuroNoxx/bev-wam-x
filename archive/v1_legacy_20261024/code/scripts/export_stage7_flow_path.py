"""Save real vector-field integration states for the interactive Flow lesson."""
import json
from pathlib import Path
import numpy as np
import torch
from dataset.flow_features import FlowFeaturesDataset
from models.flow_matcher import ConditionalFlowMatcher
from mini_wa_jepa.provenance import provenance,write_json
from configs.stage7 import Stage7Config
from training.flow_matching import file_sha256


def export():
    torch.set_num_threads(4);root=Path('artifacts/restart_stage7');out=root/'flow_paths';out.mkdir(exist_ok=False)
    data=FlowFeaturesDataset(root/'flow_features_001','test');cfg=Stage7Config()
    bev=np.load('outputs/data/stage2_v2_100k/baseline_cache/test/bev.npy',mmap_mode='r')
    manifest=dict(**provenance(cfg),completed=False,records=[])
    with torch.inference_mode():
        for seed in cfg.training_seeds:
            checkpoint=root/f'flow_{seed}/best.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model=ConditionalFlowMatcher(**{k:v for k,v in saved['model_arch'].items() if k!='time_emb_dim'}).to('cuda').eval()
            model.load_state_dict(saved['model'])
            for index in (0,16,99):
                item=data[index];h=item['history'][None].to('cuda');e=item['ego'][None].to('cuda')
                condition=model.extract_condition(h,e);rng_seed=42+item['anchor']
                gen=torch.Generator(device='cuda').manual_seed(rng_seed)
                x=torch.randn(8,10,4,generator=gen,device='cuda');path=[x.cpu().numpy()]
                for step in range(10):
                    x=x+model.vector_field(x,torch.full((8,),step*.1,device='cuda'),condition.expand(8,-1))*.1
                    path.append(x.cpu().numpy())
                reference=model.sample(condition,num_steps=10,num_candidates=8,seed=rng_seed)[0].cpu().numpy()
                np.testing.assert_allclose(path[-1],reference,atol=1e-6,rtol=1e-6)
                file=out/f'{seed}_{index}.npz'
                np.savez_compressed(file,path=data.normalizer.inverse_trajectory(np.asarray(path)),
                    ground_truth=data.normalizer.inverse_trajectory(item['target'].numpy()),
                    bev=bev[item['anchor']],anchor=item['anchor'],noise_seed=rng_seed)
                manifest['records'].append(dict(seed=seed,index=index,path=str(file),sha256=file_sha256(file),
                    checkpoint_sha256=file_sha256(checkpoint)))
    manifest['completed']=True;write_json(out/'manifest.json',manifest)


if __name__=='__main__':export()
