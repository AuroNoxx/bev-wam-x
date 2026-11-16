"""Real BF16 Flow optimizer resume: same next loss and update on this ROCm GPU."""
import json,torch
from torch.utils.data import DataLoader
from train_v1 import ROOT,CONFIG,Samples,seed,rng,restore_rng,atomic_checkpoint,move
from models_v1 import FlowPlanner,loss

def main():
    torch.set_num_threads(6);cfg=json.loads(CONFIG.read_text());seed(cfg['seed'])
    manifest=json.loads((ROOT/'data/occ_v1/pilot.json').read_text())
    batch=move(next(iter(DataLoader(Samples(manifest,'train','flow'),batch_size=2,generator=torch.Generator().manual_seed(1)))))
    model=FlowPlanner().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4);scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,.99)
    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):value=loss('flow',model,batch)
        value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step();torch.cuda.synchronize()
        return float(value.detach())
    step()
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng(),'epoch':0,'next_batch':1,'global_step':1,'config':cfg,'manifest_sha':'gpu-resume-test','order':[0,1],'best_loss':1.,'bad_epochs':0}
    path=ROOT/'artifacts/v1/contract-test/gpu-latest.pt';atomic_checkpoint(path,state)
    expected_loss=step();expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};expected_lr=scheduler.get_last_lr()
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);restore_rng(saved['rng'])
    resumed_loss=step();delta=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in model.state_dict().items())
    assert abs(expected_loss-resumed_loss)<1e-6 and delta<1e-6 and scheduler.get_last_lr()==expected_lr,(expected_loss,resumed_loss,delta)
    result={'gpu':torch.cuda.get_device_name(0),'expected_loss':expected_loss,'resumed_loss':resumed_loss,'max_parameter_difference':delta,'saved_step':1,'resumed_next_step':2,'pass':True}
    (path.parent/'gpu-resume-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
