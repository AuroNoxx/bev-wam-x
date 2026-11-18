"""One native-resolution calibrated frame through the128x128 BEV World extension.

VLM and World are tested sequentially to isolate geometry/operator faults. This
does NOT certify joint training memory, speed, future prediction quality or batch8.
"""
from .runtime import ROOT, configure, math_attention
configure()
import gc
import json
import time
import traceback
import torch
from qwen_drive_perception import QwenDrivePerception
from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor
from .components import load_planning, load_perception
from .world import ActionConditionedWorld


def main():
    torch.set_num_threads(6); torch.manual_seed(71)
    torch.cuda.set_per_process_memory_fraction(.85,0)
    out = ROOT / "artifacts/v5-qwen-drive-world/world-verification-original-resolution.json"
    report = {"phase":"loading_vlm", "training_started":False,"optimizer_updates":0,
              "scope":"one calibrated official frame at unchanged896x512 processor sizes; VLM and World staged separately"}
    def save(): out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    save()
    try:
        model = load_planning(); model.eval()
        folder = sorted((ROOT / "runtime/qwen-drive-official/data/demo/perception").iterdir())[0]
        frame = PerceptionFrame(folder)
        processor = PerceptionProcessor(model.processor.tokenizer)
        inputs, meta = processor(frame,device="cuda")
        captured = {}
        def tap(module,args,output): captured["patches"] = args[0]
        handle = model.vlm.model.visual.merger.register_forward_hook(tap)
        report.update(phase="extracting_native_features",frame=frame.token,cameras=len(frame.cam_order)); save()
        with torch.no_grad(), math_attention(), torch.autocast("cuda",dtype=torch.bfloat16):
            result = model.vlm.model(**inputs,mm_token_type_ids=model._modality_ids(inputs["input_ids"]),use_cache=False)
            # Match the released perception tap convention exactly.
            hidden = model.vlm.model.language_model.norm(result.last_hidden_state)
            patches = model.vlm.model.visual.merger.norm(captured["patches"])
            vit = torch.stack(QwenDrivePerception._premerge_grids(patches,inputs["image_grid_thw"])).detach().clone()
            language = hidden[0][inputs["input_ids"][0] == model.config.vlm_config.image_token_id]
            grid = inputs["image_grid_thw"]
            assert (grid[:,1:] == grid[0,1:]).all()
            language = language.reshape(len(frame.cam_order),int(grid[0,1])//2,int(grid[0,2])//2,-1).detach().clone()
        handle.remove()
        del hidden, patches, captured, result, model, inputs, processor
        gc.collect(); torch.cuda.empty_cache()
        report["phase"]="loading_world"; save()
        perception, _ = load_perception()
        world = ActionConditionedWorld(perception,image_size=(896,512)).cuda().eval()
        del perception
        torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
        report["phase"]="native_corridor_encoding"; save()
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            current = world.encode_current(vit,language,[meta])
        assert current.shape == (1,256,128,128) and torch.isfinite(current).all()
        report["phase"]="future_world_forward_backward"; save()
        world.train()
        current = current.detach()
        trajectory = torch.zeros(1,50,2,device="cuda")
        trajectory[0,:,0]=torch.arange(1,51,device="cuda")*.5
        with torch.autocast("cuda",dtype=torch.bfloat16):
            prediction=world(current,trajectory,torch.tensor([5.],device="cuda"))
            loss=prediction["occupancy"].mean()+prediction["delta_latent"].float().square().mean()
        loss.backward()
        assert prediction["occupancy"].shape == (1,3,128,128)
        assert torch.isfinite(prediction["occupancy"]).all()
        assert world.delta[-1].weight.grad is not None and torch.isfinite(world.delta[-1].weight.grad).all()
        torch.cuda.synchronize()
        report.update(phase="passed",seconds=time.perf_counter()-started,
                      occupancy_shape=list(prediction["occupancy"].shape),
                      peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                      residual_head_grad_norm=world.delta[-1].weight.grad.float().norm().item(),
                      migration=world.migration,full_joint_backward_tested=False,
                      action_condition_initialization="zero residual, action dependence learned after updates")
    except Exception:
        report.update(phase="failed",error=traceback.format_exc()); raise
    finally:
        save();print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__": main()
