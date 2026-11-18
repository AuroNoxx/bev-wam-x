"""One real official RGB scene: verify gradients through VLM KV to control Flow."""
from .runtime import ROOT, configure, math_attention
configure()
import json
import time
import traceback
import torch
from qwen_drive.benchmarks import read_scene_file
from .components import load_planning
from .native_scene import encode_native_scene
from .planning import ControlFlowPlanner


def main():
    torch.set_num_threads(6); torch.manual_seed(71)
    torch.cuda.set_per_process_memory_fraction(.85,0)
    path = ROOT / "artifacts/v5-qwen-drive-world/native-rgb-gradient-original-resolution.json"
    report = {"phase": "loading", "training_started": False, "optimizer_updates": 0,
              "scope": "one official demo, 12 history images at unchanged official processor sizes"}
    def save(): path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save()
    try:
        model = load_planning()
        # Keep planner weights under one owner to avoid registering duplicate keys.
        expert = model.planning_expert
        del model.planning_expert
        planner = ControlFlowPlanner(expert, model.config.trajectory_scale).cuda()
        demo = ROOT / "runtime/qwen-drive-official/data/demo"
        scene = next(read_scene_file(demo / "planning_scenes.jsonl", image_root=demo)).scene
        inputs = model.processor(scene, with_reasoning=True, device="cuda")
        model.train(); planner.train()
        torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        report.update(phase="native_forward", tokens=int(inputs["input_ids"].shape[1]), images=int(inputs["image_grid_thw"].shape[0])); save()
        with math_attention(), torch.autocast("cuda", dtype=torch.bfloat16):
            native = encode_native_scene(model, inputs)
            context = planner.context(native, inputs)
            u = torch.randn(1, 50, 2, device="cuda")
            velocity = planner.velocity(u, torch.tensor([.5], device="cuda"), context)
            loss = velocity.square().mean()
        report["phase"] = "native_backward"; save()
        with math_attention(): loss.backward()
        grads = {name: parameter.grad for name, parameter in model.vlm.named_parameters() if parameter.requires_grad}
        lora_norm = sum(g.float().abs().sum().item() for g in grads.values() if g is not None)
        assert lora_norm > 0 and all(torch.isfinite(g).all() for g in grads.values() if g is not None)
        assert planner.expert.out_proj.weight.grad is not None and torch.isfinite(planner.expert.out_proj.weight.grad).all()
        torch.cuda.synchronize()
        report.update(phase="passed", seconds=time.perf_counter()-started,
                      scene_cache_groups=len(native["scene_cache"]), velocity_shape=list(velocity.shape),
                      finite_lora_grad_sum=lora_norm, peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                      official_processor_resolution_tested=True, sensor_1920_resolution_tested=False,
                      batch8_tested=False, world_integrated=False)
    except Exception:
        report.update(phase="failed", error=traceback.format_exc())
        raise
    finally:
        save(); print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
