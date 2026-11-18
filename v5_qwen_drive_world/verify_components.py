"""Evidence-producing component checks, not a training or driving benchmark."""
from .runtime import ROOT, configure, math_attention
configure()
import argparse
import gc
import json
import traceback
import torch
from safetensors import safe_open
from .kinematics import integrate, normalize_controls, denormalize_controls, euler_flow


def numerical_checks(device):
    speeds = torch.tensor([0., 5/3.6, 20/3.6, 60/3.6, 100/3.6, 120/3.6], device=device)
    controls = torch.zeros(6, 50, 2, device=device, requires_grad=True)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        trajectory = integrate(controls, speeds)
        roundtrip = denormalize_controls(normalize_controls(controls))
        solved = euler_flow(lambda u, s: torch.ones_like(u), torch.zeros_like(controls))
    expected = speeds * 5
    torch.testing.assert_close(trajectory[:, -1, 0], expected, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(roundtrip, controls)
    torch.testing.assert_close(solved, torch.ones_like(solved))
    assert trajectory.dtype == torch.float32 and solved.dtype == torch.float32
    trajectory[..., :2].sum().backward()
    assert torch.isfinite(controls.grad).all() and controls.grad.abs().sum() > 0
    braking = torch.zeros_like(controls); braking[..., 0] = -8
    braking_path = integrate(braking, speeds)
    assert braking_path[..., 3].min() >= 0
    # SDPA suffix gradient test covers the actual backend, not just forward masks.
    x = torch.randn(2, 2, 12, 16, device=device, requires_grad=True)
    with math_attention(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        q = x.to(torch.bfloat16) if device == "cuda" else x
        output = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        output[..., :6, :].float().square().sum().backward()
    suffix = x.grad[..., 6:, :].abs().max().item()
    assert suffix == 0, suffix
    return {"fp32_kinematics": True, "fp32_euler": True, "finite_nonzero_control_gradients": True,
            "tested_speeds_kmh": (speeds * 3.6).tolist(), "sdpa_math_suffix_gradient_max": suffix}


def check_pretrained(device, report, save):
    from .components import load_planning, load_perception, settings
    cfg = settings(); path = ROOT / cfg["model_dir"]
    original_config = json.loads((path / "config.json").read_text())
    kinds = original_config["vlm_config"]["text_config"]["layer_types"]
    report["interface"] = {"full_attention_sources_32": kinds.count("full_attention"),
                           "full_attention_sources_24": kinds[:24].count("full_attention"),
                           "planner_required_sources": original_config["expert_config"]["num_hidden_layers"] // original_config["expert_config"]["layers_per_kv"],
                           "24_layer_whole_system_compatible": False,
                           "selected_policy": cfg["layer_policy"]}
    report["phase"] = "loading_native_planner"; save()
    model = load_planning(cfg, device)
    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as handle:
        stored = set(handle.keys())
    # PEFT adds adapter keys and wraps projection weights as base_layer.weight.
    expected = {"vlm." + k.replace(".base_layer.", ".") for k in model.vlm.state_dict() if "lora_" not in k}
    missing = sorted(expected - stored)
    unexpected = sorted(stored - expected)
    # Safetensors omits tied lm_head.weight, whose storage is embed_tokens.weight.
    missing = [k for k in missing if not k.endswith("lm_head.weight")]
    assert not missing and not unexpected, {"missing": missing, "unexpected": unexpected}
    report["native_weights"] = {"vlm_keys_matched": len(stored), "planner_strict_load": True,
        "lora_trainable_parameters": sum(p.numel() for p in model.vlm.parameters() if p.requires_grad),
        "trainable_dtypes": sorted({str(p.dtype) for p in model.vlm.parameters() if p.requires_grad})}
    report["phase"] = "native_causal_gradient"; save()
    model.train()
    text_model = model.vlm.model.language_model
    embeds = text_model.embed_tokens(torch.arange(100, 112, device=device)[None]).detach().requires_grad_(True)
    with math_attention(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        result = text_model(inputs_embeds=embeds, use_cache=False).last_hidden_state
        result[:, :6, :128].float().square().sum().backward()
    suffix = embeds.grad[:, 6:].abs().max().item()
    assert suffix == 0, f"Native VLM suffix gradient leak: {suffix}"
    grads = [p.grad for p in model.vlm.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)
    report["native_causal_gradient"] = {"suffix_max": suffix, "lora_finite_nonzero": True,
                                        "scope": "12 text tokens through pretrained hybrid backbone, full multimodal test pending"}
    model.zero_grad(set_to_none=True)
    # Exercise the configured 24-layer text truncation independently; it is not
    # a claim that the unchanged full planner can consume its six KV groups.
    layers = text_model.layers
    text_model.layers = torch.nn.ModuleList(list(layers)[:24])
    with torch.no_grad(), math_attention(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        short = text_model(inputs_embeds=embeds.detach(), use_cache=False).last_hidden_state
    assert torch.isfinite(short).all()
    text_model.layers = layers
    report["24_layer_text_only"] = {"finite": True, "shape": list(short.shape), "planner_attached_tested": False}
    del short, result, embeds, grads, text_model, layers, model
    gc.collect()
    if device == "cuda": torch.cuda.empty_cache()
    report["phase"] = "loading_native_perception"; save()
    perception, info = load_perception(cfg, device)
    report["perception"] = {"strict_load": True, "parameters": sum(p.numel() for p in perception.parameters()),
                            "native_bev_shape": [perception.config.bev_h, perception.config.bev_w],
                            "native_frustum": perception.config.frustum_range,
                            "future_action_conditioning": False, "corridor_128_adaptation": False}
    del perception
    gc.collect()
    if device == "cuda": torch.cuda.empty_cache()


def check_perception_operator(device):
    from qwen_drive_perception.attention import multi_scale_deformable_attn_cuda
    value = torch.randn(1, 4, 1, 8, device=device, dtype=torch.bfloat16, requires_grad=True)
    loc = torch.rand(1, 2, 1, 1, 2, 2, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.full((1, 2, 1, 1, 2), .5, device=device, dtype=torch.bfloat16, requires_grad=True)
    shapes = torch.tensor([[2, 2]], device=device)
    out = multi_scale_deformable_attn_cuda(value, shapes, torch.tensor([0], device=device), loc, weight, 64)
    out.float().sum().backward()
    assert value.grad is not None and loc.grad is not None and weight.grad is not None
    return {"native_gpu_forward_backward": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--native-ops", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(6)
    report = {"phase": "numerical_checks", "training_started": False, "torch": torch.__version__,
              "hip": torch.version.hip, "device": args.device, "checks": {}}
    destination = ROOT / "artifacts/v5-qwen-drive-world/component-verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    def save(): destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save()
    try:
        report["checks"] = numerical_checks(args.device); save()
        if args.pretrained: check_pretrained(args.device, report, save)
        if args.native_ops:
            try:
                report["native_perception_operator"] = check_perception_operator(args.device)
            except Exception:
                report["native_perception_operator"] = {"passed": False, "error": traceback.format_exc()}
        report["phase"] = "component_checks_complete"
        report["full_training_ready"] = False
        report["remaining"] = ["native RGB and perception training forward", "differentiable perception ops",
                               "action-conditioned corridor world", "continuous Flow GRPO likelihood",
                               "highway dataset", "batch8 memory and Lightning resume", "CARLA closed loop"]
    except Exception:
        report["phase"] = "failed"; report["error"] = traceback.format_exc()
        raise
    finally:
        save()
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
