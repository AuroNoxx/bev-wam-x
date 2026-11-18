"""Load original components strictly and attach PEFT adapters without replacing inputs."""
from .runtime import ROOT, configure, math_attention
configure()
import json
import re
import torch
from peft import LoraConfig, inject_adapter_in_model
from qwen_drive import QwenDriveForPlanning
from qwen_drive_perception import QwenDrivePerception


def settings():
    return json.loads((ROOT / "configs/v5-qwen-drive-world/default.json").read_text())


def load_planning(cfg=None, device="cuda"):
    cfg = cfg or settings()
    path = ROOT / cfg["model_dir"]
    from .accelerators import configure_native_kernels
    configure_native_kernels(cfg.get('acceleration', {}))
    # The official wrapper assumes from_pretrained returns a model (not a tuple).
    # Planner load_planner already uses strict=True; VLM compatibility is separately
    # checked against the safetensors key inventory in verify_components.
    model = QwenDriveForPlanning.from_pretrained(
        path, planner=path / "planner-rl", dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )
    if cfg.get('acceleration',{}).get('fla_causal_conv1d'):
        from .accelerators import attach_fla_convolution
        count = attach_fla_convolution(model.vlm)
        if count != 24:
            raise ValueError(f'Expected 24 native linear-attention convolutions, got {count}')
    model.requires_grad_(False)
    target_names = [name for name, _ in model.vlm.named_modules()
                    if ".self_attn." in name and name.rsplit(".", 1)[-1] in cfg["lora"]["target_modules"]]
    last_start = cfg['lora'].get('layer_start',0)
    if last_start:
        target_names = [name for name in target_names if int(re.search(r'language_model.layers.(\d+)',name)[1]) >= last_start]
        target_names += [name for name,_ in model.vlm.named_modules()
                         if '.linear_attn.' in name and name.rsplit('.',1)[-1] in ('in_proj_qkv','out_proj')
                         and int(re.search(r'language_model.layers.(\d+)',name)[1]) >= last_start]
    if not target_names:
        raise ValueError("No native full-attention q/k/v/o targets found")
    lora = cfg["lora"]
    inject_adapter_in_model(LoraConfig(r=lora["r"], lora_alpha=lora["alpha"],
                                      lora_dropout=lora["dropout"], target_modules=target_names), model.vlm)
    for name, parameter in model.vlm.named_parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    # Full official stack preserves all 8 KV sources. Truncation requires planner
    # migration and is rejected instead of silently reusing unrelated cache groups.
    if cfg["layer_policy"] != "full_with_gradient_checkpointing":
        raise ValueError("24-layer full-system planner migration is not implemented")
    if cfg["gradient_checkpointing"]:
        model.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "context_fn": lambda: (math_attention(), math_attention()),
        })
        for i,layer in enumerate(model.vlm.model.language_model.layers):
            layer.gradient_checkpointing = i >= last_start
        for module in model.vlm.model.visual.modules():
            if hasattr(module,'gradient_checkpointing'): module.gradient_checkpointing=False
    if last_start:
        assert all(not p.requires_grad for layer in model.vlm.model.language_model.layers[:last_start] for p in layer.parameters())
    return model.to(device)


def load_perception(cfg=None, device="cuda"):
    cfg = cfg or settings()
    model, info = QwenDrivePerception.from_pretrained(
        ROOT / cfg["model_dir"] / "perception", dtype=torch.bfloat16,
        local_files_only=True, output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys"):
        raise ValueError(f"Perception checkpoint mismatch: {info}")
    return model.to(device), info
