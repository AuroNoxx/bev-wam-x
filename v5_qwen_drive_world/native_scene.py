"""Differentiable native VLM taps without an inference-only mutable KV cache."""
from .runtime import math_attention
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb


def encode_native_scene(model, inputs):
    """Use original ViT/VLM, capture exact native K/V projections with gradients.

    Capture hooks only retain tensors; reconstruction occurs outside checkpointed
    layers, so recomputation does not mutate an external inference cache. Actual
    PEFT dropout outputs are retained rather than re-evaluated with new masks.
    No future label fields are accepted by the VLM call.
    """
    captured = {}; handles = []
    for index in model.config.full_attention_layers:
        attn = model.vlm.model.language_model.layers[index].self_attn
        bucket = captured[index] = {}
        def retain_key(module, args, output, bucket=bucket): bucket["key"] = output
        def retain_value(module, args, output, bucket=bucket): bucket["value"] = output
        def retain_position(module, args, kwargs, bucket=bucket): bucket["rotary"] = kwargs["position_embeddings"]
        handles.append(attn.k_norm.register_forward_hook(retain_key))
        handles.append(attn.v_proj.register_forward_hook(retain_value))
        handles.append(attn.register_forward_pre_hook(retain_position, with_kwargs=True))
    try:
        with math_attention():
            if 'llm_grid_thw' in inputs:
                from .lightweight_input import compressed_forward
                result=compressed_forward(model,inputs)
            else:
                result = model.vlm.model(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                     image_grid_thw=inputs["image_grid_thw"],
                                     mm_token_type_ids=model._modality_ids(inputs["input_ids"]),
                                     use_cache=False)
    finally:
        for handle in handles: handle.remove()
    caches = []
    for index in model.config.full_attention_layers:
        bucket = captured[index]
        key = bucket["key"].transpose(1, 2)
        cos, sin = bucket["rotary"]
        _, key = apply_rotary_pos_emb(key, key, cos, sin)
        value = bucket["value"].view(*bucket["value"].shape[:2], key.shape[1], key.shape[-1])
        caches.append((key.transpose(1, 2), value))
    return {"hidden": result.last_hidden_state, "scene_cache": caches,
            "position_anchor": model._rope_positions(inputs["input_ids"], inputs.get('llm_grid_thw',inputs["image_grid_thw"]))[:, :, -1]}
