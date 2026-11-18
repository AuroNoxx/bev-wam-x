"""Bounded RAM cache for the genuinely frozen ViT/layers0..23 only.

No disk feature copies. The planner still consumes all eight real KV sources.
Last-eight LoRA, final norm and World always execute with current parameters.
"""
from collections import OrderedDict
import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
from .runtime import math_attention
from .native_scene import encode_native_scene


class PassFrozenLayer(nn.Module):
    def forward(self,hidden_states,**kwargs):return hidden_states


class FrozenPrefixCache:
    def __init__(self,max_bytes=3*2**30):
        self.entries=OrderedDict();self.max_bytes=max_bytes;self.bytes=0;self.hits=0;self.misses=0

    def encode(self,model,inputs,key,planner=True):
        key=(key,planner)
        if key not in self.entries:
            self.misses+=1;capture={}
            handle=model.vlm.model.language_model.layers[23].register_forward_hook(lambda m,a,o:capture.update(hidden=o.detach()))
            try:native=encode_native_scene(model,inputs)
            finally:handle.remove()
            assert not capture['hidden'].requires_grad
            positions=model._rope_positions(inputs['input_ids'],inputs['llm_grid_thw'])
            entry={'hidden':capture['hidden'].cpu(),'positions':positions.cpu(),
                   'prefix_kv':[(k.detach().cpu(),v.detach().cpu()) for k,v in native['scene_cache'][:6]] if planner else []}
            entry['inputs']={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in inputs.items() if k!='pixel_values'}
            tensors=[entry['hidden'],entry['positions']]+[t for pair in entry['prefix_kv'] for t in pair]+[v for v in entry['inputs'].values() if isinstance(v,torch.Tensor)]
            entry['bytes']=sum(t.numel()*t.element_size() for t in tensors)
            while self.entries and self.bytes+entry['bytes']>self.max_bytes:
                _,old=self.entries.popitem(last=False);self.bytes-=old['bytes']
            if entry['bytes']<=self.max_bytes:self.entries[key]=entry;self.bytes+=entry['bytes']
            return native
        self.hits+=1;entry=self.entries.pop(key);self.entries[key]=entry
        text=model.vlm.model.language_model;original=text.layers
        assert not any(p.requires_grad for layer in original[:24] for p in layer.parameters())
        device=next(model.parameters()).device
        positions=entry['positions'].to(device);hidden=entry['hidden'].to(device)
        captured={};handles=[]
        for index in model.config.full_attention_layers[6:]:
            attn=original[index].self_attn;bucket=captured[index]={}
            handles.append(attn.k_norm.register_forward_hook(lambda m,a,o,b=bucket:b.update(key=o)))
            handles.append(attn.v_proj.register_forward_hook(lambda m,a,o,b=bucket:b.update(value=o)))
            handles.append(attn.register_forward_pre_hook(lambda m,a,kw,b=bucket:b.update(rotary=kw['position_embeddings']),with_kwargs=True))
        try:
            text.layers=nn.ModuleList([PassFrozenLayer() for _ in range(24)]+list(original[24:]))
            with math_attention():result=text(inputs_embeds=hidden,position_ids=positions,use_cache=False)
        finally:
            text.layers=original
            for h in handles:h.remove()
        caches=[(k.to(device),v.to(device)) for k,v in entry['prefix_kv']]
        for index in model.config.full_attention_layers[6:]:
            b=captured[index];k=b['key'].transpose(1,2);cos,sin=b['rotary'];_,k=apply_rotary_pos_emb(k,k,cos,sin)
            v=b['value'].view(*b['value'].shape[:2],k.shape[1],k.shape[-1]);caches.append((k.transpose(1,2),v))
        if planner:assert len(caches)==8
        return {'hidden':result.last_hidden_state,'scene_cache':caches,'position_anchor':positions[:,:,-1]}
