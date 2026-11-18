"""Adapters to existing FLA kernels; no custom attention or convolution math."""
from .runtime import configure
configure()


def configure_native_kernels(options):
    # Transformers selects these implementations when constructing each native
    # layer. Explicit flags prevent an installed package silently choosing paths.
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    if options.get('fla_linear_attention'):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        native.chunk_gated_delta_rule = chunk_gated_delta_rule
    else:
        native.chunk_gated_delta_rule = None
    if options.get('fla_gated_norm'):
        from fla.modules import FusedRMSNormGated
        native.FusedRMSNormGated = FusedRMSNormGated
    else:
        native.FusedRMSNormGated = None
    # Incremental inference is not covered by the training-kernel verification.
    native.fused_recurrent_gated_delta_rule = None
    native.causal_conv1d_fn = None
    native.causal_conv1d_update = None
    from transformers import AttentionInterface
    from functools import partial
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    chunk = options.get('math_query_chunk', 0)
    AttentionInterface.register('sdpa', partial(tiled_native_sdpa, chunk_size=chunk)
                                if chunk else sdpa_attention_forward)


def tiled_native_sdpa(module, query, key, value, attention_mask, dropout=0.,
                      scaling=None, is_causal=None, position_bias=None, *, chunk_size=256, **kwargs):
    """Query tiling around the original Transformers SDPA MATH operator.

    Every query attends to its full permitted key set. Explicit masks preserve
    absolute query positions. Checkpoint each tile to avoid retaining T*T maps.
    No attention math/kernel or model parameters are replaced.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from .runtime import math_attention
    length = query.shape[2]
    if length <= chunk_size or dropout or position_bias is not None:
        with math_attention():
            return sdpa_attention_forward(module, query, key, value, attention_mask,
                    dropout=dropout, scaling=scaling, is_causal=is_causal,
                    position_bias=position_bias, **kwargs)
    causal = is_causal if is_causal is not None else getattr(module, 'is_causal', True)
    causal = length > 1 and attention_mask is None and causal
    if causal and key.shape[2] > length:
        key, value = key[:,:,:length], value[:,:,:length]
    outputs = []
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        mask = attention_mask
        if causal:
            mask = (torch.arange(start,end,device=query.device)[:,None] >=
                    torch.arange(key.shape[2],device=query.device)[None,:])[None,None]
        elif mask is not None and mask.shape[-2] != 1:
            mask = mask[...,start:end,:]
        def run(q, k, v, m):
            with math_attention():
                return sdpa_attention_forward(module,q,k,v,m,dropout=0.,scaling=scaling,
                                               is_causal=False,**kwargs)[0]
        args = (query[:,:,start:end],key,value,mask)
        if torch.is_grad_enabled() and any(x.requires_grad for x in args[:3]):
            outputs.append(checkpoint(run,*args,use_reentrant=False,
                                      context_fn=lambda: (math_attention(),math_attention())))
        else:
            outputs.append(run(*args))
    return torch.cat(outputs,dim=1), None


def fla_causal_conv1d(x, weight, bias=None, activation=None, seq_idx=None):
    from fla.modules.causal_conv1d import causal_conv1d
    if seq_idx is not None:
        raise ValueError('Packed sequence IDs need explicit conversion before using this adapter')
    output, _ = causal_conv1d(x.transpose(1,2),weight=weight,bias=bias,
                              activation=activation,backend='triton')
    return output.transpose(1,2)


def attach_fla_convolution(model):
    count=0
    for module in model.modules():
        if module.__class__.__name__ == 'Qwen3_5GatedDeltaNet':
            module.causal_conv1d_fn = fla_causal_conv1d
            count += 1
    return count
