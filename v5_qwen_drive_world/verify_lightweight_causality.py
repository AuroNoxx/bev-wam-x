"""Actual pretrained hybrid-backbone causal gradient with last-eight LoRA."""
from .runtime import ROOT,configure,math_attention
configure()
import json,torch
from .components import load_planning


def main():
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.8)
    cfg=json.loads((ROOT/'configs/v5-qwen-drive-world/lightweight.json').read_text())
    model=load_planning(cfg);model.train()
    text=model.vlm.model.language_model
    # Force input gradients for this diagnostic only. Production prefix is frozen.
    embeds=text.embed_tokens(torch.arange(100,112,device='cuda')[None]).detach().requires_grad_()
    with math_attention(),torch.autocast('cuda',dtype=torch.bfloat16):
        output=text(inputs_embeds=embeds,use_cache=False).last_hidden_state
        output[:,:6,:128].float().square().sum().backward()
    assert torch.isfinite(embeds.grad).all()
    suffix=float(embeds.grad[:,6:].abs().max());assert suffix==0
    grads={}
    for i,layer in enumerate(text.layers):
        if i<24:assert all(p.grad is None for p in layer.parameters())
        else:
            gs=[p.grad for n,p in layer.named_parameters() if 'lora_' in n and p.grad is not None]
            assert gs and all(torch.isfinite(g).all() for g in gs)
            grads[i]=sum(float(g.float().square().sum()) for g in gs)**.5
            assert grads[i]>0
    report={'phase':'passed','suffix_gradient_max':suffix,'prefix_parameter_gradients_absent':True,
            'last_eight_lora_gradient_norms':grads,
            'scope':'actual pretrained 32-layer hybrid model, 12 text tokens, checkpoint recomputation enabled only last8; not multimodal quality validation'}
    (ROOT/'artifacts/v5-qwen-drive-world/lightweight-causal-gradient.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
