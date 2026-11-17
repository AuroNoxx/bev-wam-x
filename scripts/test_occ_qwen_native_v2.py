"""Bounded CPU engineering tests for isolated v2; no pretrained/GPU training."""
import inspect,json,copy,time
from pathlib import Path
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from occ_qwen_native_v2 import NativeOccQwenV2

def main():
 torch.set_num_threads(2);torch.manual_seed(71);started=time.perf_counter()
 c=Qwen3VLTextConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,vocab_size=32,rope_scaling={'rope_type':'default','mrope_section':[2,3,3],'mrope_interleaved':True});c._attn_implementation='sdpa'
 m=NativeOccQwenV2(Qwen3VLTextModel(c),decoder_width=64,decoder_layers=2,heads=4)
 history=torch.randint(0,8,(2,4,20,128,128),dtype=torch.uint8);future=torch.randint(0,8,(2,2,20,128,128),dtype=torch.uint8)
 state=torch.randn(2,20)*.1;state[:,0]=.2
 trajectory=torch.zeros(2,25,5);trajectory[:,:,0]=torch.arange(1,26)[None]*1.2/50;trajectory[:,:,3]=1;trajectory[:,:,4]=.2
 assert list(inspect.signature(m.sample).parameters)==['history','state','steps','generator']
 assert m.encoder(history).shape==(2,256,64)
 assert m.teacher.frames(future).shape==(2,2,64,64)
 assert float(m.world_action_gate)==0 and torch.count_nonzero(m.world_out.weight)==0
 m.eval()
 for steps in [1,3,8]:
  sample=m.sample(history,state,steps=steps,generator=torch.Generator().manual_seed(19))
  assert torch.equal(sample['world_latent'],m.persistence(history)),steps
  assert torch.count_nonzero(sample['world_delta'])==0
  assert sample['trajectory_xy'].shape==(2,25,2) and torch.isfinite(sample['trajectory_xy']).all()
 scene,_=m.encode_history(history);a=torch.randn(2,25,2);t=torch.full((2,1),.5);w=torch.randn(2,128,64)
 v0,_=m.velocity(scene,state,a,t,w);v1,_=m.velocity(scene,state,a,t,w+10)
 assert torch.equal(v0,v1),'closed world gate leaks to Action'
 changed=state.clone();changed[:,6:]+=1
 vstate,_=m.velocity(scene,changed,a,t,w)
 assert torch.equal(v0,vstate),'AdaLN-Zero should start condition-neutral'
 frozen={k:v.clone() for k,v in m.backbone.state_dict().items()};before={k:v.clone() for k,v in m.encoder.state_dict().items()}
 opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3)
 grad={};losses=[]
 for step in range(3):
  m.train();opt.zero_grad(set_to_none=True);loss,parts=m.loss(history,state,trajectory,future);assert torch.isfinite(loss);loss.backward()
  for name,p in m.named_parameters():
   if p.grad is not None:assert torch.isfinite(p.grad).all(),name
  for name,prefix in [('encoder','encoder.'),('state','state_in.'),('world','world_out.'),('action','action_out.')]:grad[name]=max(grad.get(name,0),sum(float(p.grad.abs().sum()) for k,p in m.named_parameters() if k.startswith(prefix) and p.grad is not None))
  torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step();m.update_teacher();losses.append(float(loss.detach()))
 assert all(x>0 for x in grad.values()),grad
 assert any(not torch.equal(v,before[k]) for k,v in m.encoder.state_dict().items())
 assert all(torch.equal(v,frozen[k]) for k,v in m.backbone.state_dict().items())
 m.eval();scene,_=m.encode_history(history);v0,_=m.velocity(scene,state,a,t,w);vstate,_=m.velocity(scene,changed,a,t,w)
 state_effect=float((v0-vstate).abs().max());assert state_effect>1e-7,state_effect
 # Strict serialization roundtrip including EMA and gate, same fixed Action source noise.
 m2=NativeOccQwenV2(Qwen3VLTextModel(c),decoder_width=64,decoder_layers=2,heads=4);m2.load_state_dict(copy.deepcopy(m.state_dict()),strict=True);m2.eval()
 x=m.sample(history,state,steps=2,generator=torch.Generator().manual_seed(29));y=m2.sample(history,state,steps=2,generator=torch.Generator().manual_seed(29))
 assert all(torch.equal(x[k],y[k]) for k in x)
 result={'passed':True,'test_only':True,'device':'cpu','backbone':'tiny random 2 layers; not pretrained','updates':3,'history_tokens':256,'per_frame_tokens':64,'world_tokens':128,'initial_world_equals_persistence_all_tested_steps':True,'world_source':'deterministic zero/Dirac source; not stochastic world hypotheses','closed_world_gate_isolation_exact':True,'adaln_zero_initial_state_neutral':True,'state_condition_effect_after_updates':state_effect,'future_absent_from_sample_signature':True,'frozen_backbone_unchanged':True,'encoder_updated':True,'gradient_l1_max':grad,'serialization_sample_exact':True,'losses':losses,'seconds':time.perf_counter()-started,'production_architecture':'first12 frozen Qwen blocks + width256,3layer decoders; GPU not tested'}
 out=Path('artifacts/occ-qwen-native-v2-tests');out.mkdir(exist_ok=True);(out/'cpu-model.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
