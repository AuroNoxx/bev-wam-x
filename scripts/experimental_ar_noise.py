"""Experimental matched training/sampling XY Gaussian AR noise; default untouched."""
from experimental_residual_representation import ResidualJEPA
from train_fresh_jepa_first1024_v2 import FreshTask
import torch,hashlib
from torch.nn import functional as F

def correlate_xy(e,rho):
 if rho==0:return e
 z=e.clone();scale=(1-rho*rho)**.5
 for i in range(1,z.shape[1]):z[:,i,:2]=rho*z[:,i-1,:2]+scale*e[:,i,:2]
 return z
class NoiseResidualJEPA(ResidualJEPA):
 def set_noise(self,rho):
  assert 0<=rho<1;self.noise_rho=float(rho)
 def noise_like(self,x):return correlate_xy(torch.randn_like(x),self.noise_rho)
 @torch.no_grad()
 def sample(self,batch,k=8,steps=16):
  context=self.context(batch).repeat_interleave(k,0);x=correlate_xy(torch.randn(len(context),25,5,device=context.device),self.noise_rho)
  for i in range(steps):x=x+self.velocity(context,x,torch.full((len(x),1),i/steps,device=x.device))/steps
  x=self.from_flow_space({'state':batch['state'].repeat_interleave(k,0)},x);x[...,2:4]=F.normalize(x[...,2:4],dim=-1,eps=1e-6);x[...,4]=x[...,4].clamp_min(0)
  return x.reshape(len(batch['state']),k,25,5)
def noise_flow_loss(model,batch,config):
 from stage1_objective import kinematic_residual
 physical_target=batch['trajectory'].float();target=model.to_flow_space(batch,physical_target);noise=model.noise_like(target);t=torch.rand(len(target),1,device=target.device);xt=(1-t[:,:,None])*noise+t[:,:,None]*target;prediction=model(batch,xt,t).float();fm=F.mse_loss(prediction,target-noise);endpoint=model.from_flow_space(batch,xt+(1-t[:,:,None])*prediction);gate=(t[:,0]>=config['auxiliary_min_t']).float()/(1-config['auxiliary_min_t'])
 def gated(v):return (v.flatten(1).mean(1)*gate).mean()
 xy=(endpoint[...,:2]-physical_target[...,:2])*endpoint.new_tensor([50,10]);position=gated(F.smooth_l1_loss(xy/config['position_scale_m'],torch.zeros_like(xy),reduction='none'));res=kinematic_residual(endpoint,batch['state'].float())/config['velocity_scale_mps'];kin=gated(F.smooth_l1_loss(res,torch.zeros_like(res),reduction='none'));heading=gated((endpoint[...,2:4].square().sum(-1)-1).square());negative=gated(F.relu(-endpoint[...,4]*30/config['velocity_scale_mps']).square());terms={'fm':fm,'position':position,'kinematics':kin,'unit_heading':heading,'negative_speed':negative};return sum(config['weights'][k]*v for k,v in terms.items()),terms
class NoiseTask(FreshTask):
 def task_loss(self,b,split):
  with torch.random.fork_rng(devices=[self.device.index]):
   torch.manual_seed(self.seed*100000+self.global_step);before=torch.cuda.get_rng_state().clone();v,terms=noise_flow_loss(self.model,b,self.settings['objective'])
   for k,x in zip(['latent','variance','semantic'],self.model.auxiliary_loss(b)):v=v+self.settings['aux_weights'][k]*x;terms[k]=x
   after=torch.cuda.get_rng_state().clone()
  if self.global_step<771:self.trace.append({'step':self.global_step,'before':hashlib.sha256(before.cpu().numpy().tobytes()).hexdigest(),'after':hashlib.sha256(after.cpu().numpy().tobytes()).hexdigest()})
  self.log_dict({f'{split}/{k}':x for k,x in terms.items()},on_epoch=True,batch_size=len(b['state']));return v
 def on_save_checkpoint(self,c):
  super().on_save_checkpoint(c);c['noise_runtime']={'xy_ar1_rho':self.model.noise_rho,'training_and_sampler_matched':True,'source_was_iid':True}
