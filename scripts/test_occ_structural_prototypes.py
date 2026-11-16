import unittest,json,numpy as np,torch
from pathlib import Path
from occ_structural_prototypes import PrototypeJoint,future_positive_targets,prototype_loss
from train_masked_teacher_bounded import DiskSamples
from train_joint_occ import batch_to
from stage1_objective import kinematic_residual
from lightning_task import ROOT,configure_runtime
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  configure_runtime();cls.sel=json.loads((ROOT/'artifacts/residual-representation-paired128/selection.json').read_text());cls.b=batch_to([DiskSamples(cls.sel['train'])[i] for i in range(2)],'cpu');cls.scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m']
 def model(self,action='velocity2',world='current_support'):
  m=PrototypeJoint(128,2,4,action,world);m.set_scales(self.scales);return m.eval()
 def test_velocity_roundtrip_and_integration(self):
  m=self.model();p=self.b['trajectory'];q=m.to_flow_space(self.b,p);z=m.from_flow_space(self.b,q);v=p[...,4:5]*30*p[...,[3,2]];vz=z[...,4:5]*30*z[...,[3,2]];torch.testing.assert_close(v,vz,rtol=1e-5,atol=2e-6);self.assertLess(float(kinematic_residual(z,self.b['state']).abs().max()),2e-4)
 def test_constant_velocity_and_t0(self):
  m=self.model();b={**self.b,'state':self.b['state'].clone()};b['state'][:,0]=10/30;z=m.from_flow_space(b,torch.zeros(2,25,2));torch.testing.assert_close(z[...,:2]*torch.tensor([50,10]),torch.stack([torch.arange(1,26)*2,torch.zeros(25)],-1)[None].expand(2,-1,-1));torch.testing.assert_close(z[...,4],torch.full((2,25),10/30))
 def test_left_axis_and_zero_speed_finite(self):
  m=self.model();b={**self.b,'state':self.b['state'].clone()};b['state'][:,0]=0;q=torch.zeros(2,25,2);z=m.from_flow_space(b,q);self.assertTrue(torch.isfinite(z).all());q[...,1]=.5;z=m.from_flow_space(b,q);self.assertTrue((z[...,1]>0).all());self.assertTrue((z[...,2]>0).all())
 def test_future_fields_not_in_prior_or_actor(self):
  m=self.model();b=self.b;poison={**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)};q=torch.randn(2,25,2);t=torch.ones(2,1)*.3
  with torch.no_grad():self.assertTrue(torch.equal(m(b,q,t),m(poison,q,t)));self.assertTrue(torch.equal(m.from_flow_space(b,q),m.from_flow_space(poison,q)))
 def test_new_future_positive_included_unknown_excluded(self):
  b={**self.b,'history':torch.full_like(self.b['history'],255),'geometry_valid':torch.zeros_like(self.b['geometry_valid']),'future':torch.full_like(self.b['future'],255)};b['future'][:,0,0,0,0]=4;b['future'][:,0,0,1,0]=0;old=self.model('xy5').target_data(b);new=self.model('xy5','future_positive').target_data(b);self.assertEqual(int(old[1].sum()),0);self.assertGreater(int(new[1].sum()),0);self.assertTrue((new[0][:,0,0,1,0]==255).all())
 def test_world_mask_does_not_change_actor_same_weights(self):
  a=self.model('xy5');b=self.model('xy5','future_positive');b.load_state_dict(a.state_dict());x=torch.randn(2,25,5);t=torch.ones(2,1)*.3
  with torch.no_grad():self.assertTrue(torch.equal(a(self.b,x,t),b(self.b,x,t)))
 def test_loss_backward_finite_no_optimizer(self):
  m=self.model();cfg=json.loads((ROOT/'configs/joint-occ-v1.json').read_text())['objective'];v,terms=prototype_loss(m,self.b,cfg);v.backward();self.assertTrue(torch.isfinite(v));self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
if __name__=='__main__':unittest.main()
