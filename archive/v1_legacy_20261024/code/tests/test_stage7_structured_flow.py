"""Train-only fitting, projection, masking, gradients and deployed-sampler parity."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from mini_wa_jepa.normalization import Normalizer
from models.flow_matcher_structured import StructuredFlowMatcher, fit_trajectory_basis, sample_structured_anchors


class StructuredFlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(3)
        basis,_ = np.linalg.qr(np.random.default_rng(3).normal(size=(40,12)))
        self.model = StructuredFlowMatcher(token_count=4,token_dim=8,d_model=16,n_heads=2,
                  n_layers=1,dim_feedforward=32,basis=basis,coefficient_std=np.linspace(.2,2,12),
                  residual_mean=np.arange(40)*.001,normalizer=Normalizer()).eval()
        self.h=torch.randn(2,4,4,8); self.ego=torch.randn(2,4,2)*.1

    def test_train_only_fit_rejects_validation_and_partial_targets(self):
        with self.assertRaisesRegex(ValueError,'train split'):
            fit_trajectory_basis(SimpleNamespace(split='val'))
        with self.assertRaisesRegex(ValueError,'complete valid'):
            fit_trajectory_basis(SimpleNamespace(split='train',action_mask=np.array([False]),waypoint_valid=np.ones((1,10),bool)))

    def test_fitted_basis_is_orthonormal_and_coefficients_are_standardized(self):
        d=SimpleNamespace(split='train',action_mask=np.ones(80,bool),waypoint_valid=np.ones((80,10),bool),
             ego=np.zeros((80,4,2),np.float32),targets=np.random.default_rng(2).normal(size=(80,10,4)).astype(np.float32),
             normalizer=Normalizer())
        fit=fit_trajectory_basis(d,12)
        np.testing.assert_allclose(fit['basis'].T@fit['basis'],np.eye(12),atol=1e-6)
        z=(d.targets/fit['residual_std']).reshape(80,40)
        coefficients=((z-fit['residual_mean'])@fit['basis'])/fit['coefficient_std']
        np.testing.assert_allclose(coefficients.mean(0),0,atol=1e-6)
        np.testing.assert_allclose(coefficients.std(0),1,atol=1e-6)

    def test_shared_transformer_initialization_matches_v2(self):
        from models.flow_matcher_metric import MetricFlowMatcher
        torch.manual_seed(91)
        old=MetricFlowMatcher()
        torch.manual_seed(91)
        new=StructuredFlowMatcher()
        old_params=dict(old.named_parameters());matched=0
        for key,value in new.named_parameters():
            if key in old_params:
                torch.testing.assert_close(value,old_params[key],rtol=0,atol=0);matched+=value.numel()
        self.assertGreater(matched,1800000)

    def test_coefficient_roundtrip_and_generated_paths_stay_in_basis(self):
        c=torch.randn(2,8,12)
        torch.testing.assert_close(self.model.encode_residual(self.model.decode_residual(c)),c,rtol=1e-5,atol=1e-6)
        out=self.model.sample(self.h,self.ego,num_candidates=8,seed=14)
        residual=(out-self.model.compute_norm_prior(self.ego)[:,None])/self.model.residual_std
        reconstructed=self.model.decode_residual(self.model.encode_residual(residual))
        torch.testing.assert_close(residual,reconstructed,rtol=1e-5,atol=1e-6)

    def test_per_anchor_sampling_matches_serial_model_sampler(self):
        batched=sample_structured_anchors(self.model,self.h,self.ego,[7,29],.8)
        for i,a in enumerate([7,29]):
            serial=self.model.sample(self.h[i:i+1],self.ego[i:i+1],num_candidates=8,temperature=.8,seed=42+a)
            torch.testing.assert_close(batched[i],serial[0],rtol=1e-5,atol=1e-6)

    def test_training_gradients_and_no_nan_pollution_from_invalid_sample(self):
        target=torch.randn(2,10,4); target[1]=float('nan')
        loss,_,parts=self.model.compute_loss(self.h,self.ego,target,torch.tensor([True,False]),
                           torch.ones(2,10,dtype=torch.bool),return_parts=True)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertGreater(self.model.coefficient_head.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
        self.assertTrue(all(torch.isfinite(x) for x in parts.values()))

    def test_partial_valid_trajectory_is_rejected_explicitly(self):
        mask=torch.ones(2,10,dtype=torch.bool);mask[0,-1]=False
        with self.assertRaisesRegex(ValueError,'partial labels'):
            self.model.compute_loss(self.h,self.ego,torch.randn(2,10,4),torch.ones(2,dtype=torch.bool),mask)

    def test_checkpoint_roundtrip_preserves_samples(self):
        import copy
        other=copy.deepcopy(self.model)
        other.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(self.model.sample(self.h,self.ego,num_candidates=8,seed=8),
                                   other.sample(self.h,self.ego,num_candidates=8,seed=8),rtol=0,atol=0)

    def test_twenty_coefficient_control_keeps_explicit_trajectory_and_gradients(self):
        model=StructuredFlowMatcher(dimensions=20,token_count=4,token_dim=8,d_model=16,n_heads=2,
                  n_layers=1,dim_feedforward=32,normalizer=Normalizer()).eval()
        out=model.sample(self.h,self.ego,num_candidates=8,seed=42)
        self.assertEqual(out.shape,(2,8,10,4))
        loss,_=model.compute_loss(self.h,self.ego,torch.randn(2,10,4),torch.ones(2,dtype=torch.bool),
                                  torch.ones(2,10,dtype=torch.bool))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.coefficient_head.weight.grad.shape,(20,160))


if __name__=='__main__':unittest.main()
