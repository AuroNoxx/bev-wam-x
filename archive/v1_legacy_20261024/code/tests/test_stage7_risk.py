import unittest
import numpy as np
import torch
from models.risk_head import ConsequenceRisk, masked_risk_loss
from evaluation.risk import binary_metrics, validation_threshold, paired_seed_interval


class RiskTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(3)

    def test_zero_world_control_ignores_predicted_future(self):
        model=ConsequenceRisk(False,token_dim=16,tokens=4,hidden=16,dropout=0)
        h=torch.randn(3,4,4,16);e=torch.randn(3,4,2);c=torch.randn(3,10,4)
        torch.testing.assert_close(model(h,e,c,h),model(h,e,c,h*100))

    def test_world_head_uses_predictions_and_preserves_spatial_positions(self):
        model=ConsequenceRisk(True,token_dim=16,tokens=4,hidden=16,dropout=0)
        h=torch.randn(3,4,4,16);e=torch.randn(3,4,2);c=torch.randn(3,10,4)
        p=torch.randn_like(h,requires_grad=True);model(h,e,c,p).sum().backward()
        self.assertGreater(float(p.grad.abs().sum()),0)
        self.assertFalse(torch.allclose(model(h,e,c,p),model(h,e,c,p.flip(2))))

    def test_censored_labels_cannot_change_loss_or_gradient(self):
        logits=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True)
        valid=torch.tensor([[True,False],[False,True]])
        y=torch.tensor([[1.,float('nan')],[float('nan'),0.]])
        loss=masked_risk_loss(logits,y,valid);self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertEqual(float(logits.grad[~valid].abs().sum()),0.)

    def test_ties_are_not_perfect_discrimination(self):
        m=binary_metrics([0,1,0,1],[.5]*4)
        self.assertAlmostEqual(m['auroc'],.5);self.assertAlmostEqual(m['average_precision'],.5)
        perfect=binary_metrics([0,1,0,1],[0.,1.,0.,1.])
        self.assertEqual(perfect['auroc'],1.);self.assertEqual(perfect['brier'],0.)
        self.assertIsNone(binary_metrics([0,0],[.1,.2])['auroc'])

    def test_threshold_is_from_validation_and_cluster_bootstrap(self):
        self.assertEqual(validation_threshold([0,0,1,1],[.1,.2,.7,.8]),.7)
        self.assertIsNone(validation_threshold([0,0],[.1,.2]))
        r=paired_seed_interval(np.ones(8),np.ones(8)*2,np.repeat([1,2],4),100)
        self.assertEqual(r['seed_clusters'],2);self.assertEqual(r['ci95'],[-1.,-1.])


if __name__=='__main__':unittest.main()
