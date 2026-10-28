"""Stage 6 causality, masks and fixed-command tracking invariants."""
import copy
from dataclasses import replace
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch

from configs.stage4 import Stage4Config
from configs.stage6 import Stage6Config,BRANCH_SPLITS
from dataset.action_branches import remaining_command,realized_waypoints
from mini_wa_jepa.geometry import ego_to_world,wrap_angle
from mini_wa_jepa.normalization import Normalizer
from models.joint import JointModel
from models.action_conditioned import ActionConditionedWorld,masked_world_error
from evaluation.action_conditioned import consequence_metrics,cluster_interval


class Stage6Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(19)

    def model(self,conditioned=True):
        cfg=replace(Stage4Config(),encoder_dim=32,encoder_layers=1,encoder_heads=4,encoder_mlp=64,
            predictor_dim=48,predictor_layers=1,predictor_heads=4,predictor_mlp=96,
            action_projection=8,action_hidden=16)
        norm=Normalizer(trajectory_mean=[0.]*4,trajectory_std=[1.]*4)
        return ActionConditionedWorld(JointModel(cfg,norm),conditioned)

    def test_fixed_command_reexpression_preserves_world_points(self):
        command=np.column_stack([np.arange(1,11),np.ones(10),np.full(10,5.),np.zeros(10)]).astype(np.float32)
        anchor=np.array([10.,20.,.7,5.,0.,0.]);current=np.array([11.,21.,1.1,5.,0.,0.])
        remaining=remaining_command(command,anchor,current,0.)
        np.testing.assert_allclose(ego_to_world(remaining[:,:2],current[:2],current[2]),
                                   ego_to_world(command[:,:2],anchor[:2],anchor[2]),atol=1e-6)
        np.testing.assert_allclose(wrap_angle(remaining[:,3]+current[2]),wrap_angle(command[:,3]+anchor[2]),atol=1e-6)

    def test_realized_labels_use_anchor_not_current_frame(self):
        states=np.array([[0,0,np.pi/2,2,0,2],[0,2,np.pi/2,3,0,3],[0,5,np.pi/2+.2,4,0,4]])
        labels=realized_waypoints(states,0,[1,2])
        np.testing.assert_allclose(labels,[[2,0,3,0],[5,0,4,.2]],atol=1e-6)

    def test_unconditional_ignores_command_and_initial_parity(self):
        conditional=self.model();unconditional=copy.deepcopy(conditional);unconditional.conditioned=False
        h=torch.randn(2,4,64,32);ego=torch.randn(2,4,2);a=torch.randn(2,10,4)
        torch.testing.assert_close(conditional(h,ego,a),unconditional(h,ego,a))
        with torch.no_grad():unconditional.command_embedding[-1].weight.normal_()
        torch.testing.assert_close(unconditional(h,ego,a),unconditional(h,ego,a*4))

    def test_only_predictor_and_command_have_gradient(self):
        model=self.model();h=torch.randn(2,4,64,32);ego=torch.randn(2,4,2);a=torch.randn(2,10,4)
        model(h,ego,a).square().mean().backward()
        predictor_blocks = model.blocks if hasattr(model, 'blocks') else model.base.predictor
        self.assertTrue(any(p.grad is not None for p in predictor_blocks.parameters()))
        self.assertTrue(model.command_embedding[-1].weight.grad.abs().sum()>0)
        for module in (model.base.encoder,model.base.action_projection,model.base.action_head):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))

    def test_missing_futures_do_not_enter_loss(self):
        prediction=torch.zeros(2,4,3,5,requires_grad=True);target=torch.ones_like(prediction)
        valid=torch.tensor([[1,1,0,0],[1,0,0,0]],dtype=torch.bool)
        target=target.detach();target[~valid]=9999
        loss=masked_world_error(prediction,target,valid);self.assertAlmostEqual(float(loss.detach()),1.)
        loss.backward();self.assertEqual(float(prediction.grad[~valid].abs().sum()),0.)

    def test_split_seed_and_horizon_contract(self):
        cfg=Stage6Config();sets=[]
        for split,spec in BRANCH_SPLITS.items():
            count=getattr(cfg,split+'_seeds_per_map')
            sets.append({spec['seed_start']+i*10000+j for i in range(len(spec['maps'])) for j in range(count)})
        for i in range(3):
            for j in range(i):self.assertFalse(sets[i]&sets[j])
        self.assertEqual(tuple(np.rint(np.array(cfg.future_times_s)/cfg.control_dt_s).astype(int)),(5,10,15,20))
        self.assertEqual(len(cfg.candidates),4)

    def test_correct_consequences_score_better_than_sensitivity_alone(self):
        target=np.broadcast_to(np.tile(np.arange(4),(2,1)).reshape(8,1,1,1),(8,4,2,3)).astype(np.float32)
        valid=np.ones((8,4),bool);groups=np.repeat(np.arange(2),4);seeds=np.repeat([10,20],4)
        perfect,_=consequence_metrics(target,target,valid,groups,seeds)
        wrong=target.reshape(2,4,4,2,3)[:,::-1].reshape(target.shape)
        sensitive,_=consequence_metrics(wrong,target,valid,groups,seeds)
        constant,_=consequence_metrics(np.zeros_like(target),target,valid,groups,seeds)
        self.assertEqual(perfect['retrieval_accuracy'],1.)
        self.assertEqual(perfect['delta_mse'],0.)
        self.assertGreater(sensitive['delta_mse'],constant['delta_mse'])
        self.assertEqual(sensitive['retrieval_accuracy'],0.)

    def test_bootstrap_clusters_repeated_anchors_by_seed(self):
        interval=cluster_interval(np.ones(8),np.full(8,2.),np.ones(8),np.repeat([10,20],4),100,19)
        self.assertEqual(interval['seed_clusters'],2)
        np.testing.assert_allclose(interval['delta_ci95'],[-1.,-1.])

    def test_public_logger_retains_multiple_candidates_at_same_step(self):
        from mini_wa_jepa.visualization import RerunLogger
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'branches.rrd'
            with RerunLogger(spawn=False,save_path=path) as log:
                log.set_step(12,0.)
                for name in ('keep','left'):
                    log.log_candidate(name,commanded=np.zeros((10,4)),actual=np.zeros((10,4)),
                        predicted_latent=np.zeros((4,64,256)),target_latent=np.zeros((4,64,256)))
                with self.assertRaises(ValueError):
                    log.log_candidate('../unsafe',commanded=None,actual=None,predicted_latent=None,target_latent=None)
            self.assertGreater(path.stat().st_size,1000)


if __name__=='__main__':unittest.main()
