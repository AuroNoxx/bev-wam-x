import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from configs.stage4 import Stage4Config
from dataset.joint import JointSequences
from dataset.storage import ShardWriter
from evaluation.history import ObservedHistory
from mini_wa_jepa.normalization import Normalizer
from models.joint import JointModel,constant_turn_trajectory
from models.latent import LatentModel
from training.joint_loss import joint_loss


class SequenceTests(unittest.TestCase):
    def test_startup_padding_gaps_future_masks_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'manifest.json').write_text(json.dumps(dict(schema_version=2,completed=True,
                config=dict(dt_s=.1,future_steps=10,waypoint_dt_s=.2))))
            (root/'normalization.json').write_text(json.dumps(Normalizer().to_dict()))
            state=np.zeros((6,6),np.float32);state[:,2]=np.arange(6)*.1
            writer=ShardWriter(root/'train',2)
            writer.add(dict(bev=np.zeros((6,4,128,128),np.uint8),state=state,
                episode_id=np.array([0,0,0,1,1,1]),step_id=np.array([0,1,2,0,1,3]),
                future_trajectory=np.zeros((6,10,4),np.float32),future_valid=np.ones((6,10),bool),
                action_mask=np.array([True,True,False,True,True,True]),noise_active=np.ones(6,bool),
                recovery_active=np.ones(6,bool)))
            writer.close();data=JointSequences(root)
            np.testing.assert_array_equal(data.history_indices[2],[0,0,1,2])
            np.testing.assert_array_equal(data.history_indices[4],[3,3,3,4])
            np.testing.assert_array_equal(data.history_indices[5],[5,5,5,5])
            np.testing.assert_array_equal(data[0]['world_valid'],[True,True,False,False])
            self.assertFalse(data[2]['world_valid'].any());self.assertFalse(data[4]['world_valid'].any())
            self.assertEqual(len(data),6);self.assertTrue(data[1]['noisy']);self.assertTrue(data[1]['recovery'])
            self.assertEqual(data[3]['history_padding'],3);self.assertEqual(float(data[3]['ego'][-1,1]),0.)

    def test_cache_time_yaw_and_reset_are_observed_only(self):
        cache=ObservedHistory()
        for step in range(23):
            z=torch.full((1,2,3),float(step));history,motion,indices=cache.append(step,z,7,step*.002)
        self.assertEqual(indices,[7,12,17,22]);self.assertEqual(len(cache.frames),16)
        np.testing.assert_allclose(motion[0,:,1],.1,atol=1e-6)
        np.testing.assert_allclose(history[0,:,0,0],[7,12,17,22])
        with self.assertRaises(ValueError):cache.append(0,z,0,0)
        cache.reset();history,motion,indices=cache.append(0,z,0,0)
        self.assertEqual(indices,[0,0,0,0]);self.assertTrue((motion==0).all())


class JointModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def small(self):
        cfg=replace(Stage4Config(),encoder_layers=1,predictor_layers=1)
        return cfg,JointModel(cfg,Normalizer(scale=2.,trajectory_mean=[1,2,3,.1],trajectory_std=[2,3,4,.2]))

    def data(self):
        torch.manual_seed(7)
        past=torch.randint(0,256,(2,4,4,128,128),dtype=torch.uint8)
        future=torch.randint(0,256,past.shape,dtype=torch.uint8)
        ego=torch.randn(2,4,2)*.05;ego[:,:,0]=.7
        return past,ego,future

    def test_parameter_budgets(self):
        model=JointModel(Stage4Config(),Normalizer())
        total=sum(p.numel() for p in model.parameters())
        predictor=sum(p.numel() for p in model.predictor.parameters())+sum(p.numel() for p in model.history_fusion.parameters())
        self.assertTrue(15e6<=total<=30e6);self.assertTrue(8e6<=predictor<=12e6)

    def test_kinematic_arc_and_normalization_roundtrip(self):
        cfg,model=self.small();past,ego,future=self.data();ego[:,:,1]=0
        with sdpa_kernel(SDPBackend.MATH):out=model(past,ego)
        normalizer=Normalizer(scale=2.,trajectory_mean=[1,2,3,.1],trajectory_std=[2,3,4,.2])
        raw=normalizer.inverse_trajectory(out['trajectory'].detach().numpy())
        np.testing.assert_allclose(raw,constant_turn_trajectory(ego).numpy(),atol=1e-6)
        np.testing.assert_allclose(raw[0,:,0],np.arange(1,11)*1.4,atol=2e-6)
        turn=ego.clone();turn[:,:,1]=.2;arc=constant_turn_trajectory(turn)
        self.assertTrue((arc[:,:,1]>0).all());self.assertTrue((arc[:,:,3]>0).all())

    def test_future_is_target_only_and_cached_encoding_matches_full(self):
        cfg,model=self.small();past,ego,future=self.data()
        torch.nn.init.normal_(model.action_head[-1].weight,std=.01)
        torch.nn.init.normal_(model.history_fusion.weight,std=.01)
        with sdpa_kernel(SDPBackend.MATH):
            output=model(past,ego,future)
            alternate=model(past,ego,future.flip(0))
            torch.testing.assert_close(output['trajectory'],alternate['trajectory'])
            torch.testing.assert_close(output['latent'],alternate['latent'])
            encoded=torch.cat([model.encoder(frame[None]) for frame in past.flatten(0,1)]).reshape(2,4,64,256)
            cached=model.forward_encoded(encoded,ego)
            torch.testing.assert_close(cached['trajectory'],output['trajectory'],atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(cached['latent'],output['latent'],atol=2e-5,rtol=2e-5)

    def test_stage3_initialization_preserves_first_horizon(self):
        cfg,model=self.small();old=LatentModel(cfg);model.initialize_stage3(dict(model=old.state_dict()))
        past,ego,future=self.data()
        with sdpa_kernel(SDPBackend.MATH):
            expected=old(past[:,-1],ego[:,-1]);actual=model(past,ego)['latent'][:,0]
        torch.testing.assert_close(actual,expected)

    def test_action_only_gradient_reaches_world_head_and_encoder(self):
        cfg,model=self.small();past,ego,future=self.data()
        torch.nn.init.normal_(model.action_head[-1].weight,std=.01)
        batch=dict(trajectory=torch.zeros(2,10,4),valid=torch.ones(2,10,dtype=torch.bool),
                   action_mask=torch.ones(2,dtype=torch.bool),world_valid=torch.ones(2,4,dtype=torch.bool))
        with sdpa_kernel(SDPBackend.MATH):
            out=model(past,ego);loss,terms=joint_loss(out,batch,replace(cfg,latent_weight=0));loss.backward()
        self.assertEqual(float(terms['latent']),0.)
        self.assertGreater(float(model.predictor.output.weight.grad.norm()),0)
        self.assertGreater(float(model.encoder.stem[0].weight.grad.norm()),0)

    def test_masked_future_does_not_contribute_and_target_has_gradients(self):
        cfg,model=self.small();past,ego,future=self.data()
        batch=dict(trajectory=torch.zeros(2,10,4),valid=torch.ones(2,10,dtype=torch.bool),
                   action_mask=torch.ones(2,dtype=torch.bool),world_valid=torch.tensor([[1,1,1,0],[1,1,1,0]],dtype=torch.bool))
        with sdpa_kernel(SDPBackend.MATH):
            out=model(past,ego,future);out['target'].retain_grad();loss,_=joint_loss(out,batch,cfg)
            changed=dict(out);changed['target']=out['target'].clone();changed['target'][:,3]+=10000
            alternative,_=joint_loss(changed,batch,cfg);torch.testing.assert_close(loss,alternative)
            loss.backward()
        self.assertGreater(float(out['target'].grad[:,:3].norm()),0)
        self.assertEqual(float(out['target'].grad[:,3].norm()),0.)


if __name__=='__main__':unittest.main()
