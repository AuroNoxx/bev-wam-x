"""Unit checks for BEWWAJointDataset."""

import unittest
from pathlib import Path
import torch
from dataset.bev_wa_dataset import BEWWAJointDataset

TRAIN_MANIFEST = "artifacts/stage7_integrated/mixed_train_010/manifest.json"
VAL_MANIFEST = "artifacts/stage7_integrated/mixed_validation_010/manifest.json"


class BEWWAJointDatasetTests(unittest.TestCase):
    def test_train_dataset_loading_and_shapes(self):
        ds = BEWWAJointDataset(TRAIN_MANIFEST, split="train", eligible_action_only=False)
        self.assertEqual(len(ds), 13952)
        stats = ds.stats()
        self.assertEqual(stats["episodes"], 48)
        self.assertEqual(stats["eligible_action_anchors"], 12186)
        
        # Test item fetching at step 0 (boundary step)
        item0 = ds[0]
        self.assertEqual(item0["history_bev"].shape, (4, 4, 128, 128))
        self.assertEqual(item0["history_bev"].dtype, torch.uint8)
        self.assertEqual(item0["history_valid"].tolist(), [False, False, False, True])
        self.assertEqual(item0["ego_motion"].shape, (4, 2))
        self.assertEqual(item0["applied_action"].shape, (4, 2))
        self.assertEqual(item0["route_points"].shape, (8, 2))
        self.assertEqual(item0["future_bev"].shape, (4, 4, 128, 128))
        self.assertEqual(item0["future_bev_mask"].shape, (4,))
        self.assertEqual(item0["trajectory"].shape, (10, 4))
        self.assertEqual(item0["future_traj_mask"].shape, (10,))
        
        # Test item at step 50 (middle of episode, all history valid)
        item50 = ds[50]
        self.assertEqual(item50["history_valid"].tolist(), [True, True, True, True])
        self.assertTrue(item50["future_bev_mask"].all().item())
        self.assertTrue(item50["future_traj_mask"].all().item())

    def test_action_only_subset(self):
        ds_action = BEWWAJointDataset(TRAIN_MANIFEST, split="train", eligible_action_only=True)
        self.assertEqual(len(ds_action), 12186)
        # All items must have action_mask == True
        for i in [0, 100, 1000, 5000, 12000]:
            self.assertTrue(ds_action[i]["action_mask"].item())

    def test_validation_dataset(self):
        ds_val = BEWWAJointDataset(VAL_MANIFEST, split="validation", eligible_action_only=False)
        self.assertEqual(len(ds_val), 7371)
        stats = ds_val.stats()
        self.assertEqual(stats["episodes"], 24)
        self.assertEqual(stats["eligible_action_anchors"], 6453)


if __name__ == "__main__":
    unittest.main()
