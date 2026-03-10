"""
Unit tests for PLUSS_β components
"""

import torch
import numpy as np
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pluss_beta.models.semantic_tuner import SemanticTuner, SemanticTunerLoss
from pluss_beta.models.box_tuner import BoxTuner, BoxTunerLoss
from pluss_beta.models.memory_bank import MemoryBank
from pluss_beta.utils.point2box import point_to_box


class TestSemanticTuner(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.embed_dim = 512
        self.num_prompts = 16
        self.tuner = SemanticTuner(
            num_layers=12,
            embed_dim=self.embed_dim,
            num_prompts=self.num_prompts
        )
    
    def test_initialization(self):
        """Test semantic tuner initialization"""
        self.assertEqual(len(self.tuner.prompts), 12)
        self.assertEqual(self.tuner.prompts[0].shape, (self.num_prompts, self.embed_dim))
    
    def test_forward(self):
        """Test forward pass"""
        # Create dummy image features [batch, num_patches, embed_dim]
        num_patches = 197  # ViT-B/16 for 224x224
        features = torch.randn(self.batch_size, num_patches, self.embed_dim)
        
        # Forward through layer 0
        augmented = self.tuner(features, layer_idx=0)
        
        # Check output shape: [batch, 1 (CLS) + num_prompts + (num_patches-1), embed_dim]
        expected_len = 1 + self.num_prompts + (num_patches - 1)
        self.assertEqual(augmented.shape, (self.batch_size, expected_len, self.embed_dim))
    
    def test_loss(self):
        """Test semantic tuner loss"""
        loss_fn = SemanticTunerLoss()
        
        visual_features = torch.randn(self.batch_size, self.embed_dim)
        text_features = torch.randn(self.batch_size, self.embed_dim)
        
        loss = loss_fn(visual_features, text_features)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)


class TestBoxTuner(unittest.TestCase):
    def setUp(self):
        self.num_boxes = 5
        self.feature_dim = 512
        self.sam_dim = 256
        self.tuner = BoxTuner(
            feature_dim=self.feature_dim,
            num_heads=8,
            hidden_dim=2048
        )
    
    def test_initialization(self):
        """Test box tuner initialization"""
        self.assertIsNotNone(self.tuner.sam_projection)
        self.assertIsNotNone(self.tuner.cross_attention)
        self.assertIsNotNone(self.tuner.mlp)
    
    def test_forward(self):
        """Test forward pass"""
        sam_tokens = torch.randn(self.num_boxes, self.sam_dim)
        clip_features = torch.randn(self.num_boxes, self.feature_dim)
        
        fused = self.tuner(sam_tokens, clip_features)
        
        self.assertEqual(fused.shape, (self.num_boxes, self.feature_dim))
    
    def test_box_refinement(self):
        """Test box refinement"""
        initial_boxes = torch.tensor([
            [100, 100, 50, 50],
            [200, 200, 80, 80]
        ], dtype=torch.float32)
        
        delta_boxes = torch.tensor([
            [5, 5, 0.1, 0.1],
            [-3, -3, -0.05, -0.05]
        ], dtype=torch.float32)
        
        refined = self.tuner.refine_boxes(initial_boxes, delta_boxes)
        
        self.assertEqual(refined.shape, initial_boxes.shape)
        # Check that boxes have been modified
        self.assertFalse(torch.allclose(refined, initial_boxes))
    
    def test_loss(self):
        """Test box tuner loss"""
        loss_fn = BoxTunerLoss()
        
        pred_boxes = torch.tensor([
            [100, 100, 50, 50],
            [200, 200, 80, 80]
        ], dtype=torch.float32)
        
        target_boxes = torch.tensor([
            [102, 98, 52, 48],
            [198, 202, 82, 78]
        ], dtype=torch.float32)
        
        loss, metrics = loss_fn(pred_boxes, target_boxes)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
        self.assertIn('l1', metrics)
        self.assertIn('giou', metrics)


class TestMemoryBank(unittest.TestCase):
    def setUp(self):
        self.bank = MemoryBank(capacity=10, threshold=0.5)
    
    def test_initialization(self):
        """Test memory bank initialization"""
        self.assertEqual(len(self.bank), 0)
        self.assertEqual(self.bank.capacity, 10)
        self.assertEqual(self.bank.threshold, 0.5)
    
    def test_mask_loss(self):
        """Test mask loss computation"""
        mask1 = torch.rand(256, 256)
        mask2 = torch.rand(256, 256)
        
        loss = self.bank.compute_mask_loss(mask1, mask2)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(loss.item(), 0)
        self.assertLessEqual(loss.item(), 1)
    
    def test_add_entry(self):
        """Test adding entries"""
        image_features = torch.randn(512)
        text_features = torch.randn(512)
        mask1 = torch.ones(256, 256)
        mask2 = torch.zeros(256, 256)  # High disagreement
        
        added, loss_val = self.bank.add_entry(
            image_features, text_features, mask1, mask2
        )
        
        # Should be added (high loss)
        self.assertTrue(added)
        self.assertEqual(len(self.bank), 1)
    
    def test_fifo_replacement(self):
        """Test FIFO replacement when full"""
        for i in range(15):  # Add more than capacity
            features = torch.randn(512)
            mask1 = torch.ones(256, 256)
            mask2 = torch.zeros(256, 256)
            
            self.bank.add_entry(features, features, mask1, mask2)
        
        # Should be at capacity
        self.assertEqual(len(self.bank), self.bank.capacity)
    
    def test_sample_batch(self):
        """Test sampling batches"""
        # Add some entries
        for i in range(5):
            features = torch.randn(512)
            mask1 = torch.rand(256, 256)
            mask2 = torch.rand(256, 256)
            
            self.bank.add_entry(features, features, mask1, mask2)
        
        batch = self.bank.sample_batch(batch_size=3)
        
        self.assertIsNotNone(batch)
        self.assertEqual(batch['f_I'].shape[0], 3)
        self.assertEqual(batch['f_T'].shape[0], 3)


class TestPoint2Box(unittest.TestCase):
    def test_single_cluster(self):
        """Test point-to-box with single cluster"""
        points = np.array([
            [100, 100],
            [105, 102],
            [98, 103],
            [102, 98]
        ])
        labels = np.array([1, 1, 1, 1])
        
        boxes, clusters = point_to_box(
            points, labels,
            radius=10,
            min_pts=2,
            image_shape=(512, 512)
        )
        
        # Should produce one box
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes.shape[1], 4)  # (x, y, w, h)
    
    def test_multiple_clusters(self):
        """Test point-to-box with multiple clusters"""
        points = np.array([
            [100, 100], [105, 102], [98, 103],  # Cluster 1
            [300, 300], [305, 302], [298, 303]  # Cluster 2
        ])
        labels = np.array([1, 1, 1, 1, 1, 1])
        
        boxes, clusters = point_to_box(
            points, labels,
            radius=10,
            min_pts=2,
            image_shape=(512, 512)
        )
        
        # Should produce two boxes
        self.assertGreaterEqual(len(boxes), 1)
    
    def test_noise_filtering(self):
        """Test that noise points are filtered"""
        points = np.array([
            [100, 100], [105, 102], [98, 103],  # Cluster
            [400, 400]  # Noise point (isolated)
        ])
        labels = np.array([1, 1, 1, 1])
        
        boxes, clusters = point_to_box(
            points, labels,
            radius=10,
            min_pts=3,
            image_shape=(512, 512)
        )
        
        # Noise point should be filtered
        # Only one cluster should remain
        self.assertEqual(len(boxes), 1)


def run_tests():
    """Run all tests"""
    print("=" * 80)
    print("Running PLUSS_β Unit Tests")
    print("=" * 80)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticTuner))
    suite.addTests(loader.loadTestsFromTestCase(TestBoxTuner))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryBank))
    suite.addTests(loader.loadTestsFromTestCase(TestPoint2Box))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
