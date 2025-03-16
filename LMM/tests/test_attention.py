#!/usr/bin/env python
# test_attention.py - Tests for the attention mechanism components

import os
import sys
import unittest
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LMM.core.base.component import ComponentConfig, DevelopmentalStage
from LMM.core.components.attention import (
    AttentionConfig, 
    DotProductAttention, 
    MultiHeadAttention,
    SelfAttention,
    CrossAttention,
    PositionalEncoding,
    create_attention_component
)

class TestAttentionComponents(unittest.TestCase):
    """Test suite for the attention components"""
    
    def setUp(self):
        """Set up test environment"""
        self.batch_size = 8
        self.seq_len = 16
        self.input_dim = 64
        self.output_dim = 64
        self.num_heads = 8
        
        # Create random inputs
        self.query = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        self.key = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        self.value = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
        # Create a simple attention mask
        self.mask = torch.ones(self.batch_size, self.seq_len, self.seq_len)
        for i in range(self.batch_size):
            # Randomly mask some positions
            random_len = torch.randint(1, self.seq_len, (1,)).item()
            self.mask[i, :, random_len:] = 0
    
    def test_attention_config(self):
        """Test the attention configuration class"""
        config = AttentionConfig(
            component_type="test_attention",
            name="Test Attention",
            description="Attention component for testing",
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_heads=self.num_heads,
            attention_type="multi_head",
            use_positional_encoding=True
        )
        
        # Test computed properties
        self.assertEqual(config.attention_dim, self.input_dim)
        
        # Test with explicit head_dim
        head_dim = 16
        config = AttentionConfig(
            component_type="test_attention",
            name="Test Attention",
            description="Attention component for testing",
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_heads=self.num_heads,
            head_dim=head_dim,
            attention_type="multi_head"
        )
        
        self.assertEqual(config.attention_dim, head_dim * self.num_heads)
    
    def test_dot_product_attention(self):
        """Test dot product attention component"""
        config = AttentionConfig(
            component_type="dot_product_attention",
            name="Dot Product Attention",
            description="Dot product attention for testing",
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )
        
        attention = DotProductAttention(config)
        
        # Test forward pass
        context, weights = attention(self.query, self.key, self.value)
        
        # Check output shape
        self.assertEqual(context.shape, (self.batch_size, self.seq_len, self.input_dim))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len))
        
        # Check if weights sum to 1 across attention dimension
        weight_sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5))
        
        # Test with mask
        context_masked, weights_masked = attention(self.query, self.key, self.value, self.mask)
        
        # Check that masked positions have zero attention weight
        for i in range(self.batch_size):
            for j in range(self.seq_len):
                masked_positions = (self.mask[i, j] == 0).nonzero().squeeze(-1)
                if masked_positions.numel() > 0:
                    self.assertTrue(torch.allclose(
                        weights_masked[i, j, masked_positions],
                        torch.zeros_like(weights_masked[i, j, masked_positions]),
                        atol=1e-5
                    ))
    
    def test_multi_head_attention(self):
        """Test multi-head attention component"""
        config = AttentionConfig(
            component_type="multi_head_attention",
            name="Multi-Head Attention",
            description="Multi-head attention for testing",
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_heads=self.num_heads
        )
        
        attention = MultiHeadAttention(config)
        
        # Test forward pass
        output, weights = attention(self.query, self.key, self.value)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len))
        
        # Check if weights sum to 1 across attention dimension
        weight_sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5))
        
        # Test self-attention case (no key/value)
        output_self, weights_self = attention(self.query)
        self.assertEqual(output_self.shape, (self.batch_size, self.seq_len, self.output_dim))
        
        # Test with mask
        output_masked, weights_masked = attention(self.query, self.key, self.value, self.mask)
        
        # Check that masked positions have zero attention weight
        for i in range(self.batch_size):
            for j in range(self.seq_len):
                masked_positions = (self.mask[i, j] == 0).nonzero().squeeze(-1)
                if masked_positions.numel() > 0:
                    self.assertTrue(torch.allclose(
                        weights_masked[i, j, masked_positions],
                        torch.zeros_like(weights_masked[i, j, masked_positions]),
                        atol=1e-5
                    ))
    
    def test_self_attention(self):
        """Test self-attention component"""
        config = AttentionConfig(
            component_type="self_attention",
            name="Self Attention",
            description="Self-attention for testing",
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_heads=self.num_heads
        )
        
        attention = SelfAttention(config)
        
        # Test forward pass
        output, weights = attention(self.query)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len))
        
        # Test with mask
        output_masked, weights_masked = attention(self.query, self.mask)
        
        # Check that masked positions have zero attention weight
        for i in range(self.batch_size):
            for j in range(self.seq_len):
                masked_positions = (self.mask[i, j] == 0).nonzero().squeeze(-1)
                if masked_positions.numel() > 0:
                    self.assertTrue(torch.allclose(
                        weights_masked[i, j, masked_positions],
                        torch.zeros_like(weights_masked[i, j, masked_positions]),
                        atol=1e-5
                    ))
    
    def test_cross_attention(self):
        """Test cross-attention component"""
        config = AttentionConfig(
            component_type="cross_attention",
            name="Cross Attention",
            description="Cross-attention for testing",
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_heads=self.num_heads
        )
        
        attention = CrossAttention(config)
        
        # Test forward pass
        output, weights = attention(self.query, self.key)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len))
        
        # Test with mask
        output_masked, weights_masked = attention(self.query, self.key, self.mask)
        
        # Check that masked positions have zero attention weight
        for i in range(self.batch_size):
            for j in range(self.seq_len):
                masked_positions = (self.mask[i, j] == 0).nonzero().squeeze(-1)
                if masked_positions.numel() > 0:
                    self.assertTrue(torch.allclose(
                        weights_masked[i, j, masked_positions],
                        torch.zeros_like(weights_masked[i, j, masked_positions]),
                        atol=1e-5
                    ))
    
    def test_positional_encoding(self):
        """Test positional encoding module"""
        d_model = self.input_dim
        max_len = 100
        pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Test forward pass
        x = torch.zeros(1, max_len, d_model)
        output = pos_encoding(x)
        
        # Check output shape
        self.assertEqual(output.shape, (1, max_len, d_model))
        
        # Check if positional encodings are different for different positions
        # Extract the positional encoding from the buffer
        pe = pos_encoding.pe.squeeze(1)
        
        # Check some positions
        for i in range(1, max_len):
            # Positions should have different encodings
            self.assertFalse(torch.allclose(pe[0], pe[i], atol=1e-5))
    
    def test_factory_function(self):
        """Test the attention component factory function"""
        for attn_type in ["dot_product", "multi_head", "self_attention", "cross_attention"]:
            # Create component using factory function
            component = create_attention_component(
                attn_type,
                self.input_dim,
                self.output_dim,
                num_heads=self.num_heads
            )
            
            # Check if component was created with correct type
            if attn_type == "dot_product":
                self.assertIsInstance(component, DotProductAttention)
            elif attn_type == "multi_head":
                self.assertIsInstance(component, MultiHeadAttention)
            elif attn_type == "self_attention":
                self.assertIsInstance(component, SelfAttention)
            elif attn_type == "cross_attention":
                self.assertIsInstance(component, CrossAttention)
    
    def test_developmental_adaptation(self):
        """Test adaptation to different developmental stages"""
        # Create components
        dot_product = create_attention_component(
            "dot_product", self.input_dim, self.output_dim)
        multi_head = create_attention_component(
            "multi_head", self.input_dim, self.output_dim, num_heads=self.num_heads)
        
        # Test initial stage (should be SENSORIMOTOR)
        self.assertEqual(dot_product.state.developmental_stage, DevelopmentalStage.SENSORIMOTOR)
        self.assertEqual(multi_head.state.developmental_stage, DevelopmentalStage.SENSORIMOTOR)
        
        # Collect attention patterns across stages
        stages = [
            DevelopmentalStage.SENSORIMOTOR,
            DevelopmentalStage.PREOPERATIONAL,
            DevelopmentalStage.CONCRETE_OPERATIONAL,
            DevelopmentalStage.FORMAL_OPERATIONAL
        ]
        
        dot_product_patterns = []
        multi_head_patterns = []
        
        for stage in stages:
            # Adapt to stage
            dot_product.adapt_to_developmental_stage(stage)
            multi_head.adapt_to_developmental_stage(stage)
            
            # Get attention patterns
            _, dp_weights = dot_product(self.query, self.key, self.value)
            _, mh_weights = multi_head(self.query, self.key, self.value)
            
            # Store sample pattern
            dot_product_patterns.append(dp_weights[0, 0].detach().cpu().numpy())
            multi_head_patterns.append(mh_weights[0, 0].detach().cpu().numpy())
            
            # Check if stage was updated
            self.assertEqual(dot_product.state.developmental_stage, stage)
            self.assertEqual(multi_head.state.developmental_stage, stage)
        
        # Plot attention patterns
        self._plot_attention_patterns(stages, dot_product_patterns, multi_head_patterns)
    
    def _plot_attention_patterns(self, stages, dot_product_patterns, multi_head_patterns):
        """Helper method to visualize attention patterns across stages"""
        try:
            plt.figure(figsize=(15, 10))
            
            for i, (stage, dp_pattern, mh_pattern) in enumerate(zip(
                stages, dot_product_patterns, multi_head_patterns)):
                
                # Dot product attention
                plt.subplot(2, len(stages), i + 1)
                plt.imshow(dp_pattern, cmap='viridis')
                plt.title(f"Dot Product - {stage.value}")
                plt.colorbar()
                
                # Multi-head attention
                plt.subplot(2, len(stages), i + 1 + len(stages))
                plt.imshow(mh_pattern, cmap='viridis')
                plt.title(f"Multi-Head - {stage.value}")
                plt.colorbar()
            
            plt.tight_layout()
            
            # Save the plot
            output_dir = Path("./outputs")
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / "attention_patterns.png")
            plt.close()
        except Exception as e:
            print(f"Could not generate plot: {e}")
    
    def test_causal_attention(self):
        """Test causal (masked) attention"""
        config = AttentionConfig(
            component_type="causal_attention",
            name="Causal Attention",
            description="Causal attention for testing",
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_heads=self.num_heads,
            causal=True
        )
        
        attention = MultiHeadAttention(config)
        
        # Test forward pass
        output, weights = attention(self.query)
        
        # Check if causal mask was applied (lower triangular)
        for i in range(self.seq_len):
            for j in range(i + 1, self.seq_len):
                # Future positions should have zero attention weight
                self.assertTrue(torch.allclose(
                    weights[0, i, j],
                    torch.zeros_like(weights[0, i, j]),
                    atol=1e-5
                ))

def visualize_attention_weights():
    """
    Standalone function to visualize attention weights across developmental stages
    """
    # Parameters
    batch_size = 1
    seq_len = 20
    input_dim = 64
    output_dim = 64
    num_heads = 8
    
    # Create inputs
    query = torch.randn(batch_size, seq_len, input_dim)
    
    # Create components
    stages = [
        DevelopmentalStage.SENSORIMOTOR,
        DevelopmentalStage.PREOPERATIONAL,
        DevelopmentalStage.CONCRETE_OPERATIONAL,
        DevelopmentalStage.FORMAL_OPERATIONAL
    ]
    
    # Create attentions for each stage
    attentions = {}
    
    for stage in stages:
        # Create and adapt dot product attention
        dp_config = AttentionConfig(
            component_type="dot_product_attention",
            name=f"Dot Product - {stage.value}",
            description=f"Dot product attention at {stage.value} stage",
            input_dim=input_dim,
            output_dim=output_dim,
            initial_stage=stage
        )
        
        # Create and adapt multi-head attention
        mh_config = AttentionConfig(
            component_type="multi_head_attention",
            name=f"Multi-Head - {stage.value}",
            description=f"Multi-head attention at {stage.value} stage",
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            initial_stage=stage
        )
        
        # Create attention components
        attentions[f"dp_{stage.value}"] = DotProductAttention(dp_config)
        attentions[f"mh_{stage.value}"] = MultiHeadAttention(mh_config)
    
    # Run all attentions on the same input
    results = {}
    for name, attn in attentions.items():
        if name.startswith("dp"):
            _, weights = attn(query, query, query)
        else:
            _, weights = attn(query)
        
        results[name] = weights[0, 0].detach().cpu().numpy()
    
    # Plot all results
    plt.figure(figsize=(15, 10))
    
    for i, stage in enumerate(stages):
        # Dot product attention
        plt.subplot(2, len(stages), i + 1)
        plt.imshow(results[f"dp_{stage.value}"], cmap='viridis')
        plt.title(f"Dot Product - {stage.value}")
        plt.colorbar()
        
        # Multi-head attention
        plt.subplot(2, len(stages), i + 1 + len(stages))
        plt.imshow(results[f"mh_{stage.value}"], cmap='viridis')
        plt.title(f"Multi-Head - {stage.value}")
        plt.colorbar()
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "developmental_attention_patterns.png")
    
    return output_dir / "developmental_attention_patterns.png"

if __name__ == "__main__":
    # Run tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Visualize attention patterns
    plot_path = visualize_attention_weights()
    print(f"Attention visualization saved to: {plot_path}") 