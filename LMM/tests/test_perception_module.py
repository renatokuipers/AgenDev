#!/usr/bin/env python
# test_perception_module.py - Tests for the perception module

import os
import sys
import torch
import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LMM.core.base.component import DevelopmentalStage
from LMM.core.modules.perception_module import PerceptionModule, PerceptionConfig, create_perception_module

class TestPerceptionModule(unittest.TestCase):
    """Test suite for the perception module"""
    
    def setUp(self):
        """Set up test environment"""
        # Model parameters
        self.batch_size = 8
        self.seq_len = 16
        self.memory_len = 10
        self.input_dim = 64
        self.hidden_dim = 128
        self.output_dim = 32
        self.num_heads = 4
        
        # Create random test data
        self.input_data = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        self.memory_data = torch.randn(self.batch_size, self.memory_len, self.hidden_dim)
        
        # Create perception module
        self.perception = create_perception_module(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_heads=self.num_heads
        )
    
    def test_forward_pass(self):
        """Test forward pass through the perception module"""
        # Forward pass without memory
        output, attention_info = self.perception(self.input_data)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
        
        # Check attention weights
        self.assertEqual(attention_info['self_attention'].shape, 
                        (self.batch_size, self.seq_len, self.seq_len))
        self.assertIsNone(attention_info['cross_attention'])
        
        # Forward pass with memory
        output_with_memory, attention_info_with_memory = self.perception(
            self.input_data, self.memory_data)
        
        # Check output shape
        self.assertEqual(output_with_memory.shape, 
                        (self.batch_size, self.seq_len, self.output_dim))
        
        # Check attention weights
        self.assertEqual(attention_info_with_memory['cross_attention'].shape, 
                        (self.batch_size, self.seq_len, self.memory_len))
    
    def test_developmental_adaptation(self):
        """Test adaptation to different developmental stages"""
        stages = [
            DevelopmentalStage.SENSORIMOTOR,
            DevelopmentalStage.PREOPERATIONAL,
            DevelopmentalStage.CONCRETE_OPERATIONAL,
            DevelopmentalStage.FORMAL_OPERATIONAL
        ]
        
        # Store attention patterns for each stage
        self_attention_patterns = []
        cross_attention_patterns = []
        
        for stage in stages:
            # Adapt to stage
            self.perception.adapt_to_developmental_stage(stage)
            
            # Check if stage was updated
            self.assertEqual(self.perception.state.developmental_stage, stage)
            self.assertEqual(self.perception.feature_extractor.state.developmental_stage, stage)
            self.assertEqual(self.perception.self_attention.state.developmental_stage, stage)
            self.assertEqual(self.perception.cross_attention.state.developmental_stage, stage)
            
            # Process data
            _, attention_info = self.perception(self.input_data, self.memory_data)
            
            # Store sample attention pattern
            self_attention_patterns.append(
                attention_info['self_attention'][0, 0].detach().cpu().numpy())
            cross_attention_patterns.append(
                attention_info['cross_attention'][0, 0].detach().cpu().numpy())
        
        # Plot attention patterns
        self._plot_attention_patterns(stages, self_attention_patterns, cross_attention_patterns)
    
    def test_save_load(self):
        """Test saving and loading the perception module"""
        # Create a directory for saving
        save_dir = Path("./test_outputs/perception_module")
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Process input data and get original output
        original_output, _ = self.perception(self.input_data, self.memory_data)
        
        # Save the module
        self.perception.save(str(save_dir))
        
        # Create a new module with the same configuration
        new_perception = create_perception_module(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_heads=self.num_heads
        )
        
        # Load the saved module
        new_perception.load(str(save_dir))
        
        # Process input data with the loaded module
        loaded_output, _ = new_perception(self.input_data, self.memory_data)
        
        # Check that outputs are the same
        self.assertTrue(torch.allclose(original_output, loaded_output, atol=1e-5))
    
    def _plot_attention_patterns(self, stages, self_attention_patterns, cross_attention_patterns):
        """Helper method to visualize attention patterns across stages"""
        try:
            plt.figure(figsize=(15, 10))
            
            for i, (stage, self_attn, cross_attn) in enumerate(zip(
                stages, self_attention_patterns, cross_attention_patterns)):
                
                # Self-attention
                plt.subplot(2, len(stages), i + 1)
                plt.imshow(self_attn, cmap='viridis')
                plt.title(f"Self-Attention - {stage.value}")
                plt.colorbar()
                
                # Cross-attention
                plt.subplot(2, len(stages), i + 1 + len(stages))
                plt.imshow(cross_attn, cmap='viridis')
                plt.title(f"Cross-Attention - {stage.value}")
                plt.colorbar()
            
            plt.tight_layout()
            
            # Save the plot
            output_dir = Path("./test_outputs")
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / "perception_attention_patterns.png")
            plt.close()
        except Exception as e:
            print(f"Could not generate plot: {e}")


def run_perception_demo():
    """Standalone function to demonstrate the perception module"""
    print("Running Perception Module Demo")
    print("=" * 40)
    
    # Create synthetic sensory data
    batch_size = 1
    seq_len = 20
    memory_len = 15
    input_dim = 64
    hidden_dim = 128
    output_dim = 32
    
    # Create random patterns with some structure
    def create_pattern(n_samples, n_features):
        centers = np.random.randn(5, n_features)  # 5 cluster centers
        samples = []
        for _ in range(n_samples):
            # Select a random center and add noise
            center_idx = np.random.randint(0, 5)
            sample = centers[center_idx] + 0.1 * np.random.randn(n_features)
            samples.append(sample)
        return np.array(samples)
    
    # Create input sequence - simulate sequential observations
    input_sequence = create_pattern(seq_len, input_dim)
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)
    
    # Create memory data - simulate memory of past observations
    memory_sequence = create_pattern(memory_len, hidden_dim)
    memory_tensor = torch.tensor(memory_sequence, dtype=torch.float32).unsqueeze(0)
    
    # Create perception modules for each developmental stage
    stages = [
        DevelopmentalStage.SENSORIMOTOR,
        DevelopmentalStage.PREOPERATIONAL,
        DevelopmentalStage.CONCRETE_OPERATIONAL,
        DevelopmentalStage.FORMAL_OPERATIONAL
    ]
    
    perception_modules = {}
    for stage in stages:
        module = create_perception_module(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=4,
            dropout_rate=0.1,
            use_positional_encoding=True,
            initial_stage=stage
        )
        perception_modules[stage.value] = module
    
    # Store results for each stage
    results = {}
    for stage_name, module in perception_modules.items():
        # Process with memory
        output_with_memory, attn_info_with_memory = module(input_tensor, memory_tensor)
        
        # Process without memory
        output_without_memory, attn_info_without_memory = module(input_tensor)
        
        results[stage_name] = {
            'output_with_memory': output_with_memory.squeeze(0).detach().cpu().numpy(),
            'output_without_memory': output_without_memory.squeeze(0).detach().cpu().numpy(),
            'self_attention': attn_info_with_memory['self_attention'].squeeze(0).detach().cpu().numpy(),
            'cross_attention': attn_info_with_memory['cross_attention'].squeeze(0).detach().cpu().numpy()
        }
    
    # Calculate similarity matrices between outputs at different stages
    # This demonstrates how processing evolves with developmental stage
    stage_names = [stage.value for stage in stages]
    similarity_with_memory = np.zeros((len(stage_names), len(stage_names)))
    similarity_without_memory = np.zeros((len(stage_names), len(stage_names)))
    
    for i, stage1 in enumerate(stage_names):
        for j, stage2 in enumerate(stage_names):
            out1_with_mem = results[stage1]['output_with_memory'].flatten()
            out2_with_mem = results[stage2]['output_with_memory'].flatten()
            
            out1_without_mem = results[stage1]['output_without_memory'].flatten()
            out2_without_mem = results[stage2]['output_without_memory'].flatten()
            
            # Calculate cosine similarity
            sim_with_mem = np.dot(out1_with_mem, out2_with_mem) / (
                np.linalg.norm(out1_with_mem) * np.linalg.norm(out2_with_mem))
            
            sim_without_mem = np.dot(out1_without_mem, out2_without_mem) / (
                np.linalg.norm(out1_without_mem) * np.linalg.norm(out2_without_mem))
            
            similarity_with_memory[i, j] = sim_with_mem
            similarity_without_memory[i, j] = sim_without_mem
    
    # Visualize results
    output_dir = Path("./test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Plot attention patterns
    plt.figure(figsize=(20, 15))
    
    # Plot self-attention for each stage
    for i, stage in enumerate(stage_names):
        plt.subplot(3, len(stage_names), i + 1)
        plt.imshow(results[stage]['self_attention'], cmap='viridis')
        plt.title(f"Self-Attention - {stage}")
        plt.colorbar()
    
    # Plot cross-attention for each stage
    for i, stage in enumerate(stage_names):
        plt.subplot(3, len(stage_names), i + 1 + len(stage_names))
        plt.imshow(results[stage]['cross_attention'], cmap='viridis')
        plt.title(f"Cross-Attention - {stage}")
        plt.colorbar()
    
    # Plot output heatmaps
    for i, stage in enumerate(stage_names):
        plt.subplot(3, len(stage_names), i + 1 + 2*len(stage_names))
        plt.imshow(results[stage]['output_with_memory'], cmap='viridis', aspect='auto')
        plt.title(f"Output (with memory) - {stage}")
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(output_dir / "perception_developmental_comparison.png")
    
    # Plot similarity matrices
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(similarity_with_memory, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Output Similarity (with memory)")
    plt.xticks(range(len(stage_names)), stage_names, rotation=45)
    plt.yticks(range(len(stage_names)), stage_names)
    
    plt.subplot(1, 2, 2)
    plt.imshow(similarity_without_memory, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Output Similarity (without memory)")
    plt.xticks(range(len(stage_names)), stage_names, rotation=45)
    plt.yticks(range(len(stage_names)), stage_names)
    
    plt.tight_layout()
    plt.savefig(output_dir / "perception_stage_similarity.png")
    
    # Calculate difference between outputs with and without memory
    plt.figure(figsize=(15, 5))
    
    for i, stage in enumerate(stage_names):
        plt.subplot(1, len(stage_names), i + 1)
        diff = np.abs(results[stage]['output_with_memory'] - results[stage]['output_without_memory'])
        plt.imshow(diff, cmap='hot', aspect='auto')
        plt.colorbar()
        plt.title(f"Memory Influence - {stage}")
    
    plt.tight_layout()
    plt.savefig(output_dir / "perception_memory_influence.png")
    
    print(f"Visualizations saved to {output_dir}")
    
    # Return paths to generated files
    return [
        output_dir / "perception_developmental_comparison.png",
        output_dir / "perception_stage_similarity.png",
        output_dir / "perception_memory_influence.png"
    ]


if __name__ == "__main__":
    # Run tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Run demo
    output_files = run_perception_demo()
    print("Demo completed. Generated files:")
    for file in output_files:
        print(f"- {file}") 