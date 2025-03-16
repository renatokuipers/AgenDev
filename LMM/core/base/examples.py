#!/usr/bin/env python
# examples.py - Example implementations of neural modules and components
# Demonstrates how to use the base classes

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

from LMM.core.base.module import NeuralModule, ModuleConfig, DevelopmentalStage
from LMM.core.base.component import NeuralComponent, ComponentConfig

# Example Neural Component
class FeedForwardComponent(NeuralComponent):
    """
    Example implementation of a neural component with a feedforward network.
    
    This component demonstrates how to implement a concrete neural component
    based on the abstract base class. It implements a simple multi-layer
    perceptron with developmental adaptation.
    """
    
    def _build_network(self) -> None:
        """Build the feedforward neural network architecture"""
        layers = []
        
        # Input layer
        in_features = self.config.input_dim
        
        # Add hidden layers based on configuration
        if self.config.hidden_dims:
            for hidden_dim in self.config.hidden_dims:
                # Add linear layer
                layers.append(nn.Linear(in_features, hidden_dim))
                
                # Add batch normalization if specified
                if self.config.use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                
                # Add activation
                layers.append(self.get_activation_function())
                
                # Add dropout if rate > 0
                if self.config.dropout_rate > 0:
                    layers.append(nn.Dropout(self.config.dropout_rate))
                
                in_features = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_features, self.config.output_dim))
        
        # Create sequential model
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the feedforward network"""
        return self.net(x)
    
    def _adapt_to_developmental_stage(self, stage: DevelopmentalStage) -> None:
        """
        Adapt the component to a developmental stage.
        
        In this example, we adjust the dropout rate based on the developmental stage.
        - More dropout in earlier stages (more exploration)
        - Less dropout in later stages (more exploitation)
        """
        # Find dropout layers
        dropout_layers = [module for module in self.net.modules() if isinstance(module, nn.Dropout)]
        
        # Adjust dropout rate based on stage
        if stage == DevelopmentalStage.SENSORIMOTOR:
            new_rate = min(0.5, self.config.dropout_rate * 2)
        elif stage == DevelopmentalStage.PREOPERATIONAL:
            new_rate = self.config.dropout_rate
        elif stage == DevelopmentalStage.CONCRETE_OPERATIONAL:
            new_rate = max(0.1, self.config.dropout_rate / 2)
        else:  # FORMAL_OPERATIONAL
            new_rate = max(0.05, self.config.dropout_rate / 4)
        
        # Update dropout layers
        for layer in dropout_layers:
            layer.p = new_rate
        
        if self.debug_mode:
            print(f"Adapted dropout rate to {new_rate} for stage {stage.value}")

# Example Neural Module
class SimplePerceptionModule(NeuralModule):
    """
    Example implementation of a neural module with multiple components.
    
    This module demonstrates how to implement a concrete neural module
    based on the abstract base class. It simulates a simple perception
    module with feature extraction and recognition components.
    """
    
    def _build_network(self) -> None:
        """Build the neural network architecture for this module"""
        # Feature extraction component
        self.feature_extractor = FeedForwardComponent(
            ComponentConfig(
                component_type="feature_extractor",
                name="Feature Extractor",
                description="Extracts features from input data",
                input_dim=self.config.input_dim,
                output_dim=128,  # Feature dimension
                hidden_dims=[256, 192],
                activation="relu",
                dropout_rate=0.2,
                use_batch_norm=True,
                initial_stage=self.state.developmental_stage,
                device=self.config.device
            )
        )
        
        # Recognition component
        self.recognizer = FeedForwardComponent(
            ComponentConfig(
                component_type="recognizer",
                name="Recognizer",
                description="Recognizes patterns in extracted features",
                input_dim=128,  # Feature dimension
                output_dim=self.config.output_dim,
                hidden_dims=[64],
                activation="tanh",
                dropout_rate=0.1,
                use_batch_norm=False,
                initial_stage=self.state.developmental_stage,
                device=self.config.device
            )
        )
        
        # Developmental adaptation based on initial stage
        self._adapt_to_developmental_stage()
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass through the module"""
        # Extract features
        features = self.feature_extractor(x)
        
        # Generate recognition output
        output = self.recognizer(features)
        
        # If in debug mode, return both features and output
        if self.debug_mode:
            return output, features
        
        return output
    
    def _adapt_to_developmental_stage(self) -> None:
        """
        Adapt the module's architecture to its current developmental stage.
        
        This method adjusts the module's structure based on its developmental stage:
        - SENSORIMOTOR: Very simple processing with limited components
        - PREOPERATIONAL: Basic pattern recognition with more connections
        - CONCRETE_OPERATIONAL: More complex representations with refined connections
        - FORMAL_OPERATIONAL: Sophisticated processing with rich interconnections
        """
        stage = self.state.developmental_stage
        
        # Update components to match module's stage
        self.feature_extractor.adapt_to_developmental_stage(stage)
        self.recognizer.adapt_to_developmental_stage(stage)
        
        # Adjust architecture based on developmental stage
        if stage == DevelopmentalStage.SENSORIMOTOR:
            # Simplest form - just basic pattern detection
            self.use_attention = False
            self.use_skip_connections = False
            
        elif stage == DevelopmentalStage.PREOPERATIONAL:
            # Add simple attention mechanism
            self.use_attention = True
            self.use_skip_connections = False
            
        elif stage == DevelopmentalStage.CONCRETE_OPERATIONAL:
            # Add skip connections for more complex processing
            self.use_attention = True
            self.use_skip_connections = True
            
        elif stage == DevelopmentalStage.FORMAL_OPERATIONAL:
            # Full functionality
            self.use_attention = True
            self.use_skip_connections = True

# Usage examples
def example_create_component():
    """Example of creating and using a neural component"""
    # Create configuration
    config = ComponentConfig(
        component_type="feedforward",
        name="Example Feedforward",
        description="Example feedforward neural network component",
        input_dim=10,
        output_dim=5,
        hidden_dims=[20, 15],
        activation="relu",
        dropout_rate=0.2,
        use_batch_norm=True
    )
    
    # Create component
    component = FeedForwardComponent(config)
    
    # Create sample input
    x = torch.randn(32, 10)  # batch_size=32, input_dim=10
    
    # Forward pass
    output = component(x)
    
    print(f"Component: {component}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    return component

def example_create_module():
    """Example of creating and using a neural module"""
    # Create configuration
    config = ModuleConfig(
        module_type="visual_perception",
        name="Example Visual Perception",
        description="Example visual perception module",
        input_dim=784,  # e.g., flattened 28x28 image
        output_dim=10,  # e.g., 10 classes
        hidden_dims=[512, 256],
        initial_stage=DevelopmentalStage.SENSORIMOTOR
    )
    
    # Create module
    module = SimplePerceptionModule(config)
    
    # Create sample input
    x = torch.randn(32, 784)  # batch_size=32, input_dim=784
    
    # Forward pass
    output = module(x)
    
    print(f"Module: {module}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Advance developmental stage
    print(f"Current stage: {module.state.developmental_stage.value}")
    module.advance_developmental_stage()
    print(f"New stage: {module.state.developmental_stage.value}")
    
    # Test again with new stage
    output = module(x)
    print(f"Output shape after stage advancement: {output.shape}")
    
    return module

def example_save_load_module(module, tmp_dir="./tmp"):
    """Example of saving and loading a module"""
    import os
    import shutil
    
    # Create temporary directory
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Save module
    saved_path = module.save(tmp_dir)
    print(f"Module saved to: {saved_path}")
    
    # Load module
    loaded_module = SimplePerceptionModule.load(saved_path)
    print(f"Loaded module: {loaded_module}")
    
    # Compare original and loaded
    print(f"Original stage: {module.state.developmental_stage.value}")
    print(f"Loaded stage: {loaded_module.state.developmental_stage.value}")
    
    # Clean up
    shutil.rmtree(tmp_dir)
    
    return loaded_module

if __name__ == "__main__":
    # Create and test component
    component = example_create_component()
    
    # Create and test module
    module = example_create_module()
    
    # Test save and load
    loaded_module = example_save_load_module(module) 