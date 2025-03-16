#!/usr/bin/env python
# perception_module.py - Neural module for sensory perception with attention

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path

from LMM.core.base.component import NeuralComponent, ComponentConfig, ComponentState, DevelopmentalStage
from LMM.core.base.module import NeuralModule, ModuleConfig
from LMM.core.components.attention import (
    create_attention_component,
    AttentionConfig,
    SelfAttention,
    CrossAttention
)

class PerceptionConfig(ModuleConfig):
    """
    Configuration for the perception module.
    
    Attributes:
        input_dim (int): Dimension of the input features
        hidden_dim (int): Dimension of hidden layers
        output_dim (int): Dimension of the output features
        num_layers (int): Number of layers in feature extractor
        num_heads (int): Number of attention heads
        dropout_rate (float): Dropout rate
        use_positional_encoding (bool): Whether to use positional encoding for attention
    """
    def __init__(
        self,
        module_type: str = "perception_module",
        name: str = "PerceptionModule",
        description: str = "Perception module with attentional processing",
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        use_positional_encoding: bool = True,
        initial_stage: DevelopmentalStage = DevelopmentalStage.SENSORIMOTOR,
        **kwargs
    ):
        super().__init__(
            module_type=module_type,
            name=name,
            description=description,
            initial_stage=initial_stage,
            **kwargs
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_positional_encoding = use_positional_encoding


class FeatureExtractorComponent(NeuralComponent):
    """
    Neural component for extracting features from sensory input.
    """
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout_rate
    
    def build(self):
        """Build the feature extractor network."""
        layers = []
        current_dim = self.input_dim
        
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                next_dim = self.hidden_dim
            else:
                next_dim = self.output_dim
                
            layers.append(nn.Linear(current_dim, next_dim))
            
            if i < self.num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(next_dim))
                layers.append(nn.Dropout(self.dropout_rate))
            
            current_dim = next_dim
        
        self.layers = nn.ModuleList(layers)
        self.built = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            
        Returns:
            Processed tensor of shape [batch_size, seq_len, output_dim] or [batch_size, output_dim]
        """
        original_shape = x.shape
        if len(original_shape) == 3:
            # Handle sequential data
            batch_size, seq_len, _ = original_shape
            # Reshape to [batch_size * seq_len, input_dim] for batch norm
            x = x.reshape(-1, self.input_dim)
        
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                # Handle batch norm
                x = layer(x)
            else:
                x = layer(x)
        
        if len(original_shape) == 3:
            # Reshape back to [batch_size, seq_len, output_dim]
            x = x.reshape(batch_size, seq_len, self.output_dim)
            
        return x
    
    def adapt_to_developmental_stage(self, stage: DevelopmentalStage):
        """Adapt the component to the given developmental stage."""
        self.state.developmental_stage = stage
        
        # Adjust dropout rate based on stage
        if not self.built:
            return
            
        dropout_rates = {
            DevelopmentalStage.SENSORIMOTOR: 0.5,
            DevelopmentalStage.PREOPERATIONAL: 0.3,
            DevelopmentalStage.CONCRETE_OPERATIONAL: 0.2,
            DevelopmentalStage.FORMAL_OPERATIONAL: 0.1
        }
        
        current_dropout = dropout_rates[stage]
        
        for layer in self.layers:
            if isinstance(layer, nn.Dropout):
                layer.p = current_dropout


class PerceptionModule(NeuralModule):
    """
    Neural module for sensory perception with attention mechanisms.
    
    This module:
    1. Extracts features from sensory input
    2. Applies self-attention to model relationships within the input
    3. Can process memory context using cross-attention
    4. Produces perceptual features that can be used by higher-level modules
    
    The module adapts across developmental stages, with increasingly sophisticated
    attention mechanisms in later stages.
    """
    def __init__(self, config: PerceptionConfig):
        """
        Initialize the perception module.
        
        Args:
            config: Configuration for the perception module
        """
        super().__init__(config)
        self.config = config
        
        # Create components
        self.feature_extractor_config = ComponentConfig(
            component_type="feature_extractor",
            name="FeatureExtractor",
            description="Extracts features from sensory input",
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout_rate=config.dropout_rate,
            initial_stage=config.initial_stage
        )
        
        self.self_attention_config = AttentionConfig(
            component_type="self_attention",
            name="SelfAttention",
            description="Self-attention for input features",
            input_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout_attn=config.dropout_rate,
            use_positional_encoding=config.use_positional_encoding,
            initial_stage=config.initial_stage
        )
        
        self.cross_attention_config = AttentionConfig(
            component_type="cross_attention",
            name="CrossAttention",
            description="Cross-attention for memory integration",
            input_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout_attn=config.dropout_rate,
            initial_stage=config.initial_stage
        )
        
        self.output_projection_config = ComponentConfig(
            component_type="output_projection",
            name="OutputProjection",
            description="Projects attended features to output dimension",
            input_dim=config.hidden_dim,
            output_dim=config.output_dim,
            initial_stage=config.initial_stage
        )
    
    def build(self):
        """Build the perception module components."""
        # Create and build components
        self.feature_extractor = FeatureExtractorComponent(self.feature_extractor_config)
        self.feature_extractor.build()
        
        self.self_attention = create_attention_component(
            "self_attention",
            input_dim=self.config.hidden_dim,
            output_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            dropout_attn=self.config.dropout_rate,
            use_positional_encoding=self.config.use_positional_encoding,
            initial_stage=self.config.initial_stage
        )
        
        self.cross_attention = create_attention_component(
            "cross_attention",
            input_dim=self.config.hidden_dim,
            output_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            dropout_attn=self.config.dropout_rate,
            initial_stage=self.config.initial_stage
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(self.config.hidden_dim, self.config.output_dim)
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(self.config.hidden_dim)
        self.norm2 = nn.LayerNorm(self.config.hidden_dim)
        self.norm3 = nn.LayerNorm(self.config.hidden_dim)
        
        self.dropout = nn.Dropout(self.config.dropout_rate)
        
        self.built = True
    
    def forward(
        self, 
        x: torch.Tensor, 
        memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the perception module.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            memory: Optional memory tensor for cross-attention
                   of shape [batch_size, memory_len, hidden_dim]
        
        Returns:
            tuple:
                - output tensor of shape [batch_size, seq_len, output_dim]
                - dictionary of attention weights and intermediate activations
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply self-attention
        attended_features, self_attn_weights = self.self_attention(features)
        features = self.norm1(features + self.dropout(attended_features))
        
        # Apply cross-attention if memory is provided
        if memory is not None:
            attended_memory, cross_attn_weights = self.cross_attention(features, memory)
            features = self.norm2(features + self.dropout(attended_memory))
        else:
            cross_attn_weights = None
        
        # Apply output projection
        output = self.output_projection(features)
        output = self.norm3(output)
        
        # Collect attention weights and activations
        attention_info = {
            'self_attention': self_attn_weights,
            'cross_attention': cross_attn_weights,
            'features': features
        }
        
        return output, attention_info
    
    def adapt_to_developmental_stage(self, stage: DevelopmentalStage):
        """
        Adapt the module to the given developmental stage.
        
        Args:
            stage: The developmental stage to adapt to
        """
        self.state.developmental_stage = stage
        
        # Adapt components
        self.feature_extractor.adapt_to_developmental_stage(stage)
        self.self_attention.adapt_to_developmental_stage(stage)
        self.cross_attention.adapt_to_developmental_stage(stage)
        
        # Adjust dropout
        dropout_rates = {
            DevelopmentalStage.SENSORIMOTOR: 0.5,
            DevelopmentalStage.PREOPERATIONAL: 0.3,
            DevelopmentalStage.CONCRETE_OPERATIONAL: 0.2,
            DevelopmentalStage.FORMAL_OPERATIONAL: 0.1
        }
        
        self.dropout.p = dropout_rates[stage]
        
        # Modify behavior based on developmental stage
        if stage == DevelopmentalStage.SENSORIMOTOR:
            # In sensorimotor stage, perception is more focused on immediate sensory input
            # and less influenced by memory
            self._use_memory_in_forward = False
            
        elif stage == DevelopmentalStage.PREOPERATIONAL:
            # In preoperational stage, begin to integrate memory but with limited capacity
            self._use_memory_in_forward = True
            
        elif stage == DevelopmentalStage.CONCRETE_OPERATIONAL:
            # In concrete operational stage, better integration of memory 
            # and sensory information
            self._use_memory_in_forward = True
            
        elif stage == DevelopmentalStage.FORMAL_OPERATIONAL:
            # In formal operational stage, full integration of all components
            self._use_memory_in_forward = True
    
    def save(self, directory: str) -> None:
        """
        Save the module to the given directory.
        
        Args:
            directory: Directory to save the module to
        """
        directory_path = Path(directory)
        directory_path.mkdir(exist_ok=True, parents=True)
        
        # Save component states and weights
        self.feature_extractor.save(str(directory_path / "feature_extractor"))
        self.self_attention.save(str(directory_path / "self_attention"))
        self.cross_attention.save(str(directory_path / "cross_attention"))
        
        # Save the rest of the module
        torch.save({
            'output_projection': self.output_projection.state_dict(),
            'norm1': self.norm1.state_dict(),
            'norm2': self.norm2.state_dict(),
            'norm3': self.norm3.state_dict(),
            'state': self.state.dict(),
            'config': self.config.dict()
        }, directory_path / "module.pt")
    
    def load(self, directory: str) -> None:
        """
        Load the module from the given directory.
        
        Args:
            directory: Directory to load the module from
        """
        if not self.built:
            self.build()
            
        directory_path = Path(directory)
        
        # Load component states and weights
        self.feature_extractor.load(str(directory_path / "feature_extractor"))
        self.self_attention.load(str(directory_path / "self_attention"))
        self.cross_attention.load(str(directory_path / "cross_attention"))
        
        # Load the rest of the module
        checkpoint = torch.load(directory_path / "module.pt")
        self.output_projection.load_state_dict(checkpoint['output_projection'])
        self.norm1.load_state_dict(checkpoint['norm1'])
        self.norm2.load_state_dict(checkpoint['norm2'])
        self.norm3.load_state_dict(checkpoint['norm3'])
        self.state = ComponentState(**checkpoint['state'])
        
        # Load configuration (optional, as config should be passed during initialization)
        # self.config = PerceptionConfig(**checkpoint['config'])


def create_perception_module(
    input_dim: int = 128,
    hidden_dim: int = 256,
    output_dim: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout_rate: float = 0.1,
    use_positional_encoding: bool = True,
    initial_stage: DevelopmentalStage = DevelopmentalStage.SENSORIMOTOR
) -> PerceptionModule:
    """
    Factory function to create a perception module.
    
    Args:
        input_dim: Dimension of the input features
        hidden_dim: Dimension of hidden layers
        output_dim: Dimension of the output features
        num_layers: Number of layers in feature extractor
        num_heads: Number of attention heads
        dropout_rate: Dropout rate
        use_positional_encoding: Whether to use positional encoding
        initial_stage: Initial developmental stage
        
    Returns:
        Initialized and built perception module
    """
    config = PerceptionConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        use_positional_encoding=use_positional_encoding,
        initial_stage=initial_stage
    )
    
    module = PerceptionModule(config)
    module.build()
    
    return module 