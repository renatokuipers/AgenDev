#!/usr/bin/env python
# component.py - Neural Component Base Class
# Base class for smaller neural network components used within modules

import os
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import json
import time
import torch
import torch.nn as nn
from pydantic import BaseModel, Field, field_validator

from LMM.core.base.module import DevelopmentalStage

class ComponentConfig(BaseModel):
    """Base configuration for neural components"""
    component_id: str = Field(default_factory=lambda: str(uuid.uuid4()), 
        description="Unique identifier for the component")
    component_type: str = Field(..., 
        description="Type of neural component")
    name: str = Field(..., 
        description="Human-readable name for the component")
    description: str = Field("", 
        description="Description of the component's function")
    input_dim: int = Field(..., 
        description="Input dimension for the component")
    output_dim: int = Field(..., 
        description="Output dimension for the component")
    hidden_dims: List[int] = Field(default_factory=list, 
        description="Hidden layer dimensions")
    activation: str = Field("relu", 
        description="Activation function to use")
    dropout_rate: float = Field(0.0, ge=0.0, le=1.0, 
        description="Dropout rate")
    use_batch_norm: bool = Field(False, 
        description="Whether to use batch normalization")
    initial_stage: DevelopmentalStage = Field(DevelopmentalStage.SENSORIMOTOR, 
        description="Initial developmental stage")
    device: str = Field("cuda" if torch.cuda.is_available() else "cpu", 
        description="Device to run the component on")
    
    model_config = {"extra": "forbid"}
    
    @field_validator('activation')
    @classmethod
    def validate_activation(cls, v: str) -> str:
        """Validate that activation is one of the supported types"""
        valid_activations = ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'gelu', 'elu', 'selu', 'none']
        if v.lower() not in valid_activations:
            raise ValueError(f"Activation must be one of: {valid_activations}")
        return v.lower()
    
    @field_validator('dropout_rate')
    @classmethod
    def validate_dropout_rate(cls, v: float) -> float:
        """Validate that dropout_rate is between 0 and 1"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Dropout rate must be between 0.0 and 1.0")
        return v

class ComponentState(BaseModel):
    """Represents the current state of a neural component"""
    developmental_stage: DevelopmentalStage = Field(..., 
        description="Current developmental stage of the component")
    active: bool = Field(True, 
        description="Whether the component is currently active")
    trainable: bool = Field(True, 
        description="Whether the component's parameters can be updated")
    last_activation_time: float = Field(default_factory=time.time, 
        description="Timestamp of when the component was last active")
    
    model_config = {"extra": "forbid"}

class NeuralComponent(nn.Module, ABC):
    """
    Base class for smaller neural network components used within modules.
    
    Components are simpler than full modules and are used as building blocks
    within larger neural modules. They provide a standardized interface and
    include developmental adaptation.
    """
    
    def __init__(self, config: ComponentConfig):
        """
        Initialize a neural component with the given configuration.
        
        Args:
            config: Configuration for the component
        """
        super().__init__()
        self.config = config
        
        # Initialize state
        self.state = ComponentState(
            developmental_stage=config.initial_stage,
            active=True,
            trainable=True,
            last_activation_time=time.time()
        )
        
        # Device setup
        self.device = torch.device(config.device)
        
        # Component metadata
        self.creation_time = time.time()
        self.update_time = self.creation_time
        
        # Debug mode flag
        self.debug_mode = False
        
        # Initialize neural network architecture
        self._build_network()
        
        # Move to specified device
        self.to(self.device)
    
    @abstractmethod
    def _build_network(self) -> None:
        """
        Build the neural network architecture for this component.
        
        This method must be implemented by subclasses to create their
        specific neural network structures.
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through the component.
        
        This method must be implemented by subclasses to define how inputs
        are processed by the component.
        
        Args:
            x: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor
        """
        pass
    
    def update_state(self, **kwargs) -> None:
        """
        Update the component's state with the provided values.
        
        Args:
            **kwargs: State attributes to update
        """
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        
        self.state.last_activation_time = time.time()
        self.update_time = time.time()
    
    def adapt_to_developmental_stage(self, stage: DevelopmentalStage) -> None:
        """
        Adapt the component to the given developmental stage.
        
        Args:
            stage: Developmental stage to adapt to
        """
        if self.state.developmental_stage == stage:
            return  # Already at the target stage
        
        # Update state
        self.update_state(developmental_stage=stage)
        
        # Implement developmental adaptations
        self._adapt_to_developmental_stage(stage)
    
    @abstractmethod
    def _adapt_to_developmental_stage(self, stage: DevelopmentalStage) -> None:
        """
        Adapt the component's architecture to a developmental stage.
        
        This method should be implemented by subclasses to define how the
        neural architecture changes with developmental stage.
        
        Args:
            stage: Developmental stage to adapt to
        """
        pass
    
    def get_activation_function(self, name: str = None) -> nn.Module:
        """
        Get the specified activation function.
        
        Args:
            name: Name of the activation function (defaults to config value if None)
            
        Returns:
            Activation function module
        """
        name = name.lower() if name else self.config.activation.lower()
        
        if name == 'relu':
            return nn.ReLU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'elu':
            return nn.ELU()
        elif name == 'selu':
            return nn.SELU()
        elif name == 'none':
            return nn.Identity()
        else:
            return nn.ReLU()  # Default to ReLU if not recognized
    
    def save(self, directory: Union[str, Path]) -> str:
        """
        Save the component's state and weights to the specified directory.
        
        Args:
            directory: Directory where to save the component
            
        Returns:
            Path to the saved component directory
        """
        directory = Path(directory)
        os.makedirs(directory, exist_ok=True)
        
        # Create component-specific directory
        component_dir = directory / f"{self.config.component_type}_{self.config.component_id}"
        os.makedirs(component_dir, exist_ok=True)
        
        # Save configuration
        config_path = component_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.model_dump(), f, indent=2)
        
        # Save state
        state_path = component_dir / "state.json"
        with open(state_path, 'w') as f:
            json.dump(self.state.model_dump(), f, indent=2)
        
        # Save network weights
        weights_path = component_dir / "weights.pt"
        torch.save(self.state_dict(), weights_path)
        
        return str(component_dir)
    
    @classmethod
    def load(cls, directory: Union[str, Path]) -> 'NeuralComponent':
        """
        Load a component from the specified directory.
        
        Args:
            directory: Directory where the component is saved
            
        Returns:
            Loaded neural component
        """
        directory = Path(directory)
        
        # Load configuration
        config_path = directory / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = ComponentConfig(**config_dict)
        
        # Create component instance
        component = cls(config)
        
        # Load state
        state_path = directory / "state.json"
        with open(state_path, 'r') as f:
            state_dict = json.load(f)
        
        component.state = ComponentState(**state_dict)
        
        # Load network weights
        weights_path = directory / "weights.pt"
        component.load_state_dict(torch.load(weights_path))
        
        return component
    
    def enable_debug(self) -> None:
        """Enable debug mode for additional logging and visualization"""
        self.debug_mode = True
    
    def disable_debug(self) -> None:
        """Disable debug mode"""
        self.debug_mode = False
    
    def __repr__(self) -> str:
        """String representation of the component"""
        return (f"{self.__class__.__name__}("
                f"id={self.config.component_id}, "
                f"type={self.config.component_type}, "
                f"in={self.config.input_dim}, "
                f"out={self.config.output_dim})") 