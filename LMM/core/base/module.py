#!/usr/bin/env python
# module.py - Core Neural Module Base Class
# Abstract base class for all neural modules in the LMM project

import os
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import json
import time
import torch
import torch.nn as nn
from pydantic import BaseModel, Field, field_validator

class DevelopmentalStage(str, Enum):
    """Developmental stages of the LMM system, inspired by Piaget's theory"""
    SENSORIMOTOR = "sensorimotor"  # Infancy
    PREOPERATIONAL = "preoperational"  # Early childhood
    CONCRETE_OPERATIONAL = "concrete_operational"  # Middle childhood 
    FORMAL_OPERATIONAL = "formal_operational"  # Adolescence and beyond

class ModuleState(BaseModel):
    """Represents the current state of a neural module"""
    developmental_stage: DevelopmentalStage = Field(..., 
        description="Current developmental stage of the module")
    plasticity: float = Field(..., ge=0.0, le=1.0, 
        description="Learning rate multiplier, high in early stages, lower in later stages")
    connection_density: float = Field(..., ge=0.0, le=1.0, 
        description="Proportion of possible connections that are active")
    experience_count: int = Field(0, ge=0, 
        description="Number of learning experiences processed by this module")
    last_activation_time: float = Field(default_factory=time.time, 
        description="Timestamp of when the module was last active")
    active: bool = Field(True, 
        description="Whether the module is currently active")
    
    model_config = {"extra": "forbid"}
    
    @field_validator('plasticity')
    @classmethod
    def validate_plasticity(cls, v: float) -> float:
        """Validate that plasticity is between 0 and 1"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Plasticity must be between 0.0 and 1.0")
        return v
    
    @field_validator('connection_density')
    @classmethod
    def validate_connection_density(cls, v: float) -> float:
        """Validate that connection_density is between 0 and 1"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Connection density must be between 0.0 and 1.0")
        return v

class ModuleConfig(BaseModel):
    """Base configuration for neural modules"""
    module_id: str = Field(default_factory=lambda: str(uuid.uuid4()), 
        description="Unique identifier for the module")
    module_type: str = Field(..., 
        description="Type of neural module (e.g., 'visual_perception', 'working_memory')")
    name: str = Field(..., 
        description="Human-readable name for the module")
    description: str = Field("", 
        description="Description of the module's function")
    input_dim: Optional[int] = Field(None, 
        description="Input dimension for the module")
    output_dim: Optional[int] = Field(None, 
        description="Output dimension for the module")
    hidden_dims: List[int] = Field(default_factory=list, 
        description="Hidden layer dimensions")
    initial_stage: DevelopmentalStage = Field(DevelopmentalStage.SENSORIMOTOR, 
        description="Initial developmental stage")
    initial_plasticity: float = Field(0.9, ge=0.0, le=1.0, 
        description="Initial plasticity (learning rate)")
    initial_connection_density: float = Field(0.2, ge=0.0, le=1.0, 
        description="Initial connection density")
    device: str = Field("cuda" if torch.cuda.is_available() else "cpu", 
        description="Device to run the module on")
    
    model_config = {"extra": "forbid"}
    
    @field_validator('module_type')
    @classmethod
    def validate_module_type(cls, v: str) -> str:
        """Validate that module type is one of the accepted types"""
        valid_types = [
            'brain_hub',
            'visual_perception', 'auditory_perception',
            'working_memory', 'episodic_memory', 'semantic_memory',
            'language_processing',
            'emotional_processing',
            'self_model', 'planning', 'social_cognition',
            'error_monitoring', 'temporal_processing',
            'creativity', 'unconscious_processing'
        ]
        if v not in valid_types:
            raise ValueError(f"Module type must be one of: {valid_types}")
        return v

class NeuralModule(nn.Module, ABC):
    """
    Abstract base class for all neural modules in the LMM system.
    
    This provides a standardized interface and functionality for all modules,
    including state management, developmental stages, and serialization.
    """
    
    def __init__(self, config: ModuleConfig):
        """
        Initialize a neural module with the given configuration.
        
        Args:
            config: Configuration for the module
        """
        super().__init__()
        self.config = config
        
        # Initialize state
        self.state = ModuleState(
            developmental_stage=config.initial_stage,
            plasticity=config.initial_plasticity,
            connection_density=config.initial_connection_density,
            experience_count=0,
            last_activation_time=time.time(),
            active=True
        )
        
        # Device setup
        self.device = torch.device(config.device)
        
        # Module metadata
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
        Build the neural network architecture for this module.
        
        This method must be implemented by subclasses to create their
        specific neural network structures.
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through the module.
        
        This method must be implemented by subclasses to define how inputs
        are processed by the module.
        
        Args:
            x: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor(s)
        """
        pass
    
    def update_state(self, **kwargs) -> None:
        """
        Update the module's state with the provided values.
        
        Args:
            **kwargs: State attributes to update
        """
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        
        self.state.last_activation_time = time.time()
        self.update_time = time.time()
    
    def advance_developmental_stage(self) -> None:
        """
        Advance to the next developmental stage.
        
        This adjusts plasticity and connection density according to developmental principles.
        """
        current_stage = self.state.developmental_stage
        
        # Define stage transitions
        if current_stage == DevelopmentalStage.SENSORIMOTOR:
            next_stage = DevelopmentalStage.PREOPERATIONAL
            new_plasticity = 0.7  # Reduced but still high
            new_density = 0.4  # Increased connections
            
        elif current_stage == DevelopmentalStage.PREOPERATIONAL:
            next_stage = DevelopmentalStage.CONCRETE_OPERATIONAL
            new_plasticity = 0.5  # Moderate plasticity
            new_density = 0.6  # More connections
            
        elif current_stage == DevelopmentalStage.CONCRETE_OPERATIONAL:
            next_stage = DevelopmentalStage.FORMAL_OPERATIONAL
            new_plasticity = 0.3  # Lower plasticity
            new_density = 0.8  # Dense connections
            
        else:  # Already at highest stage
            return
        
        # Update state with new stage parameters
        self.update_state(
            developmental_stage=next_stage,
            plasticity=new_plasticity,
            connection_density=new_density
        )
        
        # Implement architectural changes appropriate for the new stage
        self._adapt_to_developmental_stage()
    
    @abstractmethod
    def _adapt_to_developmental_stage(self) -> None:
        """
        Adapt the module's architecture to its current developmental stage.
        
        This method should be implemented by subclasses to define how the
        neural architecture changes with developmental stage.
        """
        pass
    
    def process_experience(self, *args, **kwargs) -> None:
        """
        Process a learning experience.
        
        This increments the experience counter and may trigger developmental transitions.
        
        Args:
            *args: Experience-related positional arguments
            **kwargs: Experience-related keyword arguments
        """
        # Increment experience counter
        self.state.experience_count += 1
        
        # Check if ready to advance to next developmental stage
        # This is a simple example; real implementation would use more sophisticated criteria
        if (self.state.developmental_stage == DevelopmentalStage.SENSORIMOTOR and 
                self.state.experience_count >= 1000):
            self.advance_developmental_stage()
            
        elif (self.state.developmental_stage == DevelopmentalStage.PREOPERATIONAL and 
                self.state.experience_count >= 5000):
            self.advance_developmental_stage()
            
        elif (self.state.developmental_stage == DevelopmentalStage.CONCRETE_OPERATIONAL and 
                self.state.experience_count >= 20000):
            self.advance_developmental_stage()
    
    def save(self, directory: Union[str, Path]) -> str:
        """
        Save the module's state and weights to the specified directory.
        
        Args:
            directory: Directory where to save the module
            
        Returns:
            Path to the saved module directory
        """
        directory = Path(directory)
        os.makedirs(directory, exist_ok=True)
        
        # Create module-specific directory
        module_dir = directory / f"{self.config.module_type}_{self.config.module_id}"
        os.makedirs(module_dir, exist_ok=True)
        
        # Save configuration
        config_path = module_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.model_dump(), f, indent=2)
        
        # Save state
        state_path = module_dir / "state.json"
        with open(state_path, 'w') as f:
            json.dump(self.state.model_dump(), f, indent=2)
        
        # Save network weights
        weights_path = module_dir / "weights.pt"
        torch.save(self.state_dict(), weights_path)
        
        # Save module metadata
        metadata = {
            "creation_time": self.creation_time,
            "update_time": self.update_time,
            "saved_time": time.time()
        }
        metadata_path = module_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(module_dir)
    
    @classmethod
    def load(cls, directory: Union[str, Path]) -> 'NeuralModule':
        """
        Load a module from the specified directory.
        
        Args:
            directory: Directory where the module is saved
            
        Returns:
            Loaded neural module
        """
        directory = Path(directory)
        
        # Load configuration
        config_path = directory / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = ModuleConfig(**config_dict)
        
        # Create module instance
        module = cls(config)
        
        # Load state
        state_path = directory / "state.json"
        with open(state_path, 'r') as f:
            state_dict = json.load(f)
        
        module.state = ModuleState(**state_dict)
        
        # Load network weights
        weights_path = directory / "weights.pt"
        module.load_state_dict(torch.load(weights_path))
        
        # Load module metadata
        metadata_path = directory / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        module.creation_time = metadata.get("creation_time", time.time())
        module.update_time = metadata.get("update_time", time.time())
        
        return module
    
    def to_json(self) -> str:
        """
        Convert the module's configuration and state to a JSON string.
        
        Returns:
            JSON string representation of the module
        """
        data = {
            "config": self.config.model_dump(),
            "state": self.state.model_dump(),
            "metadata": {
                "creation_time": self.creation_time,
                "update_time": self.update_time,
                "module_type": self.config.module_type,
                "module_id": self.config.module_id
            }
        }
        return json.dumps(data, indent=2)
    
    def enable_debug(self) -> None:
        """Enable debug mode for additional logging and visualization"""
        self.debug_mode = True
    
    def disable_debug(self) -> None:
        """Disable debug mode"""
        self.debug_mode = False
        
    def __repr__(self) -> str:
        """String representation of the module"""
        return (f"{self.__class__.__name__}("
                f"id={self.config.module_id}, "
                f"type={self.config.module_type}, "
                f"stage={self.state.developmental_stage.value}, "
                f"exp={self.state.experience_count})") 