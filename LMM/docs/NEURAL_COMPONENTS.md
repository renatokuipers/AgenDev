# Neural Components in the LMM Project

## Overview

Neural Components are the foundational building blocks of the Large Mind Model (LMM) architecture. They are designed to be modular, reusable, and adaptable units that can be combined to create complex neural networks that mimic aspects of cognitive development.

The component system is inspired by developmental psychology theories, particularly those of Jean Piaget, and implements a developmental stage-based approach to neural network architecture and learning.

## Core Concepts

### 1. Neural Components

Neural Components are the smallest functional units in the LMM architecture. Each component:

- Has a specific function (e.g., feature extraction, pattern recognition)
- Can be trained independently or as part of a larger network
- Adapts its behavior based on developmental stages
- Can be combined with other components to form more complex modules

### 2. Developmental Stages

All components support four developmental stages inspired by Piaget's theory:

1. **Sensorimotor Stage**: The most basic form of processing, focused on simple pattern detection with high regularization
2. **Preoperational Stage**: More complex but still somewhat rigid processing with evolving representations
3. **Concrete Operational Stage**: Advanced processing with refined connections and lower regularization
4. **Formal Operational Stage**: The most sophisticated processing capabilities with minimal regularization

Components can adapt their architecture, hyperparameters, and learning strategies based on their current developmental stage.

## Using the Component System

### Creating a New Component

To create a neural component, you need to:

1. Subclass the `NeuralComponent` base class
2. Implement the required methods
3. Define how the component adapts to different developmental stages

Here's a basic example:

```python
from LMM.core.base.component import NeuralComponent, ComponentConfig, DevelopmentalStage

class MyComponent(NeuralComponent):
    def _build_network(self) -> None:
        """Build the neural network architecture"""
        # Define your network architecture here
        self.layers = nn.Sequential(
            nn.Linear(self.config.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.config.output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the component"""
        return self.layers(x)
    
    def _adapt_to_developmental_stage(self, stage: DevelopmentalStage) -> None:
        """Adapt component behavior based on developmental stage"""
        # Implement stage-specific adaptations here
        pass
```

### Component Configuration

When creating a component, you must provide a `ComponentConfig` that defines its properties:

```python
config = ComponentConfig(
    component_id="my_component_1",
    component_type="custom_type",
    name="My Custom Component",
    description="This component does something specific",
    input_dim=10,
    output_dim=5,
    hidden_dims=[20, 15],  # Optional
    activation="relu",
    dropout_rate=0.2,
    initial_stage=DevelopmentalStage.SENSORIMOTOR,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

component = MyComponent(config)
```

### Using Components in Forward Pass

Once created, a component can be used like any PyTorch module:

```python
# Create input tensor
x = torch.randn(32, 10)  # batch_size=32, input_dim=10

# Forward pass
output = component(x)
```

### Developmental Adaptation

Components can transition between developmental stages:

```python
# Advance to the next developmental stage
component.advance_developmental_stage()

# Or set a specific stage
component.adapt_to_developmental_stage(DevelopmentalStage.CONCRETE_OPERATIONAL)
```

### Saving and Loading Components

Components can be saved and loaded with their state and configuration:

```python
# Save component
saved_path = component.save("./model_storage")

# Load component
loaded_component = MyComponent.load(saved_path)
```

## Best Practices

1. **Modular Design**: Design components to perform a single, well-defined function
2. **Developmental Adaptation**: Implement meaningful adaptations for each developmental stage
3. **Configuration**: Use the configuration system to make components flexible and reusable
4. **Testing**: Use the provided test utilities to verify component behavior across stages
5. **Documentation**: Document the purpose and behavior of each component thoroughly

## Example Implementation

See `LMM/core/base/examples.py` for example implementations of neural components and modules.

## Testing Components

The `LMM/tests/test_neural_component.py` file provides a framework for testing components through developmental stages using a classification task. It demonstrates:

1. Creating a test component
2. Training it through all developmental stages
3. Visualizing how its behavior changes with each stage
4. Measuring performance metrics across stages
5. Saving and loading the trained component

To run the test:

```bash
python -m LMM.tests.test_neural_component
```

## Advanced Features

### Component State

Each component has a `ComponentState` that tracks:

- Current developmental stage
- Active/inactive status
- Trainable status 
- Custom state variables

### Custom Parameters

You can add custom parameters to the configuration by extending the `ComponentConfig` class:

```python
from LMM.core.base.component import ComponentConfig

class MyCustomConfig(ComponentConfig):
    custom_param: float = 0.5
    special_feature: bool = True
```

### Debugging Support

Components have a built-in debug mode:

```python
component.debug_mode = True
```

When enabled, components will print detailed information about their operations and state changes.

## Integration with Neural Modules

Components are designed to be integrated into larger Neural Modules. A module can contain multiple components and coordinate their interactions. See the `NeuralModule` class documentation for more details. 