# Attention Components in LMM

## Overview

Attention mechanisms are a critical part of the Large Mind Model (LMM) architecture, enabling the system to focus on relevant information while filtering out noise. The attention components in LMM implement various attention mechanisms that can be used across different neural modules such as perception, working memory, and executive functions.

These attention components follow the developmental adaptation principles of the LMM system, becoming more sophisticated as the system progresses through developmental stages.

## Core Attention Mechanisms

### DotProductAttention

The simplest form of attention that computes compatibility between queries and keys using dot product operations.

```python
attention = create_attention_component(
    "dot_product", 
    input_dim=64, 
    output_dim=64
)
context, weights = attention(query, key, value)
```

In early developmental stages, this attention mechanism has simple focusing capabilities, but evolves to develop:
- More precise attention patterns
- Better distinction between relevant and irrelevant information
- Reduced sensitivity to noise

### MultiHeadAttention

A more sophisticated attention mechanism that allows the model to jointly attend to information from different representation subspaces.

```python
attention = create_attention_component(
    "multi_head", 
    input_dim=64, 
    output_dim=64, 
    num_heads=8
)
context, weights = attention(query, key, value)
```

Multi-head attention provides several benefits:
- Ability to focus on different parts of the input simultaneously
- Better representation of complex relationships
- Efficient parallel computation

### SelfAttention

A specialized form of attention where the input attends to itself, identifying relationships between different positions within the same sequence.

```python
attention = create_attention_component(
    "self_attention", 
    input_dim=64, 
    output_dim=64, 
    num_heads=8
)
context, weights = attention(input_sequence)
```

### CrossAttention

Attention mechanism that allows one sequence to attend to another sequence, useful for integrating information from different sources.

```python
attention = create_attention_component(
    "cross_attention", 
    input_dim=64, 
    output_dim=64, 
    num_heads=8
)
context, weights = attention(query_sequence, key_value_sequence)
```

## Developmental Adaptation

Attention components in LMM adapt across developmental stages:

1. **Sensorimotor Stage**:
   - Basic attention patterns
   - Higher dropout rates to encourage exploration
   - Simple attention mechanisms with limited heads
   - Focus primarily on the most salient features

2. **Preoperational Stage**:
   - More structured attention patterns
   - Begin to develop the ability to focus on relevant details
   - Reduced dropout for more stable attention
   - Introduction of positional awareness

3. **Concrete Operational Stage**:
   - Refined attention mechanisms with clearer focus
   - Lower dropout rates for consistent processing
   - Enhanced ability to discriminate between similar inputs
   - Better retention of contextual information

4. **Formal Operational Stage**:
   - Sophisticated attention with precise focus
   - Minimal dropout for reliable performance
   - Full multi-head capabilities
   - Advanced contextual understanding
   - Ability to handle abstract relationships

## Configuration Options

Attention components are configured using the `AttentionConfig` class:

```python
config = AttentionConfig(
    component_type="multi_head_attention",
    name="Working Memory Attention",
    description="Attention for working memory module",
    input_dim=128,
    output_dim=128,
    num_heads=8,
    head_dim=16,  # Optional, computed from input_dim/num_heads if not provided
    dropout_attn=0.1,
    attention_type="multi_head",
    use_positional_encoding=True,
    max_sequence_length=512,
    causal=False,  # Set to True for causal attention (future masking)
    initial_stage=DevelopmentalStage.SENSORIMOTOR
)
```

## Positional Encoding

For sequence-based attention, positional encoding can be enabled to preserve order information:

```python
config = AttentionConfig(
    # Other parameters...
    use_positional_encoding=True,
    max_sequence_length=512
)
```

The default implementation uses sinusoidal positional encoding, which encodes position information using sine and cosine functions of different frequencies.

## Causal Attention

For autoregressive models or sequences where future information should be masked, causal attention can be enabled:

```python
config = AttentionConfig(
    # Other parameters...
    causal=True
)
```

This creates a triangular mask that prevents positions from attending to subsequent positions.

## Factory Function

For convenience, a factory function is provided to create attention components:

```python
# Create dot product attention
dot_product = create_attention_component(
    "dot_product", 
    input_dim=64, 
    output_dim=64
)

# Create multi-head attention
multi_head = create_attention_component(
    "multi_head", 
    input_dim=64, 
    output_dim=64,
    num_heads=8
)

# Create self-attention
self_attn = create_attention_component(
    "self_attention", 
    input_dim=64, 
    output_dim=64,
    num_heads=8
)

# Create cross-attention
cross_attn = create_attention_component(
    "cross_attention", 
    input_dim=64, 
    output_dim=64,
    num_heads=8
)
```

## Integration with Neural Modules

Attention components can be integrated into larger neural modules:

```python
class PerceptionModule(NeuralModule):
    def __init__(self, config):
        super().__init__(config)
        
        # Create an attention component
        self.attention = create_attention_component(
            "self_attention", 
            input_dim=config.feature_dim, 
            output_dim=config.feature_dim,
            num_heads=config.num_heads
        )
        
        # Other layers...
        
    def build(self):
        # Build the attention component
        self.attention.build()
        
        # Build other components...
        
    def forward(self, x):
        # Apply attention
        attended_features, attention_weights = self.attention(x)
        
        # Process further...
        return output
        
    def adapt_to_developmental_stage(self, stage):
        # Adapt attention component
        self.attention.adapt_to_developmental_stage(stage)
        
        # Adapt other components...
```

## Visualizing Attention Patterns

Attention weights can be visualized to understand what the model is focusing on:

```python
# Get attention weights
_, attention_weights = attention(query, key, value)

# Visualize (example for a single query)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.imshow(attention_weights[0, 0].detach().cpu().numpy(), cmap='viridis')
plt.colorbar()
plt.title("Attention Weights")
plt.xlabel("Key Position")
plt.ylabel("Query Position")
plt.tight_layout()
plt.savefig("attention_pattern.png")
```

## Testing

A comprehensive test suite is available in `LMM/tests/test_attention.py` that demonstrates:
- Basic functionality of all attention types
- Developmental adaptation across stages
- Visualization of attention patterns
- Masked and causal attention

Run the tests with:

```bash
python -m LMM.tests.test_attention
```

This will also generate visualizations of attention patterns across developmental stages in the `outputs` directory.

## Best Practices

1. **Choose the right attention type** for your module's needs:
   - Use `DotProductAttention` for simple relationships
   - Use `MultiHeadAttention` for complex relationships
   - Use `SelfAttention` for intra-sequence relationships
   - Use `CrossAttention` for inter-sequence relationships

2. **Scale attention dimensions appropriately**:
   - For `MultiHeadAttention`, ensure that `input_dim` is divisible by `num_heads`
   - Or explicitly specify `head_dim` to control the dimension of each head

3. **Consider developmental stages**:
   - Test your module across all developmental stages
   - Ensure that attention patterns evolve meaningfully
   - Use higher dropout in early stages and lower dropout in later stages

4. **Use positional encoding** for sequence data where order matters

5. **Enable causal masking** for autoregressive models or when future information should be masked

6. **Visualize attention patterns** to understand what your model is focusing on and debug issues 