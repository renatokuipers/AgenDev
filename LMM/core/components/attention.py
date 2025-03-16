#!/usr/bin/env python
# attention.py - Attention mechanism components for the LMM
# These components can be used within various modules requiring attention capabilities

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, Dict, Any, List

from LMM.core.base.component import NeuralComponent, ComponentConfig, ComponentState, DevelopmentalStage

class AttentionConfig(ComponentConfig):
    """Extended configuration for attention components"""
    num_heads: int = 1
    head_dim: Optional[int] = None
    dropout_attn: float = 0.0
    attention_type: str = "dot_product"  # dot_product, general, additive, self_attention, multi_head
    use_positional_encoding: bool = False
    max_sequence_length: int = 512
    causal: bool = False  # For causal/masked attention (can't see future tokens)
    
    @property
    def attention_dim(self) -> int:
        """Calculated attention dimension"""
        return self.head_dim * self.num_heads if self.head_dim else self.input_dim

class DotProductAttention(NeuralComponent):
    """
    Basic dot-product attention mechanism.
    
    Computes attention weights using dot product between query and key vectors,
    then uses these weights to compute a weighted sum of value vectors.
    """
    
    def _build_network(self) -> None:
        """Build the attention network architecture"""
        config = self.config
        
        # Ensure we have AttentionConfig
        if not isinstance(config, AttentionConfig):
            raise TypeError("DotProductAttention requires AttentionConfig")
            
        # Calculate dimensions
        self.attn_dim = config.attention_dim
        self.scale = 1.0 / math.sqrt(self.attn_dim)  # Scaling factor for numerical stability
        
        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(config.dropout_attn)
        
        # Positional encoding if specified
        if config.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(
                config.attention_dim, 
                max_len=config.max_sequence_length
            )
        else:
            self.pos_encoding = None
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for dot product attention
        
        Args:
            query: Query vectors [batch_size, query_len, dim]
            key: Key vectors [batch_size, key_len, dim]
            value: Value vectors [batch_size, key_len, dim]
            mask: Optional mask for attention [batch_size, query_len, key_len]
                  or [batch_size, 1, key_len]
                  
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # Apply positional encoding if needed
        if self.pos_encoding is not None:
            query = self.pos_encoding(query)
            key = self.pos_encoding(key)
            value = self.pos_encoding(value)
        
        # Calculate attention scores
        # [batch_size, query_len, key_len]
        scores = torch.bmm(query, key.transpose(1, 2)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # If specified as causal attention
        if self.config.causal and mask is None:
            # Create a causal mask
            query_len, key_len = query.size(1), key.size(1)
            causal_mask = torch.triu(
                torch.ones(query_len, key_len, device=query.device), 
                diagonal=1
            ).bool()
            # Expand to batch size
            causal_mask = causal_mask.unsqueeze(0).expand(query.size(0), -1, -1)
            # Apply causal mask
            scores = scores.masked_fill(causal_mask, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.bmm(attn_weights, value)
        
        return context, attn_weights
    
    def _adapt_to_developmental_stage(self, stage: DevelopmentalStage) -> None:
        """
        Adapt the attention component based on developmental stage.
        
        In earlier stages, attention is more diffuse and less selective.
        As development progresses, attention becomes more focused and efficient.
        """
        # Adjust dropout based on developmental stage
        if stage == DevelopmentalStage.SENSORIMOTOR:
            # High dropout in early stages - more random exploration
            self.attn_dropout.p = min(0.5, self.config.dropout_attn * 2)
            self.scale = 0.5 / math.sqrt(self.attn_dim)  # Less sharp attention distribution
            
        elif stage == DevelopmentalStage.PREOPERATIONAL:
            # Moderate dropout - starting to focus better
            self.attn_dropout.p = self.config.dropout_attn
            self.scale = 0.8 / math.sqrt(self.attn_dim)
            
        elif stage == DevelopmentalStage.CONCRETE_OPERATIONAL:
            # Lower dropout - more consistent attention
            self.attn_dropout.p = max(0.1, self.config.dropout_attn / 2)
            self.scale = 1.0 / math.sqrt(self.attn_dim)
            
        else:  # FORMAL_OPERATIONAL
            # Minimal dropout - highly selective attention
            self.attn_dropout.p = max(0.05, self.config.dropout_attn / 4)
            self.scale = 1.2 / math.sqrt(self.attn_dim)  # Sharper attention distribution

class MultiHeadAttention(NeuralComponent):
    """
    Multi-head attention component.
    
    Splits input into multiple heads to allow the model to jointly attend to
    information from different representation subspaces at different positions.
    """
    
    def _build_network(self) -> None:
        """Build the multi-head attention network architecture"""
        config = self.config
        
        # Ensure we have AttentionConfig
        if not isinstance(config, AttentionConfig):
            raise TypeError("MultiHeadAttention requires AttentionConfig")
            
        # Calculate dimensions
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim if config.head_dim else config.input_dim // config.num_heads
        self.attn_dim = self.head_dim * self.num_heads
        
        # Check if dimensions are compatible
        if self.head_dim * self.num_heads != config.input_dim:
            raise ValueError(
                f"Input dimension ({config.input_dim}) must be divisible by num_heads ({config.num_heads})"
            )
        
        # Projection layers for query, key, value
        self.q_proj = nn.Linear(config.input_dim, self.attn_dim)
        self.k_proj = nn.Linear(config.input_dim, self.attn_dim)
        self.v_proj = nn.Linear(config.input_dim, self.attn_dim)
        self.out_proj = nn.Linear(self.attn_dim, config.output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        self.attn_dropout = nn.Dropout(config.dropout_attn)
        
        # Scaling factor for attention
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Positional encoding if specified
        if config.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(
                config.input_dim, 
                max_len=config.max_sequence_length
            )
        else:
            self.pos_encoding = None
    
    def forward(self, 
                query: torch.Tensor, 
                key: Optional[torch.Tensor] = None, 
                value: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention
        
        Args:
            query: Query vectors [batch_size, query_len, dim]
            key: Key vectors [batch_size, key_len, dim], if None, use query
            value: Value vectors [batch_size, key_len, dim], if None, use key
            mask: Optional mask [batch_size, query_len, key_len] or [batch_size, 1, key_len]
                  
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        
        # For self-attention, use query for all inputs if key/value not provided
        if key is None:
            key = query
        if value is None:
            value = key
            
        key_len = key.size(1)
        value_len = value.size(1)
        
        # Apply positional encoding if needed
        if self.pos_encoding is not None:
            query = self.pos_encoding(query)
            key = self.pos_encoding(key)
            value = self.pos_encoding(value)
        
        # Linear projections
        q = self.q_proj(query)  # [batch_size, query_len, attn_dim]
        k = self.k_proj(key)    # [batch_size, key_len, attn_dim]
        v = self.v_proj(value)  # [batch_size, value_len, attn_dim]
        
        # Reshape for multi-head attention
        # [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, value_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate scaled dot-product attention
        # [batch_size, num_heads, query_len, key_len]
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to account for multiple heads
            if mask.dim() == 3:  # [batch_size, query_len, key_len]
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # If specified as causal attention
        if self.config.causal and mask is None:
            # Create a causal mask
            causal_mask = torch.triu(
                torch.ones(query_len, key_len, device=query.device), 
                diagonal=1
            ).bool()
            # Expand to batch size and num_heads
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
            # Apply causal mask
            scores = scores.masked_fill(causal_mask, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention weights to values
        # [batch_size, num_heads, query_len, head_dim]
        context = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        # [batch_size, query_len, attn_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.attn_dim)
        
        # Final linear projection
        output = self.out_proj(context)
        output = self.dropout(output)
        
        # Average attention weights across heads for return value
        avg_attn_weights = attn_weights.mean(dim=1)
        
        return output, avg_attn_weights
    
    def _adapt_to_developmental_stage(self, stage: DevelopmentalStage) -> None:
        """
        Adapt the multi-head attention component based on developmental stage.
        
        Different stages show different attention capabilities:
        - Earlier stages use fewer effective heads (masking some)
        - Later stages use the full power of multi-head attention
        """
        # Adjust attention parameters based on stage
        if stage == DevelopmentalStage.SENSORIMOTOR:
            # Early stage: use only a fraction of heads (the first 1 or 2)
            used_heads = max(1, self.num_heads // 4)
            self._mask_unused_heads(used_heads)
            self.attn_dropout.p = min(0.5, self.config.dropout_attn * 2)
            self.scale = 0.5 / math.sqrt(self.head_dim)
            
        elif stage == DevelopmentalStage.PREOPERATIONAL:
            # More heads become available
            used_heads = max(1, self.num_heads // 2)
            self._mask_unused_heads(used_heads)
            self.attn_dropout.p = self.config.dropout_attn
            self.scale = 0.8 / math.sqrt(self.head_dim)
            
        elif stage == DevelopmentalStage.CONCRETE_OPERATIONAL:
            # Most heads are now functional
            used_heads = max(1, int(self.num_heads * 0.75))
            self._mask_unused_heads(used_heads)
            self.attn_dropout.p = max(0.1, self.config.dropout_attn / 2)
            self.scale = 1.0 / math.sqrt(self.head_dim)
            
        else:  # FORMAL_OPERATIONAL
            # All heads are fully functional
            self._reset_head_masks()
            self.attn_dropout.p = max(0.05, self.config.dropout_attn / 4)
            self.scale = 1.2 / math.sqrt(self.head_dim)
    
    def _mask_unused_heads(self, num_active_heads: int) -> None:
        """
        Masks unused attention heads by zeroing out their key/query projections
        
        Args:
            num_active_heads: Number of heads to keep active
        """
        if num_active_heads >= self.num_heads:
            return
            
        # Save original parameters if not already saved
        if not hasattr(self, '_original_q_weight'):
            self._original_q_weight = self.q_proj.weight.data.clone()
            self._original_k_weight = self.k_proj.weight.data.clone()
            self._original_q_bias = self.q_proj.bias.data.clone() if self.q_proj.bias is not None else None
            self._original_k_bias = self.k_proj.bias.data.clone() if self.k_proj.bias is not None else None
        
        # Create a binary mask for weights
        # Each head corresponds to a chunk of the output dimension
        head_size = self.attn_dim // self.num_heads
        mask = torch.ones(self.attn_dim, device=self.q_proj.weight.device)
        
        # Set mask to 0 for unused heads
        for h in range(num_active_heads, self.num_heads):
            start_idx = h * head_size
            end_idx = (h + 1) * head_size
            mask[start_idx:end_idx] = 0.0
            
        # Apply mask to projection layers (only needed for query and key)
        # We reshape the mask to match weight matrices
        q_mask = mask.repeat(self.q_proj.weight.size(0), 1)
        k_mask = mask.repeat(self.k_proj.weight.size(0), 1)
        
        # Apply the mask
        self.q_proj.weight.data = self._original_q_weight * q_mask
        self.k_proj.weight.data = self._original_k_weight * k_mask
        
        # Mask biases if they exist
        if self.q_proj.bias is not None and self._original_q_bias is not None:
            q_bias_mask = mask.repeat(self.q_proj.bias.size(0) // mask.size(0))
            self.q_proj.bias.data = self._original_q_bias * q_bias_mask
            
        if self.k_proj.bias is not None and self._original_k_bias is not None:
            k_bias_mask = mask.repeat(self.k_proj.bias.size(0) // mask.size(0))
            self.k_proj.bias.data = self._original_k_bias * k_bias_mask
    
    def _reset_head_masks(self) -> None:
        """Restore the original parameters for all heads"""
        if hasattr(self, '_original_q_weight'):
            self.q_proj.weight.data = self._original_q_weight.clone()
            self.k_proj.weight.data = self._original_k_weight.clone()
            
            if self.q_proj.bias is not None and self._original_q_bias is not None:
                self.q_proj.bias.data = self._original_q_bias.clone()
                
            if self.k_proj.bias is not None and self._original_k_bias is not None:
                self.k_proj.bias.data = self._original_k_bias.clone()

class SelfAttention(MultiHeadAttention):
    """
    Self-attention component that attends to different positions of a single sequence.
    This is a convenience wrapper around MultiHeadAttention.
    """
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for self-attention
        
        Args:
            x: Input sequence [batch_size, seq_len, dim]
            mask: Optional mask [batch_size, seq_len, seq_len] or [batch_size, 1, seq_len]
                  
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        return super().forward(x, x, x, mask)

class CrossAttention(MultiHeadAttention):
    """
    Cross-attention component that attends from one sequence to another.
    This is a convenience wrapper around MultiHeadAttention.
    """
    
    def forward(self, 
                query: torch.Tensor, 
                context: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-attention
        
        Args:
            query: Query sequence [batch_size, query_len, dim]
            context: Context sequence [batch_size, context_len, dim]
            mask: Optional mask [batch_size, query_len, context_len] or [batch_size, 1, context_len]
                  
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        return super().forward(query, context, context, mask)

class PositionalEncoding(nn.Module):
    """
    Positional encoding adds information about the position of tokens in a sequence.
    
    This implementation uses sine and cosine functions of different frequencies
    to create unique encodings for each position.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum length of sequences
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and transpose to [1, max_len, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register buffer (persistent state that's not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        # Add positional encoding
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

# Factory function to create attention components
def create_attention_component(
    attention_type: str, 
    input_dim: int, 
    output_dim: int, 
    num_heads: int = 8,
    **kwargs
) -> NeuralComponent:
    """
    Factory function to create appropriate attention component
    
    Args:
        attention_type: Type of attention ('dot_product', 'multi_head', 'self_attention', 'cross_attention')
        input_dim: Input dimension
        output_dim: Output dimension
        num_heads: Number of attention heads for multi-head variants
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured attention component
    """
    config = AttentionConfig(
        component_type=f"{attention_type}_attention",
        name=f"{attention_type.replace('_', ' ').title()} Attention",
        description=f"{attention_type.replace('_', ' ').title()} attention component",
        input_dim=input_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        **kwargs
    )
    
    if attention_type.lower() == "dot_product":
        return DotProductAttention(config)
    elif attention_type.lower() == "multi_head":
        return MultiHeadAttention(config)
    elif attention_type.lower() == "self_attention":
        return SelfAttention(config)
    elif attention_type.lower() == "cross_attention":
        return CrossAttention(config)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}") 