"""
Scaled Dot-Product Attention Implementation
============================================
This module implements the core attention mechanism from "Attention is All You Need".

The formula:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Where:
    - Q (Query): What we're looking for
    - K (Key): What each position offers
    - V (Value): The actual information to aggregate
    - d_k: Dimension of the key vectors (for scaling)
"""

import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute Scaled Dot-Product Attention.
    
    Args:
        Q (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_k)
        K (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_k)
        V (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_v)
        mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len, seq_len)
                                       or (seq_len, seq_len). Use -inf for positions to mask.
    
    Returns:
        output (torch.Tensor): Attention output of shape (batch_size, seq_len, d_v)
        attention_weights (torch.Tensor): Attention weights of shape (batch_size, seq_len, seq_len)
    
    Shape explanation:
        - Q @ K^T gives us (batch_size, seq_len, seq_len) - similarity scores
        - softmax normalizes these scores into probabilities
        - scores @ V gives us (batch_size, seq_len, d_v) - weighted sum of values
    """
    # Get the dimension of the key vectors
    d_k = Q.size(-1)
    
    # Step 1: Compute attention scores: Q @ K^T
    # Shape: (batch_size, seq_len, d_k) @ (batch_size, d_k, seq_len) 
    #     -> (batch_size, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Step 2: Scale by sqrt(d_k) to prevent vanishing gradients
    # This scaling is crucial! Without it, the dot products can grow large,
    # pushing the softmax into regions with tiny gradients.
    scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply mask (if provided)
    # For causal attention, we'll mask out future positions
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 4: Apply softmax to get attention weights (probabilities)
    # Softmax is applied along the last dimension (over keys)
    attention_weights = F.softmax(scores, dim=-1)
    
    # Step 5: Compute weighted sum of values
    # Shape: (batch_size, seq_len, seq_len) @ (batch_size, seq_len, d_v)
    #     -> (batch_size, seq_len, d_v)
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Scaled Dot-Product Attention")
    print("=" * 70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define dimensions
    batch_size = 2
    seq_len = 4
    d_k = 8  # dimension of queries and keys
    d_v = 8  # dimension of values (often same as d_k)
    
    # Create random Q, K, V tensors
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)
    
    print(f"\nInput Shapes:")
    print(f"  Q: {Q.shape} (batch_size={batch_size}, seq_len={seq_len}, d_k={d_k})")
    print(f"  K: {K.shape}")
    print(f"  V: {V.shape}")
    
    # Test 1: Basic attention without mask
    print("\n" + "-" * 70)
    print("Test 1: Basic Attention (No Mask)")
    print("-" * 70)
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"\nFirst batch's attention weights:")
    print(attention_weights[0])
    print(f"\nSum of attention weights (should be ~1.0 for each row):")
    print(attention_weights[0].sum(dim=-1))
    
    # Test 2: Attention with causal mask (for autoregressive models)
    print("\n" + "-" * 70)
    print("Test 2: Attention with Causal Mask")
    print("-" * 70)
    
    # Create a causal mask: lower triangular matrix
    # This prevents positions from attending to future positions
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    print(f"\nCausal mask (1=attend, 0=mask):")
    print(causal_mask)
    
    output_masked, attention_weights_masked = scaled_dot_product_attention(
        Q, K, V, mask=causal_mask
    )
    
    print(f"\nFirst batch's masked attention weights:")
    print(attention_weights_masked[0])
    print(f"\nNotice: Each position only attends to itself and previous positions!")
    print(f"Sum of attention weights (should still be ~1.0 for each row):")
    print(attention_weights_masked[0].sum(dim=-1))
    
    # Test 3: Verify scaling importance
    print("\n" + "-" * 70)
    print("Test 3: Importance of Scaling")
    print("-" * 70)
    
    # Without scaling
    d_k_test = Q.size(-1)
    scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))
    scores_scaled = scores_unscaled / math.sqrt(d_k_test)
    
    print(f"\nUnscaled scores - mean: {scores_unscaled[0].mean():.4f}, std: {scores_unscaled[0].std():.4f}")
    print(f"Scaled scores   - mean: {scores_scaled[0].mean():.4f}, std: {scores_scaled[0].std():.4f}")
    print(f"\nScaling factor (sqrt(d_k)): {math.sqrt(d_k_test):.4f}")
    print(f"This prevents scores from becoming too large, which would cause")
    print(f"softmax to saturate and produce tiny gradients.")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully! ✓")
    print("=" * 70)

