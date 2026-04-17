import numpy as np
from typing import Optional, Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: Optional[np.ndarray] = None,
    dropout_rate: float = 0.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention in pure NumPy.

    Args:
        query: Shape (batch, heads, seq_len, d_k)
        key: Shape (batch, heads, seq_len, d_k)
        value: Shape (batch, heads, seq_len, d_v)
        mask: Optional mask shape broadcastable to (batch, heads, seq_len, seq_len)
        dropout_rate: Dropout probability (0.0 = no dropout)
        seed: Random seed for reproducible dropout

    Returns:
        output: Shape (batch, heads, seq_len, d_v)
        attention_weights: Shape (batch, heads, seq_len, seq_len)
    """
    d_k = query.shape[-1]
    scores = np.matmul(query, key.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    attention_weights = softmax(scores, axis=-1)

    if dropout_rate > 0.0:
        rng = np.random.RandomState(seed)
        dropout_mask = rng.binomial(1, 1 - dropout_rate, attention_weights.shape)
        attention_weights = attention_weights * dropout_mask / (1 - dropout_rate)

    output = np.matmul(attention_weights, value)
    return output, attention_weights


def multi_head_attention(
    x: np.ndarray,
    num_heads: int,
    d_model: int,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified multi-head attention using random projections.

    Args:
        x: Input shape (batch, seq_len, d_model)
        num_heads: Number of attention heads
        d_model: Model dimension
        mask: Optional attention mask

    Returns:
        output: Shape (batch, seq_len, d_model)
        attention_weights: Shape (batch, num_heads, seq_len, seq_len)
    """
    d_k = d_model // num_heads
    batch_size, seq_len, _ = x.shape

    rng = np.random.RandomState(42)
    w_q = rng.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    w_k = rng.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    w_v = rng.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    w_o = rng.randn(d_model, d_model) * np.sqrt(2.0 / d_model)

    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    q = q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    k = k.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    v = v.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)

    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
        batch_size, seq_len, d_model
    )
    output = attn_output @ w_o

    return output, attn_weights


def causal_mask(seq_len: int) -> np.ndarray:
    """
    Generate a causal (lower-triangular) mask for autoregressive attention.

    Args:
        seq_len: Sequence length

    Returns:
        Mask of shape (1, 1, seq_len, seq_len)
    """
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask[np.newaxis, np.newaxis, :, :]


def attention_pattern_visualization(
    attention_weights: np.ndarray,
    tokens: list,
    head: int = 0,
) -> str:
    """
    Generate a text-based visualization of attention weights.

    Args:
        attention_weights: Shape (batch, num_heads, seq_len, seq_len)
        tokens: List of token strings
        head: Which head to visualize

    Returns:
        Formatted string showing attention pattern
    """
    weights = attention_weights[0, head]
    seq_len = len(tokens)

    max_token_len = max(len(t) for t in tokens) + 2
    header = " " * max_token_len + "".join(f"{t:>6}" for t in tokens)
    lines = [header]

    for i in range(seq_len):
        row = f"{tokens[i]:<{max_token_len}}"
        for j in range(seq_len):
            row += f"{weights[i, j]:>6.2f}"
        lines.append(row)

    return "\n".join(lines)
