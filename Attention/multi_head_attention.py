"""
Simple Multi-Head Self-Attention implementation using NumPy.

This file shows the core attention block:

1. Project input X into Q, K, V
2. Split Q, K, V into multiple heads
3. Run scaled dot-product attention in each head
4. Concatenate heads
5. Apply output projection

Run:
    python Attention/multi_head_attention.py
"""

import argparse
import math

try:
    import numpy as np
except ModuleNotFoundError:
    raise SystemExit("Missing dependency: numpy. Install it with: python -m pip install numpy")


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def split_heads(x, num_heads):
    """
    Convert:
        x: (batch, seq_len, d_model)

    Into:
        x: (batch, num_heads, seq_len, head_dim)
    """

    batch_size, seq_len, d_model = x.shape

    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    head_dim = d_model // num_heads

    x = x.reshape(batch_size, seq_len, num_heads, head_dim)
    x = np.transpose(x, (0, 2, 1, 3))

    return x


def combine_heads(x):
    """
    Convert:
        x: (batch, num_heads, seq_len, head_dim)

    Into:
        x: (batch, seq_len, d_model)
    """

    batch_size, num_heads, seq_len, head_dim = x.shape

    x = np.transpose(x, (0, 2, 1, 3))
    x = x.reshape(batch_size, seq_len, num_heads * head_dim)

    return x


def scaled_dot_product_attention(q, k, v):
    """
    q, k, v shape:
        (batch, num_heads, seq_len, head_dim)

    scores shape:
        (batch, num_heads, seq_len, seq_len)

    output shape:
        (batch, num_heads, seq_len, head_dim)
    """

    head_dim = q.shape[-1]

    scores = q @ np.swapaxes(k, -1, -2)
    scores = scores / math.sqrt(head_dim)

    attention_weights = softmax(scores, axis=-1)
    output = attention_weights @ v

    return output, attention_weights


def multi_head_self_attention(x, w_q, w_k, w_v, w_o, num_heads):
    """
    x shape:
        (batch, seq_len, d_model)

    w_q, w_k, w_v, w_o shape:
        (d_model, d_model)

    output shape:
        (batch, seq_len, d_model)
    """

    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)

    head_outputs, attention_weights = scaled_dot_product_attention(q, k, v)

    combined = combine_heads(head_outputs)
    output = combined @ w_o

    return output, attention_weights


def run_demo(batch_size, seq_len, d_model, num_heads, seed):
    rng = np.random.default_rng(seed)

    x = rng.normal(size=(batch_size, seq_len, d_model)).astype(np.float32)

    scale = 1.0 / math.sqrt(d_model)

    w_q = (rng.normal(size=(d_model, d_model)) * scale).astype(np.float32)
    w_k = (rng.normal(size=(d_model, d_model)) * scale).astype(np.float32)
    w_v = (rng.normal(size=(d_model, d_model)) * scale).astype(np.float32)
    w_o = (rng.normal(size=(d_model, d_model)) * scale).astype(np.float32)

    output, attention_weights = multi_head_self_attention(
        x=x,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        num_heads=num_heads,
    )

    print()
    print("Multi-Head Self-Attention Demo")
    print("=" * 52)

    print(f"batch_size: {batch_size}")
    print(f"seq_len:    {seq_len}")
    print(f"d_model:    {d_model}")
    print(f"num_heads:  {num_heads}")
    print(f"head_dim:   {d_model // num_heads}")

    print()
    print("Shapes")
    print("-" * 52)
    print(f"Input X:             {x.shape}")
    print(f"Attention weights:   {attention_weights.shape}")
    print(f"Output:              {output.shape}")

    print()
    print("Attention check")
    print("-" * 52)
    row_sums = attention_weights.sum(axis=-1)
    print("Each attention row should sum to 1.")
    print(f"Min row sum: {row_sums.min():.6f}")
    print(f"Max row sum: {row_sums.max():.6f}")

    print()
    print("One attention matrix")
    print("-" * 52)
    print("Showing batch 0, head 0:")
    print(np.round(attention_weights[0, 0], 4))


def main():
    parser = argparse.ArgumentParser(
        description="Simple NumPy implementation of multi-head self-attention."
    )

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_demo(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
