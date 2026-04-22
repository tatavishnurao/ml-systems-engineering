import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
    scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    d_k = query.shape[-1]
    if scale is None:
        scale = np.sqrt(d_k)

    scores = np.matmul(query, key.transpose(0, 1)) / scale

    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    attn_weights = softmax(scores, axis=-1)
    output = np.matmul(attn_weights, value)

    return output, attn_weights


if __name__ == "__main__":
    seq_len, d_k, d_v = 4, 8, 8
    np.random.seed(42)
    q = np.random.randn(seq_len, d_k).astype(np.float32)
    k = np.random.randn(seq_len, d_k).astype(np.float32)
    v = np.random.randn(seq_len, d_v).astype(np.float32)

    out, weights = scaled_dot_product_attention(q, k, v)
    print(f"Output shape: {out.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Attention weights:\n{weights}")