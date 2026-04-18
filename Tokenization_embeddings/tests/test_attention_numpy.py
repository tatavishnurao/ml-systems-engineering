import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_emb.attention_numpy import (
    softmax,
    scaled_dot_product_attention,
    multi_head_attention,
    causal_mask,
    attention_pattern_visualization,
)



def test_softmax():
    x = np.array([[1.0, 2.0, 3.0]])
    result = softmax(x)
    assert np.allclose(result.sum(), 1.0)
    assert result[0, 2] > result[0, 1] > result[0, 0]


def test_softmax_numerical_stability():
    x = np.array([[1000.0, 1001.0, 1002.0]])
    result = softmax(x)
    assert not np.any(np.isnan(result))
    assert np.allclose(result.sum(), 1.0)


def test_causal_mask():
    mask = causal_mask(4)
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 0] == 1
    assert mask[0, 0, 0, 3] == 0
    assert mask[0, 0, 3, 3] == 1
    assert mask[0, 0, 3, 0] == 1


def test_sdpa_output_shape():
    batch, heads, seq_len, d_k = 2, 4, 8, 16
    q = np.random.randn(batch, heads, seq_len, d_k)
    k = np.random.randn(batch, heads, seq_len, d_k)
    v = np.random.randn(batch, heads, seq_len, d_k)

    output, weights = scaled_dot_product_attention(q, k, v)

    assert output.shape == (batch, heads, seq_len, d_k)
    assert weights.shape == (batch, heads, seq_len, seq_len)


def test_sdpa_weights_sum_to_one():
    batch, heads, seq_len, d_k = 2, 4, 8, 16
    q = np.random.randn(batch, heads, seq_len, d_k)
    k = np.random.randn(batch, heads, seq_len, d_k)
    v = np.random.randn(batch, heads, seq_len, d_k)

    _, weights = scaled_dot_product_attention(q, k, v)
    sums = weights.sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_sdpa_with_causal_mask():
    batch, heads, seq_len, d_k = 2, 4, 8, 16
    q = np.random.randn(batch, heads, seq_len, d_k)
    k = np.random.randn(batch, heads, seq_len, d_k)
    v = np.random.randn(batch, heads, seq_len, d_k)
    mask = causal_mask(seq_len)

    output, weights = scaled_dot_product_attention(q, k, v, mask=mask)

    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert weights[0, 0, i, j] < 0.01


def test_multi_head_attention_shape():
    batch, seq_len, d_model, num_heads = 2, 8, 64, 4
    x = np.random.randn(batch, seq_len, d_model)

    output, attn_weights = multi_head_attention(x, num_heads, d_model)

    assert output.shape == (batch, seq_len, d_model)
    assert attn_weights.shape == (batch, num_heads, seq_len, seq_len)


def test_attention_visualization():
    tokens = ["the", "cat", "sat"]
    weights = np.array([[[[0.5, 0.3, 0.2], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]]]])
    viz = attention_pattern_visualization(weights, tokens, head=0)
    assert isinstance(viz, str)
    assert "the" in viz


def test_self_attention_identity():
    seq_len, d_k = 4, 16
    q = k = v = np.random.randn(1, 1, seq_len, d_k).astype(np.float64)
    output, weights = scaled_dot_product_attention(q, k, v)

    assert output.shape == (1, 1, seq_len, d_k)
    assert weights.shape == (1, 1, seq_len, seq_len)
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-6)


if __name__ == "__main__":
    test_softmax()
    test_softmax_numerical_stability()
    test_causal_mask()
    test_sdpa_output_shape()
    test_sdpa_weights_sum_to_one()
    test_sdpa_with_causal_mask()
    test_multi_head_attention_shape()
    test_attention_visualization()
    test_self_attention_identity()
    print("All attention tests passed!")
