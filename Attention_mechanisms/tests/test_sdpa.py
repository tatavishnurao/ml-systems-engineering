import numpy as np
from sdpa_numpy import softmax, scaled_dot_product_attention
from multi_head import MultiHeadAttention
from causal_mask import create_causal_mask, apply_causal_mask


def test_softmax():
    x = np.array([[1.0, 2.0, 3.0]])
    result = softmax(x)
    expected_sum = np.sum(result)
    assert np.isclose(expected_sum, 1.0), f"Softmax sum = {expected_sum}"
    print("test_softmax: PASSED")


def test_sdpa():
    seq_len, d_k, d_v = 4, 8, 8
    np.random.seed(42)
    q = np.random.randn(seq_len, d_k).astype(np.float32)
    k = np.random.randn(seq_len, d_k).astype(np.float32)
    v = np.random.randn(seq_len, d_v).astype(np.float32)

    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (seq_len, d_v)
    assert weights.shape == (seq_len, seq_len)
    assert np.allclose(np.sum(weights, axis=-1), 1.0)
    print("test_sdpa: PASSED")


def test_multi_head_attention():
    batch, seq_len, d_model, num_heads = 2, 4, 16, 4
    np.random.seed(42)
    x = np.random.randn(batch, seq_len, d_model).astype(np.float32)

    mha = MultiHeadAttention(d_model, num_heads)
    out = mha.forward(x, x, x)
    assert out.shape == (batch, seq_len, d_model)
    print("test_multi_head_attention: PASSED")


def test_causal_mask():
    seq_len = 4
    mask = create_causal_mask(seq_len)
    assert mask.shape == (seq_len, seq_len)
    assert mask[0, 0] == 0
    assert mask[1, 0] == 1
    print("test_causal_mask: PASSED")


if __name__ == "__main__":
    test_softmax()
    test_sdpa()
    test_multi_head_attention()
    test_causal_mask()
    print("\nAll tests passed!")