import numpy as np
from sdpa_numpy import scaled_dot_product_attention, softmax


class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.1

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        batch_size = query.shape[0]

        Q = self.split_heads(self.W_q @ query)
        K = self.split_heads(self.W_k @ key)
        V = self.split_heads(self.W_v @ value)

        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, -1, self.d_model)

        return self.W_o @ attn_output


if __name__ == "__main__":
    batch, seq_len, d_model, num_heads = 2, 4, 16, 4
    np.random.seed(42)
    x = np.random.randn(batch, seq_len, d_model).astype(np.float32)

    mha = MultiHeadAttention(d_model, num_heads)
    out = mha.forward(x, x, x)
    print(f"Output shape: {out.shape}")