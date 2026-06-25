import torch
import torch.nn as nn
import torch.nn.functional as F


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """
    Linear attention needs a positive feature map φ(x).

    Instead of using softmax(QK^T), we transform Q and K using φ.
    ELU(x) + 1 keeps the values positive.
    """
    return F.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape:
            [batch_size, seq_len, embed_dim]

        output shape:
            [batch_size, seq_len, embed_dim]
        """

        batch_size, seq_len, embed_dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split embedding into multiple heads.
        # [batch_size, seq_len, embed_dim]
        # -> [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Move heads before sequence.
        # [batch_size, seq_len, num_heads, head_dim]
        # -> [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = elu_feature_map(q)
        k = elu_feature_map(k)

        # Standard attention computes:
        # softmax(QK^T)V
        #
        # Linear attention computes:
        # φ(Q)(φ(K)^T V)
        #
        # First compute K^T V.
        # k: [batch_size, num_heads, seq_len, head_dim]
        # v: [batch_size, num_heads, seq_len, head_dim]
        # kv: [batch_size, num_heads, head_dim, head_dim]
        kv = torch.einsum("bhnd,bhne->bhde", k, v)

        # Normalization term:
        # 1 / (φ(Q) sum(φ(K)))
        k_sum = k.sum(dim=2)

        normalizer = 1.0 / (
            torch.einsum("bhnd,bhd->bhn", q, k_sum) + 1e-6
        )

        # Apply Q to precomputed KV.
        # out: [batch_size, num_heads, seq_len, head_dim]
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, normalizer)

        # Merge heads back.
        # [batch_size, num_heads, seq_len, head_dim]
        # -> [batch_size, seq_len, num_heads, head_dim]
        # -> [batch_size, seq_len, embed_dim]
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, embed_dim)

        return self.out_proj(out)


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    embed_dim = 16
    num_heads = 4

    x = torch.randn(batch_size, seq_len, embed_dim)

    model = LinearAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
    )

    y = model(x)

    print("Linear Attention Example")
    print("------------------------")
    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)

    print()
    print("First input token, first 5 values:")
    print(x[0, 0, :5])

    print()
    print("First output token, first 5 values:")
    print(y[0, 0, :5])