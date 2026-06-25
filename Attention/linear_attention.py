import torch
import torch.nn as nn
import torch.nn.functional as F


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = elu_feature_map(q)
        k = elu_feature_map(k)

        kv = torch.einsum("bhnd,bhne->bhde", k, v)

        k_sum = k.sum(dim=2)
        normalizer = 1.0 / (
            torch.einsum("bhnd,bhd->bhn", q, k_sum) + 1e-6
        )

        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, normalizer)

        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, embed_dim)

        return self.out_proj(out)


if __name__ == "__main__":
    x = torch.randn(2, 128, 256)

    model = LinearAttention(
        embed_dim=256,
        num_heads=8,
    )

    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)