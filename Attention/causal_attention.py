import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:      [batch, seq_len, embed_dim]
        output: [batch, seq_len, embed_dim]

        Each token can only attend to itself and previous tokens.
        """

        batch_size, seq_len, embed_dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(embed_dim)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1,
        ).bool()

        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        out = attn_weights @ v
        return self.out_proj(out)


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 6
    embed_dim = 16

    x = torch.randn(batch_size, seq_len, embed_dim)

    model = CausalAttention(embed_dim=embed_dim)

    output = model(x)

    print("Causal Attention Example")
    print("------------------------")
    print("Input shape: ", x.shape)
    print("Output shape:", output.shape)
