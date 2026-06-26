import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        """
        query_tokens:   [batch, query_len, embed_dim]
        context_tokens: [batch, context_len, embed_dim]

        output:         [batch, query_len, embed_dim]
        """

        q = self.q_proj(query_tokens)
        k = self.k_proj(context_tokens)
        v = self.v_proj(context_tokens)

        embed_dim = q.shape[-1]

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(embed_dim)

        attn_weights = F.softmax(scores, dim=-1)

        out = attn_weights @ v
        return self.out_proj(out)

if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    query_len = 4
    context_len = 6
    embed_dim = 16

    query_tokens = torch.randn(batch_size, query_len, embed_dim)
    context_tokens = torch.randn(batch_size, context_len, embed_dim)

    model = CrossAttention(embed_dim=embed_dim)

    output = model(query_tokens, context_tokens)

    print("Cross Attention Example")
    print("-----------------------")
    print("Query shape:  ", query_tokens.shape)
