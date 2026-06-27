import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SlidingWindowAttention(nn.Module):
    def __init__(self, embed_dim: int, window_size: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.window_size = window_size

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x:      [batch, seq_len, embed_dim]
            output: [batch, seq_len, embed_dim]
    
            Each token only attends to nearby tokens inside a fixed window.
            """
    
            batch_size, seq_len, embed_dim = x.shape
    
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
    
            scores = q @ k.transpose(-2, -1)
            scores = scores / math.sqrt(embed_dim)
    
            positions = torch.arange(seq_len, device=x.device)
    
            distance = positions[None, :] - positions[:, None]
            distance = distance.abs()
    
            window_mask = distance > self.window_size
    
            scores = scores.masked_fill(window_mask, float("-inf"))
    
            attn_weights = F.softmax(scores, dim=-1)
    
            out = attn_weights @ v
            return self.out_proj(out)

if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    embed_dim = 16
    window_size = 2

    x = torch.randn(batch_size, seq_len, embed_dim)

    model = SlidingWindowAttention(
            embed_dim=embed_dim,
            window_size=window_size,
        )
    output = model(x)
    
    print("Sliding Window Attention Example")
    print("--------------------------------")
    print("Input shape: ", x.shape)
    print("Output shape:", output.shape)
    print("Window size: ", window_size)