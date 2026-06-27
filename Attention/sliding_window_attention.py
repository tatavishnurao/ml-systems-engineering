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

    def build_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Builds a mask where positions outside the local window are blocked.

        mask[i, j] = True means token i cannot attend to token j.
        """

        positions = torch.arange(seq_len, device=device)

        query_positions = positions[:, None]
        key_positions = positions[None, :]

        distance = (query_positions - key_positions).abs()

        mask = distance > self.window_size

        return mask

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        """
        x: [batch_size, seq_len, embed_dim]

        output:
            [batch_size, seq_len, embed_dim]
        """

        batch_size, seq_len, embed_dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(embed_dim)

        window_mask = self.build_sliding_window_mask(
            seq_len=seq_len,
            device=x.device,
        )

        scores = scores.masked_fill(window_mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)

        out = attention_weights @ v
        out = self.out_proj(out)

        if return_debug:
            debug_info = {
                "scores_shape": scores.shape,
                "window_mask": window_mask,
                "attention_weights": attention_weights,
            }

            return out, debug_info

        return out


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 16
    embed_dim = 32
    window_size = 3

    x = torch.randn(batch_size, seq_len, embed_dim)

    model = SlidingWindowAttention(
        embed_dim=embed_dim,
        window_size=window_size,
    )

    output, debug_info = model(x, return_debug=True)

    print("Sliding Window Attention Example")
    print("--------------------------------")
    print("Input shape:             ", x.shape)
    print("Output shape:            ", output.shape)
    print("Attention scores shape:  ", debug_info["scores_shape"])
    print("Window mask shape:       ", debug_info["window_mask"].shape)

    selected_token_index = 8

    allowed_positions = (
        ~debug_info["window_mask"][selected_token_index]
    ).nonzero(as_tuple=True)[0]

    print()
    print("Selected token index:", selected_token_index)
    print("Window size:         ", window_size)
    print(f"Token {selected_token_index} can attend to positions:")
    print(allowed_positions.tolist())

    print()
    print("First token output, first 5 values:")
    print(output[0, 0, :5])
