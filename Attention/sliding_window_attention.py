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