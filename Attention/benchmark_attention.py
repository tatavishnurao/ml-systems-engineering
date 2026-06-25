import time
import torch
import torch.nn.functional as F





def linear_attention(q, k, v):
    q = F.elu(q) + 1
    k = F.elu(k) + 1

    kv = k.transpose(-2, -1) @ v
    z = 1 / ((q @ k.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6)

    return (q @ kv) * z


def time_fn(fn, q, k, v, repeats=20):
    start = time.time()

    for _ in range(repeats):
        _ = fn(q, k, v)

    end = time.time()
    return (end - start) / repeats


if __name__ == "__main__":
    torch.manual_seed(42)

    batch = 1
    dim = 64

    for seq_len in [128, 256, 512, 1024]:
        q = torch.randn(batch, seq_len, dim)
        k = torch.randn(batch, seq_len, dim)
        v = torch.randn(batch, seq_len, dim)

        standard_time = time_fn(standard_attention, q, k, v)
        linear_time = time_fn(linear_attention, q, k, v)

        print(f"seq_len={seq_len}")
        print(f"  standard attention: {standard_time:.6f} sec")
        print(f"  linear attention:   {linear_time:.6f} sec")
        print()