import numpy as np


def create_causal_mask(seq_len: int, dtype: np.dtype = np.float32) -> np.ndarray:
    mask = np.triu(np.ones((seq_len, seq_len), dtype=dtype), k=1)
    return mask


def apply_causal_mask(scores: np.ndarray) -> np.ndarray:
    seq_len = scores.shape[-1]
    mask = create_causal_mask(seq_len, scores.dtype)
    return np.where(mask == 1, -1e9, scores)


if __name__ == "__main__":
    seq_len = 4
    mask = create_causal_mask(seq_len)
    print(f"Causal mask shape: {mask.shape}")
    print(mask)

    scores = np.random.randn(seq_len, seq_len).astype(np.float32)
    masked_scores = apply_causal_mask(scores)
    print(f"\nMasked scores:\n{masked_scores}")