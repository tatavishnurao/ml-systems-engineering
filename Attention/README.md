# Attention

This folder explains attention from an ML systems perspective.

The first example calculates the arithmetic intensity of standard attention and connects it to the difference between prefill and decode in LLM inference.

## Prefill vs Decode

The essential difference is simple:

- Prefill processes the entire input sequence in parallel.
- Decode generates tokens one at a time.

During prefill, model weights are loaded once and reused across large matrix multiplications over the whole prompt. This creates a lot of computation per byte loaded from memory, so arithmetic intensity is high.

During decode, the model repeatedly loads weights and KV-cache data while doing less work per generated token. This gives lower arithmetic intensity, so decode is often memory-bound.

## Standard Attention

Standard attention has three major steps:

1. S = QK^T
2. P = softmax(S)
3. O = PV

For this example:

- N = 4096 sequence positions
- d = 128 attention head dimension
- FP16 = 2 bytes per value

Matrix shapes:

- Q, K, V = N x d = 4096 x 128
- S, P = N x N = 4096 x 4096
- O = N x d = 4096 x 128

A 4096 x 4096 FP16 matrix uses:

4096 * 4096 * 2 = 33,554,432 bytes = 32 MiB

## Memory Movement

For standard attention:

memory = 8N^2 + 8Nd bytes

For N = 4096 and d = 128:

memory = 138,412,032 bytes = 132 MiB

## Compute

compute = 4N^2d + 3N^2 ops

For N = 4096 and d = 128:

compute = 8,640,266,240 ops

## Arithmetic Intensity

arithmetic_intensity = compute / memory

For this example:

arithmetic_intensity = 62.42 ops/byte

This is much lower than a reference H100-style ratio of about 295 ops/byte, so this standard attention example is memory-bound.

## Run

From the repo root:

python Attention/attention_arithmetic_intensity.py

Try different values:

python Attention/attention_arithmetic_intensity.py --N 8192 --d 128
python Attention/attention_arithmetic_intensity.py --N 4096 --d 64

Expected default result:

- Total memory movement: 138,412,032 bytes = 132.00 MiB
- Total compute: 8,640,266,240 ops
- Arithmetic intensity: 62.42 ops/byte
- Verdict: memory-bound

## Multi-Head Attention

Multi-head attention runs several attention heads in parallel.

Instead of doing one attention operation over the full model dimension, the model splits the hidden dimension into smaller heads.

Example:

- d_model = 512
- num_heads = 8
- head_dim = 64

Each head learns a different attention pattern. After attention is computed in each head, the heads are concatenated and passed through a final output projection.

The basic flow is:

1. X is projected into Q, K, and V
2. Q, K, and V are split into heads
3. Each head runs scaled dot-product attention
4. Head outputs are concatenated
5. A final output projection mixes the heads

Run the demo:

python Attention/multi_head_attention.py

Try different dimensions:

python Attention/multi_head_attention.py --seq-len 8 --d-model 16 --num-heads 4

# Linear Attention

This folder contains a minimal PyTorch implementation of **linear attention**.

## Why Linear Attention?

Standard attention computes:

```text
Attention(Q, K, V) = softmax(QK^T)V
```

This creates a full `seq_len × seq_len` attention matrix.

That means every token compares with every other token.

For long sequences, this becomes expensive because the cost grows quadratically:

```text
standard attention: O(n²d)
```

## Core Idea

Linear attention avoids building the full attention matrix.

Instead of computing:

```text
softmax(QK^T)V
```

we use a positive feature map `φ` and rewrite attention as:

```text
φ(Q)(φ(K)^T V)
```

The important trick is this:

```text
φ(K)^T V
```

is computed first.

So instead of comparing every query token with every key token directly, the keys and values are compressed into a smaller summary first. Then each query reads from that summary.

## Feature Map

In this implementation, the feature map is:

```python
φ(x) = elu(x) + 1
```

This keeps the transformed `Q` and `K` values positive, which is useful for linear attention.

## Example

The script uses this example input:

```python
batch_size = 2
seq_len = 8
embed_dim = 16
num_heads = 4
```

So the input tensor shape is:

```text
[2, 8, 16]
```

The output shape is also:

```text
[2, 8, 16]
```

Linear attention changes the token representations, but preserves the original input shape.

## Run

From the repo root:

```bash
uv run python attention/linear_attention.py
```

Expected output:

```text
Linear Attention Example
------------------------
Input shape:  torch.Size([2, 8, 16])
Output shape: torch.Size([2, 8, 16])
```

## Standard Attention vs Linear Attention

Standard attention:

```text
QK^T creates a [seq_len, seq_len] matrix
```

Linear attention:

```text
K^T V creates a smaller compressed summary
```

So the rough complexity changes from:

```text
O(n²d)
```

to:

```text
O(nd²)
```

where:

```text
n = sequence length
d = head dimension
```