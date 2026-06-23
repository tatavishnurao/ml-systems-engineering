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
