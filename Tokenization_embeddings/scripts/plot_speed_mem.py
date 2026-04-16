#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

df_fp16 = pd.DataFrame(
    {
        "model": ["Llama-2-7B", "Mistral-7B"],
        "throughput": [4.08, 3.73],
        "memory": [13800, 14800],
    }
)

df_int8 = pd.DataFrame(
    {
        "model": ["Llama-2-7B", "Mistral-7B"],
        "throughput": [6.58, 5.94],
        "memory": [7200, 7800],
    }
)

df_int4 = pd.DataFrame(
    {
        "model": ["Llama-2-7B", "Mistral-7B"],
        "throughput": [11.19, 10.18],
        "memory": [4100, 4400],
    }
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

models = ["Llama-2-7B", "Mistral-7B"]
x = range(len(models))
width = 0.25

ax1.bar([i - width for i in x], df_fp16["memory"], width, label="FP16", color="#ff6b6b")
ax1.bar(x, df_int8["memory"], width, label="INT8", color="#51cf66")
ax1.bar([i + width for i in x], df_int4["memory"], width, label="INT4", color="#339af0")
ax1.set_ylabel("Memory (MB)")
ax1.set_title("Memory Usage by Precision")
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()

ax2.bar(
    [i - width for i in x], df_fp16["throughput"], width, label="FP16", color="#ff6b6b"
)
ax2.bar(x, df_int8["throughput"], width, label="INT8", color="#51cf66")
ax2.bar(
    [i + width for i in x], df_int4["throughput"], width, label="INT4", color="#339af0"
)
ax2.set_ylabel("Throughput (tokens/sec)")
ax2.set_title("Throughput by Precision")
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()

plt.tight_layout()
plt.savefig("docs/img/speed_vs_mem.png", dpi=150)
print("Saved to docs/img/speed_vs_mem.png")
