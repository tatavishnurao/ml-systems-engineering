# Example Outputs

This directory contains sample outputs from the tokenization and embedding tools.

## Tokenization Example

When running:
```bash
python scripts/count_tokens.py sample_text.txt --model gpt-4
```

Example output:
```
==================================================
File: sample_text.txt
Model: gpt-4
Backend: tiktoken
==================================================
Token count: 156
Character count: 847
Avg chars per token: 5.43
==================================================
```

## Embedding Example

When running:
```bash
python scripts/dump_embeddings.py sample_texts.txt --output outputs/sample_embeddings.npy
```

This creates:
- `sample_embeddings.npy` - NumPy array of shape (N, 384) where N is number of texts
- `sample_texts.txt` - Original texts used

## PCA Visualization Example

When running:
```bash
python scripts/vis_pca.py sample_embeddings.npy --labels sample_labels.txt --output outputs/sample_plot.png
```

This creates a PNG showing:
- 2D scatter plot of embeddings reduced via PCA
- Labels annotated on each point
- Color coding by category if provided
- Explained variance ratio for each component