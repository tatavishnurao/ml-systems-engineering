# Token & Embedding Module Documentation

## Overview

This module provides a unified interface for:
- **Tokenization**: Wrap both OpenAI (tiktoken) and HuggingFace tokenizers
- **Embeddings**: Generate embeddings via OpenAI API or sentence-transformers
- **Visualization**: PCA-based dimensionality reduction and plotting

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    token_emb Library                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────┐  │
│  │ tokenizer_utils │  │ embedding_utils  │  │   pca_    │  │
│  │                 │  │                  │  │visualizer │  │
│  │ • tiktoken      │  │ • OpenAI API     │  │           │  │
│  │ • HuggingFace   │  │ • sent-transform │  │ • sklearn │  │
│  └─────────────────┘  └──────────────────┘  │ • plotly  │  │
│                                              └───────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

### Tokenization

```python
from token_emb import TokenizerWrapper, count_tokens

# Wrap any tokenizer
tokenizer = TokenizerWrapper("gpt-4")  # or "bert-base-uncased"

# Count tokens
count = tokenizer.count("Hello, world!")

# Encode/decode
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)
```

### Embeddings

```python
from token_emb import EmbeddingProvider, get_embeddings

# Local model (fast, free)
provider = EmbeddingProvider("sentence-transformers/all-MiniLM-L6-v2")

# Or OpenAI API
provider = EmbeddingProvider("text-embedding-ada-002", api_key="sk-...")

# Get embeddings
embeddings = provider.embed(["text 1", "text 2"])

# Quick function
embeddings = get_embeddings("Hello world")
```

### Visualization

```python
from token_emb import PCAVisualizer, plot_2d, plot_3d

# 2D plot
plot_2d(embeddings, labels=["A", "B", "C"], save_path="plot.png")

# 3D plot
plot_3d(embeddings, colors=[0, 1, 0], title="My Plot")

# Full control
viz = PCAVisualizer(n_components=2)
fig = viz.plot(embeddings, labels=texts, save_path="output.png")
```

---

## Supported Models

### Tokenizers

| Backend | Models | Dependencies |
|---------|--------|--------------|
| tiktoken | gpt-4, gpt-3.5-turbo, text-davinci-003, ... | `tiktoken` |
| HuggingFace | Any HF tokenizer | `transformers` |

### Embedding Models

| Backend | Models | Dependencies |
|---------|--------|--------------|
| OpenAI | text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large | `openai` |
| sentence-transformers | all-MiniLM-L6-v2, all-mpnet-base-v2, ... | `sentence-transformers` |

---

## CLI Scripts

### Count Tokens

```bash
# Count tokens in a file
python scripts/count_tokens.py essay.txt --model gpt-4

# Verbose output with token IDs
python scripts/count_tokens.py essay.txt -v
```

### Generate Embeddings

```bash
# Embed a single text
python scripts/dump_embeddings.py "Hello world" -o embeddings.npy

# Embed lines from file
python scripts/dump_embeddings.py sentences.txt --model all-MiniLM-L6-v2

# Use OpenAI API
python scripts/dump_embeddings.py texts.txt --model text-embedding-ada-002 --api-key $OPENAI_API_KEY
```

### Visualize

```bash
# 2D PCA plot
python scripts/vis_pca.py embeddings.npy --labels labels.txt

# 3D plot with auto-clustering
python scripts/vis_pca.py embeddings.npy --dim 3 --colors auto -o plot_3d.png

# Save without opening
python scripts/vis_pca.py embeddings.npy --no-show
```

---

## Examples

### Example 1: Document Similarity Search

```python
from token_emb import EmbeddingProvider, cosine_similarity
import numpy as np

# Documents
docs = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Python is a programming language",
    "Neural networks can learn patterns",
]

# Get embeddings
provider = EmbeddingProvider("sentence-transformers/all-MiniLM-L6-v2")
embeddings = provider.embed(docs)

# Query
query = "artificial intelligence"
query_emb = provider.embed(query)

# Find most similar
similarities = [cosine_similarity(query_emb, doc_emb) for doc_emb in embeddings]
most_similar_idx = np.argmax(similarities)

print(f"Query: {query}")
print(f"Most similar: {docs[most_similar_idx]}")
print(f"Similarity: {similarities[most_similar_idx]:.3f}")
```

### Example 2: Token Cost Estimation

```python
from token_emb import count_tokens

# OpenAI pricing (as of 2024)
PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
}

# Your text
text = "..."  # Your prompt

# Estimate cost
model = "gpt-4"
tokens = count_tokens(text, model=model)
cost = (tokens / 1000) * PRICING[model]["input"]

print(f"Tokens: {tokens}")
print(f"Estimated cost: ${cost:.4f}")
```

### Example 3: Clustering Visualization

```python
from token_emb import EmbeddingProvider, PCAVisualizer
from sklearn.cluster import KMeans

# Texts
texts = ["..."]  # Your texts

# Embeddings
provider = EmbeddingProvider("all-MiniLM-L6-v2")
embeddings = provider.embed(texts)

# Cluster
n_clusters = 5
labels = KMeans(n_clusters=n_clusters).fit_predict(embeddings)

# Visualize
visualizer = PCAVisualizer(n_components=2)
fig = visualizer.plot(
    embeddings,
    colors=labels,
    title=f"Text Clusters (k={n_clusters})",
    cmap='tab10'
)
```

---

## Performance Tips

### 1. Batch Processing

Always batch your texts when possible:

```python
# ❌ Slow - one at a time
embeddings = [provider.embed(text) for text in texts]

# ✅ Fast - batched
embeddings = provider.embed(texts, batch_size=32)
```

### 2. Model Selection

| Model | Speed | Quality | Size |
|-------|-------|---------|------|
| all-MiniLM-L6-v2 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 384-dim |
| all-mpnet-base-v2 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 768-dim |
| text-embedding-3-small | ⭐⭐⭐⭐ | ⭐⭐⭐ | 1536-dim |
| text-embedding-3-large | ⭐⭐ | ⭐⭐⭐⭐⭐ | 3072-dim |

### 3. Caching

For sentence-transformers models, embeddings are cached automatically in `~/.cache/torch/sentence_transformers/`.

---

## Testing

Run tests with pytest:

```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_tokenizer_utils.py -v

# With coverage
pytest tests/ --cov=src/token_emb --cov-report=html
```

---

## Troubleshooting

### Import Error: No module named 'tiktoken'

```bash
pip install tiktoken
```

### Import Error: No module named 'sentence_transformers'

```bash
pip install sentence-transformers
```

### CUDA Out of Memory

Reduce batch size:

```python
embeddings = provider.embed(texts, batch_size=8)
```

Or use CPU:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

---

## API Reference

See docstrings in source files for detailed API documentation:

- `src/token_emb/tokenizer_utils.py`
- `src/token_emb/embedding_utils.py`
- `src/token_emb/pca_visualizer.py`