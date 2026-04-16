# Tokenization & Embeddings

A Python library for text tokenization and embeddings with visualization.

## Getting Started

```bash
# Count tokens in a file
python scripts/count_tokens.py sample_text.txt --model gpt-4

# Generate embeddings
python scripts/dump_embeddings.py sentences.txt --output embeddings.npy

# Visualize embeddings
python scripts/vis_pca.py embeddings.npy --labels labels.txt --output plot.png
```

## What is This?

This library provides two main capabilities:

**Tokenization** - Convert text into tokens (numbers) that LLMs can process
**Embeddings** - Convert text into vectors that capture meaning

## Quick Examples

### Tokenization

```python
from token_emb import TokenizerWrapper

tokenizer = TokenizerWrapper("gpt-4")
tokens = tokenizer.encode("Hello world")
print(tokenizer.count("Hello world"))  # 2 tokens
```

### Embeddings

```python
from token_emb import EmbeddingProvider, cosine_similarity

provider = EmbeddingProvider("all-MiniLM-L6-v2")
emb1 = provider.embed("The cat is sleeping")
emb2 = provider.embed("A dog is resting")
print(cosine_similarity(emb1, emb2))  # High similarity!
```

## CLI Usage

### Count Tokens in a File

```bash
python scripts/count_tokens.py essay.txt --model gpt-4
```

Output:
```
==================================================
File: essay.txt
Model: gpt-4
Backend: tiktoken
==================================================
Token count: 1,247
Character count: 6,582
Avg chars per token: 5.28
==================================================
```

### Generate Embeddings

```bash
python scripts/dump_embeddings.py sentences.txt --output embeddings.npy
```

### Visualize with PCA

```bash
python scripts/vis_pca.py embeddings.npy --labels sentences.txt --output plot.png
```

## Benchmark Results

See `benchmarks/` for INT4 and INT8 benchmark results.

![Speed vs Memory](./docs/img/speed_vs_mem.png)

## Project Structure

```
Tokenization_embeddings/
├── src/token_emb/           # Library code
├── scripts/                 # CLI tools
├── tests/                   # Unit tests
├── benchmarks/              # Benchmark results
├── examples/               # Jupyter demo
└── docs/                   # Documentation
```

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- numpy, pyyaml (core)
- tiktoken (for OpenAI tokenizers)
- transformers (for HuggingFace models)
- sentence-transformers (for local embeddings)
- matplotlib, scikit-learn (for visualization)

## Testing

```bash
pytest tests/ -v
```

## License

MIT