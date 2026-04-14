# Tokenization & Embeddings Module

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reusable Python library for text tokenization, embeddings, and visualization.

## Features

- **Unified Tokenizers** - Wrap OpenAI (tiktoken) and HuggingFace tokenizers with one API
- **Multiple Embedding Providers** - OpenAI API or local sentence-transformers models
- **PCA Visualization** - 2D and 3D plots with matplotlib
- **CLI Tools** - Quick command-line scripts for common tasks
- **Tested** - Comprehensive unit tests with pytest

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd Tokenization_embeddings

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

```python
from token_emb import TokenizerWrapper, EmbeddingProvider, plot_2d

# 1. Tokenize
tokenizer = TokenizerWrapper("gpt-4")
count = tokenizer.count("Hello, world!")  # Returns: 4

# 2. Embed
provider = EmbeddingProvider("sentence-transformers/all-MiniLM-L6-v2")
embeddings = provider.embed(["Hello world", "Goodbye world"])

# 3. Visualize
plot_2d(embeddings, labels=["Hello", "Goodbye"], save_path="plot.png")
```

## CLI Usage

```bash
# Count tokens in a file
python scripts/count_tokens.py essay.txt --model gpt-4

# Generate embeddings from a file (one text per line)
python scripts/dump_embeddings.py sentences.txt --output embeddings.npy

# Create PCA visualization
python scripts/vis_pca.py embeddings.npy --labels labels.txt --output plot.png
```

## Project Structure

```
Tokenization_embeddings/
│
├── src/token_emb/               # Reusable library
│   ├── __init__.py
│   ├── tokenizer_utils.py       # Tokenization (tiktoken/HF)
│   ├── embedding_utils.py       # Embeddings (OpenAI/local)
│   └── pca_visualizer.py        # PCA visualization
│
├── scripts/                     # CLI entry points
│   ├── count_tokens.py          # Count tokens in files
│   ├── dump_embeddings.py       # Save embeddings to .npy
│   └── vis_pca.py               # Quick PCA plots
│
├── tests/                       # Unit tests
│   ├── test_tokenizer_utils.py
│   └── test_embedding_utils.py
│
├── examples/                    # Demos
│   └── notebook.ipynb           # Jupyter notebook
│
├── configs/
│   └── model_paths.yaml         # Model configurations
│
└── docs/
    └── token_emb.md             # Full documentation
```

## Dependencies

Core:
- numpy
- pyyaml

Optional (install as needed):
```bash
# For OpenAI tokenizers & embeddings
pip install tiktoken openai

# For HuggingFace models
pip install transformers sentence-transformers

# For visualization
pip install matplotlib scikit-learn

# For testing
pip install pytest
```

## Examples

### Tokenization

```python
from token_emb import TokenizerWrapper

# Works with OpenAI models
tokenizer = TokenizerWrapper("gpt-4")
tokens = tokenizer.encode("Hello, world!")
count = tokenizer.count(text)

# Or HuggingFace models
tokenizer = TokenizerWrapper("bert-base-uncased")
```

### Embeddings

```python
from token_emb import EmbeddingProvider, cosine_similarity

# Local model (fast, free)
provider = EmbeddingProvider("all-MiniLM-L6-v2")

# Or OpenAI API
provider = EmbeddingProvider("text-embedding-ada-002", api_key="sk-...")

# Get embeddings
emb1 = provider.embed("Machine learning is amazing")
emb2 = provider.embed("Deep learning is powerful")

# Compare similarity
sim = cosine_similarity(emb1, emb2)
print(f"Similarity: {sim:.3f}")
```

### Visualization

```python
from token_emb import PCAVisualizer

# 2D plot
visualizer = PCAVisualizer(n_components=2)
fig = visualizer.plot(embeddings, labels=texts, save_path="plot.png")

# 3D plot
plot_3d(embeddings, colors=labels, title="My 3D Plot")
```

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=src/token_emb --cov-report=html

# Specific test
pytest tests/test_tokenizer_utils.py -v
```

## Configuration

See `configs/model_paths.yaml` for model configurations.

You can override cache directory:
```python
provider = EmbeddingProvider(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir="/path/to/cache"
)
```

## Documentation

Full documentation in `docs/token_emb.md`.

## License

MIT License - see LICENSE file for details.