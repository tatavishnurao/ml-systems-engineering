# Tokenization & Embeddings

A Python library for text tokenization and embeddings with visualization.

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

tokenizer = TokenizerWrapper("bert-base-uncased")
```

**Use cases:**
- Estimate API costs before sending a request
- Debug why model outputs seem cut off
- Optimize prompts to fit within token limits

### Embeddings

```python
from token_emb import EmbeddingProvider, cosine_similarity

provider = EmbeddingProvider("all-MiniLM-L6-v2")
emb = provider.embed("Machine learning is amazing")

emb1 = provider.embed("The cat is sleeping")
emb2 = provider.embed("A dog is resting")
print(cosine_similarity(emb1, emb2))  # High similarity!
```

**Use cases:**
- Semantic search - find documents by meaning
- Clustering - group similar content automatically
- Classification - categorize text by similarity
- Recommendations - suggest related content

### Visualization

```python
from token_emb import plot_2d, PCAVisualizer

plot_2d(embeddings, labels=text_labels, save_path="clusters.png")

viz = PCAVisualizer(n_components=2)
viz.plot(embeddings, colors=category_labels, title="Document Clusters")
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

This saves embeddings to `embeddings.npy` and texts to `sentences.txt`.

### Visualize with PCA

```bash
python scripts/vis_pca.py embeddings.npy --labels sentences.txt --output plot.png
```

## Project Structure

```
Tokenization_embeddings/
├── src/token_emb/           # Library code
│   ├── tokenizer_utils.py   # Tokenization
│   ├── embedding_utils.py   # Embeddings
│   └── pca_visualizer.py   # Visualization
├── scripts/                 # CLI tools
│   ├── count_tokens.py      # Count tokens in files
│   ├── dump_embeddings.py   # Generate embeddings
│   └── vis_pca.py          # Visualize embeddings
├── examples/               # Jupyter demo
├── outputs/                # Example outputs
└── tests/                  # Unit tests
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

## Tokenization Use Cases

### 1. Cost Estimation

```python
from token_emb import count_tokens

PRICING = {"gpt-4": 0.03, "gpt-3.5-turbo": 0.0015}

text = open("prompt.txt").read()
tokens = count_tokens(text, "gpt-4")
cost = (tokens / 1000) * PRICING["gpt-4"]
print(f"Estimated cost: ${cost:.4f}")
```

### 2. Debug Token Limits

```python
tokenizer = TokenizerWrapper("gpt-4")
long_text = load_document()

if tokenizer.count(long_text) > 8000:
    print("Warning: Document may exceed context limit!")
```

### 3. Optimize Prompts

```python
verbose = "Please provide a comprehensive and detailed explanation of..."
tokens_verbose = count_tokens(verbose, "gpt-4")

concise = "Explain thoroughly..."
tokens_concise = count_tokens(concise, "gpt-4")

print(f"Saving {tokens_verbose - tokens_concise} tokens per prompt!")
```

## Embedding Use Cases

### 1. Semantic Search

```python
from token_emb import EmbeddingProvider, cosine_similarity
import numpy as np

provider = EmbeddingProvider("all-MiniLM-L6-v2")

documents = [
    "Python is a programming language",
    "JavaScript is used for web development",
    "Machine learning uses algorithms to learn from data",
    "The weather is nice today",
]

query = "Tell me about coding"
doc_embeddings = provider.embed(documents)
query_embedding = provider.embed(query)

similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
best_match = np.argmax(similarities)
print(f"Most relevant: {documents[best_match]}")
```

### 2. Document Clustering

```python
from token_emb import EmbeddingProvider
from sklearn.cluster import KMeans

provider = EmbeddingProvider("all-MiniLM-L6-v2")
embeddings = provider.embed(document_list)

clusters = KMeans(n_clusters=5).fit_predict(embeddings)

for i in range(5):
    cluster_docs = [doc for doc, c in zip(documents, clusters) if c == i]
    print(f"Cluster {i}: {len(cluster_docs)} documents")
```

### 3. Text Classification

```python
from token_emb import get_embeddings
from sklearn.linear_model import LogisticRegression

X_train = get_embeddings(train_texts)
X_test = get_embeddings(test_texts)

clf = LogisticRegression()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

### 4. Duplicate Detection

```python
from token_emb import cosine_similarity_matrix

embeddings = provider.embed(candidate_texts)
similarity_matrix = cosine_similarity_matrix(embeddings)

for i in range(len(texts)):
    for j in range(i+1, len(texts)):
        if similarity_matrix[i, j] > 0.95:
            print(f"Potential duplicate: {texts[i]} <-> {texts[j]}")
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT