"""
Unit tests for embedding_utils module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_emb.embedding_utils import (
    EmbeddingProvider,
    get_embeddings,
    cosine_similarity,
    cosine_similarity_matrix,
)


class TestEmbeddingProvider:
    """Tests for EmbeddingProvider class."""

    def test_init_local(self):
        """Test initialization with local model."""
        # Note: This will download model on first run
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        assert provider.model_name == "all-MiniLM-L6-v2"
        assert provider.backend == "sentence-transformers"

    def test_embed_single(self):
        """Test embedding a single text."""
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        embedding = provider.embed("Hello world")

        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1  # Single vector
        assert len(embedding) == provider.dimension

    def test_embed_multiple(self):
        """Test embedding multiple texts."""
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        texts = ["Hello world", "Goodbye world", "Test text"]
        embeddings = provider.embed(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2  # Matrix
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == provider.dimension

    def test_dimension_consistency(self):
        """Test that dimension matches embedding size."""
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        embedding = provider.embed("Test")

        assert len(embedding) == provider.dimension

    def test_empty_text(self):
        """Test with empty text."""
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        embedding = provider.embed("")

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == provider.dimension


class TestCosineSimilarity:
    """Tests for cosine similarity functions."""

    def test_identical_vectors(self):
        """Test that identical vectors have similarity 1.0."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])

        sim = cosine_similarity(a, b)
        assert pytest.approx(sim, abs=1e-6) == 1.0

    def test_opposite_vectors(self):
        """Test that opposite vectors have similarity -1.0."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])

        sim = cosine_similarity(a, b)
        assert pytest.approx(sim, abs=1e-6) == -1.0

    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have similarity 0.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])

        sim = cosine_similarity(a, b)
        assert pytest.approx(sim, abs=1e-6) == 0.0

    def test_similarity_range(self):
        """Test that similarity is always in [-1, 1]."""
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        texts = ["cat", "dog", "car", "house"]
        embeddings = provider.embed(texts)

        for i in range(len(texts)):
            for j in range(len(texts)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                assert -1.0 <= sim <= 1.0

    def test_similar_texts(self):
        """Test that similar texts have high similarity."""
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        embeddings = provider.embed(["happy", "joyful", "sad"])

        sim_happy_joyful = cosine_similarity(embeddings[0], embeddings[1])
        sim_happy_sad = cosine_similarity(embeddings[0], embeddings[2])

        # Happy and joyful should be more similar than happy and sad
        assert sim_happy_joyful > sim_happy_sad


class TestCosineSimilarityMatrix:
    """Tests for similarity matrix function."""

    def test_matrix_shape(self):
        """Test that matrix has correct shape."""
        embeddings = np.random.randn(5, 10)
        matrix = cosine_similarity_matrix(embeddings)

        assert matrix.shape == (5, 5)

    def test_diagonal_is_one(self):
        """Test that diagonal is all 1s (self-similarity)."""
        embeddings = np.random.randn(5, 10)
        matrix = cosine_similarity_matrix(embeddings)

        for i in range(5):
            assert pytest.approx(matrix[i, i], abs=1e-6) == 1.0

    def test_symmetry(self):
        """Test that matrix is symmetric."""
        embeddings = np.random.randn(5, 10)
        matrix = cosine_similarity_matrix(embeddings)

        assert np.allclose(matrix, matrix.T)


class TestGetEmbeddings:
    """Tests for get_embeddings convenience function."""

    def test_get_embeddings_single(self):
        """Test getting embedding for single text."""
        embedding = get_embeddings("Hello world", model_name="all-MiniLM-L6-v2")

        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1

    def test_get_embeddings_multiple(self):
        """Test getting embeddings for multiple texts."""
        texts = ["Hello world", "Goodbye world"]
        embeddings = get_embeddings(texts, model_name="all-MiniLM-L6-v2")

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
