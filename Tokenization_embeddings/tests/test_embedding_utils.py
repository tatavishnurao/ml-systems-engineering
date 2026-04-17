import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_emb.embedding_utils import (
    EmbeddingProvider,
    get_embeddings,
    cosine_similarity,
    cosine_similarity_matrix,
)


class TestEmbeddingProvider:
    def test_init_local(self):
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        assert provider.model_name == "all-MiniLM-L6-v2"
        assert provider.backend == "sentence-transformers"

    def test_embed_single(self):
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        embedding = provider.embed("Hello world")
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert len(embedding) == provider.dimension

    def test_embed_multiple(self):
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        texts = ["Hello world", "Goodbye world", "Test text"]
        embeddings = provider.embed(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == provider.dimension

    def test_dimension_consistency(self):
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        embedding = provider.embed("Test")
        assert len(embedding) == provider.dimension

    def test_empty_text(self):
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        embedding = provider.embed("")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == provider.dimension


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(a, b)
        assert pytest.approx(sim, abs=1e-6) == 1.0

    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])
        sim = cosine_similarity(a, b)
        assert pytest.approx(sim, abs=1e-6) == -1.0

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        sim = cosine_similarity(a, b)
        assert pytest.approx(sim, abs=1e-6) == 0.0

    def test_similarity_range(self):
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        texts = ["cat", "dog", "car", "house"]
        embeddings = provider.embed(texts)
        for i in range(len(texts)):
            for j in range(len(texts)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                assert -1.0 <= sim <= 1.0 + 1e-6

    def test_similar_texts(self):
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        embeddings = provider.embed(["happy", "joyful", "sad"])
        sim_happy_joyful = cosine_similarity(embeddings[0], embeddings[1])
        sim_happy_sad = cosine_similarity(embeddings[0], embeddings[2])
        assert sim_happy_joyful > sim_happy_sad


class TestCosineSimilarityMatrix:
    def test_matrix_shape(self):
        embeddings = np.random.randn(5, 10)
        matrix = cosine_similarity_matrix(embeddings)
        assert matrix.shape == (5, 5)

    def test_diagonal_is_one(self):
        embeddings = np.random.randn(5, 10)
        matrix = cosine_similarity_matrix(embeddings)
        for i in range(5):
            assert pytest.approx(matrix[i, i], abs=1e-6) == 1.0

    def test_symmetry(self):
        embeddings = np.random.randn(5, 10)
        matrix = cosine_similarity_matrix(embeddings)
        assert np.allclose(matrix, matrix.T)


class TestGetEmbeddings:
    def test_get_embeddings_single(self):
        embedding = get_embeddings("Hello world", model_name="all-MiniLM-L6-v2")
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1

    def test_get_embeddings_multiple(self):
        texts = ["Hello world", "Goodbye world"]
        embeddings = get_embeddings(texts, model_name="all-MiniLM-L6-v2")
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)

    def test_cat_dog_more_similar_than_cat_car(self):
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        embeddings = provider.embed(["cat", "dog", "car"])
        sim_cat_dog = cosine_similarity(embeddings[0], embeddings[1])
        sim_cat_car = cosine_similarity(embeddings[0], embeddings[2])
        assert sim_cat_dog > sim_cat_car

    def test_identical_sentences_high_similarity(self):
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        embeddings = provider.embed(
            ["The weather is nice today", "Today the weather is nice"]
        )
        sim = cosine_similarity(embeddings[0], embeddings[1])
        assert sim > 0.7

    def test_unrelated_sentences_low_similarity(self):
        provider = EmbeddingProvider("all-MiniLM-L6-v2")
        embeddings = provider.embed(
            ["Machine learning algorithms", "The recipe needs flour"]
        )
        sim = cosine_similarity(embeddings[0], embeddings[1])
        assert sim < 0.5
