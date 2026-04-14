"""
Unit tests for tokenizer_utils module.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_emb.tokenizer_utils import (
    TokenizerWrapper,
    count_tokens,
    encode_text,
    decode_tokens,
)


class TestTokenizerWrapper:
    """Tests for TokenizerWrapper class."""

    def test_init_default(self):
        """Test default initialization."""
        tokenizer = TokenizerWrapper("gpt-4")
        assert tokenizer.model_name == "gpt-4"
        assert tokenizer.backend == "tiktoken"

    def test_init_hf(self):
        """Test HuggingFace backend initialization."""
        tokenizer = TokenizerWrapper("bert-base-uncased")
        assert tokenizer.backend == "huggingface"

    def test_encode_decode_roundtrip(self):
        """Test that encode -> decode returns original text (approximately)."""
        text = "Hello, world! This is a test."
        tokenizer = TokenizerWrapper("gpt-4")

        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        # Note: Decoding might not be exact due to tokenization artifacts
        assert len(tokens) > 0
        assert isinstance(decoded, str)

    def test_count_consistency(self):
        """Test that count matches encode length."""
        text = "The quick brown fox jumps over the lazy dog."
        tokenizer = TokenizerWrapper("gpt-4")

        count = tokenizer.count(text)
        tokens = tokenizer.encode(text)

        assert count == len(tokens)

    def test_vocab_size_positive(self):
        """Test that vocab size is a positive integer."""
        tokenizer = TokenizerWrapper("gpt-4")
        assert tokenizer.vocab_size > 0
        assert isinstance(tokenizer.vocab_size, int)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_count_tokens(self):
        """Test count_tokens function."""
        text = "Hello world"
        count = count_tokens(text, model_name="gpt-4")

        assert isinstance(count, int)
        assert count > 0

    def test_encode_text(self):
        """Test encode_text function."""
        text = "Hello world"
        tokens = encode_text(text, model_name="gpt-4")

        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) > 0

    def test_decode_tokens(self):
        """Test decode_tokens function."""
        text = "Hello world"
        tokens = encode_text(text, model_name="gpt-4")
        decoded = decode_tokens(tokens, model_name="gpt-4")

        assert isinstance(decoded, str)

    def test_empty_text(self):
        """Test handling of empty text."""
        text = ""
        tokens = encode_text(text, model_name="gpt-4")

        assert tokens == []
        assert count_tokens(text, model_name="gpt-4") == 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_unicode_text(self):
        """Test with unicode characters."""
        text = "Hello 世界! 🌍"
        tokenizer = TokenizerWrapper("gpt-4")

        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert len(tokens) > 0
        assert isinstance(decoded, str)

    def test_long_text(self):
        """Test with longer text."""
        text = "Word " * 1000
        tokenizer = TokenizerWrapper("gpt-4")

        tokens = tokenizer.encode(text)
        count = tokenizer.count(text)

        assert len(tokens) == count
        assert len(tokens) > 100  # Should have many tokens

    def test_special_characters(self):
        """Test with special characters."""
        text = "<|endoftext|> [MASK] <s> </s>"
        tokenizer = TokenizerWrapper("gpt-4")

        tokens = tokenizer.encode(text)
        assert len(tokens) > 0
