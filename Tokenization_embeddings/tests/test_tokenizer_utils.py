import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_emb.tokenizer_utils import (
    TokenizerWrapper,
    count_tokens,
    encode_text,
    decode_tokens,
)


class TestTokenizerWrapper:
    def test_init_default(self):
        tokenizer = TokenizerWrapper("gpt-4")
        assert tokenizer.model_name == "gpt-4"
        assert tokenizer.backend == "tiktoken"

    def test_init_hf(self):
        tokenizer = TokenizerWrapper("bert-base-uncased")
        assert tokenizer.backend == "huggingface"

    def test_encode_decode_roundtrip(self):
        text = "Hello, world! This is a test."
        tokenizer = TokenizerWrapper("gpt-4")
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert len(tokens) > 0
        assert isinstance(decoded, str)

    def test_count_consistency(self):
        text = "The quick brown fox jumps over the lazy dog."
        tokenizer = TokenizerWrapper("gpt-4")
        count = tokenizer.count(text)
        tokens = tokenizer.encode(text)
        assert count == len(tokens)

    def test_vocab_size_positive(self):
        tokenizer = TokenizerWrapper("gpt-4")
        assert tokenizer.vocab_size > 0
        assert isinstance(tokenizer.vocab_size, int)


class TestUtilityFunctions:
    def test_count_tokens(self):
        text = "Hello world"
        count = count_tokens(text, model_name="gpt-4")
        assert isinstance(count, int)
        assert count > 0

    def test_encode_text(self):
        text = "Hello world"
        tokens = encode_text(text, model_name="gpt-4")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) > 0

    def test_decode_tokens(self):
        text = "Hello world"
        tokens = encode_text(text, model_name="gpt-4")
        decoded = decode_tokens(tokens, model_name="gpt-4")
        assert isinstance(decoded, str)

    def test_empty_text(self):
        text = ""
        tokens = encode_text(text, model_name="gpt-4")
        assert tokens == []
        assert count_tokens(text, model_name="gpt-4") == 0


class TestEdgeCases:
    def test_unicode_text(self):
        text = "Hello 世界! 🌍"
        tokenizer = TokenizerWrapper("gpt-4")
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert len(tokens) > 0
        assert isinstance(decoded, str)

    def test_long_text(self):
        text = "Word " * 1000
        tokenizer = TokenizerWrapper("gpt-4")
        tokens = tokenizer.encode(text)
        count = tokenizer.count(text)
        assert len(tokens) == count
        assert len(tokens) > 100

    def test_special_characters(self):
        text = "<|endoftext|> [MASK] <s> </s>"
        tokenizer = TokenizerWrapper("gpt-4")
        tokens = tokenizer.encode(text)
        assert len(tokens) > 0
