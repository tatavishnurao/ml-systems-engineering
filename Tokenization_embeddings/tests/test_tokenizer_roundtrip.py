import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_emb.tokenizer_utils import TokenizerWrapper

   
def test_encode_decode_roundtrip():
    texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming AI.",
        "こんにちは世界",  # Japanese
        "Hello 世界! 🌍",  
    ]

    for text in texts:
        tokenizer = TokenizerWrapper("gpt-4")
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert len(tokens) > 0, f"Failed for text: {text}"
        assert isinstance(decoded, str), f"Decoded should be str for: {text}"
        print(f"OK: '{text[:30]}...' -> {len(tokens)} tokens")


def test_count_consistency():
    tokenizer = TokenizerWrapper("gpt-4")
    text = "The quick brown fox jumps over the lazy dog."

    count = tokenizer.count(text)
    tokens = tokenizer.encode(text)

    assert count == len(tokens), "count() should match len(encode())"
    print(f"Count consistency: {count} == {len(tokens)}")


if __name__ == "__main__":
    test_encode_decode_roundtrip()
    test_count_consistency()
    print("\nAll roundtrip tests passed!")
