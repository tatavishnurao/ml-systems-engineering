"""
Tokenizer utilities - wraps tiktoken and HuggingFace tokenizers.
"""

from typing import List, Union, Optional
import warnings

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from transformers import AutoTokenizer

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class TokenizerWrapper:
    """Unified wrapper for tiktoken and HuggingFace tokenizers."""

    def __init__(self, model_name: str = "gpt-4", backend: Optional[str] = None):
        """
        Initialize tokenizer.

        Args:
            model_name: Model identifier (e.g., 'gpt-4', 'bert-base-uncased')
            backend: 'tiktoken', 'huggingface', or None (auto-detect)
        """
        self.model_name = model_name
        self.backend = backend or self._detect_backend(model_name)
        self._tokenizer = None
        self._initialize()

    def _detect_backend(self, model_name: str) -> str:
        """Auto-detect appropriate backend."""
        if model_name.startswith(("gpt-", "text-")):
            return "tiktoken"
        return "huggingface"

    def _initialize(self):
        """Initialize the tokenizer."""
        if self.backend == "tiktoken":
            if not TIKTOKEN_AVAILABLE:
                raise ImportError("tiktoken not installed. Run: pip install tiktoken")
            encoding_name = self._map_to_tiktoken(self.model_name)
            self._tokenizer = tiktoken.get_encoding(encoding_name)
        else:
            if not HF_AVAILABLE:
                raise ImportError(
                    "transformers not installed. Run: pip install transformers"
                )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _map_to_tiktoken(self, model_name: str) -> str:
        """Map model name to tiktoken encoding."""
        mapping = {
            "gpt-4": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "text-embedding-ada-002": "cl100k_base",
            "text-davinci-003": "p50k_base",
            "text-davinci-002": "p50k_base",
            "davinci": "r50k_base",
        }
        return mapping.get(model_name, "cl100k_base")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.backend == "tiktoken":
            return self._tokenizer.encode(text)
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        if self.backend == "tiktoken":
            return self._tokenizer.decode(tokens)
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    def count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encode(text))

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.backend == "tiktoken":
            return self._tokenizer.n_vocab
        return len(self._tokenizer)


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    Quick function to count tokens in text.

    Args:
        text: Input text
        model_name: Model identifier

    Returns:
        Number of tokens
    """
    tokenizer = TokenizerWrapper(model_name)
    return tokenizer.count(text)


def encode_text(text: str, model_name: str = "gpt-4") -> List[int]:
    """
    Encode text to token IDs.

    Args:
        text: Input text
        model_name: Model identifier

    Returns:
        List of token IDs
    """
    tokenizer = TokenizerWrapper(model_name)
    return tokenizer.encode(text)


def decode_tokens(tokens: List[int], model_name: str = "gpt-4") -> str:
    """
    Decode token IDs to text.

    Args:
        tokens: List of token IDs
        model_name: Model identifier

    Returns:
        Decoded text
    """
    tokenizer = TokenizerWrapper(model_name)
    return tokenizer.decode(tokens)
