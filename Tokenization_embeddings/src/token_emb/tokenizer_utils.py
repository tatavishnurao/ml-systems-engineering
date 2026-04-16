from typing import List, Optional

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
    def __init__(self, model_name: str = "gpt-4", backend: Optional[str] = None):
        self.model_name = model_name
        self.backend = backend or self._detect_backend(model_name)
        self._tokenizer = None
        self._initialize()

    def _detect_backend(self, model_name: str) -> str:
        if model_name.startswith(("gpt-", "text-")):
            return "tiktoken"
        return "huggingface"

    def _initialize(self):
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
        if self.backend == "tiktoken":
            return self._tokenizer.encode(text)
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: List[int]) -> str:
        if self.backend == "tiktoken":
            return self._tokenizer.decode(tokens)
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    @property
    def vocab_size(self) -> int:
        if self.backend == "tiktoken":
            return self._tokenizer.n_vocab
        return len(self._tokenizer)


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    tokenizer = TokenizerWrapper(model_name)
    return tokenizer.count(text)


def encode_text(text: str, model_name: str = "gpt-4") -> List[int]:
    tokenizer = TokenizerWrapper(model_name)
    return tokenizer.encode(text)


def decode_tokens(tokens: List[int], model_name: str = "gpt-4") -> str:
    tokenizer = TokenizerWrapper(model_name)
    return tokenizer.decode(tokens)
