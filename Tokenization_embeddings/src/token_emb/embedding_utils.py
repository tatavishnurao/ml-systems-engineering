from typing import List, Union, Optional
import numpy as np

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class EmbeddingProvider:
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        backend: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.backend = backend or self._detect_backend(model_name)
        self.api_key = api_key
        self.cache_dir = cache_dir
        self._model = None
        self._initialize()

    def _detect_backend(self, model_name: str) -> str:
        if model_name.startswith(("text-embedding",)):
            return "openai"
        return "sentence-transformers"

    def _initialize(self):
        if self.backend == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not installed. Run: pip install openai")
            if self.api_key:
                openai.api_key = self.api_key
        else:
            if not ST_AVAILABLE:
                raise ImportError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(
                self.model_name, cache_folder=self.cache_dir
            )

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        single = isinstance(texts, str)
        texts = [texts] if single else texts

        if self.backend == "openai":
            embeddings = self._embed_openai(texts)
        else:
            embeddings = self._embed_local(texts, batch_size, show_progress)

        return embeddings[0] if single else embeddings

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.embeddings.create(model=self.model_name, input=texts)
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        except ImportError:
            response = openai.Embedding.create(model=self.model_name, input=texts)
            embeddings = [item["embedding"] for item in response["data"]]
            return np.array(embeddings)

    def _embed_local(
        self, texts: List[str], batch_size: int, show_progress: bool
    ) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    @property
    def dimension(self) -> int:
        if self.backend == "openai":
            dims = {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
            }
            return dims.get(self.model_name, 1536)
        return self._model.get_sentence_embedding_dimension()


def get_embeddings(
    texts: Union[str, List[str]], model_name: str = "all-MiniLM-L6-v2", **kwargs
) -> np.ndarray:
    provider = EmbeddingProvider(model_name=model_name, **kwargs)
    return provider.embed(texts)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return np.dot(normalized, normalized.T)
