from .tokenizer_utils import (
    TokenizerWrapper,
    count_tokens,
    encode_text,
    decode_tokens,
)

from .embedding_utils import (
    EmbeddingProvider,
    get_embeddings,
    cosine_similarity,
)

from .pca_visualizer import (
    PCAVisualizer,
    plot_2d,
    plot_3d,
)

from .attention_numpy import (
    scaled_dot_product_attention,
    multi_head_attention,
    causal_mask,
    softmax,
)

__version__ = "0.1.0"
__all__ = [
    "TokenizerWrapper",
    "count_tokens",
    "encode_text",
    "decode_tokens",
    "EmbeddingProvider",
    "get_embeddings",
    "cosine_similarity",
    "PCAVisualizer",
    "plot_2d",
    "plot_3d",
    "scaled_dot_product_attention",
    "multi_head_attention",
    "causal_mask",
    "softmax",
]
