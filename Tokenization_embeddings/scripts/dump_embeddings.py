#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_emb import EmbeddingProvider


def read_input(input_path: str) -> list:
    path = Path(input_path)
    if path.exists() and path.is_file():
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        return [line.strip() for line in lines if line.strip()]
    return [input_path]


def main():
    parser = argparse.ArgumentParser(
        description="Generate and save embeddings to .npy file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python dump_embeddings.py "Hello world" --output embeddings.npy
    python dump_embeddings.py sentences.txt --output embeddings.npy
    python dump_embeddings.py sentences.txt --model all-MiniLM-L6-v2 --output embeddings.npy
        """,
    )

    parser.add_argument(
        "input", type=str, help="Input text or path to file (one text per line)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="embeddings.npy",
        help="Output file path (default: embeddings.npy)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None, help="API key for OpenAI models"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )

    args = parser.parse_args()
    texts = read_input(args.input)

    if args.verbose:
        print(f"Input: {args.input}")
        print(f"Number of texts: {len(texts)}")
        print(f"Model: {args.model}")
        print(f"Output: {args.output}")

    try:
        provider = EmbeddingProvider(model_name=args.model, api_key=args.api_key)
    except Exception as e:
        print(f"Error initializing embedding provider: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Generating embeddings...")
    try:
        embeddings = provider.embed(
            texts, batch_size=args.batch_size, show_progress=True
        )
    except Exception as e:
        print(f"Error generating embeddings: {e}", file=sys.stderr)
        sys.exit(1)

    np.save(args.output, embeddings)

    print(f"\n{'=' * 50}")
    print(f"Embeddings saved to: {args.output}")
    print(f"Shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Number of texts: {embeddings.shape[0]}")
    print(f"{'=' * 50}\n")

    texts_path = str(Path(args.output).with_suffix(".txt"))
    with open(texts_path, "w") as f:
        for text in texts:
            f.write(text + "\n")
    print(f"Texts saved to: {texts_path}")


if __name__ == "__main__":
    main()
