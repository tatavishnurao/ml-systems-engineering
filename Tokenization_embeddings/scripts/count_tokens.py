#!/usr/bin/env python3
"""
CLI script: Count tokens in a text file.

Usage:
    python count_tokens.py <file_path> [--model MODEL]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_emb import TokenizerWrapper


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens in a text file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python count_tokens.py essay.txt
    python count_tokens.py essay.txt --model gpt-4
    python count_tokens.py essay.txt --model bert-base-uncased --backend huggingface
        """,
    )

    parser.add_argument("file", type=str, help="Path to text file")

    parser.add_argument(
        "--model", type=str, default="gpt-4", help="Model name (default: gpt-4)"
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["tiktoken", "huggingface"],
        default=None,
        help="Tokenizer backend (auto-detected if not specified)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including token IDs",
    )

    args = parser.parse_args()

    # Check file exists
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)

    # Read file
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize tokenizer
    try:
        tokenizer = TokenizerWrapper(args.model, backend=args.backend)
    except Exception as e:
        print(f"Error initializing tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    # Count tokens
    tokens = tokenizer.encode(text)
    token_count = len(tokens)

    # Print results
    print(f"\n{'=' * 50}")
    print(f"File: {args.file}")
    print(f"Model: {args.model}")
    print(f"Backend: {tokenizer.backend}")
    print(f"{'=' * 50}")
    print(f"Token count: {token_count:,}")
    print(f"Character count: {len(text):,}")
    print(f"Avg chars per token: {len(text) / token_count:.2f}")

    if args.verbose:
        print(f"\nFirst 20 token IDs: {tokens[:20]}")
        print(f"First 20 tokens decoded: {tokenizer.decode(tokens[:20])}")

    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
