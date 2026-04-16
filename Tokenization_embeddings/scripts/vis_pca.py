#!/usr/bin/env python3
import argparse
import sys
import subprocess
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from token_emb import PCAVisualizer


def open_image(path: str):
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", path], check=True)
        elif sys.platform == "linux":
            subprocess.run(["xdg-open", path], check=True)
        elif sys.platform == "win32":
            subprocess.run(["start", path], shell=True, check=True)
    except Exception as e:
        print(f"Could not auto-open image: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Create PCA visualization from embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python vis_pca.py embeddings.npy
    python vis_pca.py embeddings.npy --dim 3 --labels labels.txt
    python vis_pca.py embeddings.npy --output plot.png --no-show
        """,
    )

    parser.add_argument("embeddings", type=str, help="Path to .npy embeddings file")
    parser.add_argument(
        "--dim",
        type=int,
        choices=[2, 3],
        default=2,
        help="PCA dimensions (2 or 3, default: 2)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to labels file (one label per line)",
    )
    parser.add_argument(
        "--colors",
        type=str,
        default=None,
        help="Path to colors file (one value per line, or 'auto' for clustering)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="pca_plot.png",
        help="Output image file (default: pca_plot.png)",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Don't auto-open the image"
    )
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument(
        "--size", type=float, default=50, help="Scatter point size (default: 50)"
    )

    args = parser.parse_args()

    embeddings_path = Path(args.embeddings)
    if not embeddings_path.exists():
        print(f"Error: Embeddings file '{args.embeddings}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        embeddings = np.load(args.embeddings)
    except Exception as e:
        print(f"Error loading embeddings: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded embeddings: {embeddings.shape}")

    labels = None
    if args.labels:
        labels_path = Path(args.labels)
        if labels_path.exists():
            with open(labels_path) as f:
                labels = [line.strip() for line in f]
            print(f"Loaded {len(labels)} labels")
        else:
            print(f"Warning: Labels file '{args.labels}' not found")

    colors = None
    if args.colors:
        if args.colors == "auto":
            from sklearn.cluster import KMeans

            n_clusters = min(5, len(embeddings))
            colors = KMeans(
                n_clusters=n_clusters, random_state=42, n_init=10
            ).fit_predict(embeddings)
            print(f"Auto-generated colors using KMeans (k={n_clusters})")
        else:
            colors_path = Path(args.colors)
            if colors_path.exists():
                with open(colors_path) as f:
                    colors = [float(line.strip()) for line in f]
                print(f"Loaded {len(colors)} color values")

    title = (
        args.title or f"PCA Visualization ({args.dim}D) - {embeddings.shape[0]} points"
    )

    print(f"\nGenerating {args.dim}D PCA plot...")

    visualizer = PCAVisualizer(n_components=args.dim)
    fig = visualizer.plot(
        embeddings,
        labels=labels,
        colors=colors,
        title=title,
        save_path=args.output,
        show=False,
        s=args.size,
        alpha=0.7,
    )

    print(f"\n{'=' * 50}")
    print(f"Explained variance ratio:")
    for i, ratio in enumerate(visualizer.explained_variance_ratio):
        print(f"  PC{i + 1}: {ratio:.2%}")
    print(f"Total variance explained: {sum(visualizer.explained_variance_ratio):.2%}")
    print(f"{'=' * 50}")

    if not args.no_show:
        print(f"\nOpening {args.output}...")
        open_image(args.output)
    else:
        print(f"\nPlot saved to: {args.output}")


if __name__ == "__main__":
    main()
