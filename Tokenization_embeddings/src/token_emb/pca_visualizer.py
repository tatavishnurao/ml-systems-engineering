"""
PCA visualization utilities for embedding visualization.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

try:
    from sklearn.decomposition import PCA

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class PCAVisualizer:
    """PCA-based embedding visualizer."""

    def __init__(self, n_components: int = 2, random_state: int = 42):
        """
        Initialize PCA visualizer.

        Args:
            n_components: Number of PCA components (2 or 3)
            random_state: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn not installed. Run: pip install scikit-learn"
            )

        if n_components not in [2, 3]:
            raise ValueError("n_components must be 2 or 3")

        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self._is_fitted = False

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform embeddings.

        Args:
            embeddings: Array of shape (N, D)

        Returns:
            Transformed embeddings of shape (N, n_components)
        """
        self._is_fitted = True
        return self.pca.fit_transform(embeddings)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using fitted PCA.

        Args:
            embeddings: Array of shape (N, D)

        Returns:
            Transformed embeddings of shape (N, n_components)
        """
        if not self._is_fitted:
            raise RuntimeError("PCA not fitted. Call fit_transform first.")
        return self.pca.transform(embeddings)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio."""
        if not self._is_fitted:
            raise RuntimeError("PCA not fitted.")
        return self.pca.explained_variance_ratio_

    def plot(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        colors: Optional[Union[List, np.ndarray]] = None,
        title: str = "PCA Visualization",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs,
    ) -> plt.Figure:
        """
        Create PCA plot.

        Args:
            embeddings: Embeddings to visualize
            labels: Optional labels for points
            colors: Optional colors for points
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            show: Whether to show the plot
            **kwargs: Additional arguments for scatter plot

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib not installed. Run: pip install matplotlib")

        # Transform embeddings
        transformed = self.fit_transform(embeddings)

        # Create figure
        fig = plt.figure(figsize=figsize)

        if self.n_components == 2:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(
                transformed[:, 0], transformed[:, 1], c=colors, **kwargs
            )
            ax.set_xlabel(f"PC1 ({self.explained_variance_ratio[0]:.1%} variance)")
            ax.set_ylabel(f"PC2 ({self.explained_variance_ratio[1]:.1%} variance)")

            # Add labels if provided
            if labels:
                for i, label in enumerate(labels):
                    ax.annotate(label, (transformed[i, 0], transformed[i, 1]))

        else:  # 3D
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(
                transformed[:, 0],
                transformed[:, 1],
                transformed[:, 2],
                c=colors,
                **kwargs,
            )
            ax.set_xlabel(f"PC1 ({self.explained_variance_ratio[0]:.1%} variance)")
            ax.set_ylabel(f"PC2 ({self.explained_variance_ratio[1]:.1%} variance)")
            ax.set_zlabel(f"PC3 ({self.explained_variance_ratio[2]:.1%} variance)")

            # Add labels if provided
            if labels:
                for i, label in enumerate(labels):
                    ax.text(
                        transformed[i, 0], transformed[i, 1], transformed[i, 2], label
                    )

        ax.set_title(title)

        # Add colorbar if colors are numeric
        if (
            colors is not None
            and isinstance(colors, (list, np.ndarray))
            and len(set(colors)) > 1
        ):
            plt.colorbar(scatter, ax=ax)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig


def plot_2d(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    colors: Optional[Union[List, np.ndarray]] = None,
    title: str = "2D PCA Visualization",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> plt.Figure:
    """
    Quick 2D PCA plot.

    Args:
        embeddings: Embeddings to visualize
        labels: Optional labels
        colors: Optional colors
        title: Plot title
        save_path: Optional save path
        show: Whether to show plot
        **kwargs: Additional scatter arguments

    Returns:
        Matplotlib figure
    """
    visualizer = PCAVisualizer(n_components=2)
    return visualizer.plot(
        embeddings,
        labels=labels,
        colors=colors,
        title=title,
        save_path=save_path,
        show=show,
        **kwargs,
    )


def plot_3d(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    colors: Optional[Union[List, np.ndarray]] = None,
    title: str = "3D PCA Visualization",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> plt.Figure:
    """
    Quick 3D PCA plot.

    Args:
        embeddings: Embeddings to visualize
        labels: Optional labels
        colors: Optional colors
        title: Plot title
        save_path: Optional save path
        show: Whether to show plot
        **kwargs: Additional scatter arguments

    Returns:
        Matplotlib figure
    """
    visualizer = PCAVisualizer(n_components=3)
    return visualizer.plot(
        embeddings,
        labels=labels,
        colors=colors,
        title=title,
        save_path=save_path,
        show=show,
        **kwargs,
    )
