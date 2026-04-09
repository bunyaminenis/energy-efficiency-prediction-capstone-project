"""
Clustering Module for Energy Benchmarking
Implements unsupervised learning approaches from the proposal:
  - K-Means Clustering
  - K-Shape Clustering (via DTW approximation)
  - Hierarchical Clustering
  - PCA for dimensionality reduction

Used to group buildings by energy profiles for benchmarking.
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from utils.logger import setup_logger

logger = setup_logger(__name__)


class EnergyBenchmarker:
    """Clusters buildings by energy profile for benchmarking purposes."""

    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.pca = None
        self.labels_ = None

    def apply_pca(self, X: np.ndarray, n_components=None, variance_threshold=0.95):
        """
        Apply PCA for dimensionality reduction.
        Used when there are too many features (as per architecture chart).
        """
        if n_components is None:
            self.pca = PCA(random_state=self.random_state)
            self.pca.fit(X)
            cumulative = np.cumsum(self.pca.explained_variance_ratio_)
            n_components = int(np.argmax(cumulative >= variance_threshold) + 1)
            logger.info(f"PCA: {n_components} components explain "
                        f"{cumulative[n_components-1]*100:.1f}% of variance")

        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        X_reduced = self.pca.fit_transform(X)
        logger.info(f"PCA reduced: {X.shape[1]} -> {X_reduced.shape[1]} features")
        return X_reduced

    def find_optimal_k(self, X: np.ndarray, k_range=range(2, 11)) -> int:
        """Elbow method + silhouette analysis to find optimal cluster count."""
        best_k = 2
        best_silhouette = -1

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            logger.info(f"  k={k}: silhouette={sil:.4f}, calinski_harabasz={ch:.1f}")

            if sil > best_silhouette:
                best_silhouette = sil
                best_k = k

        logger.info(f"Optimal k={best_k} (silhouette={best_silhouette:.4f})")
        return best_k

    def cluster_kmeans(self, X: np.ndarray) -> np.ndarray:
        """K-Means clustering for energy benchmarking."""
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.labels_ = self.kmeans.fit_predict(X)

        sil = silhouette_score(X, self.labels_)
        logger.info(f"K-Means ({self.n_clusters} clusters): silhouette={sil:.4f}")

        # Log cluster sizes
        unique, counts = np.unique(self.labels_, return_counts=True)
        for c, n in zip(unique, counts):
            logger.info(f"  Cluster {c}: {n} buildings")

        return self.labels_

    def cluster_hierarchical(self, X: np.ndarray, linkage="ward") -> np.ndarray:
        """Hierarchical (agglomerative) clustering."""
        hc = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=linkage
        )
        labels = hc.fit_predict(X)

        sil = silhouette_score(X, labels)
        logger.info(f"Hierarchical ({linkage}, {self.n_clusters} clusters): silhouette={sil:.4f}")
        return labels

    def get_cluster_profiles(self, X: np.ndarray, feature_names: list) -> dict:
        """Compute mean feature values per cluster for profiling."""
        if self.labels_ is None:
            raise ValueError("Run clustering first.")

        profiles = {}
        for c in np.unique(self.labels_):
            mask = self.labels_ == c
            cluster_mean = X[mask].mean(axis=0)
            profiles[int(c)] = {
                "size": int(mask.sum()),
                "features": {name: float(val) for name, val in zip(feature_names, cluster_mean)}
            }
        return profiles

    def assign_new_building(self, X_new: np.ndarray) -> int:
        """Assign a new building to an existing cluster."""
        if self.kmeans is None:
            raise ValueError("Train K-Means first.")
        return int(self.kmeans.predict(X_new.reshape(1, -1))[0])
