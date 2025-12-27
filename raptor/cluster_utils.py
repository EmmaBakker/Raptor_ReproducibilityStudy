import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import tiktoken
import umap
from sklearn.mixture import GaussianMixture
import hdbscan

from .tree_structures import Node

try:
    import pacmap
except ImportError:
    pacmap = None

try:
    import trimap
except ImportError:
    trimap = None

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

RANDOM_SEED = 224
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Dimensionality reduction utils

def _dr_with_method(
    X: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
    dr_method: str = "umap",
):
    """
    Shared helper for UMAP / PaCMAP / TriMAP / none.

    Used by both global and local DR.
    """
    if X.shape[0] == 0:
        return X

    dr_method = (dr_method or "umap").lower()

    if dr_method == "none":
        target_dim = min(dim, X.shape[1])
        return X[:, :target_dim]

    if dr_method == "umap":
        if n_neighbors is None:
            n_neighbors = int((len(X) - 1) ** 0.5)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=dim,
            metric=metric,
            random_state=RANDOM_SEED,
        )
        return reducer.fit_transform(X)

    if dr_method == "trimap":
        if trimap is None:
            raise ImportError("TriMAP is not installed, but dr_method='trimap' was requested.")

        n_samples = X.shape[0]
        if n_samples <= 3:
            # Too few samples for proper DR → just truncate
            target_dim = min(dim, X.shape[1])
            return X[:, :target_dim]

        target_dim = max(1, min(dim, n_samples - 2))

        # Make TriMAP config valid for small n
        n_inliers = min(10, max(1, n_samples - 2))
        n_outliers = min(5, max(1, n_samples - 3))
        n_random = min(5, max(1, n_samples - 3))

        reducer = trimap.TRIMAP(
            n_dims=target_dim,
            distance=metric,   # "cosine"
            n_inliers=n_inliers,
            n_outliers=n_outliers,
            n_random=n_random,
            apply_pca=True,
            verbose=False,
        )
        return reducer.fit_transform(X)

    if dr_method == "pacmap":
        if pacmap is None:
            raise ImportError("PaCMAP is not installed, but dr_method='pacmap' was requested.")

        n_samples, n_features = X.shape

        if n_samples <= 3:
            target_dim = min(dim, n_features)
            return X[:, :target_dim]

        target_dim = max(1, min(dim, n_features, n_samples - 2))

        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

        if n_neighbors is None:
            n_neighbors = int((n_samples - 1) ** 0.5)
        n_neighbors = max(2, min(int(n_neighbors), n_samples - 1))

        reducer = pacmap.PaCMAP(
            n_components=target_dim,
            n_neighbors=n_neighbors,
            MN_ratio=0.5,
            FP_ratio=2.0,
            random_state=RANDOM_SEED,
        )
        return reducer.fit_transform(Xn)



    raise ValueError(f"Unsupported dr_method '{dr_method}'. Use 'umap', 'pacmap', 'trimap', or 'none'.")


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
    dr_method: str = "umap",
) -> np.ndarray:
    """
    Global DR before the first clustering step.

    - dr_method = "umap" / "pacmap" / "trimap": standard DR to `dim`.
    - dr_method = "none": return embeddings (optionally truncated).
    """
    X = np.asarray(embeddings, dtype=np.float64)
    if X.shape[0] == 0:
        return X

    return _dr_with_method(
        X,
        dim=dim,
        n_neighbors=n_neighbors,
        metric=metric,
        dr_method=dr_method,
    )


def local_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    num_neighbors: int = 10,
    metric: str = "cosine",
    dr_method: str = "umap",
) -> np.ndarray:
    """
    Local DR inside each global cluster.

    In the original paper this is UMAP; here we generalize to:
    - UMAP / PaCMAP / TriMAP / none (same interface as global).
    """
    X = np.asarray(embeddings, dtype=np.float64)
    if X.shape[0] == 0:
        return X

    return _dr_with_method(
        X,
        dim=dim,
        n_neighbors=num_neighbors,
        metric=metric,
        dr_method=dr_method,
    )


# GMM clustering (original RAPTOR)

def get_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int = 50,
    random_state: int = RANDOM_SEED,
) -> int:
    """
    Pick n_components by BIC, but be robust to small sample sizes.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    n_samples = X.shape[0]

    if n_samples <= 2:
        return 1

    max_clusters = min(max_clusters, n_samples)
    if max_clusters <= 1:
        return 1

    candidate_ks = np.arange(1, max_clusters)
    bics = []
    for n in candidate_ks:
        gm = GaussianMixture(
            n_components=n,
            random_state=random_state,
        )
        gm.fit(X)
        bics.append(gm.bic(X))

    optimal_clusters = candidate_ks[int(np.argmin(bics))]
    return int(optimal_clusters)


def GMM_cluster(
    embeddings: np.ndarray,
    threshold: float,
    random_state: int = RANDOM_SEED,
):
    X = np.asarray(embeddings, dtype=np.float64)
    if X.shape[0] == 0:
        return [], 0

    n_clusters = get_optimal_clusters(X, random_state=random_state)
    gm = GaussianMixture(
        n_components=n_clusters,
        random_state=random_state,
    )
    gm.fit(X)
    probs = gm.predict_proba(X)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
    verbose: bool = False,
    dr_method: str = "umap",
) -> List[np.ndarray]:
    """
    RAPTOR two-stage clustering (GMM version):

      1) Global DR (umap/pacmap/trimap/none) + GMM
      2) Local DR (same dr_method) + GMM in each global cluster
    """
    X = np.asarray(embeddings, dtype=np.float64)
    if X.shape[0] == 0:
        return []

    reduced_embeddings_global = global_cluster_embeddings(
        X,
        dim=min(dim, len(X) - 2),
        metric="cosine",
        dr_method=dr_method,
    )
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global,
        threshold,
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(X))]
    total_clusters = 0

    for i in range(n_global_clusters):
        mask = np.array([i in gc for gc in global_clusters])
        global_cluster_embeddings_ = X[mask]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue

        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_,
                dim,
                dr_method=dr_method,
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local,
                threshold,
            )

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        for j in range(n_local_clusters):
            local_mask = np.array([j in lc for lc in local_clusters])
            local_cluster_embeddings_ = global_cluster_embeddings_[local_mask]
            if local_cluster_embeddings_.size == 0:
                continue
            indices = np.where(
                (X == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    if verbose:
        logging.info(f"Total Clusters: {total_clusters}")
    return all_local_clusters


# HDBSCAN clustering (new)

def HDBSCAN_cluster(
    embeddings: np.ndarray,
    dr_dim: int,
    dr_method: str = "umap",
    min_cluster_size: int = 5,
    min_samples: int = 1,
    metric: str = "euclidean",
    verbose: bool = False,
) -> List[np.ndarray]:
    """
    Single-level clustering with HDBSCAN.

    - If dr_method = 'umap'/'pacmap'/'trimap': apply global DR to dr_dim, then cluster with HDBSCAN.
    - If dr_method = 'none': run HDBSCAN directly in the original embedding space.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    n_samples = X.shape[0]

    if n_samples == 0:
        return []

    dr_method = (dr_method or "umap").lower()

    if dr_method == "none":
        X_reduced = X
    else:
        X_reduced = global_cluster_embeddings(
            X,
            dim=min(dr_dim, max(1, n_samples - 2)),
            metric="cosine",
            dr_method=dr_method,
        )

    # IMPORTANT: HDBSCAN should see Euclidean space here
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,          # typically 'euclidean'
        gen_min_span_tree=False,
    )
    labels = clusterer.fit_predict(X_reduced)  # shape (n_samples,)

    if verbose:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logging.info(f"[HDBSCAN] Found {n_clusters} clusters (incl. noise)")

    # If everything is noise (-1), fall back to one big cluster 0
    if np.all(labels == -1):
        labels = np.zeros_like(labels)

    clusters_per_point = [np.array([lbl]) for lbl in labels]
    return clusters_per_point

"""CHANGED"""
def _safe_token_len(text, tokenizer):
    # Original code assumed text is a string; we preserve this
    # but defensively coerce anything else into a string.
    if not isinstance(text, str):
        if isinstance(text, (list, tuple)):
            text = " ".join(map(str, text))
        else:
            text = str(text)
    return len(tokenizer.encode(text))
"""CHANGED"""

# Clusterer interfaces

class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(*args, **kwargs):
        """Interface marker; not actually instantiated."""
        pass


class RAPTOR_Clustering(ClusteringAlgorithm):
    @staticmethod
    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 3500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        dr_method: str = "umap",
        verbose: bool = False,
    ) -> List[List[Node]]:
        """
        Original RAPTOR clustering:
        - Global DR (umap/pacmap/trimap/none) + GMM
        - Local DR (same dr_method) + GMM
        - recursive splitting if cluster text too long
        """
        # SAFETY: only keep real Nodes to avoid 'str'.embeddings crashes
        nodes = [n for n in nodes if isinstance(n, Node)]
        if not nodes:
            return []

        embeddings = np.array(
            [node.embeddings[embedding_model_name] for node in nodes]
        )

        clusters = perform_clustering(
            embeddings,
            dim=reduction_dimension,
            threshold=threshold,
            verbose=verbose,
            dr_method=dr_method,
        )

        node_clusters: List[List[Node]] = []
        if len(clusters) == 0:
            return node_clusters

        for label in np.unique(np.concatenate(clusters)):
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]
            cluster_nodes = [nodes[i] for i in indices]

            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            total_length = sum(_safe_token_len(node.text, tokenizer) for node in cluster_nodes)
            """CHANGED
            total_length = sum(
                len(tokenizer.encode(node.text)) for node in cluster_nodes
            )"""

            if total_length > max_length_in_cluster:
                if verbose:
                    logging.info(
                        f"[RAPTOR] Reclustering cluster with {len(cluster_nodes)} nodes"
                    )
                node_clusters.extend(
                    RAPTOR_Clustering.perform_clustering(
                        cluster_nodes,
                        embedding_model_name,
                        max_length_in_cluster=max_length_in_cluster,
                        tokenizer=tokenizer,
                        reduction_dimension=reduction_dimension,
                        threshold=threshold,
                        dr_method=dr_method,
                        verbose=verbose,
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters


class HDBSCAN_Clustering(ClusteringAlgorithm):
    @staticmethod
    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str = "SBERT",
        max_length_in_cluster: int = 3500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 10,
        threshold: float = 0.1,        # accepted for API compatibility, unused
        dr_method: str = "umap",
        min_cluster_size: int = 5,
        min_samples: int = 1,
        metric: str = "euclidean",
        verbose: bool = False,
    ) -> List[List[Node]]:
        """
        HDBSCAN-based clustering:
        - optional DR (umap/pacmap/trimap/none) to 'reduction_dimension'
        - HDBSCAN in the reduced space
        - recursive splitting *only if* the cluster actually shrinks; otherwise
          stop to avoid infinite recursion.
        """
        # SAFETY: only keep real Nodes (avoid 'str'.embeddings)
        nodes = [n for n in nodes if isinstance(n, Node)]
        if not nodes:
            return []

        embeddings = np.array(
            [node.embeddings[embedding_model_name] for node in nodes]
        )

        clusters = HDBSCAN_cluster(
            embeddings,
            dr_dim=reduction_dimension,
            dr_method=dr_method,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            verbose=verbose,
        )

        node_clusters: List[List[Node]] = []
        if len(clusters) == 0:
            return node_clusters

        unique_labels = np.unique(np.concatenate(clusters))
        for label in unique_labels:
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]
            cluster_nodes = [nodes[i] for i in indices]

            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            total_length = sum(_safe_token_len(node.text, tokenizer) for node in cluster_nodes)
            """CHANGED
            total_length = sum(
                len(tokenizer.encode(node.text)) for node in cluster_nodes
            )"""

            if total_length > max_length_in_cluster:
                if verbose:
                    logging.info(
                        f"[HDBSCAN] Cluster too long (tokens={total_length}, "
                        f"nodes={len(cluster_nodes)}); attempting recursive split."
                    )

                # Only recurse if the subset is actually smaller; otherwise we’d loop
                if len(cluster_nodes) < len(nodes):
                    node_clusters.extend(
                        HDBSCAN_Clustering.perform_clustering(
                            cluster_nodes,
                            embedding_model_name=embedding_model_name,
                            max_length_in_cluster=max_length_in_cluster,
                            tokenizer=tokenizer,
                            reduction_dimension=reduction_dimension,
                            threshold=threshold,
                            dr_method=dr_method,
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            metric=metric,
                            verbose=verbose,
                        )
                    )
                else:
                    if verbose:
                        logging.info(
                            "[HDBSCAN] Not recursing further: "
                            "cluster did not shrink; returning as single cluster."
                        )
                    node_clusters.append(cluster_nodes)
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters
