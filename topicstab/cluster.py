"""UMAP reduction and the four clustering algorithms.

Seeds are kept separate on purpose: `umap_seed` controls the reduction and
`cluster_seed` controls the clusterer. That lets us vary one while holding the
other still. Agglomerative and HDBSCAN have no randomness of their own, so for
them only the UMAP seed can change the result.
"""
import numpy as np

ALGOS = ["kmeans", "gmm", "agglomerative", "hdbscan"]
RANDOM_ALGOS = {"kmeans", "gmm"}


def reduce(X, cfg, umap_seed):
    import umap

    u = cfg["umap"]
    return umap.UMAP(n_components=u["n_components"], n_neighbors=u["n_neighbors"],
                     min_dist=u["min_dist"], metric=u["metric"],
                     random_state=umap_seed).fit_transform(X).astype(np.float32)


def kmeans(Z, k, seed):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=1, random_state=seed).fit_predict(Z)


def gmm(Z, k, seed):
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(n_components=k, n_init=1, random_state=seed).fit(Z).predict(Z)


def ward_tree(Z, cfg):
    """Build the ward tree once; it can then be cut at any number of clusters.

    Full ward needs memory that grows with n squared (about 3.3 GB at 20k), so
    above `agglomerative_graph_above` points we fall back to scikit-learn's ward
    with a k-nearest-neighbor graph, which stays small.
    """
    c = cfg["clustering"]
    if len(Z) <= c["agglomerative_graph_above"]:
        from scipy.cluster.hierarchy import linkage
        return ("scipy", linkage(Z, method="ward"))
    return ("graph", Z)


def cut_tree(tree, k, cfg):
    kind, obj = tree
    if kind == "scipy":
        from scipy.cluster.hierarchy import fcluster
        return fcluster(obj, t=k, criterion="maxclust") - 1
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph
    graph = kneighbors_graph(obj, n_neighbors=cfg["clustering"]["knn_neighbors"], include_self=False)
    return AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=graph).fit_predict(obj)


def hdbscan(Z, min_cluster_size):
    """Points HDBSCAN cannot place get -1 (noise); we score noise as its own group."""
    from sklearn.cluster import HDBSCAN
    return HDBSCAN(min_cluster_size=int(min_cluster_size)).fit_predict(Z)


def hdbscan_size(n, frac):
    return max(5, int(round(n * frac)))


def n_clusters(labels):
    s = set(np.unique(labels).tolist())
    return len(s) - (1 if -1 in s else 0)
