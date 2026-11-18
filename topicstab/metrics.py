"""Scores: stability between runs, accuracy against the arXiv categories."""
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def accuracy(true, pred):
    return dict(nmi=normalized_mutual_info_score(true, pred), ari_vs_labels=adjusted_rand_score(true, pred))


def pairwise_ari(label_list):
    """ARI for every pair of runs. 1.0 means two runs made identical groups."""
    return [adjusted_rand_score(a, b) for a, b in combinations(label_list, 2)]


def bootstrap_mean_pairwise(label_list, n_boot=500, seed=0):
    """95% interval for mean pairwise ARI, resampling runs (not pairs).

    Pairs share runs, so resampling pairs would make the interval too narrow.
    """
    rng = np.random.default_rng(seed)
    m = len(label_list)
    full = np.ones((m, m))
    for i, j in combinations(range(m), 2):
        full[i, j] = full[j, i] = adjusted_rand_score(label_list[i], label_list[j])
    means = []
    for _ in range(n_boot):
        idx = np.unique(rng.integers(0, m, m))
        if len(idx) < 2:
            continue
        sub = full[np.ix_(idx, idx)]
        means.append(sub[np.triu_indices(len(idx), 1)].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def align_to(ref, labels):
    """Rename clusters in `labels` to best match `ref` (Hungarian matching).

    Clusters with no partner get -2, so they always count as a change.
    """
    a = np.unique(ref)
    b = np.unique(labels)
    table = np.zeros((len(b), len(a)))
    bi = {v: i for i, v in enumerate(b)}
    ai = {v: i for i, v in enumerate(a)}
    np.add.at(table, (np.vectorize(bi.get)(labels), np.vectorize(ai.get)(ref)), 1)
    rows, cols = linear_sum_assignment(-table)
    mapping = {b[r]: a[c] for r, c in zip(rows, cols)}
    return np.array([mapping.get(x, -2) for x in labels])


def flip_rate(label_list):
    """Per abstract: share of runs where it lands outside its usual cluster."""
    ref = label_list[0]
    aligned = np.stack([align_to(ref, l) for l in label_list])
    flips = np.zeros(aligned.shape[1])
    for j in range(aligned.shape[1]):
        vals, counts = np.unique(aligned[:, j], return_counts=True)
        flips[j] = 1 - counts.max() / aligned.shape[0]
    return flips


def category_mixing(true, pred, n_cats):
    """How often two papers from categories i and j end up in the same cluster."""
    clusters = np.unique(pred)
    P = np.zeros((n_cats, len(clusters)))
    for ci, c in enumerate(clusters):
        mask = pred == c
        P[:, ci] = np.bincount(true[mask], minlength=n_cats)
    P = P / P.sum(axis=1, keepdims=True)
    return P @ P.T


def variance_breakdown(pairs):
    """Split the spread in pairwise ARI into parts (balanced design, descriptive only)."""
    y = pairs["ari"].to_numpy()
    grand = y.mean()
    total = ((y - grand) ** 2).sum()

    def ss(cols):
        return sum(len(g) * (g["ari"].mean() - grand) ** 2 for _, g in pairs.groupby(cols))

    s_m, s_a, s_cell = ss("model"), ss("algorithm"), ss(["model", "algorithm"])
    parts = [
        ("embedding model", s_m),
        ("clustering algorithm", s_a),
        ("model x algorithm", s_cell - s_m - s_a),
        ("seed to seed (within a setup)", total - s_cell),
    ]
    return [{"source": s, "share": v / total if total > 0 else float("nan")} for s, v in parts]
