"""Small checks for the scoring code. Run with: python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from topicstab import metrics as M
from topicstab.data import primary_of


def test_primary_of_handles_hf_list_format():
    assert primary_of(["cs.CL cs.LG"]) == "cs.CL"
    assert primary_of("math.PR math.ST") == "math.PR"
    assert primary_of(None) is None
    assert primary_of([]) is None


def test_pairwise_ari_ignores_cluster_names():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([5, 5, 3, 3, 9, 9])  # same groups, different names
    assert M.pairwise_ari([a, b]) == [1.0]


def test_align_and_flip_rate():
    ref = np.array([0, 0, 1, 1])
    renamed = np.array([1, 1, 0, 0])
    assert (M.align_to(ref, renamed) == ref).all()
    moved = np.array([0, 1, 1, 1])  # second item moved
    f = M.flip_rate([ref, renamed, moved])
    assert f[1] > 0 and f[0] == 0


def test_category_mixing_diagonal_is_one_for_perfect_clusters():
    true = np.array([0, 0, 1, 1])
    mix = M.category_mixing(true, true.copy(), 2)
    assert np.allclose(np.diag(mix), 1) and np.allclose(mix[0, 1], 0)


def test_variance_breakdown_sums_to_one():
    import pandas as pd
    rng = np.random.default_rng(0)
    rows = [dict(model=m, algorithm=a, ari=rng.random()) for m in "xy" for a in "pq" for _ in range(6)]
    parts = M.variance_breakdown(pd.DataFrame(rows))
    assert abs(sum(p["share"] for p in parts) - 1) < 1e-9
