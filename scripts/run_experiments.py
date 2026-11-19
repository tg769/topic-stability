"""Step 2: run every clustering we need and save the labels.

For each embedding and UMAP seed we reduce once, then cluster. Three seed
conditions are covered:
    joint        : UMAP seed = clusterer seed = s   (what the pilot did)
    umap_only    : UMAP seed varies, clusterer seed fixed
    cluster_only : UMAP seed fixed, clusterer seed varies
Plus sensitivity runs (number of clusters, HDBSCAN cluster size) and two
baselines (LDA topics and random labels). Category labels are used only to score.

Each (embedding, UMAP seed) pair is an independent job, so jobs run in
parallel on all CPU cores. Every job sets its own seeds, so the results are
the same whether you use 1 core or 10.

    python scripts/run_experiments.py --n 20000            # uses all cores (memory permitting)
    python scripts/run_experiments.py --n 20000 --jobs 1   # one at a time
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from topicstab import cluster as C
from topicstab.metrics import accuracy
from topicstab.utils import ROOT, Logger, load_config, run_dir


def quiet():
    # UMAP warns that a fixed seed turns off its own threading; we parallelize
    # across jobs instead, which keeps every run reproducible.
    warnings.filterwarnings("ignore", message="n_jobs value")
    warnings.filterwarnings("ignore", category=FutureWarning)


def pick_jobs(requested, n):
    """Use all cores, but not more jobs than memory allows.

    Full ward clustering needs memory that grows with n squared (about 3.3 GB at
    20k abstracts), so each job is given that much room plus 1 GB.
    """
    cores = os.cpu_count() or 1
    if requested and requested > 0:
        return min(requested, cores)
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return max(1, cores // 2)
    per_job = 1.0 + 8.5e-9 * n * n  # GB: ~1.2 at 5k, ~4.4 at 20k
    return max(1, min(cores, int(ram_gb * 0.7 // per_job)))


def make_record(true):
    def record(out, model, algo, param, u, c, lab, sec):
        key = f"{model}|{algo}|{param}|u{u}|c{c}"
        out.append((key, lab.astype(np.int16), dict(
            key=key, model=model, algorithm=algo, param=param, umap_seed=u, cluster_seed=c,
            n_clusters=C.n_clusters(lab), noise=float((lab == -1).mean()), seconds=sec,
            **accuracy(true, lab))))
    return record


def umap_job(m, u, tag, cfg, true, seeds, sens_seeds, algos, sensitivity):
    """Everything for one embedding and one UMAP seed."""
    quiet()
    k = cfg["clustering"]["k"]
    fixed = cfg["seeds"]["fixed_seed"]
    n = len(true)
    main_mcs = C.hdbscan_size(n, cfg["clustering"]["hdbscan_min_cluster_frac"])
    record = make_record(true)
    out, msg = [], ""

    zpath = ROOT / "embeddings" / "umap_cache" / f"umap_{m}_{tag}_u{u}.npy"
    if zpath.exists():
        Z = np.load(zpath)
    else:
        t = time.time()
        Z = C.reduce(np.load(ROOT / "embeddings" / f"emb_{m}_{tag}.npy"), cfg, u)
        np.save(zpath, Z)
        msg = f"UMAP {time.time() - t:.1f}s"

    # k-means and GMM: which clusterer seeds to run on this UMAP output
    cs = {u, fixed}                 # joint, umap_only
    if u == fixed:
        cs.update(seeds)            # cluster_only
    for algo in [a for a in algos if a in C.RANDOM_ALGOS]:
        fn = C.kmeans if algo == "kmeans" else C.gmm
        for c in sorted(cs):
            t = time.time()
            record(out, m, algo, f"k{k}", u, c, fn(Z, k, c), time.time() - t)
        if sensitivity and u in sens_seeds:
            for kk in cfg["sensitivity"]["k_values"]:
                if kk != k:
                    t = time.time()
                    record(out, m, algo, f"k{kk}", u, u, fn(Z, kk, u), time.time() - t)

    if "agglomerative" in algos:
        t = time.time()
        tree = C.ward_tree(Z, cfg)
        ks = [k] + ([kk for kk in cfg["sensitivity"]["k_values"] if kk != k]
                    if (sensitivity and u in sens_seeds) else [])
        for kk in ks:
            record(out, m, "agglomerative", f"k{kk}", u, "-", C.cut_tree(tree, kk, cfg), time.time() - t)
            t = time.time()
        del tree

    if "hdbscan" in algos:
        t = time.time()
        record(out, m, "hdbscan", f"mcs{main_mcs}", u, "-", C.hdbscan(Z, main_mcs), time.time() - t)
        if sensitivity and u in sens_seeds:
            for f in cfg["sensitivity"]["hdbscan_fracs"]:
                mcs = C.hdbscan_size(n, f)
                if mcs != main_mcs:
                    t = time.time()
                    record(out, m, "hdbscan", f"mcs{mcs}", u, "-", C.hdbscan(Z, mcs), time.time() - t)
    return m, u, msg, out


def lda_job(texts, true, k, s, cfg):
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    quiet()
    c = cfg["lda"]
    t = time.time()
    X = CountVectorizer(stop_words="english", max_features=c["max_features"], min_df=c["min_df"]).fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=k, max_iter=c["max_iter"], learning_method="online",
                                    random_state=s, n_jobs=1)
    lab = lda.fit_transform(X).argmax(axis=1)
    out = []
    make_record(true)(out, "lda", "lda", f"k{k}", "-", s, lab, time.time() - t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--models", default="tfidf,minilm,mpnet,specter")
    ap.add_argument("--algos", default=",".join(C.ALGOS))
    ap.add_argument("--seeds", type=int, default=None, help="override n_seeds (for quick checks)")
    ap.add_argument("--jobs", type=int, default=0, help="parallel jobs (0 = all cores, limited by memory)")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    quiet()

    cfg = load_config(args.config)
    tag = f"n{args.n}"
    out_dir = run_dir(tag)
    log = Logger(out_dir / "run_log.txt")
    log.versions()

    df = pd.read_csv(ROOT / "data" / f"sample_{tag}.csv")
    cats = cfg["data"]["categories"]
    true = pd.Categorical(df["primary"], categories=cats).codes
    n = len(df)
    k = cfg["clustering"]["k"]
    n_seeds = args.seeds or cfg["seeds"]["n_seeds"]
    seeds = list(range(n_seeds))
    sens_seeds = seeds[: min(cfg["seeds"]["sensitivity_seeds"], n_seeds)]
    algos = [a.strip() for a in args.algos.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    jobs = pick_jobs(args.jobs, n)
    log(f"n={n}, models={models}, algos={algos}, seeds={seeds}, k={k}, "
        f"hdbscan min_cluster_size={C.hdbscan_size(n, cfg['clustering']['hdbscan_min_cluster_frac'])}, "
        f"parallel jobs={jobs} (of {os.cpu_count()} cores)")
    (ROOT / "embeddings" / "umap_cache").mkdir(parents=True, exist_ok=True)

    tasks = [(m, u) for m in models for u in seeds]
    results = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(umap_job)(m, u, tag, cfg, true, seeds, sens_seeds, algos, not args.no_sensitivity)
        for m, u in tasks)
    collected = []
    for i, (m, u, msg, out) in enumerate(results, 1):
        collected.extend(out)
        log(f"[{i}/{len(tasks)}] {m} UMAP seed {u} done {msg}")

    if not args.no_baselines:
        texts = df["abstract"].tolist()
        for out in Parallel(n_jobs=min(jobs * 2, os.cpu_count() or 1))(
                delayed(lda_job)(texts, true, k, s, cfg) for s in seeds):
            collected.extend(out)
        rng = np.random.default_rng(0)
        rec = make_record(true)
        for s in seeds:
            rec(collected, "random", "random", f"k{k}", "-", s, rng.integers(0, k, n), 0.0)
        log("baselines done")

    # same order every time, whatever order the parallel jobs finished in
    collected.sort(key=lambda x: x[0])
    np.savez_compressed(out_dir / "labels.npz", **{key: lab for key, lab, _ in collected})
    pd.DataFrame([row for _, _, row in collected]).to_csv(out_dir / "runs.csv", index=False)
    log(f"saved {len(collected)} runs to results/{tag}")


if __name__ == "__main__":
    main()
