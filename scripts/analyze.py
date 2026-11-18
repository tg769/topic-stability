"""Step 3: turn saved labels into result tables.

    python scripts/analyze.py --n 20000

Writes CSV files to results/n<N>/:
    summary.csv              stability (3 seed conditions) and accuracy per setup
    stability_pairs.csv      every pairwise ARI behind summary.csv
    variance_breakdown.csv   how the spread in stability splits up
    sensitivity_k.csv        number of clusters 6 / 8 / 10 / 12
    sensitivity_hdbscan.csv  HDBSCAN minimum cluster size
    baselines.csv            LDA topics and random labels
    consensus.csv            consensus of 5 runs vs a single run
    category_mixing.csv      which categories get grouped together
    flip_by_category.csv     how often abstracts change cluster, per category
    flip_examples.csv        abstracts that change cluster the most
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from topicstab import cluster as C
from topicstab import metrics as M
from topicstab.utils import ROOT, load_config, run_dir


def consensus(label_list, k):
    """Combine runs: one-hot encode each run's clusters side by side, then k-means."""
    from sklearn.cluster import KMeans
    blocks = []
    for lab in label_list:
        vals, inv = np.unique(lab, return_inverse=True)
        oh = np.zeros((len(lab), len(vals)), dtype=np.float32)
        oh[np.arange(len(lab)), inv] = 1
        blocks.append(oh)
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(np.hstack(blocks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tag = f"n{args.n}"
    out = run_dir(tag)
    runs = pd.read_csv(out / "runs.csv", dtype=str)
    for col in ["n_clusters", "noise", "nmi", "ari_vs_labels", "seconds"]:
        runs[col] = runs[col].astype(float)
    L = np.load(out / "labels.npz")
    df = pd.read_csv(ROOT / "data" / f"sample_{tag}.csv")
    cats = cfg["data"]["categories"]
    true = pd.Categorical(df["primary"], categories=cats).codes
    n, k = len(df), cfg["clustering"]["k"]
    fixed = str(cfg["seeds"]["fixed_seed"])
    main_mcs = C.hdbscan_size(n, cfg["clustering"]["hdbscan_min_cluster_frac"])

    def main_param(algo):
        return f"mcs{main_mcs}" if algo == "hdbscan" else f"k{k}"

    def get(sel):
        sel = sel.sort_values(["umap_seed", "cluster_seed"], key=lambda s: s.map(lambda x: int(x) if x.isdigit() else -1))
        return [L[key] for key in sel["key"]], sel

    grid = runs[~runs["model"].isin(["lda", "random"])]
    models = list(dict.fromkeys(grid["model"]))
    algos = [a for a in C.ALGOS if a in set(grid["algorithm"])]

    # ---------- stability under the three seed conditions ----------
    pair_rows, summ = [], []
    for m in models:
        for a in algos:
            g = grid[(grid.model == m) & (grid.algorithm == a) & (grid.param == main_param(a))]
            if a in C.RANDOM_ALGOS:
                joint = g[g.umap_seed == g.cluster_seed]
                uonly = g[g.cluster_seed == fixed]
                conly = g[g.umap_seed == fixed]
            else:
                joint = uonly = g
                conly = None
            lj, joint = get(joint)
            lu, _ = get(uonly)
            aj = M.pairwise_ari(lj)
            au = M.pairwise_ari(lu)
            if conly is not None:
                lc, _ = get(conly)
                ac = M.pairwise_ari(lc)
            else:
                ac = [1.0]  # no randomness of its own: same UMAP output, same answer
            for cond, vals in [("joint", aj), ("umap_only", au), ("cluster_only", ac)]:
                for v in vals:
                    pair_rows.append(dict(model=m, algorithm=a, condition=cond, ari=v))
            lo, hi = M.bootstrap_mean_pairwise(lj)
            summ.append(dict(
                model=m, algorithm=a, param=main_param(a),
                stability_joint=np.mean(aj), ci_low=lo, ci_high=hi,
                stability_umap_only=np.mean(au), stability_cluster_only=np.mean(ac),
                clusterer_is_deterministic=a not in C.RANDOM_ALGOS,
                nmi=joint["nmi"].mean(), nmi_sd=joint["nmi"].std(),
                ari_vs_labels=joint["ari_vs_labels"].mean(),
                clusters_mean=joint["n_clusters"].mean(), clusters_min=joint["n_clusters"].min(),
                clusters_max=joint["n_clusters"].max(), noise=joint["noise"].mean(),
            ))
    pairs = pd.DataFrame(pair_rows)
    summary = pd.DataFrame(summ)
    pairs.to_csv(out / "stability_pairs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    vb = pd.DataFrame(M.variance_breakdown(pairs[pairs.condition == "joint"]))
    vb.to_csv(out / "variance_breakdown.csv", index=False)

    # ---------- sensitivity ----------
    sens_seeds = [str(s) for s in range(cfg["seeds"]["sensitivity_seeds"])]
    rows = []
    for m in models:
        for a in [x for x in algos if x != "hdbscan"]:
            for kk in cfg["sensitivity"]["k_values"]:
                g = grid[(grid.model == m) & (grid.algorithm == a) & (grid.param == f"k{kk}")
                         & grid.umap_seed.isin(sens_seeds)]
                if a in C.RANDOM_ALGOS:
                    g = g[g.umap_seed == g.cluster_seed]
                if len(g) < 2:
                    continue
                labs, g = get(g)
                rows.append(dict(model=m, algorithm=a, k=kk, stability=np.mean(M.pairwise_ari(labs)), nmi=g["nmi"].mean()))
    pd.DataFrame(rows).to_csv(out / "sensitivity_k.csv", index=False)

    rows = []
    if "hdbscan" in algos:
        for m in models:
            g0 = grid[(grid.model == m) & (grid.algorithm == "hdbscan") & grid.umap_seed.isin(sens_seeds)]
            for p in sorted(g0.param.unique(), key=lambda s: int(s[3:])):
                labs, g = get(g0[g0.param == p])
                if len(labs) < 2:
                    continue
                rows.append(dict(model=m, min_cluster_size=int(p[3:]), stability=np.mean(M.pairwise_ari(labs)),
                                 nmi=g["nmi"].mean(), clusters=g["n_clusters"].mean(), noise=g["noise"].mean()))
    pd.DataFrame(rows).to_csv(out / "sensitivity_hdbscan.csv", index=False)

    # ---------- baselines ----------
    rows = []
    for b in ["lda", "random"]:
        g = runs[runs.model == b]
        if len(g):
            labs, g = get(g)
            rows.append(dict(method=b, stability=np.mean(M.pairwise_ari(labs)), nmi=g["nmi"].mean(),
                             ari_vs_labels=g["ari_vs_labels"].mean()))
    pd.DataFrame(rows).to_csv(out / "baselines.csv", index=False)

    # ---------- consensus ----------
    halves = [[str(s) for s in h] for h in cfg["consensus"]["halves"]]
    rows = []
    for m in models:
        for a in algos:
            g = grid[(grid.model == m) & (grid.algorithm == a) & (grid.param == main_param(a))]
            if a in C.RANDOM_ALGOS:
                g = g[g.umap_seed == g.cluster_seed]
            parts = []
            for h in halves:
                labs, _ = get(g[g.umap_seed.isin(h)])
                if len(labs) < len(h):
                    break
                parts.append(consensus(labs, k))
            if len(parts) < 2:
                continue
            row = summary[(summary.model == m) & (summary.algorithm == a)].iloc[0]
            rows.append(dict(model=m, algorithm=a, single_stability=row.stability_joint,
                             consensus_stability=M.pairwise_ari(parts)[0], single_nmi=row.nmi,
                             consensus_nmi=np.mean([M.accuracy(true, p)["nmi"] for p in parts])))
    pd.DataFrame(rows).to_csv(out / "consensus.csv", index=False)

    # ---------- error analysis ----------
    # focus on the most accurate setup and the least stable setup
    best = summary.sort_values("nmi", ascending=False).iloc[0]
    worst = summary.sort_values("stability_joint").iloc[0]
    mix_rows, flip_rows, ex_rows = [], [], []
    for tagname, row in [("most_accurate", best), ("least_stable", worst)]:
        g = grid[(grid.model == row.model) & (grid.algorithm == row.algorithm) & (grid.param == row.param)]
        if row.algorithm in C.RANDOM_ALGOS:
            g = g[g.umap_seed == g.cluster_seed]
        labs, _ = get(g)
        mix = np.mean([M.category_mixing(true, l, len(cats)) for l in labs], axis=0)
        for i, ci in enumerate(cats):
            for j, cj in enumerate(cats):
                mix_rows.append(dict(setup=tagname, model=row.model, algorithm=row.algorithm,
                                     cat_a=ci, cat_b=cj, same_cluster_rate=mix[i, j]))
        flips = M.flip_rate(labs)
        for i, c in enumerate(cats):
            flip_rows.append(dict(setup=tagname, model=row.model, algorithm=row.algorithm,
                                  category=c, mean_flip=flips[true == i].mean(),
                                  share_ever_flipped=(flips[true == i] > 0).mean()))
        for idx in np.argsort(-flips)[:10]:
            ex_rows.append(dict(setup=tagname, model=row.model, algorithm=row.algorithm, id=df.id[idx],
                                category=df.primary[idx], flip=flips[idx], abstract_start=df.abstract[idx][:220]))
    pd.DataFrame(mix_rows).to_csv(out / "category_mixing.csv", index=False)
    pd.DataFrame(flip_rows).to_csv(out / "flip_by_category.csv", index=False)
    pd.DataFrame(ex_rows).to_csv(out / "flip_examples.csv", index=False)

    pd.set_option("display.width", 200)
    print(summary.round(3).drop(columns=["param"]).to_string(index=False))
    print(vb.round(3).to_string(index=False))
    print(pd.read_csv(out / "baselines.csv").round(3).to_string(index=False))
    print(pd.read_csv(out / "consensus.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
