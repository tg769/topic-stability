"""Step 4: make every figure from the CSV files in results/.

    python scripts/make_figures.py --n 20000

Figures go to figures/. The size comparison uses whichever results/n*/
folders exist.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from topicstab.cluster import ALGOS
from topicstab.utils import ROOT, load_config

# fixed color per series, never by rank (checked for color-blind separation)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MODEL_COLOR = {"tfidf": SERIES[0], "minilm": SERIES[1], "mpnet": SERIES[2], "specter": SERIES[3]}
MODEL_NAME = {"tfidf": "TF-IDF", "minilm": "MiniLM", "mpnet": "MPNet", "specter": "SPECTER"}
ALGO_NAME = {"kmeans": "k-means", "gmm": "GMM", "agglomerative": "Agglomerative", "hdbscan": "HDBSCAN"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e4df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name):
    # one copy for the repo, one for the website (GitHub Pages serves docs/ only)
    for out in [ROOT / "figures", ROOT / "docs" / "figures"]:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / name)
    plt.close(fig)
    print("wrote", name)


def heatmaps(s, tag):
    models = list(dict.fromkeys(s.model))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.7 * len(models) + 1.8))
    for ax, (col, title) in zip(axes, [("stability_joint", "Stability: mean pairwise ARI across 10 seeds"),
                                       ("nmi", "Accuracy: NMI vs arXiv categories")]):
        M = np.array([[s[(s.model == m) & (s.algorithm == a)][col].iloc[0] for a in ALGOS] for m in models])
        ax.imshow(M, vmin=0.5, vmax=1, cmap="Blues", aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(4), [ALGO_NAME[a] for a in ALGOS])
        ax.set_yticks(range(len(models)), [MODEL_NAME[m] for m in models])
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        for i in range(len(models)):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if M[i, j] > 0.8 else INK)
    fig.suptitle("Darker is better. Scale starts at 0.5.", x=0.01, ha="left", fontsize=9, color=MUTED, y=0.02)
    save(fig, f"fig1_heatmaps_{tag}.png")


def seed_source(s, tag):
    models = list(dict.fromkeys(s.model))
    fig, axes = plt.subplots(1, len(models), figsize=(3 * len(models), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    conds = [("stability_umap_only", "Only UMAP seed changes", SERIES[0]),
             ("stability_cluster_only", "Only clusterer seed changes", SERIES[1]),
             ("stability_joint", "Both change", SERIES[2])]
    x = np.arange(4)
    for ax, m in zip(axes, models):
        g = s[s.model == m].set_index("algorithm").loc[ALGOS]
        for i, (col, lab, c) in enumerate(conds):
            ax.bar(x + (i - 1) * 0.27, g[col], width=0.25, color=c, label=lab)
        ax.set_xticks(x, ["k-means", "GMM", "Aggl.", "HDBSCAN"], fontsize=8)
        ax.set_title(MODEL_NAME[m], fontsize=10, loc="left")
        ax.set_ylim(0.5, 1.02)
    axes[0].set_ylabel("Mean pairwise ARI")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.07))
    fig.text(0.01, -0.04, "Agglomerative and HDBSCAN have no randomness of their own, so their "
             "'clusterer seed only' bar is 1.0 by definition.", fontsize=8, color=MUTED)
    save(fig, f"fig2_seed_source_{tag}.png")


def variance(vb, tag):
    fig, ax = plt.subplots(figsize=(6, 2.3))
    ax.barh(vb.source[::-1], vb.share[::-1] * 100, color=SERIES[0], height=0.55)
    for i, v in enumerate(vb.share[::-1] * 100):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Share of the spread in pairwise ARI (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    save(fig, f"fig3_variance_{tag}.png")


def hdbscan_sens(h, tag):
    if h.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for m, g in h.groupby("model", sort=False):
        for ax, col in zip(axes, ["stability", "nmi", "clusters"]):
            ax.plot(g.min_cluster_size, g[col], marker="o", ms=6, lw=2, color=MODEL_COLOR[m], label=MODEL_NAME[m])
    for ax, t in zip(axes, ["Stability (pairwise ARI)", "Accuracy (NMI)", "Clusters found"]):
        ax.set_xscale("log")
        ax.set_xticks(sorted(h.min_cluster_size.unique()), [str(v) for v in sorted(h.min_cluster_size.unique())])
        ax.minorticks_off()
        ax.set_xlabel("HDBSCAN min_cluster_size")
        ax.set_title(t, fontsize=10, loc="left")
    axes[2].axhline(8, color=MUTED, lw=1, ls="--")
    axes[2].text(axes[2].get_xlim()[0], 8.5, "8 categories", fontsize=8, color=MUTED)
    axes[0].legend(fontsize=8)
    save(fig, f"fig4_hdbscan_sensitivity_{tag}.png")


def k_sens(ks, tag):
    if ks.empty:
        return
    algos = [a for a in ALGOS if a in set(ks.algorithm)]
    fig, axes = plt.subplots(2, len(algos), figsize=(3.4 * len(algos), 5), sharey="row", sharex=True)
    for j, a in enumerate(algos):
        for m, g in ks[ks.algorithm == a].groupby("model", sort=False):
            axes[0, j].plot(g.k, g.stability, marker="o", ms=6, lw=2, color=MODEL_COLOR[m], label=MODEL_NAME[m])
            axes[1, j].plot(g.k, g.nmi, marker="o", ms=6, lw=2, color=MODEL_COLOR[m])
        axes[0, j].set_title(ALGO_NAME[a], fontsize=10, loc="left")
        axes[1, j].set_xlabel("Number of clusters k")
        axes[1, j].set_xticks(sorted(ks.k.unique()))
    axes[0, 0].set_ylabel("Stability (ARI)")
    axes[1, 0].set_ylabel("Accuracy (NMI)")
    axes[0, 0].legend(fontsize=8)
    save(fig, f"fig5_k_sensitivity_{tag}.png")


def baselines(s, b, tag):
    rows = [(r.method.upper() if r.method == "lda" else "Random labels", r.stability, r.nmi) for r in b.itertuples()]
    for m in dict.fromkeys(s.model):
        g = s[(s.model == m) & (s.algorithm == "kmeans")].iloc[0]
        rows.append((f"{MODEL_NAME[m]} + k-means", g.stability_joint, g.nmi))
    names, stab, nmi = zip(*rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(names) + 1), sharey=True)
    y = np.arange(len(names))
    for ax, vals, t in zip(axes, [stab, nmi], ["Stability (pairwise ARI)", "Accuracy (NMI)"]):
        ax.barh(y, vals, color=[MUTED if i < len(b) else SERIES[0] for i in range(len(names))], height=0.55)
        for i, v in enumerate(vals):
            ax.text(max(v, 0) + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
        ax.set_title(t, fontsize=10, loc="left")
        ax.set_xlim(0, 1.1)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    save(fig, f"fig6_baselines_{tag}.png")


def consensus_fig(c, tag):
    if c.empty:
        return
    labels = [f"{MODEL_NAME[r.model]} + {ALGO_NAME[r.algorithm]}" for r in c.itertuples()]
    y = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(c) + 1))
    ax.hlines(y, c.single_stability, c.consensus_stability, color=GRID, lw=2)
    ax.scatter(c.single_stability, y, s=40, color=SERIES[1], label="Single run", zorder=3)
    ax.scatter(c.consensus_stability, y, s=40, color=SERIES[0], label="Consensus of 5 runs", zorder=3)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Stability (pairwise ARI)")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, f"fig7_consensus_{tag}.png")


def mixing(mx, tag, cats):
    g = mx[mx.setup == "most_accurate"]
    if g.empty:
        return
    M = g.pivot(index="cat_a", columns="cat_b", values="same_cluster_rate").loc[cats, cats].to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
    ax.grid(False)
    ax.set_xticks(range(len(cats)), cats, rotation=45, ha="right")
    ax.set_yticks(range(len(cats)), cats)
    for i in range(len(cats)):
        for j in range(len(cats)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.5 else INK)
    r = g.iloc[0]
    ax.set_title(f"How often two papers share a cluster\n({MODEL_NAME[r.model]} + {ALGO_NAME[r.algorithm]})",
                 fontsize=10, loc="left")
    save(fig, f"fig8_category_mixing_{tag}.png")


def flips(fl, tag, cats):
    if fl.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(len(cats))
    for i, (setup, g) in enumerate(fl.groupby("setup", sort=False)):
        g = g.set_index("category").loc[cats]
        r = g.iloc[0]
        ax.bar(x + (i - 0.5) * 0.4, g.share_ever_flipped * 100, width=0.38, color=SERIES[i],
               label=f"{setup.replace('_', ' ')}: {MODEL_NAME[r.model]} + {ALGO_NAME[r.algorithm]}")
    ax.set_xticks(x, cats, rotation=30, ha="right")
    ax.set_ylabel("% of abstracts that\nchanged cluster at least once")
    ax.legend(fontsize=8)
    save(fig, f"fig9_flips_{tag}.png")


def size_compare():
    found = []
    for d in sorted((ROOT / "results").glob("n*")):
        f = d / "summary.csv"
        if f.exists():
            s = pd.read_csv(f)
            s["n"] = int(d.name[1:])
            found.append(s)
    if len(found) < 2:
        return
    s = pd.concat(found)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, a in zip(axes, ALGOS):
        for m, g in s[s.algorithm == a].groupby("model", sort=False):
            g = g.sort_values("n")
            ax.plot(g.n, g.stability_joint, marker="o", ms=6, lw=2, color=MODEL_COLOR[m], label=MODEL_NAME[m])
        ax.set_xscale("log")
        ns = sorted(s.n.unique())
        ax.set_xticks(ns, [f"{v // 1000}k" for v in ns])
        ax.minorticks_off()
        ax.set_title(ALGO_NAME[a], fontsize=10, loc="left")
        ax.set_xlabel("Abstracts")
    axes[0].set_ylabel("Stability (pairwise ARI)")
    axes[0].legend(fontsize=8)
    save(fig, "fig10_size_comparison.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config()
    tag = f"n{args.n}"
    d = ROOT / "results" / tag

    def read(name):
        f = d / name
        if f.exists() and f.stat().st_size > 1:
            return pd.read_csv(f)
        return pd.DataFrame()

    s = read("summary.csv")
    heatmaps(s, tag)
    seed_source(s, tag)
    variance(read("variance_breakdown.csv"), tag)
    hdbscan_sens(read("sensitivity_hdbscan.csv"), tag)
    k_sens(read("sensitivity_k.csv"), tag)
    baselines(s, read("baselines.csv"), tag)
    consensus_fig(read("consensus.csv"), tag)
    mixing(read("category_mixing.csv"), tag, cfg["data"]["categories"])
    flips(read("flip_by_category.csv"), tag, cfg["data"]["categories"])
    size_compare()


if __name__ == "__main__":
    main()
