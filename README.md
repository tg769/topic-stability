# How Stable Are Unsupervised Topics?

**Embedding choice vs. clustering algorithm on arXiv abstracts**
CS 472 final project, Tanmay Ghate and Aryan Sharma (NAU, Fall 2026)

Topic models built by clustering text embeddings give different answers when you
rerun them with a different random seed. We measure how much they change, and
what causes it: the embedding model, the clustering algorithm, or the random
seed inside UMAP versus inside the clusterer.

## What we do

1. Sample arXiv abstracts evenly from 8 categories.
2. Embed them four ways: TF-IDF + SVD, MiniLM, MPNet, SPECTER.
3. Reduce to 5 dimensions with UMAP.
4. Cluster four ways: k-means, GMM, agglomerative (ward), HDBSCAN.
5. Repeat with 10 seeds under three conditions:
   - only the UMAP seed changes,
   - only the clusterer seed changes,
   - both change.
6. Score **stability** (pairwise ARI between runs) and **accuracy** (NMI against
   the arXiv categories). The categories are never used for clustering.

Also included: LDA and random-label baselines, sensitivity to the number of
clusters and to HDBSCAN's minimum cluster size, a consensus-clustering test, a
5k vs 20k size comparison, and an error analysis (which categories get mixed,
which abstracts change cluster).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or with conda: `conda env create -f environment.yml`.

## Run it

```bash
# 1. sample + embed (needs internet; about 40 min on a MacBook Air M4 for 20k)
python scripts/prepare_data.py --n 20000

# 2. all clustering runs (runs in parallel on all CPU cores; add --jobs 1 to run one at a time)
python scripts/run_experiments.py --n 20000

# 3. result tables
python scripts/analyze.py --n 20000

# 4. figures
python scripts/make_figures.py --n 20000

# tests for the scoring code
python -m pytest tests
```

The 5k version is the same with `--n 5000 --models tfidf,minilm,mpnet` in steps 1 and 2.
Every setting (categories, UMAP settings, seeds, k, HDBSCAN rule) is in
[`config.yaml`](config.yaml). Seeds are 0 to 9; seed 0 is the one held still in
the one-factor conditions.

## Repository layout

```
config.yaml            all settings and seeds
topicstab/             reusable code
  data.py              load and sample abstracts
  embed.py             TF-IDF and sentence-transformer embeddings
  cluster.py           UMAP and the four clustering algorithms
  metrics.py           ARI, NMI, bootstrap intervals, flip rate, category mixing
  utils.py             config, paths, logging
scripts/               one script per step (see "Run it")
results/n<N>/          CSV tables, run log, saved cluster labels
figures/               every figure, made by scripts/make_figures.py
data/                  sample id lists and data notes
tests/                 checks for the scoring code
docs/                  project website (GitHub Pages)
```

## License

MIT, see [LICENSE](LICENSE). The arXiv metadata copy we use is CC0.
