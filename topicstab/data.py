"""Loading and sampling arXiv abstracts."""
import re

import pandas as pd


def primary_of(x):
    """The first listed category is the paper's primary category.

    The Hugging Face copy stores categories as a list holding one
    space-separated string, e.g. ["cs.CL cs.LG"], so lists are joined first.
    """
    if x is None:
        return None
    if not isinstance(x, str):
        try:
            x = " ".join(str(p) for p in x)
        except TypeError:
            return None
    parts = x.split()
    return parts[0] if parts else None


def clean(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def load_hf(cfg):
    """Read the parquet shards and keep papers whose primary category we study."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    d = cfg["data"]
    keep = []
    for i in d["shards"]:
        path = hf_hub_download(
            repo_id=d["hf_repo"],
            filename=f"default/train/{i:04d}.parquet",
            repo_type="dataset",
            revision=d["hf_revision"],
        )
        df = pq.read_table(path, columns=["id", "abstract", "categories"]).to_pandas()
        df["primary"] = df["categories"].map(primary_of)
        keep.append(df[df["primary"].isin(d["categories"])][["id", "primary", "abstract"]])
    return pd.concat(keep, ignore_index=True)


def sample_even(df, cats, n_total, seed, min_chars=200):
    """Same number of abstracts from every category, then shuffle."""
    per_cat = n_total // len(cats)
    df = df.copy()
    df["abstract"] = df["abstract"].map(clean)
    df = df[df["abstract"].str.len() >= min_chars].drop_duplicates(subset="id")
    parts = []
    for c in cats:
        g = df[df["primary"] == c]
        if len(g) < per_cat:
            raise SystemExit(f"{c} has only {len(g)} usable abstracts, need {per_cat}.")
        parts.append(g.sample(n=per_cat, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
