"""Step 1: sample abstracts and make the embeddings.

This is the only step that needs the internet (Hugging Face) and, for the
neural models, a decent machine. Everything after it works from the saved files.

    python scripts/prepare_data.py --n 20000
    python scripts/prepare_data.py --n 50000 --models minilm
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from topicstab import data as D
from topicstab.embed import embed
from topicstab.utils import ROOT, Logger, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="total abstracts (split evenly across categories)")
    ap.add_argument("--models", default="tfidf,minilm,mpnet,specter")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tag = f"n{args.n}"
    (ROOT / "data").mkdir(exist_ok=True)
    emb_dir = ROOT / "embeddings"
    emb_dir.mkdir(exist_ok=True)
    log = Logger(ROOT / "data" / f"prepare_{tag}.log")
    log.versions()

    sample_path = ROOT / "data" / f"sample_{tag}.csv"
    if sample_path.exists():
        import pandas as pd
        df = pd.read_csv(sample_path)
        log(f"using existing sample {sample_path.name} ({len(df)} rows)")
    else:
        t = time.time()
        raw = D.load_hf(cfg)
        df = D.sample_even(raw, cfg["data"]["categories"], args.n,
                           cfg["data"]["data_seed"], cfg["data"]["min_chars"])
        df.to_csv(sample_path, index=False)
        df[["id", "primary"]].to_csv(ROOT / "data" / f"ids_{tag}.csv", index=False)
        log(f"sampled {len(df)} abstracts from {cfg['data']['hf_repo']} in {time.time() - t:.0f}s")

    texts = df["abstract"].tolist()
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        out = emb_dir / f"emb_{m}_{tag}.npy"
        if out.exists():
            log(f"[{m}] already done, skipping")
            continue
        t = time.time()
        X = embed(m, texts, cfg)
        np.save(out, X)
        log(f"[{m}] {X.shape} in {time.time() - t:.0f}s")
    log("done")


if __name__ == "__main__":
    main()
