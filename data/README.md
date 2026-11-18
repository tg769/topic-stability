# Data

Source: [`gfissore/arxiv-abstracts-2021`](https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021)
on Hugging Face, a CC0 copy of the arXiv metadata (papers through 2021). We read
the parquet shards listed in `config.yaml` and keep papers whose primary (first
listed) category is one of our 8 categories.

- `ids_n<N>.csv` lists the exact papers in each sample (arXiv id + category).
  It is committed so anyone can check the sample.
- `sample_n<N>.csv` holds the abstracts too. It is not committed (size); running
  `scripts/prepare_data.py --n <N>` rebuilds the same file, because sampling uses
  `data_seed` from `config.yaml`.
- Embeddings are saved to `embeddings/` (not committed) by the same script.
