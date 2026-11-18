"""Turning abstracts into vectors."""
import numpy as np


def tfidf_svd(texts, cfg):
    """Classic baseline: TF-IDF word weights, reduced to 100 dimensions with SVD."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    c = cfg["embeddings"]["tfidf"]
    tf = TfidfVectorizer(stop_words="english", max_features=c["max_features"],
                         min_df=c["min_df"], sublinear_tf=True)
    X = tf.fit_transform(texts)
    X = TruncatedSVD(n_components=c["svd_dims"], random_state=0).fit_transform(X)
    return normalize(X).astype(np.float32)


def pick_device(setting="auto"):
    """Use the GPU when there is one: CUDA (NVIDIA), MPS (Apple M-series), else CPU."""
    if setting != "auto":
        return setting
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def neural(name, texts, cfg):
    """Sentence-transformer embedding, unit length so cosine works well."""
    from sentence_transformers import SentenceTransformer

    device = pick_device(cfg["embeddings"].get("device", "auto"))
    print(f"[{name}] running on {device}", flush=True)
    model = SentenceTransformer(cfg["embeddings"]["neural"][name], device=device)
    X = model.encode(texts, batch_size=cfg["embeddings"]["batch_size"],
                     normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(X, dtype=np.float32)


def embed(name, texts, cfg):
    if name == "tfidf":
        return tfidf_svd(texts, cfg)
    return neural(name, texts, cfg)
