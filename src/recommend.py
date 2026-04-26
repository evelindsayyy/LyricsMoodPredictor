"""
Retrieval pipeline — MiniLM sentence embeddings + cosine similarity.

This is the "find me more songs like this one" half of the app. Separate from
features.py on purpose:
- TF-IDF (features.py) drives the classifier + SHAP explanation because it's
  sparse, interpretable, and works with a linear model.
- MiniLM (here) drives the retrieval because dense embeddings capture semantic
  similarity better than word overlap. Two songs that are both heartbreak-y
  but use different vocab (e.g. "cry" vs. "tears") will still come out close.

`all-MiniLM-L6-v2` is 22M params, ~80MB, outputs 384-d vectors, and is fast
enough to embed the full ~80k corpus on a laptop CPU in under 15 min.

AI attribution: implementation by Claude (Anthropic) based on my specification.
I chose MiniLM as the embedding model, the disk-cache strategy, the mood-
filter design, and the function signatures. The L2-normalize-at-embed-time
convention came from a concept-explanation conversation with Claude (small
optimization that turns retrieval into a dot product). Claude wrote the
function bodies. See ../ATTRIBUTION.md for the full breakdown.
"""

import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Frozen pretrained MiniLM from sentence-transformers."""
    return SentenceTransformer(model_name)


def embed_corpus(model: SentenceTransformer, texts, cache_path: str = "models/corpus_embeddings.npy") -> np.ndarray:
    """Embed `texts` with the model, cache to disk as .npy. Reload from cache if it exists."""
    if os.path.exists(cache_path):
        print(f"loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"embedding {len(texts)} texts — grab a coffee")
    emb = model.encode(
        list(texts),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so cosine sim = dot product later
    )

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.save(cache_path, emb)
    return emb


def recommend(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    predicted_mood: str,
    top_k: int = 5,
) -> pd.DataFrame:
    """Filter corpus rows to `predicted_mood`, rank by cosine similarity, return top-k."""
    mood_mask = (metadata_df["mood"].values == predicted_mood)
    if mood_mask.sum() == 0:
        # no rows to pull from — return an empty frame with the right columns
        return metadata_df.head(0).assign(similarity=[])

    # both embeddings were L2-normalized at embed time, so dot product = cosine
    q = query_embedding.reshape(-1)
    sims = corpus_embeddings[mood_mask] @ q

    # argsort ascending → flip for top-k descending
    k = min(top_k, len(sims))
    top_idx = np.argsort(sims)[-k:][::-1]

    out = metadata_df[mood_mask].iloc[top_idx].copy()
    out["similarity"] = sims[top_idx]
    return out
