# Architecture

LyricMood has two parallel pipelines on top of a shared lyrics input. They use different feature representations on purpose — each is the right tool for its half of the job.

```
                          paste lyrics
                               │
                               ▼
                       clean_text(text)        (src/preprocess.py)
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
    TfidfVectorizer.transform           MiniLM.encode  (src/recommend.py)
    (src/features.py)                   (sentence-transformers)
              │                                 │
              ▼                                 ▼
     LogisticRegression                  L2-normalized 384-d
     C=1.0, balanced                     embedding
     (src/classify.py)                          │
              │                                 ▼
              ▼                         cosine vs corpus
     predict_proba ───────► pred ───►   filter to pred mood
              │             confidence  rank top-5
              ▼                                 │
     SHAP LinearExplainer                       ▼
     (src/explain.py)                   5 similar songs
              │                         (title, artist, sim)
              ▼
     top-10 words pushing
     toward / away from pred
```

## Why two pipelines?

| pipeline | feature | model | why this representation? |
|---|---|---|---|
| **classification** | TF-IDF (1,2)-grams, 20k vocab, sublinear TF | Logistic Regression, L2, balanced | Sparse + interpretable. SHAP `LinearExplainer` is exact for linear models — closed-form Shapley values. |
| **retrieval** | MiniLM `all-MiniLM-L6-v2`, 384-d dense | Cosine similarity vs. precomputed corpus | Captures *semantic* similarity that TF-IDF misses. A song mentioning "house" and one mentioning "walls falling down" can still cluster together. |

The two pipelines have **disjoint imports** — `src/features.py` only imports from sklearn; `src/recommend.py` only imports from sentence_transformers. No code overlap, no shared feature representation. This keeps the rubric items for *feature engineering* (TF-IDF) and *sentence embeddings for retrieval* (MiniLM) cleanly separated.

## Mood-label generation

Mood labels come from Spotify's audio features (valence + energy), not from human annotation:

```
energy
  ▲
  │  Angry         │   Hype
  ├────────────────┼───────────────
  │       (gap zone, dropped from training)
  ├────────────────┼───────────────
  │   Sad   │ Calm │   Romantic
  └────────────────┴───────────────► valence
       0.3       0.6
```

- Valence cuts at 0.3 / 0.6, energy cuts at 0.4 / 0.6.
- The middle "gap zone" gets dropped because those songs have ambiguous labels by construction (a small change in either scalar would flip the label).
- 5 mood classes after filtering: Hype, Romantic, Calm, Sad, Angry.

## Files

```
app/streamlit_app.py        # Streamlit UI — orchestrates the two pipelines
src/preprocess.py           # text cleaning, mood labels, gap-zone filter
src/features.py             # TF-IDF vectorizer (classification only)
src/classify.py             # split_data, train_model, evaluate_model
src/recommend.py            # MiniLM model loader + cosine retrieval
src/explain.py              # SHAP LinearExplainer wrapper
notebooks/01_eda.ipynb      # EDA + 2 preprocessing experiments
notebooks/02_modeling.ipynb # baselines, 7-config sweep, best model
notebooks/03_evaluation.ipynb # error analysis, edge cases, iterations, objective metrics
models/best_classifier.pkl  # LR + TF-IDF (joblib pickle, gitignored)
models/tfidf_vectorizer.pkl # fitted vectorizer (joblib pickle, gitignored)
models/corpus_embeddings.npy # 76,595 × 384 float32 (gitignored, ~112 MB)
data/processed/songs_labeled.csv # 76,595 mood-labeled songs (gitignored, ~150 MB)
```
