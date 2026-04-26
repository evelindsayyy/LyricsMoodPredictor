# Rubric mapping

Where each ML rubric item is satisfied in the codebase. Useful for the Gradescope self-assessment.

## ML column (15 items, capped at 73 pts)

| # | Rubric item | Pts | Evidence |
|---|---|---|---|
| 1 | Modular code design | 3 | All `src/` modules — each is < 100 lines, single-responsibility, with type hints + short docstrings. See `src/preprocess.py`, `features.py`, `classify.py`, `recommend.py`, `explain.py`. |
| 2 | 80/10/10 train/val/test split | 3 | `src/classify.py` `split_data()` + `notebooks/02_modeling.ipynb` (split sizes printed: 61,275 / 7,660 / 7,660 = 80/10/10). |
| 3 | Baseline model for comparison | 3 | `notebooks/02_modeling.ipynb` first analytical section: `majority_class_baseline()` and `random_weighted_baseline()` with results table (acc 0.546/0.361, macro F1 0.141/0.201). |
| 4 | Hyperparameter tuning, 3+ configs | 5 | `notebooks/02_modeling.ipynb` sweep: 4 LR configs (C ∈ {0.01, 0.1, 1, 10}) + 3 NB configs (α ∈ {0.01, 0.1, 1}) = 7 total in a summary table. |
| 5 | Preprocessing w/ 2+ data quality challenges | 7 | `src/preprocess.py` (`clean_text`, `derive_mood_labels`, `filter_gap_zone`, `get_class_weights`) + `notebooks/01_eda.ipynb` "preprocessing experiments" section. Two A/B tests with before/after macro F1: class-weight (Δ +0.014), gap-zone-drop (Δ +0.008). Saved to `results/preprocessing_impact.csv`. |
| 6 | Feature engineering: TF-IDF unigrams + bigrams | 5 | `src/features.py` `build_tfidf_vectorizer()` — unigrams + bigrams, 20k vocab, min_df=3, sublinear_tf. 32% of vocab is bigrams. Fit on the 76,595-song corpus. |
| 7 | Sentence embeddings for retrieval | 5 | `src/recommend.py` `load_embedding_model()` + `embed_corpus()` — frozen `all-MiniLM-L6-v2`, 384-d L2-normalized vectors, cosine similarity ranking, mood-filtered top-k. |
| 8 | Multiple architectures compared quantitatively | 7 | `notebooks/02_modeling.ipynb` "LR vs NB head-to-head" — same train/val/test split, 4 metrics (acc, macro F1, val + test). |
| 9 | Interpretability via SHAP | 7 | `src/explain.py` (`explain_prediction`, `plot_shap`) — uses `shap.LinearExplainer` (exact for linear models). 5 saved SHAP charts in `results/shap_*.png`, one per mood. Faithfulness validated with deletion test (62/100 class flips when top-5 SHAP words removed). |
| 10 | 3+ evaluation metrics | 3 | `notebooks/02_modeling.ipynb` test report: accuracy, macro F1, per-class precision, per-class recall (4 distinct metrics). |
| 11 | Error analysis with visualization + causal discussion | 7 | `notebooks/03_evaluation.ipynb` — confusion matrix heatmap (`results/confusion_matrix.png`), top confused pairs identified programmatically, 6 misclassified examples with SHAP, "why the model fails" causal paragraph (vocab overlap + majority-class gravity + label noise). |
| 12 | Edge case / OOD analysis | 5 | `notebooks/03_evaluation.ipynb` "edge case / OOD testing" — 4 categories (very short, non-English, nonsense, ironic-mismatched), 10 examples total, each with input / predicted / confidence / expected / actual + per-category discussion. |
| 13 | 2 improvement iterations w/ before/after | 5 | `notebooks/03_evaluation.ipynb` "improvement iterations" — Iter 1 (unigrams → bigrams, Δ +0.012 macro F1), Iter 2 (no weighting → balanced, Δ +0.021). Spec format: What tried / measured / changed / Before / After / Delta. |
| 14 | Deployed as a functional web application | 10 | `app/streamlit_app.py` — Streamlit app with mood prediction, SHAP chart, top-5 retrieval. Cached model loading via `@st.cache_resource`. Runs end-to-end with `streamlit run app/streamlit_app.py`. |
| 15 | Solo project credit | 10 | This project has only one contributor — see `README.md` Individual Contributions section. |

**Raw total: 85 → capped at 73 ML pts.**

## Bonus: objective-aligned evaluation

Each of the three project objectives in the README has at least one quantitative metric (see `notebooks/03_evaluation.ipynb` "evaluation directly tied to project objectives"):

| objective | metric | result |
|---|---|---|
| Predict mood from lyrics | test macro F1 vs. baselines | 0.371 vs. 0.141 — **2.5× lift** |
| SHAP explanations are faithful | mean confidence drop + class-flip rate when top-5 SHAP words deleted | 0.098 / **62%** flip rate |
| MiniLM embeddings carry mood signal | unfiltered mood-match precision@5 vs. random `Σ P(c)²` | 0.478 / 0.354 — **1.35× lift** |

## Documentation column (1 pt each)

| # | item | location |
|---|---|---|
| 1 | SETUP.md exists w/ step-by-step install | [SETUP.md](../SETUP.md) — 7 numbered steps + tl;dr quick-path |
| 2 | ATTRIBUTION.md exists w/ AI-generation info | [ATTRIBUTION.md](../ATTRIBUTION.md) — 5 sections, specific examples |
| 3 | requirements.txt accurate | [requirements.txt](../requirements.txt) — 8 packages, all used |
| 4 | README "What it Does" | [README.md](../README.md) |
| 5 | README "Quick Start" | [README.md](../README.md) |
| 6 | README "Video Links" | [README.md](../README.md) — repo paths + external mirrors |
| 7 | README "Evaluation" | [README.md](../README.md) |
| 8 | README "Individual Contributions" | [README.md](../README.md) — solo project |
