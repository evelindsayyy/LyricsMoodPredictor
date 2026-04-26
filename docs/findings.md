# Findings & known limitations

Things I learned about the data and the model that aren't in the rubric checklist but are worth flagging.

## Data-quality artifacts in SpotGenTrack

### Duplicated raw lyrics across multiple records

Discovered during retrieval testing — when I queried with one Hype song, the top-5 nearest neighbors all came back with cosine similarity exactly `0.437` and titles in different scripts (Hebrew, Swedish, Latin). Investigating, all 6 records shared **identical raw lyrics text** ("Freda and Barry sat one night..."). This is a Kaggle scraper artifact: the source dataset has placeholder text replicated across multiple track IDs.

**Impact:** retrieval can return apparent duplicates with sim=1.0 even though the song titles differ.
**Mitigation considered:** drop_duplicates(subset="lyrics") before embedding the corpus. Out of scope for this submission, documented as a finding.

### Stand-up comedy transcripts in the Sad mood class

When I ran a Sad query, the top-5 retrieval included *Buff* by Daniel Tosh and *"Sam opens the show with some relatable humor"* by Sam Morril. These are stand-up comedy sets that got scraped along with music tracks. Their valence/energy thresholds happen to put them in the Sad region.

**Impact:** the Sad class is a slightly noisier label than the others.
**Mitigation:** could filter by track-name patterns or by audio metadata if available.

### Word-count distribution is asymmetric across classes

The Sad class has a long tail of near-empty lyrics that no other class has. After gap-zone filtering, per-class word-count stats:

| class | n | median word count | 25th percentile |
|---|---|---|---|
| Hype | 45,824 | 261 | 171 |
| Romantic | 10,209 | 219 | 137 |
| Calm | 8,017 | 196 | 106 |
| Angry | 8,954 | 194 | 18 |
| **Sad** | **19,814** | **146.5** | **1** |

A quarter of Sad-labeled rows have **fewer than ~5 words** of lyrics — likely the same standup-comedy / scraper-artifact pollution noted above, plus instrumentals that got auto-tagged. By comparison, every other class has 25th-percentile word count well above 100.

**Impact on the 20-word floor in `notebooks/01_eda.ipynb`:** the floor drops 7,301 of the 19,814 Sad rows (~37%), versus ~10–25% for other classes. Net class-share shift caused by the floor: Hype +5.2pp / Sad -5.0pp / Romantic +0.8pp / Angry -1.0pp / Calm ~0.

**Drop-count by candidate floor:**

| floor | total dropped | of which Sad |
|---|---|---|
| 5 | 15,253 | 7,138 |
| 10 | 15,329 | 7,152 |
| **20 (current)** | **16,223** | **7,301** |
| 30 | 16,355 | 7,337 |
| 50 | 16,885 | 7,419 |
| 100 | 20,217 | 8,160 |

Most filtering happens at 1–5 words (15.2k of the 16.2k drops at floor=20). Anywhere in [10, 50] gives nearly the same kept-row count, so the choice of 20 is reasonable but not unique. **Going below 5 starts keeping clearly-broken rows; going above 100 starts dropping legitimate short songs.**

**Mitigation:** the current 20-word floor is defensible. A more principled fix would target the underlying causes (filter standup transcripts by name/genre, drop instrumentals by duration, etc.) rather than a blanket word-count rule.

## Pipeline limitations

### `clean_text()` strips non-Latin characters silently

Documented in the Task 10 edge-case section. The regex `[a-z\s]` keeps only lowercase Latin letters and spaces. Inputs in Korean, Japanese, Arabic, Hebrew, etc. get reduced to empty strings, which means TF-IDF returns a zero vector and the classifier just predicts the class prior (Hype).

**User-visible symptom:** Korean lyrics return Hype with low confidence (~0.26).
**Mitigation:** could detect this case and surface "I only speak English (and Latin-alphabet languages) right now." For now it's documented as a known limitation rather than fixed.

## Why the model "misses" on obvious-looking inputs

Per the Task 9 error analysis, the classifier's macro F1 (0.371) is bottlenecked by **label noise**, not by classifier capacity:

- The labels come from Spotify's audio features (valence + energy), but the inputs are lyrics.
- Audio-derived labels and lyric content can disagree — Latin dancehall and reggaeton are labeled "Hype" by audio (high valence + high energy) but their lyrics read as Romantic. This drives the Romantic→Hype confusion (27% of Romantic test rows).
- Threshold-derived labels are also fragile near the cut-off — a song at valence=0.61 gets a different label than one at 0.59.

**Diagnostic recipe** (also in the notebook): run SHAP on a misclassified example and check whether the top-5 SHAP words match the predicted (wrong) class or the true class. If they match the predicted, it's likely label noise; if they match the true class but the model still got it wrong, that's a real model error.

## Model performance ceiling

Two improvement iterations (`notebooks/03_evaluation.ipynb`) gave +0.012 and +0.021 macro F1 each. Returns are diminishing — the next plausible gains would require:

1. Better labels (manual annotation on a seed set, or audio-feature ensemble).
2. More expressive lyrics features (sentence-level embeddings as classifier input, not just TF-IDF).

Both are outside the scope of this submission and documented as future work in the evaluation notebook's takeaway section.
