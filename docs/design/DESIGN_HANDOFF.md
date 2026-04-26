# LyricMood — UI Handoff

Port the design in `LyricMood Minimal.html` to `app/streamlit_app.py`. **UI-only task.** Keep all current ML behavior — model loading, SHAP explanations, MiniLM recommendations — unchanged.

---

## Scope

- **Change:** the Streamlit UI (layout, typography, colors, component styling).
- **Do NOT change:** `src/preprocess.py`, `src/features.py`, `src/classify.py`, `src/explain.py`, `src/recommend.py`, any notebook, or any model artifact.
- **Do NOT:** retrain, re-split, add new Python dependencies, or alter the `load_everything()` caching strategy.

---

## Reference files

- **`lyricmood.css`** — the design system as a standalone stylesheet. **This is the source of truth.** Copy it into `app/static/lyricmood.css` and load it in Streamlit with the snippet below. All tokens, component styles, and animations live here. Do not hand-retype styles from the HTML.
- **`streamlit_snippet.py`** — drop-in helper: `inject_design_system()` loads the CSS + Google Fonts once, and `set_mood_accent(mood)` re-injects a tiny `:root` override to swap `--accent` after each prediction.
- **`LyricMood Minimal.html`** — interactive visual reference only. Open it in a browser to see layout, rhythm, and component behavior. Do not port its markup verbatim into Python — rebuild the DOM using Streamlit primitives + `st.markdown(..., unsafe_allow_html=True)` blocks, styled by the CSS classes already defined in `lyricmood.css`.

The CSS defines these class hooks you should reuse when rendering HTML blocks from Python:
`.brand`, `.prompt`, `.paper`, `.mood-word`, `.conf`, `.probs .stack`, `.probs .legend`, `.shap`, `.shap-row`, `.shap-bar`, `.similar .row`. Match these names exactly so the styles apply.

---

## Design tokens

**Fonts** (load via Google Fonts CDN in the injected CSS):
- `Fraunces` — serif, used for the wordmark, mood words, section headings, prompts. Weight 300, italic variant used heavily.
- `Inter` — sans, used for body UI text and buttons. Weights 300/400/500.
- `IBM Plex Mono` — monospace, used for small labels, captions, tabular numbers. Weights 400/500.

**Palette (oklch):**
```css
--paper:    oklch(0.975 0.006 85);   /* warm off-white background */
--paper-2:  oklch(0.955 0.008 85);   /* card / textarea fill */
--rule:     oklch(0.88  0.008 80);   /* hairline borders */
--ink:      oklch(0.22  0.010 60);   /* primary text, button fill */
--ink-2:    oklch(0.44  0.010 60);   /* secondary text */
--ink-3:    oklch(0.62  0.008 70);   /* tertiary / label text */
```

**Mood accents** — one is active at a time, picked from the prediction:
```css
--hype:     oklch(0.72 0.13 55);    /* warm amber */
--romantic: oklch(0.70 0.12 18);    /* dusty rose */
--calm:     oklch(0.74 0.08 195);   /* pale teal */
--sad:      oklch(0.60 0.09 255);   /* slate blue */
--angry:    oklch(0.58 0.15 28);    /* burnt red */
```

The accent drives: the brand dot, positive-word chip fill, stacked-bar segment for the predicted mood, confidence bar, and song-row hover color.

---

## Layout

- Single centered column, ~720px max width.
- 72px top padding, 32px side padding, 140px bottom padding.
- Generous vertical rhythm: 56–72px gaps between major sections.
- No sidebars, no page chrome beyond the brand row.
- Use `st.set_page_config(layout="centered")`.

---

## Key UI elements (top → bottom)

1. **Brand row.** Tiny colored dot (the active mood accent) + "Lyric*Mood*" in Fraunces italic, 22px. Right-aligned mono caption: "a quiet reader for feeling".
2. **Prompt.** Fraunces italic, 30px, two lines: *"Paste a verse. I'll tell you what it feels like."* The phrase "what it feels like" is roman (non-italic), slightly muted.
3. **Textarea card.** Soft `--paper-2` fill, 1px `--rule` border, 6px radius. Fraunces 300, 18px text inside. Bottom strip with sample chips (hype / romantic / calm / sad / angry) on the left, live word count on the right.
4. **Action row.** Pill-shaped dark button "Read the mood →" + ghost "clear" link. Keyboard hint `⌘ ↵` at the right edge.
5. **Loading state.** Italic serif "reading the room" + three pulsing accent dots.
6. **Reading block** (only after prediction):
   - Mood headline: 96px Fraunces italic word + a colored mood dot floated top-right of the word.
   - Confidence: large Fraunces number (40px) + "confidence" label + thin 2px bar filling to the confidence %.
   - Probabilities: a single 6px stacked bar across all 5 moods (predicted segment full-opacity, others at 0.35), plus a 5-column legend with mood name + percentage.
   - "Why [mood]" section: a proper SHAP horizontal bar chart in a soft `--paper-2` card. Each row has the word (right-aligned, Fraunces 16px), a centered-axis bar track (positive bars extend right in the mood accent color, negative bars extend left in muted gray), and the numeric SHAP value (mono, ±0.00). Rows sorted descending by value so positive contributors sit at the top. Axis labels "pushes away / 0 / pushes toward" above; numeric scale ticks (−max, −max/2, 0, +max/2, +max) below.
   - "In a similar key" section: 5 rows with index / title (Fraunces 22px) / artist / cosine score. Hover shifts the row 6px right and colors the title.

---

## Streamlit implementation notes

- Inject all design CSS once at app start via `st.markdown(<style>...</style>, unsafe_allow_html=True)`.
- The mood headline, chips, stacked bar, and song list should be rendered as raw HTML blocks — Streamlit's native `st.metric`, `st.bar_chart`, and `st.dataframe` won't match the aesthetic.
- Keep `st.text_area` for input but restyle it by targeting `div[data-baseweb="textarea"]` and `textarea` in the injected CSS.
- Use `st.button` for the primary action, restyled via `button[kind="primary"]` selectors.
- For the accent swap: regenerate the `<style>` block (or inject a small secondary one) after prediction, with `--accent` set to the predicted mood's oklch value.
- Keep `@st.cache_resource` on `load_everything()` exactly as-is.
- Sample chips: wire them to pre-fill the textarea with short original samples (see `SAMPLES` dict in the HTML file — reuse those strings, they're original and safe).

---

## Data flow (unchanged)

```
paste lyrics → clean_text → tfidf.transform → clf.predict / predict_proba
            ↘ explain_prediction (SHAP top-10)
            ↘ minilm.encode → cosine vs corpus_embeddings → top-5 filtered by predicted mood
```

Map model outputs to the UI:
- `pred` → mood headline, accent color.
- `probs.max()` → confidence number + bar.
- `probs` across `clf.classes_` → stacked bar + legend.
- `exp["shap_values"]` + `exp["feature_names"]` → SHAP bar chart (take top-10 by `abs(shap_value)`, sort descending by signed value; positive values push toward predicted mood, negative push away).
- `recs` rows → song list (title / artist / similarity).

---

## Acceptance criteria

- Same lyric input produces the same predicted mood and the same top-5 songs as the current app.
- Design matches `LyricMood Minimal.html` within Streamlit's constraints (pixel parity is NOT required; typographic hierarchy, palette, rhythm, and component shapes ARE).
- No new entries in `requirements.txt`.
- `streamlit run app/streamlit_app.py` starts cleanly with no console warnings from the injected HTML/CSS.
- Confidence bar, stacked probabilities, chip row, and song list all render with the correct active mood accent.

---

## Out of scope

- URL/File input tabs (shown disabled in the mock).
- The Tweaks panel (design-tool only; not part of the shipped app).
- Dark mode.
- Mobile layout beyond what Streamlit's centered layout gives you for free.
