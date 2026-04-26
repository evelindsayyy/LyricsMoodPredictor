"""
SHAP explainability for the lyric classifier.

Why SHAP + LinearExplainer specifically:
- the classifier is logistic regression on TF-IDF, so the whole thing is linear
  and LinearExplainer gives exact shapley values (not approximations).
- per-word contributions make "why did it pick this mood" easy to show in the
  Streamlit UI and useful for the error analysis in task 9.

LinearExplainer needs a background distribution to compute baselines. I'm
passing a masker built from a sample of training rows so the shapley values
are interpretable as "this word pushed the score up/down relative to the
average training song".

AI attribution: implementation by Claude (Anthropic) based on my specification.
I chose SHAP LinearExplainer (because the model is linear and this gives
exact Shapley values), the background-sample-from-training strategy, the
input-vocabulary filter, and the function signatures. Claude wrote the
function bodies, including the version-defensive isinstance branch handling
LinearExplainer's list-vs-3D-array output across shap versions (which came
out of a debugging session after I hit the shape mismatch).
See ../ATTRIBUTION.md for the full breakdown.
"""

import matplotlib.pyplot as plt
import numpy as np
import shap


def explain_prediction(model, vectorizer, text: str, top_k: int = 10, background=None) -> dict:
    """Run SHAP LinearExplainer on `text`, return top-k positive/negative words for the predicted class.

    `background` is an optional (n, d) sparse matrix of TF-IDF rows used as the
    baseline. If None, the input row itself is used — fine for a quick check
    but training-set background gives more meaningful signs.
    """
    X = vectorizer.transform([text])
    pred_class = model.predict(X)[0]
    class_idx = list(model.classes_).index(pred_class)

    bg = background if background is not None else X
    explainer = shap.LinearExplainer(model, bg)
    shap_values = explainer.shap_values(X)

    # multiclass LR: shap_values can come back as list-of-arrays OR a 3-D array
    # depending on shap version. handle both.
    if isinstance(shap_values, list):
        sv_row = shap_values[class_idx][0]
    elif np.asarray(shap_values).ndim == 3:
        sv_row = np.asarray(shap_values)[0, :, class_idx]
    else:
        sv_row = np.asarray(shap_values)[0]

    feature_names = vectorizer.get_feature_names_out()

    # only rank words that actually appeared in the input — otherwise the
    # "top negatives" are just noise from words the song never used
    present = X.nonzero()[1]
    word_vals = [(str(feature_names[i]), float(sv_row[i])) for i in present]
    word_vals.sort(key=lambda kv: kv[1])

    return {
        "predicted_class": pred_class,
        "shap_values": sv_row,
        "feature_names": feature_names,
        "top_positive": word_vals[-top_k:][::-1],  # biggest positive first
        "top_negative": word_vals[:top_k],          # biggest negative first
    }


def plot_shap(shap_values, feature_names, predicted_class: str, top_k: int = 10, ax=None):
    """Horizontal bar chart of the top-k words driving the prediction (by |SHAP|)."""
    sv = np.asarray(shap_values)
    fn = np.asarray(feature_names)

    present = np.where(sv != 0)[0]
    if len(present) == 0:
        raise ValueError("no nonzero SHAP values to plot")

    # rank by absolute contribution, keep top-k
    order = np.argsort(np.abs(sv[present]))[-top_k:]
    idx = present[order]
    words = fn[idx]
    vals = sv[idx]

    colors = ["tab:green" if v > 0 else "tab:red" for v in vals]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.barh(range(len(words)), vals, color=colors)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel(f"SHAP value (positive → pushes toward {predicted_class!r})")
    ax.set_title(f"top-{top_k} words for prediction: {predicted_class}")
    return ax
