"""
Classifier training + evaluation helpers.

Thin wrappers around sklearn, mostly so I get the same split and the same
metric bundle in 02_modeling.ipynb and later in 03_evaluation.ipynb without
copy-pasting code between them.

AI attribution: implementation by Claude (Anthropic) based on my specification.
I chose the 80/10/10 stratified split (random_state=42), the four-metric
evaluation bundle (accuracy, macro F1, per-class P/R), and the function
signatures. Claude wrote the bodies. See ../ATTRIBUTION.md for the full
breakdown.
"""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def split_data(X, y, val_size: float = 0.1, test_size: float = 0.1, random_state: int = 42):
    """Three-way stratified split — returns X_train, X_val, X_test, y_train, y_val, y_test."""
    # carve off the test set first
    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )
    # val is a fraction of the *remaining* rows, not the original
    val_rel = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=val_rel, stratify=y_trval, random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, model_class, **params):
    """Instantiate `model_class(**params)`, fit, return the fitted model."""
    model = model_class(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y) -> dict:
    """Accuracy, macro F1, per-class precision + recall on (X, y)."""
    preds = model.predict(X)
    classes = sorted(set(y))
    p = precision_score(y, preds, labels=classes, average=None, zero_division=0)
    r = recall_score(y, preds, labels=classes, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro"),
        "per_class_precision": dict(zip(classes, p)),
        "per_class_recall": dict(zip(classes, r)),
    }
