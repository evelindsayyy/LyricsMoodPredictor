"""Preprocessing utilities for LyricMood.

This module owns mood-label derivation from Spotify's valence/energy audio
features, plus text cleaning and class-weight helpers used by the classifier.

The mood taxonomy follows Russell's circumplex model: two axes (valence = how
positive, energy = how intense) give a 2D mood space. We discretize it into
five moods plus a central "gap zone" that's too ambiguous to label cleanly.
"""

from __future__ import annotations

import re
import string

import numpy as np
import pandas as pd

# --- Mood thresholds (valence = positivity, energy = intensity, both in [0,1])
# Chosen so the 3x3 grid of (low / mid / high) cells partitions cleanly into
# 5 mood quadrants + a middle gap zone we discard during training.
VALENCE_LOW = 0.3
VALENCE_HIGH = 0.6
ENERGY_LOW = 0.4
ENERGY_HIGH = 0.6

MOODS = ["Hype", "Romantic", "Calm", "Sad", "Angry"]


def derive_mood_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `mood` column to df based on valence/energy thresholds.

    Mapping (valence x energy grid):
        v>=0.6, e>=0.6          -> Hype        (happy + energetic)
        v>=0.6, e<0.6           -> Romantic    (happy + mellow)
        0.3<=v<0.6, e<0.4       -> Calm        (neutral + low energy)
        0.3<=v<0.6, 0.4<=e<0.6  -> (gap zone — labeled None)
        0.3<=v<0.6, e>=0.6      -> Hype        (neutral-leaning but intense)
        v<0.3, e>=0.6           -> Angry       (negative + intense)
        v<0.3, e<0.6            -> Sad         (negative + low/mid energy)

    Returns a new DataFrame; input is not mutated.
    """
    out = df.copy()
    v = out["valence"].astype(float)
    e = out["energy"].astype(float)

    mood = pd.Series(index=out.index, dtype="object")

    # High valence
    mood[(v >= VALENCE_HIGH) & (e >= ENERGY_HIGH)] = "Hype"
    mood[(v >= VALENCE_HIGH) & (e < ENERGY_HIGH)] = "Romantic"

    # Low valence
    mood[(v < VALENCE_LOW) & (e >= ENERGY_HIGH)] = "Angry"
    mood[(v < VALENCE_LOW) & (e < ENERGY_HIGH)] = "Sad"

    # Mid valence: Calm at low energy, Hype at high energy, gap in middle
    mid_v = (v >= VALENCE_LOW) & (v < VALENCE_HIGH)
    mood[mid_v & (e < ENERGY_LOW)] = "Calm"
    mood[mid_v & (e >= ENERGY_HIGH)] = "Hype"
    # mid valence AND mid energy remains NaN (gap zone)

    out["mood"] = mood
    return out


def is_gap_zone(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: True where valence/energy fall in the ambiguous middle."""
    v = df["valence"].astype(float)
    e = df["energy"].astype(float)
    return (
        (v >= VALENCE_LOW) & (v < VALENCE_HIGH)
        & (e >= ENERGY_LOW) & (e < ENERGY_HIGH)
    )


# --- Text cleaning (used by Task 2 + modeling) ------------------------------

# Strip Genius-style section headers like [Chorus], [Verse 1: Artist]
_SECTION_HEADER = re.compile(r"\[[^\]]*\]")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, drop section headers and punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = _SECTION_HEADER.sub(" ", text)
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def get_class_weights(y: pd.Series) -> dict:
    """Balanced class weights: inversely proportional to class frequency."""
    counts = y.value_counts()
    n = len(y)
    k = len(counts)
    return {cls: n / (k * cnt) for cls, cnt in counts.items()}
