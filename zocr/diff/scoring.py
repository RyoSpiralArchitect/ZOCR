# -*- coding: utf-8 -*-
"""Lightweight heuristics for event confidence scoring."""
from __future__ import annotations

from typing import Optional


def estimate_confidence(
    similarity: Optional[float] = None,
    relative_delta: Optional[float] = None,
    numeric_delta: Optional[float] = None,
    text_changed: bool = True,
) -> float:
    """Return a 0..1 confidence score for a diff event.

    The heuristic blends textual similarity gaps with numeric magnitude. It is
    intentionally conservative so downstream queues can prioritise only
    high-signal deltas.
    """

    score = 0.35 if text_changed else 0.25

    if similarity is None:
        if text_changed:
            score += 0.2
    else:
        try:
            gap = max(0.0, min(1.0, 1.0 - float(similarity)))
        except (TypeError, ValueError):
            gap = 0.0
        score += gap * 0.45

    if relative_delta is not None:
        try:
            rel = min(4.0, abs(float(relative_delta)))
        except (TypeError, ValueError):
            rel = 0.0
        score += (rel / 4.0) * 0.4
    elif numeric_delta is not None:
        try:
            magnitude = abs(float(numeric_delta))
        except (TypeError, ValueError):
            magnitude = 0.0
        score += (magnitude / (magnitude + 100.0)) * 0.35 if magnitude else 0.0

    # clip to sensible range
    return round(max(0.05, min(1.0, score)), 3)
