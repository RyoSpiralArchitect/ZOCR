"""Column-structure helpers extracted from the consensus runtime."""
from __future__ import annotations

import math
from statistics import median
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

from .utils import clamp, has_numpy

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

__all__ = [
    "_dp_means_1d",
    "_btree_partition",
    "_btree_column_centers",
    "_find_projection_valleys",
    "_align_row_band_centers",
    "_vertical_vote_boundaries",
    "_smooth_per_column",
]


def _dp_means_1d(points, lam, iters: int = 3):
    if not points:
        return []
    centers = [float(points[0])]
    for x in points[1:]:
        d = min(abs(x - c) for c in centers)
        if d > lam:
            centers.append(float(x))
    for _ in range(iters):
        buckets = {i: [] for i in range(len(centers))}
        for x in points:
            j = min(range(len(centers)), key=lambda i: abs(x - centers[i]))
            buckets[j].append(x)
        for i, xs in buckets.items():
            if xs:
                centers[i] = sum(xs) / len(xs)
    return sorted(centers)


def _btree_partition(values: Sequence[float], min_bucket: int, max_depth: int) -> List[List[float]]:
    ordered = sorted(float(v) for v in values if math.isfinite(v))
    if not ordered:
        return []
    buckets: List[List[float]] = []

    def _split(bucket: List[float], depth: int) -> None:
        if len(bucket) <= max(1, min_bucket) or depth >= max_depth:
            buckets.append(bucket)
            return
        mid_idx = len(bucket) // 2
        pivot = bucket[mid_idx]
        left = [v for v in bucket if v <= pivot]
        right = [v for v in bucket if v > pivot]
        if not left or not right:
            buckets.append(bucket)
            return
        _split(left, depth + 1)
        _split(right, depth + 1)

    _split(ordered, 0)
    return buckets


def _btree_column_centers(points: Sequence[float], min_bucket: int = 6, max_depth: int = 5) -> List[float]:
    buckets = _btree_partition(points, min_bucket=max(1, min_bucket), max_depth=max(1, max_depth))
    centers: List[float] = []
    for bucket in buckets:
        if not bucket:
            continue
        centers.append(float(median(bucket)))
    centers.sort()
    deduped: List[float] = []
    for value in centers:
        if not deduped or abs(value - deduped[-1]) > 1.0:
            deduped.append(value)
        else:
            deduped[-1] = (deduped[-1] + value) * 0.5
    return deduped


def _find_projection_valleys(values: "np.ndarray", threshold: float, min_gap: int) -> List[int]:
    if values.size <= 2:
        return []
    valleys: List[int] = []
    for idx in range(1, values.size - 1):
        if values[idx] > threshold:
            continue
        if values[idx] > values[idx - 1] or values[idx] >= values[idx + 1]:
            continue
        if valleys and idx - valleys[-1] < min_gap:
            if values[idx] < values[valleys[-1]]:
                valleys[-1] = idx
            continue
        valleys.append(idx)
    return valleys


def _align_row_band_centers(
    row_bands: List[Tuple[int, int]], height: int, med_h: float
) -> List[Tuple[int, int]]:
    if len(row_bands) <= 1:
        return row_bands
    centers = [0.5 * (y0 + y1) for (y0, y1) in row_bands]
    diffs = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    target_gap = float(max(2.0, median([abs(d) for d in diffs]) if diffs else med_h))
    aligned = [centers[0]]
    clamp_delta = med_h * 0.6
    for idx in range(1, len(centers)):
        expected = aligned[-1] + target_gap
        delta = clamp(centers[idx] - expected, -clamp_delta, clamp_delta)
        aligned.append(expected + delta * 0.35)
    refined: List[Tuple[int, int]] = []
    for idx, center in enumerate(aligned):
        span = max(2.0, row_bands[idx][1] - row_bands[idx][0])
        start = int(round(center - span / 2.0))
        end = int(round(center + span / 2.0))
        start = max(0, start)
        end = min(height, max(start + 2, end))
        refined.append((start, end))
    return refined


def _vertical_vote_boundaries(
    binary: "np.ndarray", max_candidates: int = 16, min_gap: int = 6
) -> List[int]:
    """Return likely vertical column separators from a binary mask."""

    if not has_numpy():
        return []
    import numpy as np

    try:
        arr = np.asarray(binary, dtype=np.uint8)
    except Exception:
        return []
    if arr.ndim != 2 or arr.size == 0:
        return []
    ink = arr.sum(axis=0).astype(np.float32)
    if ink.size == 0:
        return []
    norm = ink / float(max(1.0, ink.max()))
    blank_score = 1.0 - norm
    window = max(3, min_gap // 2 * 2 + 1)
    kernel = np.ones(window, dtype=np.float32)
    smooth = np.convolve(blank_score, kernel, mode="same")
    smooth = smooth / float(window)
    valleys = _find_projection_valleys(smooth, float(np.percentile(smooth, 25)), min_gap)
    candidates = sorted({int(v) for v in valleys})
    if not candidates:
        return []
    trimmed: List[int] = []
    for c in candidates:
        if trimmed and c - trimmed[-1] < min_gap:
            continue
        trimmed.append(c)
    return trimmed[:max_candidates]


def _smooth_per_column(
    candidates_by_row: List[List[int]],
    W: int,
    lam: float,
    H_sched: int = 1000,
    *,
    thomas_solver: Optional[Callable[["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"], "np.ndarray"]] = None,
) -> List[int]:
    if not has_numpy():
        return [0, W]
    import numpy as np
    R = len(candidates_by_row)
    if R == 0:
        return [0, W]
    counts = [len(r) for r in candidates_by_row if len(r) > 0]
    if not counts:
        return [0, W]
    K = max(1, int(median(counts)))
    s = [[None] * K for _ in range(R)]
    for r, row in enumerate(candidates_by_row):
        row = sorted(row)
        if len(row) == 0:
            continue
        if len(row) >= K:
            idxs = np.linspace(0, len(row) - 1, K).round().astype(int).tolist()
            for k, ii in enumerate(idxs):
                s[r][k] = float(row[ii])
        else:
            xx = np.linspace(0, K - 1, len(row))
            for k in range(K):
                ii = int(np.argmin(np.abs(xx - k)))
                s[r][k] = float(row[ii])
    lam_eff = lam * (float(max(1, R)) / (H_sched / 20.0)) ** 0.7

    def smooth_1d(y, lam_eff, passes: int = 2):
        n = len(y)
        if n <= 2:
            return y[:]
        x = np.array(y, dtype=np.float64)
        for _ in range(passes):
            a = -lam_eff * np.ones(n - 1)
            b = np.ones(n) + 2 * lam_eff
            c = -lam_eff * np.ones(n - 1)
            b[0] = 1 + lam_eff
            b[-1] = 1 + lam_eff
            if thomas_solver is not None:
                x = thomas_solver(a, b, c, x)
            else:
                x = _thomas_solve(a, b, c, x)
        return x.tolist()

    rows_smoothed = []
    for k in range(K):
        series = [s[r][k] if s[r][k] is not None else (W * 0.5) for r in range(R)]
        rows_smoothed.append(smooth_1d(series, lam_eff))
    bounds = [0]
    for k in range(K):
        vals = [rows_smoothed[k][r] for r in range(R)]
        bounds.append(int(np.median(vals)))
    bounds.append(W)
    min_gap = 4
    cleaned = [bounds[0]]
    for x in bounds[1:]:
        if x - cleaned[-1] < min_gap:
            x = cleaned[-1] + min_gap
        cleaned.append(min(W, max(0, x)))
    cleaned[-1] = W
    return cleaned


def _thomas_solve(a, b, c, d):
    """Solve a tridiagonal system using the Thomas algorithm."""
    cp = c.copy()
    bp = b.copy()
    dp = d.copy()
    for i in range(1, len(d)):
        m = a[i - 1] / bp[i - 1]
        bp[i] -= m * cp[i - 1]
        dp[i] -= m * dp[i - 1]
    d[-1] = dp[-1] / bp[-1]
    for i in range(len(d) - 2, -1, -1):
        d[i] = (dp[i] - cp[i] * d[i + 1]) / bp[i]
    return d
