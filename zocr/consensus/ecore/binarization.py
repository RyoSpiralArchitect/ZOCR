"""Image binarization helpers extracted from the consensus runtime."""
from __future__ import annotations

from typing import Tuple

try:  # pragma: no cover - numpy is optional at runtime
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy is unavailable
    np = None  # type: ignore


__all__ = [
    "_box_mean",
    "_binarize_pure",
    "_estimate_slant_slope",
    "_shear_rows_binary",
    "_suppress_diagonal_bridges",
    "_apply_italic_guard",
]


def _box_mean(gray, k: int = 31):
    if np is None:
        raise RuntimeError("NumPy required")
    k = max(3, int(k))
    k |= 1
    r = k // 2
    H, W = gray.shape
    pad = np.pad(gray.astype(np.int64), ((1, 0), (1, 0)), mode="constant")
    ii = pad.cumsum(0).cumsum(1)
    y0 = np.clip(np.arange(H) - r, 0, H)
    y1 = np.clip(np.arange(H) + r + 1, 0, H)
    x0 = np.clip(np.arange(W) - r, 0, W)
    x1 = np.clip(np.arange(W) + r + 1, 0, W)
    Y0, X0 = np.meshgrid(y0, x0, indexing="ij")
    Y1, X1 = np.meshgrid(y1, x1, indexing="ij")
    S = ii[Y1, X1] - ii[Y0, X1] - ii[Y1, X0] + ii[Y0, X0]
    area = (Y1 - Y0) * (X1 - X0)
    area[area == 0] = 1
    return (S / area).astype(np.float32)


def _binarize_pure(gray, k: int = 31, c: int = 10):
    m = _box_mean(gray, k)
    return (gray < (m - c)).astype(np.uint8) * 255


def _estimate_slant_slope(binary: "np.ndarray") -> float:
    if np is None:
        return 0.0
    try:
        coords = np.argwhere(np.asarray(binary, dtype=np.uint8) > 0)
    except Exception:
        return 0.0
    if coords.size == 0:
        return 0.0
    y = coords[:, 0].astype(np.float64)
    x = coords[:, 1].astype(np.float64)
    if y.size < 128:
        return 0.0
    y_centered = y - float(np.mean(y))
    x_centered = x - float(np.mean(x))
    denom = float(np.dot(y_centered, y_centered))
    if denom <= 1e-6:
        return 0.0
    slope = float(np.dot(x_centered, y_centered) / denom)
    return float(max(-0.5, min(0.5, slope)))


def _shear_rows_binary(binary01: "np.ndarray", slope: float) -> "np.ndarray":
    if np is None:
        raise RuntimeError("NumPy required")
    arr = (np.asarray(binary01, dtype=np.uint8) > 0).astype(np.uint8)
    H, W = arr.shape
    out = np.zeros_like(arr)
    center = H / 2.0
    max_shift = int(min(W * 0.25, abs(slope) * H * 1.2 + 2))
    for y in range(H):
        shift = int(round((y - center) * slope))
        shift = max(-max_shift, min(max_shift, shift))
        row = arr[y]
        if shift > 0:
            out[y, shift:] = np.maximum(out[y, shift:], row[: W - shift])
        elif shift < 0:
            out[y, : W + shift] = np.maximum(out[y, : W + shift], row[-shift:])
        else:
            out[y] = np.maximum(out[y], row)
    return out


def _suppress_diagonal_bridges(binary01: "np.ndarray") -> "np.ndarray":
    if np is None:
        raise RuntimeError("NumPy required")
    arr = (np.asarray(binary01, dtype=np.uint8) > 0).astype(np.uint8)
    if arr.size == 0:
        return arr
    padded = np.pad(arr, 1, mode="constant")
    left = padded[1:-1, :-2] > 0
    right = padded[1:-1, 2:] > 0
    up = padded[:-2, 1:-1] > 0
    down = padded[2:, 1:-1] > 0
    diag = (
        padded[:-2, :-2]
        | padded[:-2, 2:]
        | padded[2:, :-2]
        | padded[2:, 2:]
    ) > 0
    horiz = left | right
    vert = up | down
    mask = diag & (~horiz) & (~vert)
    trimmed = arr.copy()
    trimmed[mask] = 0
    return trimmed


def _apply_italic_guard(binary: "np.ndarray") -> "np.ndarray":
    if np is None:
        return binary
    arr = (np.asarray(binary, dtype=np.uint8) > 0).astype(np.uint8)
    arr = _suppress_diagonal_bridges(arr)
    slope = _estimate_slant_slope(arr)
    if abs(slope) < 0.08:
        return (arr * 255).astype(np.uint8)
    deskewed = _shear_rows_binary(arr, -slope)
    combined = np.maximum(arr, deskewed)
    combined = _suppress_diagonal_bridges(combined)
    return (combined * 255).astype(np.uint8)
