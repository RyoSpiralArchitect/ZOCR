"""Morphological helpers leveraged by the consensus runtime."""
from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy is unavailable
    np = None  # type: ignore

__all__ = ["_dilate_binary_rect"]


def _dilate_binary_rect(bw: "np.ndarray", wx: int, wy: int) -> "np.ndarray":
    if np is None:
        raise RuntimeError("NumPy required")
    H, W = bw.shape
    wx = max(1, int(wx))
    r = wx // 2
    k = 2 * r + 1
    s = np.pad(bw, ((0, 0), (r, r)), mode="constant")
    s2 = np.pad(s, ((0, 0), (1, 0)), mode="constant")
    csum = s2.cumsum(axis=1)
    right = np.arange(W) + k
    left = np.arange(W)
    win = csum[:, right] - csum[:, left]
    h = (win > 0).astype(np.uint8)

    wy = max(1, int(wy))
    r = wy // 2
    k = 2 * r + 1
    s = np.pad(h, ((r, r), (0, 0)), mode="constant")
    s2 = np.pad(s, ((1, 0), (0, 0)), mode="constant")
    csum = s2.cumsum(axis=0)
    bottom = np.arange(H) + k
    top = np.arange(H)
    win = csum[bottom, :] - csum[top, :]
    v = (win > 0).astype(np.uint8)
    return v
