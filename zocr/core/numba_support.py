# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

"""Numba availability helpers."""
from __future__ import annotations

try:  # pragma: no cover - optional dependency
    from numba import njit, prange  # type: ignore
    try:
        from numba import atomic as _numba_atomic  # type: ignore
    except Exception:  # pragma: no cover - fallback when atomic is missing
        _numba_atomic = None  # type: ignore
    HAS_NUMBA = True
except Exception:  # pragma: no cover - fallback when numba is unavailable
    def njit(*a, **k):  # type: ignore
        def deco(fn):
            return fn

        return deco

    def prange(n):  # type: ignore
        return range(n)

    _numba_atomic = None  # type: ignore
    HAS_NUMBA = False

HAS_NUMBA_PARALLEL = bool(_numba_atomic)

if _numba_atomic is None:
    class _AtomicStub:
        @staticmethod
        def add(arr, idx, val):
            arr[idx] += val

    atomic = _AtomicStub()
else:  # pragma: no cover - real numba atomic implementation
    atomic = _numba_atomic

__all__ = ["HAS_NUMBA", "HAS_NUMBA_PARALLEL", "njit", "prange", "atomic"]
