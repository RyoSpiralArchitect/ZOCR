"""Utility helpers shared by consensus runtime primitives."""
from __future__ import annotations

from typing import TypeVar, cast

_T = TypeVar("_T")

try:  # pragma: no cover - numpy is optional and expensive to import in tests
    import numpy as _np
except Exception:  # pragma: no cover - gracefully handle stripped environments
    _np = None  # type: ignore


class OptionalDependencyError(RuntimeError):
    """Raised when a consensus helper requires an optional dependency."""

    def __init__(self, dependency: str, feature: str) -> None:
        hint = f"Install zocr[{dependency}] or provide {dependency} to use {feature}."
        super().__init__(f"Missing optional dependency '{dependency}' for {feature}. {hint}")


def clamp(x: _T, lo: _T, hi: _T) -> _T:
    """Clamp ``x`` between ``lo`` and ``hi`` while preserving the original type."""

    return lo if x < lo else hi if x > hi else x


def has_numpy() -> bool:
    """Return ``True`` when NumPy is importable for runtime helpers."""

    return _np is not None


def require_numpy(feature: str) -> "_np":
    """Return the shared NumPy module or raise :class:`OptionalDependencyError`."""

    if _np is None:
        raise OptionalDependencyError("numpy", feature)
    return cast("_np", _np)


__all__ = ["OptionalDependencyError", "clamp", "has_numpy", "require_numpy"]
