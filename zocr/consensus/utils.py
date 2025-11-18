"""Light-weight helpers shared by consensus runtime primitives."""
from __future__ import annotations

from typing import TypeVar

_T = TypeVar("_T")


def clamp(x: _T, lo: _T, hi: _T) -> _T:
    """Clamp ``x`` between ``lo`` and ``hi`` while preserving the original type."""

    return lo if x < lo else hi if x > hi else x


__all__ = ["clamp"]
