# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

"""Compatibility helpers for optional dependencies."""

from __future__ import annotations

from typing import Any


def optional_numpy(consumer: str) -> Any:
    """Return ``numpy`` or a stub that raises a helpful error on access."""

    try:  # pragma: no cover - optional dependency
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - fallback stub
        class _MissingNumpy:
            __slots__ = ("_consumer", "_exc")

            def __init__(self, consumer: str, exc: Exception) -> None:
                self._consumer = consumer
                self._exc = exc

            def __getattr__(self, name: str) -> Any:
                raise RuntimeError(
                    f"numpy is required for {self._consumer} but is not installed"
                ) from self._exc

        return _MissingNumpy(consumer, exc)

    return np
