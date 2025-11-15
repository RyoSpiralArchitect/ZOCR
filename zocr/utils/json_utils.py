"""JSON serialization helpers."""

from __future__ import annotations

import dataclasses
import numbers
from typing import Any

try:  # pragma: no cover - optional dependency
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - numpy is optional
    _np = None  # type: ignore


def json_ready(obj: Any):
    """Return a JSON-serializable representation of ``obj``.

    The helper recursively converts mappings, sequences, dataclasses, numpy
    arrays/scalars, and sets. Any type that exposes ``tolist``/``item`` or
    ``__json__`` is probed in a best-effort fashion. Unknown values are returned
    as-is so native ``json`` can handle str/int/bool types directly.
    """

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}

    if dataclasses.is_dataclass(obj):
        return json_ready(dataclasses.asdict(obj))

    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]

    if isinstance(obj, set):
        return [json_ready(v) for v in obj]

    if hasattr(obj, "__json__"):
        try:
            return json_ready(obj.__json__())
        except Exception:
            pass

    if hasattr(obj, "tolist"):
        try:
            return json_ready(obj.tolist())
        except Exception:
            pass

    if hasattr(obj, "item") and not isinstance(obj, (bytes, bytearray, str)):
        try:
            scalar = obj.item()  # type: ignore[attr-defined]
        except Exception:
            scalar = None
        else:
            return json_ready(scalar)

    if _np is not None:
        if isinstance(obj, _np.generic):  # type: ignore[attr-defined]
            return obj.item()
        if isinstance(obj, _np.ndarray):  # type: ignore[attr-defined]
            return obj.tolist()

    if isinstance(obj, numbers.Number):
        return obj

    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="ignore")
        except Exception:
            return obj.decode("latin-1", errors="ignore")

    return obj


__all__ = ["json_ready"]
