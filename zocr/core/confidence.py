from __future__ import annotations

import math
from typing import Any


def normalize_confidence(value: Any) -> dict[str, float]:
    source = value if isinstance(value, dict) else {"ocr": value}
    result = {}
    for key in ("ocr", "structure"):
        candidate = source.get(key)
        if candidate is None or isinstance(candidate, bool):
            continue
        try:
            score = float(candidate)
        except (ValueError, TypeError, OverflowError):
            continue
        if math.isfinite(score):
            result[key] = max(0.0, min(1.0, score))
    return result


def record_confidence(record: dict[str, Any]) -> dict[str, float]:
    value = record.get("confidence")
    if value is None:
        value = (record.get("meta") or {}).get("confidence")
    return normalize_confidence(value)
