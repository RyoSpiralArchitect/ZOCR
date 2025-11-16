# -*- coding: utf-8 -*-
"""Aggregations and derived metrics for diff events."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip(text: Optional[str], limit: int = 120) -> Optional[str]:
    if text is None:
        return None
    snippet = str(text).strip()
    if not snippet:
        return None
    snippet = " ".join(snippet.split())
    if len(snippet) > limit:
        return snippet[: limit - 1] + "…"
    return snippet


def _bucket_key(event: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    currency = event.get("numeric_currency")
    if isinstance(currency, str):
        currency = currency.upper()
    unit = event.get("numeric_unit")
    is_percent = bool(event.get("numeric_is_percent"))
    if is_percent:
        label = "%"
        key = "percent"
    elif currency:
        label = currency
        key = f"currency::{currency}"
    elif unit:
        label = unit
        key = f"unit::{unit}"
    else:
        label = "generic"
        key = "generic"
    return key, {"currency": currency, "unit": unit, "is_percent": is_percent, "label": label}


def summarize_numeric_events(events: List[Dict[str, Any]], top_n: int = 5) -> Dict[str, Any]:
    """Compute totals/leaderboards for numeric deltas across events."""

    totals: Dict[str, Dict[str, Any]] = {}
    top_candidates: List[Tuple[float, Dict[str, Any]]] = []
    total_net = 0.0
    total_abs = 0.0
    count = 0

    for event in events:
        delta = _safe_float(event.get("numeric_delta"))
        if delta is None:
            continue
        count += 1
        total_net += delta
        total_abs += abs(delta)
        key, meta = _bucket_key(event)
        bucket = totals.setdefault(
            key,
            {
                "label": meta.get("label"),
                "currency": meta.get("currency"),
                "unit": meta.get("unit"),
                "is_percent": meta.get("is_percent"),
                "count": 0,
                "increase_count": 0,
                "decrease_count": 0,
                "positive_delta": 0.0,
                "negative_delta": 0.0,
                "net_delta": 0.0,
                "abs_delta": 0.0,
            },
        )
        bucket["count"] += 1
        bucket["net_delta"] += delta
        bucket["abs_delta"] += abs(delta)
        if delta >= 0:
            bucket["increase_count"] += 1
            bucket["positive_delta"] += delta
        else:
            bucket["decrease_count"] += 1
            bucket["negative_delta"] += delta

        context = {
            "table_id": event.get("table_id"),
            "row": event.get("row_key")
            or event.get("row_key_a")
            or event.get("row_key_b"),
            "section": event.get("section_heading")
            or event.get("section_heading_a")
            or event.get("section_heading_b"),
            "row_preview": event.get("row_preview")
            or event.get("a_row_preview")
            or event.get("b_row_preview"),
            "old": event.get("old"),
            "new": event.get("new"),
            "numeric_currency": meta.get("currency"),
            "numeric_unit": meta.get("unit"),
            "numeric_is_percent": meta.get("is_percent"),
        }
        context["description"] = _clip(context.get("row_preview") or context.get("row"))
        top_candidates.append(
            (
                abs(delta),
                {
                    "delta": delta,
                    "table_id": context.get("table_id"),
                    "row": context.get("row"),
                    "section": context.get("section"),
                    "description": context.get("description"),
                    "numeric_currency": context.get("numeric_currency"),
                    "numeric_unit": context.get("numeric_unit"),
                    "numeric_is_percent": context.get("numeric_is_percent"),
                },
            )
        )

    if not count:
        return {}

    buckets = sorted(
        totals.values(),
        key=lambda item: abs(item.get("net_delta", 0.0)),
        reverse=True,
    )
    top_entries = [entry for _, entry in sorted(top_candidates, reverse=True)[:top_n]]

    return {
        "numeric_event_count": count,
        "total_net_delta": total_net,
        "total_abs_delta": total_abs,
        "buckets": buckets,
        "top_changes": top_entries,
    }
