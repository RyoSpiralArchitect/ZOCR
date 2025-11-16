# -*- coding: utf-8 -*-
"""Aggregations and derived metrics for diff events."""
from __future__ import annotations

from collections import Counter
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


def _top_counter(counter: Counter, limit: int = 8) -> List[str]:
    if not counter:
        return []
    return [token for token, _ in counter.most_common(limit)]


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
    top_entries = [
        entry
        for _, entry in sorted(top_candidates, key=lambda item: item[0], reverse=True)[:top_n]
    ]

    return {
        "numeric_event_count": count,
        "total_net_delta": total_net,
        "total_abs_delta": total_abs,
        "buckets": buckets,
        "top_changes": top_entries,
    }


def _is_textual_event(event: Dict[str, Any]) -> bool:
    if event.get("text_change_type"):
        return True
    if event.get("text_token_stats") or event.get("text_highlight"):
        return True
    if event.get("numeric_delta") is None and (event.get("old") or event.get("new")):
        return True
    return False


def summarize_textual_events(events: List[Dict[str, Any]], top_n: int = 5) -> Dict[str, Any]:
    """Aggregate rewrite-only changes for quick dashboards."""

    totals: Dict[str, Dict[str, Any]] = {}
    top_candidates: List[Tuple[float, Dict[str, Any]]] = []
    token_added: Counter = Counter()
    token_removed: Counter = Counter()
    token_common: Counter = Counter()
    count = 0

    for event in events:
        if not _is_textual_event(event):
            continue
        change_type = event.get("text_change_type") or "textual"
        bucket = totals.setdefault(
            change_type,
            {
                "change_type": change_type,
                "count": 0,
                "similarity_total": 0.0,
                "similarity_count": 0,
                "overlap_total": 0.0,
                "jaccard_total": 0.0,
                "token_stats_count": 0,
            },
        )
        count += 1
        bucket["count"] += 1
        similarity = _safe_float(event.get("similarity"))
        if similarity is not None:
            bucket["similarity_total"] += similarity
            bucket["similarity_count"] += 1
        token_stats = event.get("text_token_stats") or {}
        if token_stats:
            overlap = _safe_float(token_stats.get("overlap"))
            jaccard = _safe_float(token_stats.get("jaccard"))
            if overlap is not None:
                bucket["overlap_total"] += overlap
            if jaccard is not None:
                bucket["jaccard_total"] += jaccard
            bucket["token_stats_count"] += 1
            token_added.update(token_stats.get("added_tokens") or [])
            token_removed.update(token_stats.get("removed_tokens") or [])
            token_common.update(token_stats.get("common_tokens") or [])

        description = event.get("row_preview") or event.get("line_label")
        if not description:
            highlight = event.get("text_highlight") or {}
            description = highlight.get("new") or highlight.get("old")
        description = _clip(description)
        section = (
            event.get("section_heading")
            or event.get("section_heading_a")
            or event.get("section_heading_b")
        )
        similarity_score = similarity if similarity is not None else 0.0
        top_candidates.append(
            (
                1.0 - similarity_score,
                {
                    "change_type": change_type,
                    "similarity": similarity,
                    "section": section,
                    "row": event.get("row_key")
                    or event.get("row_key_a")
                    or event.get("row_key_b"),
                    "description": description,
                    "highlight": event.get("text_highlight"),
                },
            )
        )

    if not count:
        return {}

    buckets: List[Dict[str, Any]] = []
    for bucket in totals.values():
        similarity_count = bucket.pop("similarity_count")
        similarity_total = bucket.pop("similarity_total")
        bucket["avg_similarity"] = (
            similarity_total / similarity_count if similarity_count else None
        )
        stats_count = bucket.pop("token_stats_count")
        overlap_total = bucket.pop("overlap_total")
        jaccard_total = bucket.pop("jaccard_total")
        bucket["avg_overlap"] = (
            overlap_total / stats_count if stats_count else None
        )
        bucket["avg_jaccard"] = (
            jaccard_total / stats_count if stats_count else None
        )
        buckets.append(bucket)

    buckets.sort(key=lambda item: item.get("count", 0), reverse=True)

    top_entries = [
        entry
        for _, entry in sorted(top_candidates, key=lambda item: item[0], reverse=True)[:top_n]
    ]

    token_summary = {
        "top_added_tokens": _top_counter(token_added),
        "top_removed_tokens": _top_counter(token_removed),
        "top_common_tokens": _top_counter(token_common),
    }
    token_summary = {k: v for k, v in token_summary.items() if v}

    summary: Dict[str, Any] = {
        "textual_event_count": count,
        "change_types": buckets,
        "top_changes": top_entries,
    }
    if token_summary:
        summary["token_highlights"] = token_summary
    return summary


def _section_meta(event: Dict[str, Any]) -> Tuple[str, Optional[int], Optional[List[str]]]:
    for suffix in ("", "_a", "_b"):
        heading = event.get(f"section_heading{suffix}")
        if heading:
            level = event.get(f"section_level{suffix}")
            try:
                level = int(level) if level is not None else None
            except (TypeError, ValueError):
                level = None
            path_raw = event.get(f"section_path{suffix}") or []
            if isinstance(path_raw, str):
                path = [path_raw]
            elif isinstance(path_raw, list):
                path = [str(p) for p in path_raw if p]
            else:
                path = []
            return str(heading), level, path
    table_id = event.get("table_id")
    if table_id:
        return str(table_id), None, []
    return "document", None, []


def summarize_section_events(events: List[Dict[str, Any]], top_n: int = 5) -> Dict[str, Any]:
    """Group events by section headings / table IDs."""

    if not events:
        return {}

    totals: Dict[str, Dict[str, Any]] = {}
    top_examples: List[Tuple[float, int, Dict[str, Any]]] = []
    count = 0
    example_idx = 0

    for event in events:
        section, level, path = _section_meta(event)
        key = f"{section}|L{level}" if level is not None else section
        bucket = totals.setdefault(
            key,
            {
                "section": section,
                "level": level,
                "path": path,
                "table_id": event.get("table_id"),
                "count": 0,
                "numeric_count": 0,
                "textual_count": 0,
                "event_types": Counter(),
                "net_delta": 0.0,
                "abs_delta": 0.0,
            },
        )
        bucket["table_id"] = bucket.get("table_id") or event.get("table_id")
        bucket["count"] += 1
        count += 1
        event_type = event.get("type") or "event"
        bucket["event_types"][event_type] += 1
        delta = _safe_float(event.get("numeric_delta"))
        if delta is not None:
            bucket["numeric_count"] += 1
            bucket["net_delta"] += delta
            bucket["abs_delta"] += abs(delta)
        if _is_textual_event(event):
            bucket["textual_count"] += 1

        description = _clip(
            event.get("row_preview")
            or event.get("line_label")
            or event.get("row_key")
            or event.get("title")
        )
        score = abs(delta) if delta is not None else 1.0
        example_idx += 1
        top_examples.append(
            (
                score,
                -example_idx,
                {
                    "section": section,
                    "level": level,
                    "table_id": event.get("table_id"),
                    "event_type": event_type,
                    "row": event.get("row_key")
                    or event.get("row_key_a")
                    or event.get("row_key_b"),
                    "description": description,
                    "numeric_delta": delta,
                },
            )
        )

    sections: List[Dict[str, Any]] = []
    for bucket in totals.values():
        event_types = bucket.pop("event_types")
        bucket["event_types"] = [
            {"type": name, "count": cnt}
            for name, cnt in event_types.most_common(6)
        ]
        sections.append(bucket)

    sections.sort(key=lambda item: item.get("count", 0), reverse=True)
    top_sections = sorted(
        sections,
        key=lambda item: (item.get("abs_delta", 0.0), item.get("count", 0)),
        reverse=True,
    )[:top_n]
    top_events = [entry for _, _, entry in sorted(top_examples, reverse=True)[:top_n]]

    return {
        "section_event_count": count,
        "sections": sections,
        "top_sections": top_sections,
        "top_events": top_events,
    }
