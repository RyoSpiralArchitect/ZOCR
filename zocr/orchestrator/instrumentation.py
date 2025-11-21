# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

"""Instrumentation helpers for stage-level diagnostics."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..utils.json_utils import json_ready as _json_ready

StageTraceEntry = Dict[str, Any]
StageTrace = List[StageTraceEntry]

__all__ = [
    "set_stage_trace_sink",
    "record_stage_trace",
    "print_stage_trace_console",
    "compute_stage_stats",
]

_STAGE_TRACE_SINK: Optional[StageTrace] = None


def set_stage_trace_sink(sink: Optional[StageTrace]) -> None:
    """Register a list that should accumulate stage trace entries."""

    global _STAGE_TRACE_SINK
    _STAGE_TRACE_SINK = sink


def stage_output_preview(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        interesting = (
            "path",
            "paths",
            "count",
            "records",
            "pages",
            "tables",
            "cells",
            "reason",
            "summary",
            "output",
            "metrics",
        )
        preview: Dict[str, Any] = {}
        for key in interesting:
            if key in value:
                preview[key] = value[key]
        if preview:
            return _json_ready(preview)
        if len(value) <= 4:
            return _json_ready(value)
        return f"{len(value)} keys"
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        if not seq:
            return []
        if len(seq) <= 4 and all(isinstance(item, (bool, int, float, str)) for item in seq):
            return seq
        return f"{len(seq)} items"
    return str(type(value).__name__)


def record_stage_trace(rec: StageTraceEntry) -> None:
    """Store a sanitized snapshot of ``rec`` on the configured sink if any."""

    if _STAGE_TRACE_SINK is None:
        return
    snapshot: StageTraceEntry = {
        "name": rec.get("name"),
        "elapsed_ms": float(rec.get("elapsed_ms") or 0.0),
    }
    if rec.get("ok") is None:
        snapshot["ok"] = None
    else:
        snapshot["ok"] = bool(rec.get("ok"))
    if rec.get("error"):
        snapshot["error"] = rec.get("error")
    preview = stage_output_preview(rec.get("out"))
    if preview is not None:
        snapshot["out"] = preview
    _STAGE_TRACE_SINK.append(snapshot)


def _summarize_stage_preview(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, int, float)):
        return str(value)
    if isinstance(value, str):
        return value if len(value) <= 80 else value[:77] + "..."
    if isinstance(value, dict):
        items: List[str] = []
        for idx, (key, val) in enumerate(value.items()):
            if idx >= 3:
                items.append("…")
                break
            items.append(f"{key}={val}")
        return ", ".join(items)
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        if not seq:
            return ""
        snippet = ", ".join(str(item) for item in seq[:3])
        if len(seq) > 3:
            snippet += ", …"
        return snippet
    return str(value)


def compute_stage_stats(stage_trace: Sequence[StageTraceEntry]) -> Optional[Dict[str, Any]]:
    """Aggregate totals/failures/slowest helpers for ``stage_trace``."""

    if not stage_trace:
        return None
    total_ms = sum(float(entry.get("elapsed_ms") or 0.0) for entry in stage_trace)
    failures = sum(1 for entry in stage_trace if entry.get("ok") is False)
    slowest = max(stage_trace, key=lambda e: float(e.get("elapsed_ms") or 0.0)) if stage_trace else None
    return {
        "count": len(stage_trace),
        "failures": failures,
        "total_elapsed_ms": total_ms,
        "slowest": {"name": slowest.get("name"), "elapsed_ms": slowest.get("elapsed_ms")} if slowest else None,
    }


def print_stage_trace_console(
    stage_trace: Sequence[StageTraceEntry],
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Render ``stage_trace`` entries as a readable console table."""

    if not stage_trace:
        return
    print("\n[Stage Trace]")
    header = f"{'Stage':<28} {'OK':<4} {'Elapsed (ms)':>12}  Details"
    print(header)
    print("-" * len(header))
    for entry in stage_trace:
        name = (entry.get("name") or "?")
        ok_val = entry.get("ok")
        status = "ok" if ok_val is True else ("fail" if ok_val is False else "…")
        elapsed = float(entry.get("elapsed_ms") or 0.0)
        detail = _summarize_stage_preview(entry.get("out"))
        if entry.get("error"):
            err = str(entry.get("error"))
            detail = f"{detail} | {err}" if detail else err
        if len(detail) > 96:
            detail = detail[:93] + "..."
        print(f"{name:<28.28} {status:<4} {elapsed:>12.1f}  {detail}")
    if stats:
        total = float(stats.get("total_elapsed_ms") or 0.0)
        fail = stats.get("failures")
        count = stats.get("count")
        print("-" * len(header))
        print(f"Total stages: {count}, failures: {fail}, elapsed: {total:.1f} ms")
        slowest = stats.get("slowest") if isinstance(stats, dict) else None
        if isinstance(slowest, dict) and slowest.get("name"):
            print(f"Slowest: {slowest.get('name')} ({slowest.get('elapsed_ms')} ms)")
