# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

"""History/metadata helpers for the ZOCR orchestrator."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

__all__ = ["load_history", "print_history", "read_summary", "read_meta"]


def load_history(outdir: str) -> List[Dict[str, Any]]:
    """Load pipeline history records from ``pipeline_history.jsonl`` if present."""

    path = os.path.join(outdir, "pipeline_history.jsonl")
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def print_history(records: List[Dict[str, Any]], limit: Optional[int] = None) -> None:
    """Pretty-print a subset of history records to stdout."""

    if limit is not None and limit > 0:
        records = records[-limit:]
    if not records:
        print("(no history)")
        return
    w_step = max(4, max(len(str(r.get("name") or r.get("step"))) for r in records))
    w_status = 7
    header = f"{'timestamp':<20}  {'step':<{w_step}}  {'status':<{w_status}}  elapsed_ms  note"
    print(header)
    print("-" * len(header))
    for rec in records:
        ts = rec.get("ts", "-")
        step = rec.get("name") or rec.get("step") or "?"
        status = "OK" if rec.get("ok") else ("FAIL" if rec.get("ok") is False else "-")
        elapsed = rec.get("elapsed_ms")
        note = rec.get("error") or ""
        if rec.get("out") and status == "OK" and not isinstance(rec["out"], (str, int, float)):
            if isinstance(rec["out"], dict) and rec["out"].get("path"):
                note = rec["out"]["path"]
        print(f"{ts:<20}  {step:<{w_step}}  {status:<{w_status}}  {elapsed!s:<10}  {note}")


def read_summary(outdir: str) -> Dict[str, Any]:
    """Read the consolidated ``pipeline_summary.json`` from disk."""

    path = os.path.join(outdir, "pipeline_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as fr:
        return json.load(fr)


def read_meta(outdir: str) -> Optional[Dict[str, Any]]:
    """Read the optional snapshot metadata file if it exists."""

    path = os.path.join(outdir, "pipeline_meta.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fr:
            return json.load(fr)
    except Exception:
        return None
