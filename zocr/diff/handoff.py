# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

# -*- coding: utf-8 -*-
"""Helpers to package diff outputs for API/GUI handoffs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_handoff_bundle(
    *,
    mode: str,
    source: Dict[str, Any],
    summary: Optional[Dict[str, Any]],
    events: List[Dict[str, Any]],
    diff_text: str,
    markdown_text: Optional[str] = None,
    assist_plan: Dict[str, Any],
    artifacts: Optional[Dict[str, Any]] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a single JSON-friendly payload for downstream agents."""

    bundle: Dict[str, Any] = {
        "mode": mode,
        "source": source,
        "summary": summary or {},
        "events": events,
        "diff_text": diff_text,
        "assist_plan": assist_plan,
        "agentic_requests": assist_plan.get("agentic_requests", []),
    }
    if markdown_text:
        bundle["markdown_report"] = markdown_text
    if artifacts:
        bundle["artifacts"] = {key: val for key, val in artifacts.items() if val}
    if extras:
        bundle.update({key: val for key, val in extras.items() if val is not None})
    return bundle
