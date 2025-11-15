# -*- coding: utf-8 -*-
"""CLI entry point for semantic diff."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .assist import DiffAssistPlanner
from .differ import SemanticDiffer
from .render import render_html, render_unified


def _resolve_cells_path(value: str, label: str) -> Path:
    """Accept either a direct cells.jsonl path or a run directory."""

    path = Path(value)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = [
            path / "rag" / "cells.jsonl",
            path / "cells.jsonl",
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError(
            f"Could not locate cells.jsonl under '{value}'. "
            "Pass a file path or a directory containing rag/cells.jsonl."
        )
    raise FileNotFoundError(f"{label} path '{value}' does not exist")


def _resolve_sections_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = [
            path / "rag" / "sections.jsonl",
            path / "sections.jsonl",
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError(
            f"Could not locate sections.jsonl under '{value}'. "
            "Pass a file path or a directory containing rag/sections.jsonl."
        )
    raise FileNotFoundError(f"sections path '{value}' does not exist")


def main() -> None:
    ap = argparse.ArgumentParser(description="ZOCR semantic diff for RAG bundles")
    ap.add_argument("--a", required=True, help="cells.jsonl (version A)")
    ap.add_argument("--b", required=True, help="cells.jsonl (version B)")
    ap.add_argument("--sections_a", default=None, help="sections.jsonl (version A)")
    ap.add_argument("--sections_b", default=None, help="sections.jsonl (version B)")
    ap.add_argument("--out_json", default=None, help="save raw events JSON")
    ap.add_argument("--out_diff", default=None, help="save unified .diff-like text")
    ap.add_argument("--out_html", default=None, help="save HTML report")
    ap.add_argument(
        "--out_plan", default=None, help="save downstream reanalysis/RAG assist plan"
    )
    args = ap.parse_args()

    diff = SemanticDiffer()
    planner = DiffAssistPlanner()
    cells_a = _resolve_cells_path(args.a, "version A")
    cells_b = _resolve_cells_path(args.b, "version B")
    sec_a = _resolve_sections_path(args.sections_a)
    sec_b = _resolve_sections_path(args.sections_b)
    res = diff.compare_bundle(cells_a, cells_b, sec_a, sec_b)
    events = res["events"]
    assist_plan = planner.plan(events)
    res["assist_plan"] = assist_plan

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    txt = render_unified(events)
    if args.out_diff:
        Path(args.out_diff).write_text(txt, encoding="utf-8")
    else:
        print(txt)

    if args.out_html:
        render_html(events, Path(args.out_html))

    if args.out_plan:
        Path(args.out_plan).write_text(
            json.dumps(assist_plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if not args.out_plan:
        summary = assist_plan.get("summary", {})
        print(
            "[assist] reanalyze={reanalyze} rag={rag} profile={profile}".format(
                reanalyze=summary.get("reanalyze_candidates", 0),
                rag=summary.get("rag_followups", 0),
                profile=summary.get("profile_actions", 0),
            )
        )


if __name__ == "__main__":
    main()
