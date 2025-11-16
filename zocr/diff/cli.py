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
from .simple import SimpleTextDiffer


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
    ap.add_argument(
        "--simple_text_a",
        default=None,
        help="plain-text document A for the git-like quick differ",
    )
    ap.add_argument(
        "--simple_text_b",
        default=None,
        help="plain-text document B for the git-like quick differ",
    )
    ap.add_argument(
        "--simple_diff_out",
        default=None,
        help="save the quick differ unified diff (defaults to stdout)",
    )
    ap.add_argument(
        "--simple_json_out",
        default=None,
        help="save quick differ summary (numeric deltas, line numbers)",
    )
    ap.add_argument(
        "--simple_context",
        type=int,
        default=3,
        help="context lines for the quick differ (git-style)",
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

    if args.simple_text_a or args.simple_text_b:
        if not (args.simple_text_a and args.simple_text_b):
            raise SystemExit("--simple_text_a and --simple_text_b must be provided together")
        quick = SimpleTextDiffer(context_lines=args.simple_context)
        simple_result = quick.compare_files(
            Path(args.simple_text_a), Path(args.simple_text_b)
        )
        if args.simple_json_out:
            Path(args.simple_json_out).write_text(
                json.dumps(simple_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        summary = simple_result.get("summary", {})
        diff_payload = simple_result.get("diff", "")
        if args.simple_diff_out:
            Path(args.simple_diff_out).write_text(diff_payload, encoding="utf-8")
        else:
            print(diff_payload)
        if not args.simple_json_out:
            print(
                "[simple] diff_hunks={hunks} numeric_changes={nums}".format(
                    hunks=summary.get("diff_hunks", 0),
                    nums=summary.get("numeric_changes", 0),
                )
            )


if __name__ == "__main__":
    main()
