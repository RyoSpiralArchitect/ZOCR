# -*- coding: utf-8 -*-
"""CLI entry point for semantic diff."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .differ import SemanticDiffer
from .render import render_html, render_unified


def main() -> None:
    ap = argparse.ArgumentParser(description="ZOCR semantic diff for RAG bundles")
    ap.add_argument("--a", required=True, help="cells.jsonl (version A)")
    ap.add_argument("--b", required=True, help="cells.jsonl (version B)")
    ap.add_argument("--sections_a", default=None, help="sections.jsonl (version A)")
    ap.add_argument("--sections_b", default=None, help="sections.jsonl (version B)")
    ap.add_argument("--out_json", default=None, help="save raw events JSON")
    ap.add_argument("--out_diff", default=None, help="save unified .diff-like text")
    ap.add_argument("--out_html", default=None, help="save HTML report")
    args = ap.parse_args()

    diff = SemanticDiffer()
    sec_a = Path(args.sections_a) if args.sections_a else None
    sec_b = Path(args.sections_b) if args.sections_b else None
    res = diff.compare_bundle(Path(args.a), Path(args.b), sec_a, sec_b)
    events = res["events"]

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    txt = render_unified(events)
    if args.out_diff:
        Path(args.out_diff).write_text(txt, encoding="utf-8")
    else:
        print(txt)

    if args.out_html:
        render_html(events, Path(args.out_html))


if __name__ == "__main__":
    main()
