# -*- coding: utf-8 -*-
"""Rendering helpers for semantic diff output."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import List


def render_unified(events: List[dict]) -> str:
    lines: List[str] = []
    for e in events:
        t = e.get("type")
        if t in ("table_added", "table_removed"):
            sign = "+" if t == "table_added" else "-"
            lines.append(f"{sign} TABLE {e.get('table_id')}")
        elif t in ("row_added", "row_removed"):
            sign = "+" if t == "row_added" else "-"
            lines.append(f"{sign} ROW {e.get('table_id')} :: {e.get('row_key')}")
        elif t in ("section_added", "section_removed"):
            sign = "+" if t == "section_added" else "-"
            lines.append(
                f"{sign} SECTION L{e.get('level')} :: {e.get('title')} (page {e.get('page')})"
            )
        elif t == "section_title_changed":
            lines.append(
                f"~ SECTION TITLE :: {e.get('old')} -> {e.get('new')}"
                f" [pages {e.get('page_a')}->{e.get('page_b')}]"
            )
        elif t == "section_level_changed":
            lines.append(
                f"~ SECTION LEVEL :: {e.get('title')} [L{e.get('from_level')} -> L{e.get('to_level')}]"
            )
        elif t == "header_renamed":
            lines.append(
                f"~ HEADER {e.get('table_id')} :: {e.get('from')} -> {e.get('to')} "
                f"[a:{e.get('a_col')} b:{e.get('b_col')}]"
            )
        elif t == "col_moved":
            lines.append(
                f"~ COLMOVE {e.get('table_id')} :: {e.get('header')} "
                f"[{e.get('from_index')} -> {e.get('to_index')}]"
            )
        elif t == "cell_updated":
            nd = e.get("numeric_delta")
            rd = e.get("relative_delta")
            sim = e.get("similarity")
            extra: List[str] = []
            if nd is not None:
                extra.append(f"Δ={nd:+g}")
            if rd is not None:
                extra.append(f"rΔ={rd:+.2%}")
            if sim is not None:
                extra.append(f"sim={sim:.2f}")
            extras = (" " + " ".join(extra)) if extra else ""
            lines.append(
                f"! CELL {e.get('table_id')} [{e.get('row_key')} / a:{e.get('a_col')} b:{e.get('b_col')}]"
                f"{extras}\n- {e.get('old')}\n+ {e.get('new')}"
            )
        else:
            lines.append(json.dumps(e, ensure_ascii=False))
    return "\n".join(lines)


def render_html(events: List[dict], out_path: Path) -> None:
    css = """
    <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; color: #222; }
    .summary { margin-bottom: 16px; padding: 12px; background: #f6f8fa; border-radius: 8px; }
    .event { border-left: 4px solid #e1e4e8; padding: 8px 12px; margin: 8px 0; background: #fff; border-radius: 6px; }
    .table_added { border-color: #2da44e; }
    .table_removed { border-color: #cf222e; }
    .row_added { border-color: #2da44e; }
    .row_removed { border-color: #cf222e; }
    .header_renamed, .col_moved { border-color: #bf8700; }
    .section_added { border-color: #2da44e; }
    .section_removed { border-color: #cf222e; }
    .section_title_changed, .section_level_changed { border-color: #bf8700; }
    .cell_updated { border-color: #0969da; }
    .kv { color: #57606a; }
    pre { white-space: pre-wrap; word-break: break-word; background: #f6f8fa; padding: 8px; border-radius: 6px; }
    .minus { color: #cf222e; }
    .plus  { color: #2da44e; }
    </style>
    """
    blocks: List[str] = []
    for e in events:
        t = e.get("type")
        if t == "cell_updated":
            old = html.escape(str(e.get("old")))
            new = html.escape(str(e.get("new")))
            meta: List[str] = []
            if e.get("numeric_delta") is not None:
                meta.append(f"Δ={e['numeric_delta']:+g}")
            if e.get("relative_delta") is not None:
                meta.append(f"rΔ={e['relative_delta']:+.2%}")
            if e.get("similarity") is not None:
                meta.append(f"sim={e['similarity']:.2f}")
            meta_s = " | ".join(meta)
            blocks.append(
                f"""
            <div class="event cell_updated">
              <div><b>Cell Updated</b> <span class="kv">[{html.escape(e.get('table_id',''))}] row={html.escape(str(e.get('row_key','')))} a_col={e.get('a_col')} b_col={e.get('b_col')}</span></div>
              <div class="kv">{html.escape(meta_s)}</div>
              <pre><span class="minus">- {old}</span>\n<span class="plus">+ {new}</span></pre>
            </div>"""
            )
        elif t in (
            "table_added",
            "table_removed",
            "row_added",
            "row_removed",
            "header_renamed",
            "col_moved",
            "section_added",
            "section_removed",
            "section_title_changed",
            "section_level_changed",
        ):
            title = {
                "table_added": "Table Added",
                "table_removed": "Table Removed",
                "row_added": "Row Added",
                "row_removed": "Row Removed",
                "header_renamed": "Header Renamed",
                "col_moved": "Column Moved",
                "section_added": "Section Added",
                "section_removed": "Section Removed",
                "section_title_changed": "Section Title Changed",
                "section_level_changed": "Section Level Changed",
            }[t]
            blocks.append(
                f"""
            <div class="event {t}">
              <div><b>{title}</b> <span class="kv">{html.escape(json.dumps(e, ensure_ascii=False))}</span></div>
            </div>"""
            )
        else:
            blocks.append(
                f"""
            <div class="event">
              <div>{html.escape(json.dumps(e, ensure_ascii=False))}</div>
            </div>"""
            )

    html_doc = f"""<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>ZOCR Semantic Diff</title>{css}</head>
    <body>
      <div class=\"summary\">Events: {len(events)}</div>
      {''.join(blocks)}
    </body></html>"""
    out_path.write_text(html_doc, encoding="utf-8")
