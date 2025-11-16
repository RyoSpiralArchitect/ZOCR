# -*- coding: utf-8 -*-
"""Rendering helpers for semantic diff output."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def render_unified(events: List[dict]) -> str:
    lines: List[str] = []
    for e in events:
        t = e.get("type")
        if t in ("table_added", "table_removed"):
            sign = "+" if t == "table_added" else "-"
            lines.append(
                f"{sign} TABLE {e.get('table_id')} (page {e.get('table_page')}, idx {e.get('table_index')})"
            )
        elif t in ("row_added", "row_removed"):
            sign = "+" if t == "row_added" else "-"
            preview = e.get("row_preview")
            preview_txt = f" :: {preview}" if preview else ""
            row_ids = e.get("row_ids")
            if row_ids:
                preview_txt += f" [ids={','.join(row_ids[:3])}{'…' if len(row_ids) > 3 else ''}]"
            lines.append(
                f"{sign} ROW {e.get('table_id')} :: {e.get('row_key')}"
                f" (origin {e.get('row_origin')}){preview_txt}"
            )
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
            row_ctx = e.get("row_key_a") or e.get("row_key")
            row_pair = e.get("row_key_b")
            row_info = f"{row_ctx}" if row_pair is None or row_pair == row_ctx else f"{row_ctx} → {row_pair}"
            lines.append(
                f"! CELL {e.get('table_id')} [{row_info} / a:{e.get('a_col')} b:{e.get('b_col')}]"
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
            row_key_a = e.get("row_key_a")
            row_key_b = e.get("row_key_b")
            if row_key_b and row_key_b != row_key_a:
                row_label = f"{html.escape(str(row_key_a))} → {html.escape(str(row_key_b))}"
            else:
                row_label = html.escape(str(row_key_a or e.get("row_key")))
            blocks.append(
                f"""
            <div class="event cell_updated">
              <div><b>Cell Updated</b> <span class="kv">[{html.escape(e.get('table_id',''))}] row={row_label} a_col={e.get('a_col')} b_col={e.get('b_col')}</span></div>
              <div class="kv">{html.escape(meta_s)}</div>
              <pre><span class="minus">- {old}</span>\n<span class="plus">+ {new}</span></pre>
            </div>"""
            )
        elif t in ("row_added", "row_removed"):
            title = "Row Added" if t == "row_added" else "Row Removed"
            preview = e.get("row_preview")
            preview_html = f"<div class=\"kv\">{html.escape(preview)}</div>" if preview else ""
            row_ids = e.get("row_ids") or []
            ids_html = (
                f"<div class=\"kv\">ids={html.escape(','.join(row_ids[:3]))}{'…' if len(row_ids) > 3 else ''}</div>"
                if row_ids
                else ""
            )
            origin = e.get("row_origin")
            blocks.append(
                f"""
            <div class="event {t}">
              <div><b>{title}</b> <span class="kv">{html.escape(e.get('table_id',''))} | origin {origin}</span></div>
              <div class="kv">row_key={html.escape(str(e.get('row_key')))} | page={html.escape(str(e.get('table_page')))} | idx={html.escape(str(e.get('table_index')))}</div>
              {preview_html}
              {ids_html}
            </div>"""
            )
        elif t in ("table_added", "table_removed"):
            title = "Table Added" if t == "table_added" else "Table Removed"
            blocks.append(
                f"""
            <div class="event {t}">
              <div><b>{title}</b> <span class="kv">{html.escape(e.get('table_id',''))} | page={html.escape(str(e.get('table_page')))} | idx={html.escape(str(e.get('table_index')))}</span></div>
              <div class="kv">rows={html.escape(str(e.get('table_rows')))} cols={html.escape(str(e.get('table_columns')))}</div>
            </div>"""
            )
        elif t in (
            "header_renamed",
            "col_moved",
            "section_added",
            "section_removed",
            "section_title_changed",
            "section_level_changed",
        ):
            title = {
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


def _md_inline(value: Any, limit: int = 160) -> str:
    if value is None:
        return "∅"
    text = str(value).strip()
    if not text:
        return "∅"
    text = " ".join(text.replace("\n", " / ").split())
    text = text.replace("`", "'")
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _row_label(event: Dict[str, Any]) -> str:
    primary = event.get("row_key_a") or event.get("row_key")
    secondary = event.get("row_key_b")
    if primary and secondary and primary != secondary:
        return f"{primary} → {secondary}"
    return primary or secondary or "?"


def _section_hint(event: Dict[str, Any]) -> Optional[str]:
    for key in (
        "section_heading",
        "section_heading_a",
        "section_heading_b",
    ):
        if event.get(key):
            return str(event.get(key))
    return None


def render_markdown(
    events: List[Dict[str, Any]],
    summary: Optional[Dict[str, Any]] = None,
    title: str = "ZOCR Diff Report",
) -> str:
    """Build a Markdown summary for dashboards, chat, or reports."""

    lines: List[str] = [f"# {title}", ""]

    if summary:
        label_map = {
            "tables_matched": "Tables matched",
            "tables_added": "Tables added",
            "tables_removed": "Tables removed",
            "events": "Total events",
            "diff_hunks": "Diff hunks",
            "numeric_changes": "Numeric changes",
            "textual_changes": "Textual changes",
        }
        lines.append("## Summary")
        for key, label in label_map.items():
            if key in summary and summary[key] is not None:
                lines.append(f"- {label}: {summary[key]}")
        other_keys = [
            key
            for key in summary.keys()
            if key not in label_map and summary[key] is not None
        ]
        for key in sorted(other_keys):
            lines.append(f"- {key}: {summary[key]}")
        lines.append("")

    table_events = [
        e for e in events if e.get("type") in {"table_added", "table_removed"}
    ]
    row_events = [e for e in events if e.get("type") in {"row_added", "row_removed"}]
    section_events = [
        e
        for e in events
        if e.get("type")
        in {"section_added", "section_removed", "section_title_changed", "section_level_changed"}
    ]
    column_events = [
        e for e in events if e.get("type") in {"header_renamed", "col_moved"}
    ]
    cell_events = [e for e in events if e.get("type") == "cell_updated"]
    other_events = [
        e
        for e in events
        if e not in table_events
        and e not in row_events
        and e not in section_events
        and e not in column_events
        and e not in cell_events
    ]

    if table_events:
        lines.append("## Table changes")
        for ev in table_events:
            label = "added" if ev.get("type") == "table_added" else "removed"
            rows = ev.get("table_rows")
            cols = ev.get("table_columns")
            lines.append(
                f"- `{_md_inline(ev.get('table_id'))}` {label} (page {ev.get('table_page')}, idx {ev.get('table_index')}, rows={rows}, cols={cols})"
            )
        lines.append("")

    if row_events:
        lines.append("## Row changes")
        for ev in row_events:
            label = "added" if ev.get("type") == "row_added" else "removed"
            preview = ev.get("row_preview")
            ids = ev.get("row_ids") or []
            id_txt = f" ids={','.join(ids[:3])}{'…' if len(ids) > 3 else ''}" if ids else ""
            section = _section_hint(ev)
            section_txt = f" §{_md_inline(section)}" if section else ""
            lines.append(
                f"- `{_md_inline(ev.get('row_key'))}` {label} in `{_md_inline(ev.get('table_id'))}`{section_txt}{id_txt}"
            )
            if preview:
                lines.append(f"  - preview: `{_md_inline(preview)}`")
        lines.append("")

    if column_events:
        lines.append("## Column/header changes")
        for ev in column_events:
            if ev.get("type") == "header_renamed":
                lines.append(
                    f"- `{_md_inline(ev.get('table_id'))}`: `{_md_inline(ev.get('from'))}` → `{_md_inline(ev.get('to'))}` (col {ev.get('a_col')} → {ev.get('b_col')})"
                )
            else:
                lines.append(
                    f"- `{_md_inline(ev.get('table_id'))}`: column `{_md_inline(ev.get('header'))}` moved {ev.get('from_index')} → {ev.get('to_index')}"
                )
        lines.append("")

    if section_events:
        lines.append("## Section changes")
        for ev in section_events:
            t = ev.get("type")
            if t in {"section_added", "section_removed"}:
                label = "added" if t == "section_added" else "removed"
                lines.append(
                    f"- L{ev.get('level')} `{_md_inline(ev.get('title'))}` {label} (page {ev.get('page')})"
                )
            elif t == "section_title_changed":
                lines.append(
                    f"- Title changed: `{_md_inline(ev.get('old'))}` → `{_md_inline(ev.get('new'))}` (pages {ev.get('page_a')} → {ev.get('page_b')})"
                )
            elif t == "section_level_changed":
                lines.append(
                    f"- `{_md_inline(ev.get('title'))}` level {ev.get('from_level')} → {ev.get('to_level')}"
                )
        lines.append("")

    if cell_events:
        lines.append("## Cell / line changes")
        for ev in cell_events:
            table_label = _md_inline(ev.get("table_header_preview") or ev.get("table_id"))
            row_label = _md_inline(_row_label(ev), 96)
            metrics: List[str] = []
            if ev.get("numeric_delta") is not None:
                metrics.append(f"Δ={ev['numeric_delta']:+g}")
            if ev.get("relative_delta") is not None:
                metrics.append(f"rΔ={ev['relative_delta']:+.2%}")
            if ev.get("line_similarity") is not None:
                metrics.append(f"sim={float(ev['line_similarity']):.2f}")
            if ev.get("text_change_type"):
                metrics.append(f"text={ev['text_change_type']}")
            if ev.get("numeric_currency"):
                metrics.append(f"currency={ev['numeric_currency']}")
            if ev.get("numeric_is_percent"):
                metrics.append("percent")
            section = _section_hint(ev)
            section_txt = f" (§{_md_inline(section, 48)})" if section else ""
            metric_txt = f" ({', '.join(metrics)})" if metrics else ""
            lines.append(f"- **{table_label}** · row `{row_label}`{section_txt}{metric_txt}")
            old_text = _md_inline(ev.get("old"))
            new_text = _md_inline(ev.get("new"))
            lines.append(f"  - old: `{old_text}`")
            lines.append(f"  - new: `{new_text}`")
            if ev.get("a_row_context") or ev.get("b_row_context"):
                ctx_a = _md_inline(ev.get("a_row_context"))
                ctx_b = _md_inline(ev.get("b_row_context"))
                lines.append(f"  - context A: `{ctx_a}`")
                lines.append(f"  - context B: `{ctx_b}`")
        lines.append("")

    if other_events:
        lines.append("## Other events")
        for ev in other_events:
            lines.append(f"- `{_md_inline(ev.get('type', 'unknown'))}` :: {json.dumps(ev, ensure_ascii=False)}")
        lines.append("")

    if not events:
        lines.append("No changes detected.")

    return "\n".join(lines).strip() + "\n"
