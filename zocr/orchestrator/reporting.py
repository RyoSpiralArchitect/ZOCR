"""Reporting helpers for the all-in-one orchestrator CLI."""
from __future__ import annotations

import json
import os
import time
from html import escape
from typing import Any, Dict, List, Optional

from .history import load_history, read_meta, read_summary

__all__ = ["generate_report"]


def _render_value(value: Any) -> str:
    if value is None:
        return "<span class=\"muted\">(none)</span>"
    if isinstance(value, bool):
        return "✅ True" if value else "⚠️ False"
    if isinstance(value, (list, tuple)):
        if not value:
            return "<span class=\"muted\">[]</span>"
        items = "<br>".join(escape(str(v)) for v in value[:8])
        if len(value) > 8:
            items += "<br>…"
        return f"<pre>{items}</pre>"
    if isinstance(value, dict):
        formatted = json.dumps(value, ensure_ascii=False, indent=2)
        if len(formatted) > 600:
            formatted = formatted[:600] + "…"
        return f"<pre>{escape(formatted)}</pre>"
    if isinstance(value, (int, float)):
        return escape(f"{value:,.4g}")
    return escape(str(value))


def _render_table(data: Dict[str, Any], title: str, keys: Optional[List[str]] = None) -> str:
    if not data:
        return ""
    rows = []
    items = data.items() if keys is None else ((k, data.get(k)) for k in keys if k in data)
    for key, value in items:
        rows.append(
            f"<tr><th scope=\"row\">{escape(str(key))}</th><td>{_render_value(value)}</td></tr>"
        )
    if not rows:
        return ""
    return (
        f"<section>\n<h2>{escape(title)}</h2>\n"
        "<table class=\"kv\">\n" + "\n".join(rows) + "\n</table>\n</section>"
    )


def _render_history_table(records: List[Dict[str, Any]]) -> str:
    if not records:
        return "<p class=\"muted\">(no history recorded)</p>"
    header = "<thead><tr><th>timestamp</th><th>step</th><th>status</th><th>elapsed</th><th>note</th></tr></thead>"
    body_rows = []
    for rec in records:
        status = "ok" if rec.get("ok") else ("fail" if rec.get("ok") is False else "skip")
        cls = {
            "ok": "status-ok",
            "fail": "status-fail",
            "skip": "status-skip",
        }.get(status, "")
        elapsed = rec.get("elapsed_ms")
        if isinstance(elapsed, (int, float)):
            elapsed_s = f"{elapsed:,.1f} ms"
        else:
            elapsed_s = escape(str(elapsed)) if elapsed is not None else "–"
        note = rec.get("error") or ""
        out = rec.get("out")
        if not note and isinstance(out, dict) and out.get("path"):
            note = str(out.get("path"))
        body_rows.append(
            "<tr class=\"{cls}\"><td>{ts}</td><td>{step}</td><td><span class=\"badge {cls}\">{status}</span></td><td>{elapsed}</td><td>{note}</td></tr>".format(
                cls=cls,
                ts=escape(rec.get("ts", "–")),
                step=escape(str(rec.get("name") or rec.get("step") or "?")),
                status=escape(status.upper()),
                elapsed=elapsed_s,
                note=escape(str(note)) if note else "",
            )
        )
    return "<table class=\"history\">" + header + "<tbody>" + "".join(body_rows) + "</tbody></table>"


def _render_hotspots_section(summary: Dict[str, Any]) -> str:
    hotspots = summary.get("learning_hotspots") if isinstance(summary, dict) else None
    plan = summary.get("selective_reanalysis_plan") if isinstance(summary, dict) else None
    gallery = summary.get("hotspot_gallery") if isinstance(summary, dict) else None
    if not any([hotspots, plan, gallery]):
        return ""
    parts: List[str] = ["<section>", "<h2>ホットスポット / Hotspots</h2>"]
    if isinstance(hotspots, dict) and hotspots:
        reasons = hotspots.get("reason_counts") if isinstance(hotspots.get("reason_counts"), list) else []
        if reasons:
            parts.append("<h3>Signals</h3><ul>")
            for rec in reasons[:6]:
                if not isinstance(rec, dict):
                    continue
                label = escape(str(rec.get("reason") or "?"))
                count = escape(str(rec.get("count") or ""))
                parts.append(f"<li>{label}: {count}</li>")
            parts.append("</ul>")
        cells = hotspots.get("hot_cells") if isinstance(hotspots.get("hot_cells"), list) else []
        if cells:
            rows = ["<thead><tr><th>trace</th><th>page,row</th><th>score</th><th>reasons</th></tr></thead>"]
            body: List[str] = []
            for cell in cells[:6]:
                if not isinstance(cell, dict):
                    continue
                trace = escape(str(cell.get("trace_id") or "?"))
                loc = f"p{cell.get('page')} r{cell.get('row')}"
                score = escape(str(cell.get("score") or ""))
                reasons_txt = ", ".join(escape(str(r)) for r in cell.get("reasons", [])[:4]) if cell.get("reasons") else ""
                body.append(f"<tr><td><code>{trace}</code></td><td>{escape(loc)}</td><td>{score}</td><td>{reasons_txt}</td></tr>")
            if body:
                rows.append("<tbody>" + "".join(body) + "</tbody>")
                parts.append("<details open><summary>Top cells</summary><table class=\"history\">" + "".join(rows) + "</table></details>")
    if isinstance(plan, dict) and plan:
        parts.append(_render_table(plan, "選択的再解析計画 / Selective plan"))
    if isinstance(gallery, dict) and gallery.get("entries"):
        entries = gallery.get("entries")
        limit = min(6, len(entries)) if isinstance(entries, list) else 0
        if limit:
            parts.append("<h3>Hotspot gallery</h3>")
            parts.append("<div class=\"hotspot-gallery\">")
            for entry in entries[:limit]:
                if not isinstance(entry, dict):
                    continue
                img = entry.get("image")
                caption_bits: List[str] = []
                if entry.get("trace_id"):
                    caption_bits.append(f"trace {escape(str(entry['trace_id']))}")
                if entry.get("role"):
                    caption_bits.append(f"role {escape(str(entry['role']))}")
                if entry.get("reason_rank"):
                    caption_bits.append(f"reason #{escape(str(entry['reason_rank']))}")
                caption = " ・ ".join(caption_bits) or "cell"
                before = entry.get("before_text") or entry.get("text")
                after = entry.get("after_text")
                text_lines = []
                if before:
                    text_lines.append(f"<div class=\"muted\">before</div><div>{escape(str(before))}</div>")
                if after and after != before:
                    text_lines.append(f"<div class=\"muted\">after</div><div>{escape(str(after))}</div>")
                img_html = f"<img src=\"{escape(str(img))}\" alt=\"hotspot\">" if img else ""
                parts.append(
                    "<figure>" + img_html + f"<figcaption>{caption}</figcaption>" + "".join(text_lines) + "</figure>"
                )
            parts.append("</div>")
        if gallery.get("story"):
            parts.append(
                f"<p class=\"muted\"><a href=\"{escape(str(gallery['story']))}\">gallery notes</a></p>"
            )
    parts.append("</section>")
    return "\n".join(parts)


def generate_report(
    outdir: str,
    dest: Optional[str] = None,
    summary: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    meta: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = 50,
) -> str:
    """Render the HTML pipeline report and return the destination path."""

    summary = summary or read_summary(outdir)
    history = history or load_history(outdir)
    meta = meta if meta is not None else read_meta(outdir)
    if limit is not None and limit > 0 and len(history) > limit:
        history = history[-limit:]
    dest = dest or os.path.join(outdir, "pipeline_report.html")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

    stats = summary.get("history_stats") or {}
    total_ms = stats.get("total_elapsed_ms")
    ok_count = stats.get("ok")
    fail_count = stats.get("fail")
    total_s = None
    if isinstance(total_ms, (int, float)):
        total_s = total_ms / 1000.0

    css = """
    body { font-family: 'Inter', 'Segoe UI', 'Hiragino Sans', sans-serif; margin: 2rem; background: #0d1117; color: #e6edf3; }
    a { color: #9cdcfe; }
    h1, h2, h3 { color: #58a6ff; }
    section { margin-bottom: 2rem; }
    table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
    th, td { border: 1px solid #30363d; padding: 0.45rem 0.6rem; vertical-align: top; }
    th { background: rgba(88, 166, 255, 0.08); text-align: left; font-weight: 600; }
    table.kv th { width: 18%; }
    pre { background: #161b22; border-radius: 8px; padding: 0.75rem; overflow-x: auto; }
    .muted { opacity: 0.65; }
    .badge { display: inline-block; padding: 0.1rem 0.6rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
    .status-ok .badge { background: rgba(63, 185, 80, 0.2); color: #3fb950; }
    .status-fail .badge { background: rgba(248, 81, 73, 0.2); color: #f85149; }
    .status-skip .badge { background: rgba(201, 148, 0, 0.2); color: #c99400; }
    details { margin-top: 1rem; }
    summary { cursor: pointer; }
    .hotspot-gallery { display: flex; flex-wrap: wrap; gap: 1rem; }
    .hotspot-gallery figure { width: 220px; background: #161b22; border: 1px solid #30363d; padding: 0.5rem; border-radius: 8px; }
    .hotspot-gallery img { max-width: 100%; border-radius: 4px; margin-bottom: 0.35rem; }
    .hotspot-gallery figcaption { font-weight: 600; margin-bottom: 0.35rem; }
    footer { margin-top: 3rem; font-size: 0.85rem; opacity: 0.7; }
    """

    meta_table = _render_table(
        meta or {},
        "環境 / Environment / Environnement",
        ["seed", "python", "platform", "env", "versions"],
    ) if meta else "<p class=\"muted\">(no snapshot metadata — run with --snapshot)</p>"

    dep_table = ""
    deps = summary.get("dependencies") if isinstance(summary, dict) else None
    if isinstance(deps, dict) and deps:
        dep_table = _render_table(
            deps,
            "依存診断 / Dependency Check / Diagnostic",
        )

    core_table = _render_table(
        {
            "Output": summary.get("contextual_jsonl"),
            "Augmented": summary.get("mm_jsonl"),
            "Index": summary.get("index"),
            "Monitor": summary.get("monitor_csv"),
            "Profile": summary.get("profile_json"),
            "SQL CSV": summary.get("sql_csv"),
            "SQL schema": summary.get("sql_schema"),
            "Report": summary.get("report_html"),
        },
        "成果 / Artifacts / Artefacts",
    )

    info_table = _render_table(
        {
            "inputs": summary.get("inputs"),
            "page_images": summary.get("page_images"),
            "pages": summary.get("page_count"),
            "domain": summary.get("domain"),
            "seed": summary.get("seed"),
            "resume_requested": summary.get("resume_requested"),
            "resume_applied": summary.get("resume_applied"),
            "resume_steps": summary.get("resume_steps"),
            "snapshot": summary.get("snapshot"),
            "tune_budget": summary.get("tune_budget"),
            "generated_at": summary.get("generated_at"),
        },
        "概要 / Overview / Aperçu",
    )

    plugins = summary.get("plugins") or {}
    if plugins:
        plugin_rows = []
        for stage, fns in sorted(plugins.items()):
            names = ", ".join(escape(str(fn)) for fn in fns) or "–"
            plugin_rows.append(f"<tr><th scope=\"row\">{escape(stage)}</th><td>{names}</td></tr>")
        plugin_html = (
            "<section><h2>プラグイン / Plugins / Extensions</h2><table class=\"kv\">" +
            "".join(plugin_rows) + "</table></section>"
        )
    else:
        plugin_html = "<section><h2>プラグイン / Plugins / Extensions</h2><p class=\"muted\">(no plugins registered)</p></section>"

    monitor_html = _render_table(summary.get("monitor_row"), "モニタ / Monitor / Surveillance") if summary.get("monitor_row") else ""
    tune_html = _render_table(summary.get("tune"), "自動調整 / Tuning / Ajustement") if summary.get("tune") else ""
    learn_html = _render_table(summary.get("learn"), "学習 / Learning / Apprentissage") if summary.get("learn") else ""
    hotspot_html = _render_hotspots_section(summary)

    history_html = _render_history_table(history)

    stats_text = []
    if total_ms is not None:
        stats_text.append(f"総処理時間 / Total / Total : {total_ms:,.1f} ms")
    if total_s is not None:
        stats_text.append(f"≈ {total_s:,.2f} s")
    if ok_count is not None or fail_count is not None:
        stats_text.append(
            "成否 / Status : OK={ok} / FAIL={fail}".format(
                ok=ok_count if ok_count is not None else "–",
                fail=fail_count if fail_count is not None else "–",
            )
        )
    stats_block = "<p class=\"muted\">" + " ・ ".join(stats_text) + "</p>" if stats_text else ""

    pip_html = ""
    if meta and meta.get("pip_freeze"):
        pip_lines = "\n".join(meta["pip_freeze"][:200])
        extra = ""
        if len(meta["pip_freeze"]) > 200:
            extra = f"\n… ({len(meta['pip_freeze']) - 200} more)"
        pip_html = (
            "<details><summary>pip freeze</summary><pre>" + escape(pip_lines + extra) + "</pre></details>"
        )

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>ZOCR Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>ZOCR Pipeline Report / パイプラインレポート / Rapport</h1>
  <p>outdir: <code>{escape(os.path.abspath(outdir))}</code></p>
  {stats_block}
  {info_table}
  {core_table}
  {monitor_html}
  {tune_html}
  {learn_html}
  {hotspot_html}
  {plugin_html}
  <section>
    <h2>履歴 / History / Historique</h2>
    {history_html}
  </section>
  <section>
    <h2>環境 / Environment / Environnement</h2>
    {meta_table}
    {dep_table}
    {pip_html}
  </section>
  <footer>
    Generated at {escape(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))}
  </footer>
</body>
</html>
"""

    with open(dest, "w", encoding="utf-8") as fw:
        fw.write(html)
    return dest
