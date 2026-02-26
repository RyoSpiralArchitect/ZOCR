"""Lightweight helpers for surfacing retrieval payloads in notebooks or UIs."""
from __future__ import annotations

import csv
import html
from io import StringIO
from typing import Any, Dict, Optional

__all__ = [
    "thumbnail_img_tag",
    "table_html_fragment",
    "render_fragment",
    "data_uri",
]


def data_uri(thumbnail_b64: Optional[str]) -> str:
    if not thumbnail_b64:
        return ""
    if thumbnail_b64.startswith("data:image"):
        return thumbnail_b64
    return f"data:image/png;base64,{thumbnail_b64}"


def thumbnail_img_tag(payload: Dict[str, Any], alt: str = "figure") -> str:
    uri = data_uri(payload.get("thumbnail_b64"))
    if not uri:
        return ""
    caption = html.escape(payload.get("caption") or alt)
    return f'<img src="{uri}" alt="{caption}" style="max-width:240px;" />'


def table_html_fragment(payload: Dict[str, Any]) -> str:
    if payload.get("html"):
        return payload["html"]
    csv_data = payload.get("csv")
    if not csv_data:
        return ""
    sio = StringIO(csv_data)
    reader = csv.reader(sio)
    rows = list(reader)
    trs = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        trs.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(trs) + "</table>"


def render_fragment(payload: Dict[str, Any]) -> str:
    if not payload:
        return ""
    obj_type = payload.get("type")
    if obj_type == "figure":
        return thumbnail_img_tag(payload)
    if obj_type == "table":
        return table_html_fragment(payload)
    snippet = payload.get("snippet") or payload.get("caption")
    return html.escape(snippet) if snippet else ""
