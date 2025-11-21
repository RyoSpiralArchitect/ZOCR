# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

# -*- coding: utf-8 -*-
"""Semantic diff core utilities."""
from __future__ import annotations

import difflib
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .metrics import (
    summarize_numeric_events,
    summarize_section_events,
    summarize_textual_events,
)
from .scoring import estimate_confidence

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fuzz = None


# --------- ユーティリティ ---------
def load_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                # 壊れた行は無視（必要ならログする）
                continue
    return items


def norm_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    return " ".join(str(x).strip().split())


def try_number(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.replace(",", "").replace("¥", "").replace("$", "").strip()
        try:
            return float(s)
        except ValueError:
            return None
    return None


def str_sim(a: str, b: str) -> float:
    a, b = norm_text(a), norm_text(b)
    if not a and not b:
        return 1.0
    if fuzz:
        # token_sort_ratio is order insensitive yet fast; normalize to 0..1
        return fuzz.token_sort_ratio(a, b) / 100.0
    return difflib.SequenceMatcher(a=a, b=b).ratio()


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(map(norm_text, a)), set(map(norm_text, b))
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def clip_preview(text: Optional[str], limit: int = 160) -> Optional[str]:
    if not text:
        return None
    clean = norm_text(text)
    if not clean:
        return None
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


# --------- データ抽出 ---------
@dataclass
class Cell:
    page: int
    table_index: int
    row: int
    col: int
    text: str
    header: Optional[str]
    filters: Dict[str, Any]
    trace_id: Optional[str]


def to_cell(rec: dict) -> Cell:
    page = int(rec.get("page", -1))
    table_index = int(rec.get("table_index", rec.get("table", -1)))
    row = int(rec.get("row", -1))
    col = int(rec.get("col", -1))
    text = rec.get("text") or rec.get("value") or ""
    header = None
    if "header" in rec:
        header = rec["header"]
    elif "headers" in rec and isinstance(rec["headers"], list) and col < len(rec["headers"]):
        header = rec["headers"][col]
    filters = rec.get("filters", {}) or rec.get("norm", {}) or {}
    trace_id = rec.get("trace_id")
    return Cell(page, table_index, row, col, text, header, filters, trace_id)


@dataclass
class RowMeta:
    signature: str
    ids: List[str]
    dates: List[str]


@dataclass
class TableView:
    id_hint: str
    headers: List[str]
    rows: Dict[str, Dict[int, Cell]]
    row_keys: List[str]
    row_meta: Dict[str, RowMeta]
    column_signatures: Dict[int, str]
    page: int
    table_index: int


@dataclass
class SectionEntry:
    title: str
    level: int
    page: Optional[int]
    order: int
    trace_id: Optional[str]


def table_context(table: TableView) -> Dict[str, Any]:
    return {
        "table_id": table.id_hint,
        "table_page": table.page,
        "table_index": table.table_index,
        "table_rows": len(table.row_keys),
        "table_columns": len(table.headers),
        "table_headers": table.headers[: min(len(table.headers), 16)],
        "table_header_preview": " | ".join(table.headers[:6]),
    }


def row_meta_fields(meta: Optional[RowMeta], prefix: str = "") -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if not meta:
        return data
    preview = clip_preview(meta.signature)
    if preview:
        data[f"{prefix}row_preview"] = preview
    if meta.ids:
        data[f"{prefix}row_ids"] = meta.ids
    if meta.dates:
        data[f"{prefix}row_dates"] = meta.dates
    return data


def normalize_filter_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("value", "text", "id"):
            if key in value:
                return normalize_filter_value(value[key])
    return None


def ensure_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path if path.exists() else None


def build_tables(cells: List[Cell]) -> Dict[str, TableView]:
    groups: Dict[Tuple[int, int], List[Cell]] = defaultdict(list)
    for c in cells:
        groups[(c.page, c.table_index)].append(c)

    tables: Dict[str, TableView] = {}
    for (page, t_idx), arr in groups.items():
        by_row: Dict[int, List[Cell]] = defaultdict(list)
        for c in arr:
            by_row[c.row].append(c)

        headers_by_col: Dict[int, List[str]] = defaultdict(list)
        for c in arr:
            if c.header:
                headers_by_col[c.col].append(norm_text(c.header))
        max_col = max((c.col for c in arr), default=-1)
        headers: List[str] = []
        for col in range(max_col + 1):
            if headers_by_col[col]:
                vals = headers_by_col[col]
                headers.append(max(set(vals), key=vals.count))
            else:
                headers.append(f"col{col}")

        rows: Dict[str, Dict[int, Cell]] = {}
        row_keys: List[str] = []
        row_meta: Dict[str, RowMeta] = {}
        header_row_skip: Optional[int] = None
        if by_row:
            first_idx = min(by_row.keys())
            row_cells = by_row[first_idx]
            if row_cells and all(
                c.col < len(headers) and norm_text(c.text) == norm_text(headers[c.col]) for c in row_cells
            ):
                header_row_skip = first_idx

        for r_idx in sorted(by_row.keys()):
            if header_row_skip is not None and r_idx == header_row_skip:
                continue
            row_cells = by_row[r_idx]
            ids = [
                norm_text(v)
                for c in row_cells
                for v in [normalize_filter_value(c.filters.get("id"))]
                if v
            ]
            dates = [
                norm_text(v)
                for c in row_cells
                for key in ("date", "datetime")
                for v in [normalize_filter_value(c.filters.get(key))]
                if v
            ]
            first = next((c for c in row_cells if c.col == 0 and norm_text(c.text)), None)
            candidate = norm_text(first.text) if first else ""
            if not candidate:
                candidate = norm_text(" | ".join(c.text for c in sorted(row_cells, key=lambda x: x.col)))
            if ids:
                row_key = f"id:{ids[0]}"
            elif dates:
                row_key = f"date:{dates[0]}"
            else:
                row_key = candidate if candidate else f"r{r_idx}:{sha1(candidate)}"
            orig_key = row_key
            k_i = 1
            while row_key in rows:
                k_i += 1
                row_key = f"{orig_key}#{k_i}"
            row_dict = {c.col: c for c in row_cells}
            rows[row_key] = row_dict
            row_keys.append(row_key)
            signature = norm_text(
                " | ".join(norm_text(row_dict[c].text) for c in sorted(row_dict.keys()))
            )
            row_meta[row_key] = RowMeta(signature=signature, ids=ids, dates=dates)

        sig = f"page={page};table={t_idx}"
        hdr_sig = sha1("|".join(headers))
        table_id = f"{sig};sig={hdr_sig}"

        column_signatures: Dict[int, str] = {}
        for col in range(max_col + 1):
            values: List[str] = []
            for rk in row_keys:
                cell = rows[rk].get(col)
                if cell and norm_text(cell.text):
                    values.append(norm_text(cell.text))
            column_signatures[col] = "\n".join(values)

        tables[table_id] = TableView(
            id_hint=table_id,
            headers=headers,
            rows=rows,
            row_keys=row_keys,
            row_meta=row_meta,
            column_signatures=column_signatures,
            page=page,
            table_index=t_idx,
        )
    return tables


def build_sections(path: Optional[Path]) -> List[SectionEntry]:
    if path is None:
        return []
    entries: List[SectionEntry] = []
    for order, item in enumerate(load_jsonl(path)):
        title = norm_text(item.get("title") or item.get("heading") or item.get("text") or "")
        if not title:
            continue
        level_raw = item.get("level") or item.get("heading_level") or item.get("depth")
        try:
            level = int(level_raw)
        except (TypeError, ValueError):
            level = 1
        page = item.get("page")
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        entries.append(
            SectionEntry(
                title=title,
                level=level,
                page=page,
                order=order,
                trace_id=item.get("trace_id"),
            )
        )
    return entries


# --------- マッチング ---------
def match_tables(A: Dict[str, TableView], B: Dict[str, TableView]) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    unmatched_B = set(B.keys())
    pairs: List[Tuple[str, str]] = []
    for a_id, a in A.items():
        best, best_s = None, -1.0
        for b_id in unmatched_B:
            s = jaccard(a.headers, B[b_id].headers)
            if s > best_s:
                best, best_s = b_id, s
        if best is not None and best_s >= 0.30:
            pairs.append((a_id, best))
            unmatched_B.remove(best)
    removed = [a_id for a_id in A.keys() if a_id not in {p[0] for p in pairs}]
    added = list(unmatched_B)
    return pairs, added, removed


def column_similarity(a: TableView, b: TableView, ai: int, bi: int) -> float:
    header_sim = str_sim(a.headers[ai], b.headers[bi])
    sig_a = a.column_signatures.get(ai, "")
    sig_b = b.column_signatures.get(bi, "")
    vector_sim = str_sim(sig_a, sig_b) if sig_a or sig_b else 0.0
    return max(header_sim, vector_sim)


def linear_assignment_from_similarity(sim_matrix: List[List[float]], threshold: float) -> Dict[int, int]:
    if not sim_matrix or not sim_matrix[0]:
        return {}
    n_rows = len(sim_matrix)
    n_cols = len(sim_matrix[0])
    size = max(n_rows, n_cols)
    max_val = max((s for row in sim_matrix for s in row), default=0.0)
    cost_matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            sim = sim_matrix[i][j] if i < n_rows and j < n_cols else 0.0
            cost_matrix[i][j] = max_val - sim

    assignment = hungarian(cost_matrix)
    result: Dict[int, int] = {}
    for i, j in assignment:
        if i < n_rows and j < n_cols:
            if sim_matrix[i][j] >= threshold:
                result[i] = j
    return result


def hungarian(cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    # Implementation of the Kuhn-Munkres algorithm (Hungarian method)
    if not cost_matrix:
        return []
    n = len(cost_matrix)
    m = len(cost_matrix[0]) if cost_matrix else 0
    assert n == m, "Hungarian algorithm expects a square matrix"
    size = n
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)

    for i in range(1, size + 1):
        p[0] = i
        minv = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment: List[Tuple[int, int]] = []
    for j in range(1, size + 1):
        if p[j]:
            assignment.append((p[j] - 1, j - 1))
    return assignment


def match_columns(a: TableView, b: TableView) -> Dict[int, int]:
    if not a.headers or not b.headers:
        return {}
    sim_matrix: List[List[float]] = []
    for ai in range(len(a.headers)):
        row_vals = []
        for bi in range(len(b.headers)):
            row_vals.append(column_similarity(a, b, ai, bi))
        sim_matrix.append(row_vals)
    assignment = linear_assignment_from_similarity(sim_matrix, threshold=0.3)
    return {ai: bi for ai, bi in assignment.items()}


def row_similarity(a: TableView, b: TableView, ra: str, rb: str) -> float:
    if norm_text(ra) == norm_text(rb):
        return 1.0
    meta_a = a.row_meta.get(ra)
    meta_b = b.row_meta.get(rb)
    if not meta_a or not meta_b:
        return 0.0
    if meta_a.ids and meta_b.ids and set(meta_a.ids) & set(meta_b.ids):
        return 1.0
    if meta_a.dates and meta_b.dates and set(meta_a.dates) & set(meta_b.dates):
        return 0.9
    return str_sim(meta_a.signature, meta_b.signature)


def match_rows(a: TableView, b: TableView, col_map: Dict[int, int]) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    if not a.row_keys or not b.row_keys:
        return [], list(b.row_keys), list(a.row_keys)
    sim_matrix: List[List[float]] = []
    for ra in a.row_keys:
        row_vals = []
        for rb in b.row_keys:
            row_vals.append(row_similarity(a, b, ra, rb))
        sim_matrix.append(row_vals)
    assignment = linear_assignment_from_similarity(sim_matrix, threshold=0.5)
    used_a = set(assignment.keys())
    used_b = set(assignment.values())
    pairs = [(a.row_keys[ai], b.row_keys[bi]) for ai, bi in assignment.items()]
    removed = [rk for idx, rk in enumerate(a.row_keys) if idx not in used_a]
    added = [rk for idx, rk in enumerate(b.row_keys) if idx not in used_b]
    return pairs, added, removed


# --------- 差分生成 ---------
class SemanticDiffer:
    def __init__(self, text_sim_threshold: float = 0.85, numeric_rel_eps: float = 1e-9):
        self.text_sim_threshold = text_sim_threshold
        self.numeric_rel_eps = numeric_rel_eps

    def compare_cells(self, a: Optional[Cell], b: Optional[Cell]) -> Optional[dict]:
        if a is None and b is None:
            return None
        if a is None:
            event = {
                "type": "cell_updated",
                "old": None,
                "new": b.text,
                "similarity": 0.0,
                "numeric_delta": None,
                "relative_delta": None,
                "trace_a": None,
                "trace_b": b.trace_id,
            }
            event["confidence"] = estimate_confidence(similarity=0.0)
            return event
        if b is None:
            event = {
                "type": "cell_updated",
                "old": a.text,
                "new": None,
                "similarity": 0.0,
                "numeric_delta": None,
                "relative_delta": None,
                "trace_a": a.trace_id,
                "trace_b": None,
            }
            event["confidence"] = estimate_confidence(similarity=0.0)
            return event

        a_txt, b_txt = norm_text(a.text), norm_text(b.text)
        if a_txt == b_txt:
            return None

        a_num = (
            a.filters.get("amount")
            or a.filters.get("qty")
            or a.filters.get("rate")
            or try_number(a_txt)
        )
        b_num = (
            b.filters.get("amount")
            or b.filters.get("qty")
            or b.filters.get("rate")
            or try_number(b_txt)
        )

        numeric_delta = relative_delta = None
        sim = str_sim(a_txt, b_txt)
        if a_num is not None and b_num is not None:
            try:
                numeric_delta = float(b_num) - float(a_num)
                denom = abs(a_num) if abs(a_num) > self.numeric_rel_eps else 1.0
                relative_delta = numeric_delta / denom
            except Exception:
                pass

        event = {
            "type": "cell_updated",
            "old": a_txt,
            "new": b_txt,
            "similarity": sim,
            "numeric_delta": numeric_delta,
            "relative_delta": relative_delta,
            "trace_a": a.trace_id,
            "trace_b": b.trace_id,
        }
        event["confidence"] = estimate_confidence(
            similarity=sim, relative_delta=relative_delta, numeric_delta=numeric_delta
        )
        return event

    def compare_tables(self, a: TableView, b: TableView) -> List[dict]:
        events: List[dict] = []
        col_map = match_columns(a, b)

        def add_row_event(table: TableView, row_key: str, event_type: str, row_side: str) -> None:
            payload: Dict[str, Any] = {"type": event_type, **table_context(table)}
            payload["row_key"] = row_key
            if row_side.upper() == "A":
                payload["row_key_a"] = row_key
            else:
                payload["row_key_b"] = row_key
            payload["row_origin"] = row_side.upper()
            payload.update(row_meta_fields(table.row_meta.get(row_key)))
            events.append(payload)

        for ai, ah in enumerate(a.headers):
            if ai in col_map:
                bi = col_map[ai]
                bh = b.headers[bi]
                if norm_text(ah) != norm_text(bh):
                    if str_sim(ah, bh) >= 0.30:
                        events.append(
                            {
                                "type": "header_renamed",
                                "table_id": a.id_hint,
                                "from": ah,
                                "to": bh,
                                "a_col": ai,
                                "b_col": bi,
                            }
                        )
                if ai != bi:
                    events.append(
                        {
                            "type": "col_moved",
                            "table_id": a.id_hint,
                            "from_index": ai,
                            "to_index": bi,
                            "header": bh,
                        }
                    )

        row_pairs, row_added, row_removed = match_rows(a, b, col_map)
        for rk in row_added:
            add_row_event(b, rk, "row_added", row_side="B")
        for rk in row_removed:
            add_row_event(a, rk, "row_removed", row_side="A")

        for ra, rb in row_pairs:
            a_row = a.rows.get(ra, {})
            b_row = b.rows.get(rb, {})
            handled_b_cols = set()
            all_a_cols = set(a_row.keys()) | set(col_map.keys())
            for ai in sorted(all_a_cols):
                bi = col_map.get(ai)
                a_cell = a_row.get(ai)
                b_cell = b_row.get(bi) if bi is not None else None
                if bi is not None:
                    handled_b_cols.add(bi)
                diff = self.compare_cells(a_cell, b_cell)
                if diff:
                    diff.update({
                        **table_context(a),
                        "row_key": ra,
                        "row_key_a": ra,
                        "row_key_b": rb,
                        "a_col": ai,
                        "b_col": bi,
                    })
                    diff.update(row_meta_fields(a.row_meta.get(ra), prefix="a_"))
                    diff.update(row_meta_fields(b.row_meta.get(rb), prefix="b_"))
                    events.append(diff)

            # handle B-only columns that were not mapped
            for bi in sorted(set(b_row.keys()) - handled_b_cols):
                diff = self.compare_cells(None, b_row.get(bi))
                if diff:
                    diff.update({
                        **table_context(a),
                        "row_key": ra,
                        "row_key_a": ra,
                        "row_key_b": rb,
                        "a_col": None,
                        "b_col": bi,
                    })
                    diff.update(row_meta_fields(a.row_meta.get(ra), prefix="a_"))
                    diff.update(row_meta_fields(b.row_meta.get(rb), prefix="b_"))
                    events.append(diff)
        return events

    def compare_sections(self, sections_a: List[SectionEntry], sections_b: List[SectionEntry]) -> List[dict]:
        events: List[dict] = []
        if not sections_a and not sections_b:
            return events
        if not sections_a:
            for sec in sections_b:
                events.append({
                    "type": "section_added",
                    "title": sec.title,
                    "level": sec.level,
                    "page": sec.page,
                })
            return events
        if not sections_b:
            for sec in sections_a:
                events.append({
                    "type": "section_removed",
                    "title": sec.title,
                    "level": sec.level,
                    "page": sec.page,
                })
            return events

        sim_matrix: List[List[float]] = []
        for sa in sections_a:
            row_vals = []
            for sb in sections_b:
                title_sim = str_sim(sa.title, sb.title)
                level_bonus = 0.1 if sa.level == sb.level else 0.0
                row_vals.append(min(1.0, title_sim + level_bonus))
            sim_matrix.append(row_vals)
        assignment = linear_assignment_from_similarity(sim_matrix, threshold=0.55)
        matched_a = set(assignment.keys())
        matched_b = set(assignment.values())
        for ai, bi in assignment.items():
            sa = sections_a[ai]
            sb = sections_b[bi]
            if norm_text(sa.title) != norm_text(sb.title):
                events.append({
                    "type": "section_title_changed",
                    "old": sa.title,
                    "new": sb.title,
                    "level_a": sa.level,
                    "level_b": sb.level,
                    "page_a": sa.page,
                    "page_b": sb.page,
                })
            if sa.level != sb.level:
                events.append({
                    "type": "section_level_changed",
                    "title": sb.title,
                    "from_level": sa.level,
                    "to_level": sb.level,
                    "page_a": sa.page,
                    "page_b": sb.page,
                })
        for idx, sec in enumerate(sections_a):
            if idx not in matched_a:
                events.append({
                    "type": "section_removed",
                    "title": sec.title,
                    "level": sec.level,
                    "page": sec.page,
                })
        for idx, sec in enumerate(sections_b):
            if idx not in matched_b:
                events.append({
                    "type": "section_added",
                    "title": sec.title,
                    "level": sec.level,
                    "page": sec.page,
                })
        return events

    def compare_bundle(
        self,
        cells_a: Path,
        cells_b: Path,
        sections_a: Optional[Path] = None,
        sections_b: Optional[Path] = None,
    ) -> Dict[str, Any]:
        A = [to_cell(x) for x in load_jsonl(cells_a)]
        B = [to_cell(x) for x in load_jsonl(cells_b)]
        tables_a = build_tables(A)
        tables_b = build_tables(B)

        sections_a = ensure_path(sections_a or cells_a.with_name("sections.jsonl"))
        sections_b = ensure_path(sections_b or cells_b.with_name("sections.jsonl"))
        sec_a = build_sections(sections_a)
        sec_b = build_sections(sections_b)

        pairs, added, removed = match_tables(tables_a, tables_b)

        events: List[dict] = []
        for b_id in added:
            events.append({"type": "table_added", **table_context(tables_b[b_id])})
        for a_id in removed:
            events.append({"type": "table_removed", **table_context(tables_a[a_id])})

        for a_id, b_id in pairs:
            ev = self.compare_tables(tables_a[a_id], tables_b[b_id])
            for e in ev:
                e.setdefault("table_id_a", a_id)
                e.setdefault("table_id_b", b_id)
            events.extend(ev)

        section_events = self.compare_sections(sec_a, sec_b)
        events.extend(section_events)

        summary = {
            "tables_matched": len(pairs),
            "tables_added": len(added),
            "tables_removed": len(removed),
            "sections_compared": max(len(sec_a), len(sec_b)),
            "events": len(events),
            "section_events": len(section_events),
        }
        numeric_summary = summarize_numeric_events(events)
        if numeric_summary:
            summary["numeric_summary"] = numeric_summary
        textual_summary = summarize_textual_events(events)
        if textual_summary:
            summary["textual_summary"] = textual_summary
        section_summary = summarize_section_events(events)
        if section_summary:
            summary["section_summary"] = section_summary
        return {"events": events, "summary": summary}
