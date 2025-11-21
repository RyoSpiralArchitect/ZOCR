# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

"""Augmentation utilities (phash, filters, λ schedule)."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from PIL import Image
except Exception:  # pragma: no cover - fallback stub
    Image = None  # type: ignore

from .base import (
    RX_PHONE,
    RX_POST,
    RX_TAXID,
    infer_row_fields,
    lambda_schedule,
    norm_address,
    norm_amount,
    norm_company,
    norm_date,
    phash64,
    tiny_vec,
)

__all__ = ["augment"]


def _inject_structural_placeholders(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tables: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = defaultdict(
        lambda: {
            "cells": [],
            "rows": defaultdict(dict),
            "template": None,
        }
    )

    def _is_weak_cell(ob: Dict[str, Any]) -> bool:
        meta = ob.get("meta") or {}
        text = (ob.get("text") or "").strip()
        if not text:
            return True
        if meta.get("low_conf"):
            return True
        conf = meta.get("confidence")
        try:
            conf_f = float(conf) if conf is not None else None
        except Exception:
            conf_f = None
        return conf_f is not None and conf_f < 0.55

    def _key(ob: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        return (
            ob.get("doc_id"),
            ob.get("page"),
            ob.get("table_index"),
        )

    for ob in records:
        key = _key(ob)
        tbl = tables[key]
        tbl["cells"].append(ob)
        if tbl["template"] is None:
            tbl["template"] = ob
        try:
            r = int(ob.get("row"))
            c = int(ob.get("col"))
        except Exception:
            continue
        tbl["rows"].setdefault(r, {})[c] = ob

    def _contains_cjk(text: str) -> bool:
        return any(ord(ch) > 127 for ch in text or "")

    families = [
        ("Item", "品目"),
        ("Qty", "数量"),
        ("Unit Price", "単価"),
        ("Amount", "金額"),
    ]
    total_variants = ["Total", "Total Amount", "合計", "税込合計"]

    def _make_cell(template: Dict[str, Any], row: int, col: int, text: str, role: str) -> Dict[str, Any]:
        base_meta = dict(template.get("meta") or {})
        filters = dict(base_meta.get("filters") or {})
        filters.setdefault("row_role", role)
        base_meta["filters"] = filters
        base_meta["synthetic"] = True
        base_meta["synthetic_role"] = role
        base_meta.setdefault("confidence", 0.0)
        base_meta.setdefault("low_conf", False)
        return {
            "doc_id": template.get("doc_id"),
            "page": template.get("page"),
            "table_index": template.get("table_index"),
            "row": row,
            "col": col,
            "text": text,
            "search_unit": text,
            "synthesis_window": template.get("synthesis_window") or "",
            "bbox": template.get("bbox") or [0, 0, 0, 0],
            "meta": base_meta,
        }

    synthetic: List[Dict[str, Any]] = []
    for tbl in tables.values():
        if not tbl["cells"]:
            continue
        rows = tbl["rows"]
        template = tbl["template"]
        header_row = min(rows.keys()) if rows else 0
        header_cells = rows.get(header_row, {})
        header_text = " ".join((header_cells[c].get("text") or "") for c in sorted(header_cells))
        use_jp = _contains_cjk(header_text)
        max_col = max((c for cols in rows.values() for c in cols.keys()), default=-1)
        header_lower = header_text.lower()
        for family in families:
            present = False
            for variant in family:
                if variant.lower() in header_lower:
                    present = True
                    break
            if present:
                continue
            variant = family[1] if use_jp else family[0]
            max_col += 1
            new_cell = _make_cell(template, header_row, max_col, variant, "header")
            rows.setdefault(header_row, {})[max_col] = new_cell
            tbl["cells"].append(new_cell)
            synthetic.append(new_cell)
            header_cells[max_col] = new_cell

        total_present = False
        weak_total = None
        for ob in tbl["cells"]:
            txt = ((ob.get("text") or "") + " " + (ob.get("synthesis_window") or "")).lower()
            if any(variant.lower() in txt for variant in total_variants):
                total_present = True
                if _is_weak_cell(ob):
                    weak_total = ob
                break
        footer_text = "合計" if any(_contains_cjk((ob.get("text") or "")) for ob in tbl["cells"]) else "Total"
        if weak_total is not None:
            meta = dict(weak_total.get("meta") or {})
            filters = dict(meta.get("filters") or {})
            filters.setdefault("row_role", "footer")
            meta.update(
                {
                    "filters": filters,
                    "synthetic": True,
                    "synthetic_role": "footer",
                    "low_conf": False,
                    "confidence": max(
                        0.55,
                        float(meta.get("confidence") or 0.0)
                        if isinstance(meta.get("confidence"), (int, float))
                        else 0.55,
                    ),
                }
            )
            weak_total.update(
                {
                    "text": footer_text,
                    "search_unit": footer_text,
                    "synthesis_window": weak_total.get("synthesis_window") or footer_text,
                    "meta": meta,
                }
            )
        elif not total_present:
            footer_row = (max(rows.keys()) if rows else header_row) + 1
            footer_col = max_col if max_col >= 0 else 0
            new_footer = _make_cell(template, footer_row, footer_col, footer_text, "footer")
            rows.setdefault(footer_row, {})[footer_col] = new_footer
            tbl["cells"].append(new_footer)
            synthetic.append(new_footer)

    if synthetic:
        records = records + synthetic
    return records


def augment(
    jsonl_in: str,
    jsonl_out: str,
    lambda_shape: float = 4.5,
    lambda_refheight: int = 1000,
    lambda_alpha: float = 0.7,
    org_dict_path: Optional[str] = None,
) -> int:
    if Image is None:
        raise RuntimeError("Pillow is required for augment")
    org_dict = None
    if org_dict_path and os.path.exists(org_dict_path):
        try:
            with open(org_dict_path, "r", encoding="utf-8") as f:
                org_dict = json.load(f)
        except Exception:
            org_dict = None
    records: List[Dict[str, Any]] = []
    cur = None
    img = None
    with open(jsonl_in, "r", encoding="utf-8") as fr:
        for line in fr:
            ob = json.loads(line)
            ip = ob.get("image_path")
            bbox = ob.get("bbox", [0, 0, 0, 0])
            page_h = None
            if ip and os.path.exists(ip):
                if ip != cur:
                    img = Image.open(ip).convert("RGB")
                    cur = ip
                page_h = img.height
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.width, x2)
                y2 = min(img.height, y2)
                crop = img.crop((x1, y1, x2, y2))
                try:
                    ph = phash64(crop)
                except Exception:
                    ph = 0
                vec = tiny_vec(crop, 16).tolist()
                ob.setdefault("meta", {})
                ob["meta"]["phash64"] = ph
                ob["meta"]["img16"] = vec
                if page_h:
                    ob["meta"]["lambda_shape"] = lambda_schedule(
                        page_h, lambda_shape, lambda_refheight, lambda_alpha
                    )
            txt = (ob.get("text") or "") + " " + (ob.get("synthesis_window") or "")
            swin = ob.get("synthesis_window") or ""
            filt = (ob.get("meta") or {}).get("filters", {})
            filt["amount"] = filt.get("amount") or norm_amount(txt)
            filt["date"] = filt.get("date") or norm_date(txt)
            t = RX_TAXID.search(txt)
            filt["tax_id"] = filt.get("tax_id") or (t.group(0) if t else None)
            p = RX_POST.search(txt)
            filt["postal_code"] = filt.get("postal_code") or (p.group(0) if p else None)
            phn = RX_PHONE.search(txt)
            filt["phone"] = filt.get("phone") or (phn.group(0) if phn else None)
            comp, corp_id = norm_company(txt, org_dict)
            addr = norm_address(txt)
            if comp:
                filt["company"] = comp
            if corp_id:
                filt["corporate_id"] = corp_id
            if addr:
                filt["address"] = addr
            rowf = infer_row_fields(swin)
            for k, v in rowf.items():
                if filt.get(k) is None:
                    filt[k] = v
            if (
                filt.get("tax_amount") is None
                and filt.get("tax_rate") is not None
                and filt.get("subtotal") is not None
            ):
                filt["tax_amount"] = int(
                    round(float(filt["subtotal"]) * float(filt["tax_rate"]))
                )
            ob.setdefault("meta", {})
            ob["meta"]["filters"] = filt
            records.append(ob)

    records = _inject_structural_placeholders(records)

    with open(jsonl_out, "w", encoding="utf-8") as fw:
        for ob in records:
            fw.write(json.dumps(ob, ensure_ascii=False) + "\n")
    return len(records)
