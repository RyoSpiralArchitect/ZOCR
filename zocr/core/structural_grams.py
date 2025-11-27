"""Structural gram extractor for bootstrap RAG."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

__all__ = [
    "UNIT_PATTERN",
    "ID_PATTERN",
    "NUM_PATTERN",
    "SYMBOL_PATTERN",
    "tokenize_cell_text",
    "classify_token",
    "extract_cell_structural_gram",
    "extract_structural_grams",
]

UNIT_PATTERN = re.compile(
    r"^(a|v|w|kw|kva|kv|mm|cm|m|mpa|bar|hz|rpm|pcs?|ea|set|kN|N|Nm|℃|°c|kgf/cm2|m3/h)$",
    re.IGNORECASE,
)
ID_PATTERN = re.compile(r"^[A-Z]{2,}[A-Z0-9-]*\d{2,}$")
NUM_PATTERN = re.compile(r"^-?\d+(?:[.,]\d+)?$")
SYMBOL_PATTERN = re.compile(r"^[±+\-×x＊*φΦØø°]+$")
_TOKEN_RE = re.compile(r"[A-Za-z]+[A-Za-z0-9-]*|\d+(?:[.,]\d+)?|[一-龥ぁ-んァ-ヶー]+|\S")
_RANGE_RE = re.compile(
    r"(-?\d[\d,]*(?:\.\d+)?)[\s]*[\-~〜ー–—toTO]{1,3}[\s]*(-?\d[\d,]*(?:\.\d+)?)"
)


def tokenize_cell_text(text: str) -> List[str]:
    """Tokenize a cell's text into a compact list of tokens.

    Falls back to whitespace splitting when the heuristic regex finds nothing.
    """

    tokens = _TOKEN_RE.findall(text or "")
    if tokens:
        return tokens
    return (text or "").split()


def classify_token(token: str) -> str:
    """Classify a token into NUM/UNIT/ID/SYM/TEXT buckets."""

    tok = (token or "").strip()
    tok = tok.strip("()[]{}")
    tok = tok.rstrip(".,;:、。")
    if not tok:
        return "TEXT"
    if NUM_PATTERN.match(tok):
        return "NUM"
    if UNIT_PATTERN.match(tok):
        return "UNIT"
    if ID_PATTERN.match(tok.upper()):
        return "ID"
    if SYMBOL_PATTERN.match(tok):
        return "SYM"
    return "TEXT"


def _coerce_float(val: str) -> Optional[float]:
    try:
        return float(val.replace(",", ""))
    except Exception:
        return None


def _infer_doc_meta(record: Dict[str, Any], cell: Dict[str, Any]) -> Dict[str, Any]:
    struct = record.get("struct") or {}
    meta = record.get("meta") or {}
    return {
        "doc_id": record.get("doc_id") or cell.get("doc_id"),
        "page": record.get("page") or cell.get("page"),
        "table_id": record.get("table_id")
        or struct.get("table_id")
        or cell.get("table_id"),
        "table_index": record.get("table_index") or cell.get("table_index"),
        "region_id": record.get("region_id") or cell.get("region_id"),
        "cell_id": record.get("cell_id") or cell.get("cell_id"),
        "row_role": meta.get("filters", {}).get("row_role")
        or cell.get("row_role"),
    }


def extract_cell_structural_gram(
    cell: Dict[str, Any],
    doc_meta: Optional[Dict[str, Any]] = None,
    max_ngram: int = 4,
) -> Dict[str, Any]:
    """Build a structural gram from a single cell-like dictionary."""

    doc_meta = doc_meta or {}
    text = cell.get("search_unit") or cell.get("text") or ""
    tokens = tokenize_cell_text(text)
    lexical_ngram = tokens[:max_ngram]
    type_seq = [classify_token(tok) for tok in tokens][:max_ngram]

    layout = {
        "row_type": cell.get("row_type")
        or (cell.get("meta") or {}).get("filters", {}).get("row_role")
        or doc_meta.get("row_role")
        or "body",
        "row_idx": cell.get("row_index")
        if "row_index" in cell
        else cell.get("row"),
        "col_idx": cell.get("col_index")
        if "col_index" in cell
        else cell.get("col"),
    }

    features = {
        "has_unit": any(t == "UNIT" for t in type_seq),
        "has_id": any(t == "ID" for t in type_seq),
        "has_symbol": any(t == "SYM" for t in type_seq),
    }

    numeric_value: Optional[float] = None
    unit: Optional[str] = None
    for tok, cls in zip(tokens, type_seq):
        if cls == "NUM" and numeric_value is None:
            numeric_value = _coerce_float(tok)
        if cls == "UNIT" and unit is None:
            unit = tok
        if numeric_value is not None and unit:
            break

    if numeric_value is not None:
        features["numeric_value"] = numeric_value
    if unit is not None:
        features["unit"] = unit

    range_match = _RANGE_RE.search(text)
    if range_match:
        lower = _coerce_float(range_match.group(1))
        upper = _coerce_float(range_match.group(2))
        if lower is not None and upper is not None:
            features["range"] = [lower, upper]

    def _coalesce(seq: List[str]) -> List[str]:
        collapsed: List[str] = []
        for item in seq:
            if not collapsed or collapsed[-1] != item:
                collapsed.append(item)
        return collapsed

    return {
        "kind": "structural_gram",
        "lexical_ngram": lexical_ngram,
        "type_ngram": type_seq,
        "signatures": {
            "type_signature": "-".join(type_seq),
            "type_signature_coalesced": "-".join(_coalesce(type_seq)),
        },
        "layout": layout,
        "value_features": features,
        "doc_meta": {k: v for k, v in doc_meta.items() if v is not None},
    }


def _resolve_cell(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cell = record.get("cell")
    if isinstance(cell, dict):
        return cell
    if record:
        return record
    return None


def extract_structural_grams(
    jsonl_in: str, jsonl_out: str, max_ngram: int = 4
) -> int:
    """Extract structural grams from a ``cells.jsonl``-like file."""

    count = 0
    with open(jsonl_in, "r", encoding="utf-8") as fin, open(
        jsonl_out, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            cell = _resolve_cell(record)
            if not isinstance(cell, dict):
                continue
            gram = extract_cell_structural_gram(
                cell, _infer_doc_meta(record, cell), max_ngram=max_ngram
            )
            fout.write(json.dumps(gram, ensure_ascii=False) + "\n")
            count += 1
    return count
