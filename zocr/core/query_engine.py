# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

"""Retrieval helpers combining BM25, keywords and symbolic filters."""
from __future__ import annotations

import json
import os
import pickle
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .._compat import optional_numpy

np = optional_numpy(__name__)
try:  # pragma: no cover - optional dependency
    from PIL import Image
except Exception:  # pragma: no cover - fallback stub
    Image = None  # type: ignore

from .base import hamm64, phash64, _normalize_text
from .domains import DOMAIN_KW
from .indexer import _bm25_numba_score, _bm25_py_score
from .numba_support import HAS_NUMBA
from .tokenization import tokenize_jp

__all__ = ["query", "hybrid_query"]


def _phash_sim(q_img_path: Optional[str], ph: int) -> float:
    if not q_img_path or not os.path.exists(q_img_path) or ph == 0:
        return 0.0
    if Image is None:
        return 0.0
    try:
        qi = Image.open(q_img_path).convert("RGB")
        qh = phash64(qi)
    except Exception:
        return 0.0
    hd = hamm64(int(qh), int(ph))
    return 1.0 - (hd / 64.0)


def _kw_meta_boost(ob: Dict[str, Any], q_toks: List[str], domain: str = "invoice") -> float:
    text = ((ob.get("synthesis_window") or "") + " " + (ob.get("text") or "")).lower()
    filt = (ob.get("meta") or {}).get("filters", {})
    s = 0.0
    nums = [int(t.replace(",", "")) for t in q_toks if t.replace(",", "").isdigit()]
    if filt.get("amount") and any(abs(int(filt["amount"]) - n) <= 5 for n in nums):
        s += 1.5
    if filt.get("date"):
        for d in re.findall(r"\d+", str(filt["date"])):
            if d in q_toks:
                s += 0.3
    for kw, weight in DOMAIN_KW.get(domain, DOMAIN_KW["invoice"]):
        if kw and kw.lower() in text:
            s += weight
    return s


def _symbolic_match_score(filters: Dict[str, Any], q_text: str, tokens: Sequence[str]) -> float:
    if not filters or not q_text:
        return 0.0
    q_norm = _normalize_text(q_text)
    tokens = list(tokens)
    digits = re.findall(r"\d+(?:[.,]\d+)?", q_text)
    score = 0.0
    seen_keys = set()
    for key, val in filters.items():
        if key in (None, "trace") or val is None:
            continue
        val_str = str(val)
        if not val_str:
            continue
        val_norm = _normalize_text(val_str)
        if not val_norm:
            continue
        key_hits = 0.0
        if q_norm and val_norm in q_norm:
            key_hits += 2.0
        for tok in tokens:
            if tok and tok in val_norm:
                key_hits += 1.0
        for d in digits:
            dd = d.replace(",", "").replace(".", "")
            if dd and dd in re.sub(r"\D", "", val_str):
                key_hits += 1.5
        if key_hits > 0:
            seen_keys.add(key)
            score += key_hits
    if seen_keys:
        score += min(len(seen_keys), 4) * 0.5
    return float(score)


def _cosine_sim(a: Optional[Iterable[float]], b: Optional[Iterable[float]]) -> float:
    if a is None or b is None:
        return 0.0
    try:
        va = np.array(list(a), dtype=np.float64)
        vb = np.array(list(b), dtype=np.float64)
    except Exception:
        return 0.0
    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return 0.0
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _zone_match_score(zone: Optional[str], zone_filter: Optional[str]) -> float:
    if not zone or not zone_filter:
        return 0.0
    normalized_filter = zone_filter
    if "\\\\" in normalized_filter:
        try:
            normalized_filter = normalized_filter.encode("utf-8").decode("unicode_escape")
        except Exception:
            normalized_filter = zone_filter
    try:
        return 1.0 if re.match(normalized_filter, zone) else 0.0
    except re.error:
        return 0.0


def _header_match_score(struct: Dict[str, Any], boosts: Dict[str, Any]) -> float:
    if not boosts:
        return 0.0
    header_norm = struct.get("header_norm") or struct.get("col")
    if not header_norm:
        return 0.0
    headers = boosts.get("header_norm") or []
    header_norm = str(header_norm).lower()
    score = 0.0
    for h in headers:
        if header_norm == str(h).lower():
            score += 1.0
    if score == 0.0 and boosts.get("header_startswith"):
        for h in boosts["header_startswith"]:
            if header_norm.startswith(str(h).lower()):
                score += 0.5
    return float(score)


def _confidence_penalty(confidence: Dict[str, Any]) -> float:
    if not confidence:
        return 0.0
    try:
        ocr = float(confidence.get("ocr", 1.0))
    except Exception:
        ocr = 1.0
    try:
        struct = float(confidence.get("structure", 1.0))
    except Exception:
        struct = 1.0
    return float((1.0 - max(min(ocr, 1.0), 0.0)) * (1.0 - max(min(struct, 1.0), 0.0)))


def _neighbor_boost(
    region_id: Optional[str],
    neighbor_graph: Optional[Dict[str, Sequence[str]]],
    seeds: Sequence[str],
) -> float:
    if not region_id or not neighbor_graph or not seeds:
        return 0.0
    neigh = neighbor_graph.get(region_id) or []
    for s in seeds:
        if s in neigh:
            return 1.0
    return 0.0


def _struct_filter(struct: Dict[str, Any], struct_filters: Dict[str, Any]) -> bool:
    for key, val in (struct_filters or {}).items():
        if struct.get(key) != val:
            return False
    return True


def _passes_filters(obj: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Check top-level/meta fields against caller-provided filters.

    Supports equality checks for scalar values as well as membership checks when
    the filter value is a list/tuple/set. A string filter starting with ``"re:"``
    is treated as a regex pattern anchored at the start of the field.
    """

    for key, expected in (filters or {}).items():
        if key in (None, "trace"):
            continue

        actual = obj.get(key)
        if actual is None:
            actual = (obj.get("meta") or {}).get(key)

        if actual is None:
            return False

        # Regex filter: re:<pattern>
        if isinstance(expected, str) and expected.startswith("re:"):
            try:
                if not re.match(expected[3:], str(actual)):
                    return False
            except re.error:
                return False
            continue

        # Membership filter
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
            continue

        # Scalar equality (including dicts)
        if expected is not None and actual != expected:
            return False

    return True


def query(
    index_pkl: str,
    jsonl: str,
    q_text: str = "",
    q_image: Optional[str] = None,
    topk: int = 10,
    w_bm25: float = 1.0,
    w_kw: float = 0.6,
    w_img: float = 0.3,
    w_sym: float = 0.45,
    domain: str = "invoice",
):
    with open(index_pkl, "rb") as f:
        ix = pickle.load(f)
    vocab = ix["vocab"]
    df = np.array(ix["df"], dtype=np.int32)
    N = int(ix["N"])
    avgdl = float(ix["avgdl"])
    raws: List[Dict[str, Any]] = []
    with open(jsonl, "r", encoding="utf-8") as fr:
        for line in fr:
            raws.append(json.loads(line))
    q_ids: List[int] = []
    toks = tokenize_jp(q_text or "")
    for t in toks:
        if t in vocab:
            q_ids.append(vocab[t])
    q_arr = np.array(q_ids, dtype=np.int32) if q_ids else np.array([-1], dtype=np.int32)
    results: List[Tuple[float, Dict[str, Any], int]] = []
    for i, doc_ids in enumerate(ix["docs_tokens"]):
        di = np.array(doc_ids + [-1], dtype=np.int32)
        dl = len(doc_ids)
        sb = (
            _bm25_numba_score(N, avgdl, df, dl, q_arr, di)
            if HAS_NUMBA
            else _bm25_py_score(N, avgdl, df, dl, q_arr, di)
        )
        ob = raws[i]
        sk = _kw_meta_boost(ob, toks, domain)
        si = _phash_sim(q_image, (ob.get("meta") or {}).get("phash64") or 0)
        filters = ((ob.get("meta") or {}).get("filters") or {})
        sym = _symbolic_match_score(filters, q_text or "", list(toks))
        score = w_bm25 * sb + w_kw * sk + w_img * si + w_sym * sym
        enriched = dict(ob)
        meta = dict(enriched.get("meta") or {})
        meta.setdefault("filters", filters)
        meta["retrieval_scores"] = {
            "bm25": float(sb),
            "keyword": float(sk),
            "image": float(si),
            "symbolic": float(sym),
        }
        enriched["meta"] = meta
        results.append((score, enriched))
    results.sort(key=lambda x: -x[0])
    return results[:topk]


def hybrid_query(
    index_pkl: str,
    jsonl: str,
    q_text: str = "",
    q_embedding: Optional[Sequence[float]] = None,
    layout_embedding: Optional[Sequence[float]] = None,
    topk: int = 10,
    zone_filter: Optional[str] = None,
    struct_filters: Optional[Dict[str, Any]] = None,
    boosts: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
    filters: Optional[Dict[str, Any]] = None,
    neighbor_graph: Optional[Dict[str, Sequence[str]]] = None,
    neighbor_seeds: Optional[Sequence[str]] = None,
    domain: str = "invoice",
):
    """Hybrid retrieval that re-ranks BM25 hits using Z-OCR structural tags.

    The scorer roughly follows the blueprint described in ``docs/hybrid_rag_zocr.md``::

        S(c) = a * DenseSim + b * BM25 + c * 1[zone match]
             + d * HeaderMatch + e * NeighborBoost
             - f * (1 - conf_ocr)(1 - conf_struct)

    When ``q_embedding`` or ``layout_embedding`` are provided, cosine similarity
    is computed against the chunk's ``embeddings.text`` / ``embeddings.layout``.
    """

    weights = {
        "dense": 0.4,
        "bm25": 0.25,
        "zone": 0.2,
        "header": 0.1,
        "neighbor": 0.1,
        "penalty": 0.2,
        "symbolic": 0.0,
        **(weights or {}),
    }
    boosts = boosts or {}
    struct_filters = struct_filters or {}
    neighbor_seeds = list(neighbor_seeds or [])
    filters = filters or {}

    with open(index_pkl, "rb") as f:
        ix = pickle.load(f)
    vocab = ix["vocab"]
    df = np.array(ix["df"], dtype=np.int32)
    N = int(ix["N"])
    avgdl = float(ix["avgdl"])

    raws: List[Dict[str, Any]] = []
    with open(jsonl, "r", encoding="utf-8") as fr:
        for line in fr:
            raws.append(json.loads(line))

    q_ids: List[int] = []
    toks = tokenize_jp(q_text or "")
    for t in toks:
        if t in vocab:
            q_ids.append(vocab[t])
    q_arr = np.array(q_ids, dtype=np.int32) if q_ids else np.array([-1], dtype=np.int32)

    results: List[Tuple[float, Dict[str, Any], int]] = []
    for i, doc_ids in enumerate(ix["docs_tokens"]):
        di = np.array(doc_ids + [-1], dtype=np.int32)
        dl = len(doc_ids)
        bm25 = (
            _bm25_numba_score(N, avgdl, df, dl, q_arr, di)
            if HAS_NUMBA
            else _bm25_py_score(N, avgdl, df, dl, q_arr, di)
        )

        ob = raws[i]
        meta = dict(ob.get("meta") or {})
        struct = dict(ob.get("struct") or meta.get("struct") or {})
        zone = ob.get("zone") or meta.get("zone") or ""

        if filters and not _passes_filters(ob, filters):
            continue

        if zone_filter and not _zone_match_score(zone, zone_filter):
            # Hard filter by zone when explicitly requested.
            continue
        if struct_filters and not _struct_filter(struct, struct_filters):
            continue

        embeddings = ob.get("embeddings") or meta.get("embeddings") or {}
        dense = _cosine_sim(q_embedding, embeddings.get("text"))
        dense_layout = _cosine_sim(layout_embedding, embeddings.get("layout"))
        dense_total = dense + (0.5 * dense_layout if dense_layout else 0.0)

        kw_boost = _kw_meta_boost(ob, toks, domain)
        sym = _symbolic_match_score(filters, q_text or "", list(toks))
        zone_score = _zone_match_score(zone, zone_filter)
        header_score = _header_match_score(struct, boosts)
        neighbor_score = _neighbor_boost(
            ob.get("region_id"), neighbor_graph, neighbor_seeds
        )
        conf_penalty = _confidence_penalty(
            ob.get("confidence") or meta.get("confidence") or {}
        )

        score = (
            weights["dense"] * dense_total
            + weights["bm25"] * bm25
            + weights["zone"] * zone_score
            + weights["header"] * header_score
            + weights["neighbor"] * neighbor_score
            + weights.get("keyword", 0.0) * kw_boost
            + weights.get("symbolic", 0.0) * sym
            - weights["penalty"] * conf_penalty
        )

        enriched = dict(ob)
        enriched_meta = dict(meta)
        enriched_meta.setdefault("filters", filters)
        enriched_meta["retrieval_scores"] = {
            "bm25": float(bm25),
            "dense": float(dense_total),
            "zone": float(zone_score),
            "header": float(header_score),
            "neighbor": float(neighbor_score),
            "symbolic": float(sym),
            "keyword": float(kw_boost),
            "penalty": float(conf_penalty),
        }
        enriched["meta"] = enriched_meta
        results.append((float(score), enriched, i))

    results.sort(key=lambda x: (-round(x[0], 3), x[2]))
    return [(score, doc) for score, doc, _ in results[:topk]]
