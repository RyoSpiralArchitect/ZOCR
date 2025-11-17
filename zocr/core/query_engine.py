"""Retrieval helpers combining BM25, keywords and symbolic filters."""
from __future__ import annotations

import json
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
try:  # pragma: no cover - optional dependency
    from PIL import Image
except Exception:  # pragma: no cover - fallback stub
    Image = None  # type: ignore

from .base import hamm64, phash64, _normalize_text
from .domains import DOMAIN_KW
from .indexer import _bm25_numba_score, _bm25_py_score
from .numba_support import HAS_NUMBA
from .tokenization import tokenize_jp

__all__ = ["query"]


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
    results: List[Tuple[float, Dict[str, Any]]] = []
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
