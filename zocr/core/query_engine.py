"""Retrieval helpers combining BM25, keywords and symbolic filters."""
from __future__ import annotations

import base64
import functools
import json
import os
import pickle
import re
from io import StringIO
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .._compat import optional_numpy

np = optional_numpy(__name__)
try:  # pragma: no cover - optional dependency
    from PIL import Image
except Exception:  # pragma: no cover - fallback stub
    Image = None  # type: ignore

from .base import hamm64, phash64, _normalize_text
from .domains import DOMAIN_KW
from .indexer import (
    _bm25_numba_scores_all,
    _bm25_numba_scores_all_parallel,
    _bm25_py_scores_all,
)
from .numba_support import HAS_NUMBA, HAS_NUMBA_PARALLEL
from .tokenization import tokenize_jp

__all__ = ["query", "hybrid_query"]

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


_INDEX_CACHE_SIZE = max(0, _env_int("ZOCR_INDEX_CACHE_SIZE", 4))
_JSONL_CACHE_SIZE = max(0, _env_int("ZOCR_JSONL_CACHE_SIZE", 2))
_JSONL_CACHE_MAX_BYTES = max(0, _env_int("ZOCR_JSONL_CACHE_MAX_BYTES", 50 * 1024 * 1024))
_CACHE_ENABLED = _env_truthy("ZOCR_QUERY_CACHE", True)


def _file_sig(path: str) -> tuple[int, int]:
    st = os.stat(path)
    return int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), int(st.st_size)


def _normalize_index_payload(loaded: Any) -> Dict[str, Any]:
    if not isinstance(loaded, dict):
        raise RuntimeError("Index payload invalid")
    ix = dict(loaded)

    try:
        ix["df"] = np.asarray(ix.get("df"), dtype=np.int32)
    except Exception:
        pass

    docs_tokens_flat = ix.get("docs_tokens_flat")
    doc_offsets = ix.get("doc_offsets")
    lengths = ix.get("lengths")

    lengths_arr = None
    if lengths is not None:
        try:
            lengths_arr = np.asarray(lengths, dtype=np.int32)
        except Exception:
            lengths_arr = None

    offsets_arr = None
    if doc_offsets is not None:
        try:
            offsets_arr = np.asarray(doc_offsets, dtype=np.int64)
        except Exception:
            offsets_arr = None

    flat_arr = None
    if docs_tokens_flat is not None:
        try:
            flat_arr = np.asarray(docs_tokens_flat, dtype=np.int32)
        except Exception:
            flat_arr = None

    if (offsets_arr is None or flat_arr is None) and isinstance(ix.get("docs_tokens"), list):
        docs_tokens = ix["docs_tokens"]
        if lengths_arr is None or len(lengths_arr) != len(docs_tokens):
            lengths_arr = np.array([len(doc) for doc in docs_tokens], dtype=np.int32)
        offsets_arr = np.zeros(len(docs_tokens) + 1, dtype=np.int64)
        if len(docs_tokens):
            offsets_arr[1:] = np.cumsum(lengths_arr, dtype=np.int64)
        flat_arr = np.empty(int(offsets_arr[-1]), dtype=np.int32)
        if flat_arr.size:
            for i, doc in enumerate(docs_tokens):
                if not doc:
                    continue
                start = int(offsets_arr[i])
                flat_arr[start : start + len(doc)] = np.asarray(doc, dtype=np.int32)
        ix["doc_offsets"] = offsets_arr
        ix["docs_tokens_flat"] = flat_arr
        ix["lengths"] = lengths_arr
        try:
            del ix["docs_tokens"]
        except KeyError:
            pass
        return ix

    if offsets_arr is not None:
        ix["doc_offsets"] = offsets_arr
    if flat_arr is not None:
        ix["docs_tokens_flat"] = flat_arr
    if lengths_arr is not None:
        ix["lengths"] = lengths_arr
    elif offsets_arr is not None:
        ix["lengths"] = np.diff(offsets_arr).astype(np.int32)

    return ix


@functools.lru_cache(maxsize=_INDEX_CACHE_SIZE)
def _load_index_cached(path: str, mtime_ns: int, size: int) -> Dict[str, Any]:
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    return _normalize_index_payload(loaded)


def _load_index(path: str) -> Dict[str, Any]:
    if not _CACHE_ENABLED or _INDEX_CACHE_SIZE <= 0:
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        return _normalize_index_payload(loaded)
    mtime_ns, size = _file_sig(path)
    return _load_index_cached(path, mtime_ns, size)


def _bm25_scores(
    N: int,
    avgdl: float,
    df: "np.ndarray",
    lengths: "np.ndarray",
    doc_offsets: "np.ndarray",
    docs_tokens_flat: "np.ndarray",
    q_ids: "np.ndarray",
) -> "np.ndarray":
    if HAS_NUMBA:
        if (
            HAS_NUMBA_PARALLEL
            and _bm25_numba_scores_all_parallel is not None
            and len(lengths) >= 256
        ):
            return _bm25_numba_scores_all_parallel(
                N, avgdl, df, lengths, doc_offsets, docs_tokens_flat, q_ids
            )
        return _bm25_numba_scores_all(N, avgdl, df, lengths, doc_offsets, docs_tokens_flat, q_ids)
    return _bm25_py_scores_all(N, avgdl, df, lengths, doc_offsets, docs_tokens_flat, q_ids)


@functools.lru_cache(maxsize=_JSONL_CACHE_SIZE)
def _load_jsonl_cached(path: str, mtime_ns: int, size: int) -> List[Dict[str, Any]]:
    return _read_jsonl(path)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    raws: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            ob = json.loads(line)
            if isinstance(ob, dict):
                raws.append(ob)
    return raws


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not _CACHE_ENABLED or _JSONL_CACHE_SIZE <= 0:
        return _read_jsonl(path)
    mtime_ns, size = _file_sig(path)
    if _JSONL_CACHE_MAX_BYTES and size > _JSONL_CACHE_MAX_BYTES:
        return _read_jsonl(path)
    return _load_jsonl_cached(path, mtime_ns, size)


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


def _infer_object_type(obj: Dict[str, Any]) -> str:
    explicit = (
        (obj.get("object_type") or "")
        or ((obj.get("meta") or {}).get("object_type") or "")
    ).strip()
    if explicit:
        return explicit.lower()

    struct = obj.get("struct") or (obj.get("meta") or {}).get("struct") or {}
    zone = obj.get("zone") or (obj.get("meta") or {}).get("zone") or ""
    meta = obj.get("meta") or {}

    if struct.get("table_id") or "table" in (zone or ""):
        return "table"
    if meta.get("thumbnail_b64") or meta.get("image_path") or meta.get("figure_id"):
        return "figure"
    return "text"


def _rows_to_html(rows: Sequence[Sequence[Any]]) -> str:
    html_rows: List[str] = []
    for row in rows:
        cells = [f"<td>{cell}</td>" for cell in row]
        html_rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(html_rows) + "</table>"


def _rows_to_csv(rows: Sequence[Sequence[Any]]) -> str:
    sio = StringIO()
    for row in rows:
        sio.write(",".join(str(c) for c in row))
        sio.write("\n")
    return sio.getvalue().strip()


def _figure_thumbnail_b64(meta: Dict[str, Any]) -> Optional[str]:
    thumb = meta.get("thumbnail_b64") or meta.get("image_b64")
    if thumb:
        return thumb
    img_path = meta.get("image_path")
    if not img_path or not os.path.exists(img_path):
        return None
    try:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None


def _renderable_payload(obj: Dict[str, Any], obj_type: str) -> Dict[str, Any]:
    meta = obj.get("meta") or {}
    payload: Dict[str, Any] = {"type": obj_type}
    caption = meta.get("caption") or obj.get("caption") or obj.get("text")
    if caption:
        payload["caption"] = caption

    if obj_type == "figure":
        thumb = _figure_thumbnail_b64(meta)
        if thumb:
            payload["thumbnail_b64"] = thumb
        if meta.get("image_uri"):
            payload["image_uri"] = meta.get("image_uri")
    elif obj_type == "table":
        html = meta.get("table_html") or meta.get("html")
        csv_data = meta.get("table_csv") or meta.get("csv")
        rows = meta.get("rows")
        if not html and isinstance(rows, (list, tuple)):
            html = _rows_to_html(rows)
        if not csv_data and isinstance(rows, (list, tuple)):
            csv_data = _rows_to_csv(rows)
        if html:
            payload["html"] = html
        if csv_data:
            payload["csv"] = csv_data
    else:
        snippet = obj.get("synthesis_window") or obj.get("text")
        if snippet:
            payload["snippet"] = snippet[:240]
    return payload


def _intent_boost(q_text: str, obj_type: str, struct: Dict[str, Any]) -> float:
    if not q_text:
        return 0.0
    text_lower = q_text.lower()
    score = 0.0
    figure_signals = [
        "diagram",
        "schematic",
        "figure",
        "chart",
        "graph",
        "plot",
        "visual",
        "illustration",
        "map",
    ]
    table_signals = ["table", "grid", "column", "row", "spreadsheet", "csv"]
    spatial_signals = ["top left", "bottom right", "layout", "coordinate", "axis"]

    if obj_type == "figure" and any(sig in text_lower for sig in figure_signals):
        score += 1.0
    if obj_type == "table" and any(sig in text_lower for sig in table_signals):
        score += 1.0
    if obj_type != "text" and any(sig in text_lower for sig in spatial_signals):
        score += 0.3
    if struct.get("table_id") and "table" in text_lower:
        score += 0.3
    return score


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
    is treated as a regex pattern anchored at the start of the field. Callables
    are also allowed and are invoked with the candidate value; any falsy return
    (or raised exception) causes the object to be rejected.
    """

    for key, expected in (filters or {}).items():
        if key in (None, "trace"):
            continue

        actual = obj.get(key)
        if actual is None:
            actual = (obj.get("meta") or {}).get(key)

        if actual is None:
            return False

        # Predicate filter: callable(expected)
        if callable(expected):
            try:
                if not expected(actual):
                    return False
            except Exception:
                return False
            continue

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
    ix = _load_index(index_pkl)
    vocab = ix["vocab"]
    df = np.asarray(ix["df"], dtype=np.int32)
    N = int(ix["N"])
    avgdl = float(ix["avgdl"])
    lengths = np.asarray(ix.get("lengths", []), dtype=np.int32)
    doc_offsets = np.asarray(ix.get("doc_offsets", []), dtype=np.int64)
    docs_tokens_flat = np.asarray(ix.get("docs_tokens_flat", []), dtype=np.int32)
    raws = _load_jsonl(jsonl)
    q_ids: List[int] = []
    toks = tokenize_jp(q_text or "")
    for t in toks:
        if t in vocab:
            q_ids.append(vocab[t])
    q_arr = np.array(q_ids, dtype=np.int32) if q_ids else np.array([-1], dtype=np.int32)

    bm25_scores = _bm25_scores(N, avgdl, df, lengths, doc_offsets, docs_tokens_flat, q_arr)

    results: List[Tuple[float, Dict[str, Any], int]] = []
    for i, ob in enumerate(raws):
        sb = float(bm25_scores[i]) if i < len(bm25_scores) else 0.0
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
    hybrid_embedding: Optional[Sequence[float]] = None,
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

    When ``q_embedding``/``layout_embedding``/``hybrid_embedding`` are provided,
    cosine similarity is computed against the chunk's
    ``embeddings.text`` / ``embeddings.layout`` / ``embeddings.hybrid``.
    """

    weights = {
        "dense": 0.35,
        "hybrid": 0.25,
        "layout_dense": 0.15,
        "bm25": 0.25,
        "zone": 0.2,
        "header": 0.1,
        "neighbor": 0.1,
        "penalty": 0.2,
        "symbolic": 0.0,
        "intent": 0.2,
        **(weights or {}),
    }
    boosts = boosts or {}
    struct_filters = struct_filters or {}
    neighbor_seeds = list(neighbor_seeds or [])
    filters = filters or {}

    ix = _load_index(index_pkl)
    vocab = ix["vocab"]
    df = np.asarray(ix["df"], dtype=np.int32)
    N = int(ix["N"])
    avgdl = float(ix["avgdl"])

    raws = _load_jsonl(jsonl)

    q_ids: List[int] = []
    toks = tokenize_jp(q_text or "")
    for t in toks:
        if t in vocab:
            q_ids.append(vocab[t])
    q_arr = np.array(q_ids, dtype=np.int32) if q_ids else np.array([-1], dtype=np.int32)

    lengths = np.asarray(ix.get("lengths", []), dtype=np.int32)
    doc_offsets = np.asarray(ix.get("doc_offsets", []), dtype=np.int64)
    docs_tokens_flat = np.asarray(ix.get("docs_tokens_flat", []), dtype=np.int32)
    bm25_scores = _bm25_scores(N, avgdl, df, lengths, doc_offsets, docs_tokens_flat, q_arr)

    results: List[Tuple[float, Dict[str, Any], int]] = []
    for i, ob in enumerate(raws):
        bm25 = float(bm25_scores[i]) if i < len(bm25_scores) else 0.0
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
        dense_hybrid = _cosine_sim(hybrid_embedding, embeddings.get("hybrid"))
        dense_total = dense + (0.5 * dense_layout if dense_layout else 0.0)

        kw_boost = _kw_meta_boost(ob, toks, domain)
        sym = _symbolic_match_score(filters, q_text or "", list(toks))
        zone_score = _zone_match_score(zone, zone_filter)
        header_score = _header_match_score(struct, boosts)
        neighbor_score = _neighbor_boost(
            ob.get("region_id"), neighbor_graph, neighbor_seeds
        )
        intent_score = _intent_boost(q_text or "", _infer_object_type(ob), struct)
        conf_penalty = _confidence_penalty(
            ob.get("confidence") or meta.get("confidence") or {}
        )

        score = (
            weights["dense"] * dense_total
            + weights.get("hybrid", 0.0) * dense_hybrid
            + weights.get("layout_dense", 0.0) * dense_layout
            + weights["bm25"] * bm25
            + weights["zone"] * zone_score
            + weights["header"] * header_score
            + weights["neighbor"] * neighbor_score
            + weights.get("keyword", 0.0) * kw_boost
            + weights.get("symbolic", 0.0) * sym
            + weights.get("intent", 0.0) * intent_score
            - weights["penalty"] * conf_penalty
        )

        enriched = dict(ob)
        enriched_meta = dict(meta)
        enriched_meta.setdefault("filters", filters)
        obj_type = _infer_object_type(enriched)
        enriched_meta["object_type"] = obj_type
        enriched_meta["render"] = _renderable_payload(enriched, obj_type)
        enriched_meta["retrieval_scores"] = {
            "bm25": float(bm25),
            "dense": float(dense_total),
            "hybrid": float(dense_hybrid),
            "zone": float(zone_score),
            "header": float(header_score),
            "neighbor": float(neighbor_score),
            "symbolic": float(sym),
            "keyword": float(kw_boost),
            "intent": float(intent_score),
            "penalty": float(conf_penalty),
        }
        enriched["meta"] = enriched_meta
        results.append((float(score), enriched, i))

    results.sort(key=lambda x: (-round(x[0], 3), x[2]))
    return [(score, doc) for score, doc, _ in results[:topk]]
