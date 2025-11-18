"""Local search helpers for contextual consensus outputs."""
from __future__ import annotations

import json
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:  # pragma: no cover - optional dependency when local search is disabled
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

__all__ = [
    "LocalDocument",
    "LocalIndex",
    "LocalSearchResult",
    "ensure_local_index",
    "_tokenize",
    "_bm25_build",
    "_bm25_query",
    "_img_embed64_from_bbox",
    "_cos",
    "_img_search",
    "_rrf_merge",
    "build_local_index",
    "load_local_index",
    "query_local",
]


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class LocalDocument:
    """Compact representation of a contextual JSONL record."""

    doc_id: int
    tokens: List[str]
    raw: Dict[str, Any]
    length: int

    def as_payload(self) -> Dict[str, Any]:
        return {"id": self.doc_id, "len": self.length, "toks": self.tokens, "raw": self.raw}

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "LocalDocument":
        tokens = list(payload.get("toks", []))
        length = payload.get("len")
        return cls(
            doc_id=int(payload.get("id", 0)),
            tokens=tokens,
            raw=dict(payload.get("raw", {})),
            length=int(length) if isinstance(length, (int, float)) else len(tokens),
        )


@dataclass
class LocalIndex:
    """Structured BM25 index wrapper with provenance metadata."""

    documents: List[LocalDocument]
    df: Dict[str, int]
    total_docs: int
    avg_doc_len: float
    source_path: str | None = None
    source_mtime: float | None = None
    source_bytes: int | None = None
    created_at: str = field(default_factory=_utc_timestamp)
    version: int = 1

    def serialize(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "documents": [doc.as_payload() for doc in self.documents],
            "df": self.df,
            "total_docs": self.total_docs,
            "avg_doc_len": self.avg_doc_len,
            "source_path": self.source_path,
            "source_mtime": self.source_mtime,
            "source_bytes": self.source_bytes,
            "created_at": self.created_at,
        }

    def save(self, path: Path) -> None:
        with path.open("wb") as fw:
            pickle.dump(self.serialize(), fw)

    def is_stale(self, jsonl_path: str | os.PathLike[str]) -> bool:
        """Return ``True`` when the backing JSONL changed since the index was built."""

        if not self.source_mtime and not self.source_bytes and not self.source_path:
            return False
        try:
            stat = Path(jsonl_path).resolve().stat()
        except OSError:
            return False
        if self.source_mtime and stat.st_mtime > self.source_mtime + 1e-6:
            return True
        if self.source_bytes and stat.st_size != self.source_bytes:
            return True
        if self.source_path:
            try:
                current = Path(jsonl_path).resolve()
                stored = Path(self.source_path).resolve()
            except Exception:
                current = Path(jsonl_path)
                stored = Path(self.source_path)
            if current != stored:
                return True
        return False

    @classmethod
    def from_serialized(cls, payload: Dict[str, Any]) -> "LocalIndex":
        docs = [LocalDocument.from_payload(entry) for entry in payload.get("documents", [])]
        mtime = payload.get("source_mtime")
        size = payload.get("source_bytes")
        return cls(
            documents=docs,
            df=dict(payload.get("df", {})),
            total_docs=int(payload.get("total_docs", len(docs))),
            avg_doc_len=float(payload.get("avg_doc_len", 0.0)),
            source_path=payload.get("source_path"),
            source_mtime=float(mtime) if isinstance(mtime, (int, float)) else None,
            source_bytes=int(size) if isinstance(size, (int, float)) else None,
            created_at=payload.get("created_at", _utc_timestamp()),
            version=int(payload.get("version", 1)),
        )

    @classmethod
    def from_legacy(cls, payload: Dict[str, Any]) -> "LocalIndex":
        docs = []
        for entry in payload.get("D", []):
            tokens = list(entry.get("toks", []))
            docs.append(
                LocalDocument(
                    doc_id=int(entry.get("id", len(docs))),
                    tokens=tokens,
                    raw=dict(entry.get("raw", {})),
                    length=int(entry.get("len", len(tokens))),
                )
            )
        return cls(
            documents=docs,
            df=dict(payload.get("df", {})),
            total_docs=int(payload.get("N", len(docs))),
            avg_doc_len=float(payload.get("avgdl", 0.0)),
        )

    @classmethod
    def ensure(cls, payload: Any) -> "LocalIndex":
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict):
            if "documents" in payload:
                return cls.from_serialized(payload)
            if "D" in payload:
                return cls.from_legacy(payload)
        raise TypeError("Unsupported index payload; rebuild the local index.")


@dataclass
class LocalSearchResult:
    """Structured representation of a single local search hit."""

    score: float
    obj: Dict[str, Any]
    source: str

    def as_mapping(self) -> Dict[str, Any]:
        return {"score": float(self.score), "obj": self.obj, "source": self.source}


def _tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\-\._]+", " ", s)
    return [t for t in s.split() if t]


def _bm25_build(jsonl_path: str | os.PathLike[str]) -> LocalIndex:
    docs: List[LocalDocument] = []
    df: Dict[str, int] = {}
    total = 0
    total_len = 0
    jsonl_path = Path(jsonl_path)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            ob = json.loads(line)
            txt = ob.get("search_unit") or ob.get("text") or ""
            toks = _tokenize(txt)
            doc = LocalDocument(doc_id=total, tokens=toks, raw=ob, length=len(toks))
            docs.append(doc)
            total += 1
            total_len += doc.length
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
    avgdl = float(total_len / max(1, total))
    source_meta = None
    try:
        stat = jsonl_path.resolve().stat()
        source_meta = {
            "source_path": str(jsonl_path.resolve()),
            "source_mtime": stat.st_mtime,
            "source_bytes": stat.st_size,
        }
    except OSError:
        source_meta = {}
    return LocalIndex(
        documents=docs,
        df=df,
        total_docs=total,
        avg_doc_len=avgdl,
        **(source_meta or {}),
    )


def _bm25_query(
    ix: LocalIndex | Dict[str, Any], q: str, k1: float = 1.2, b: float = 0.75, topk: int = 20
):
    import math

    index = LocalIndex.ensure(ix)
    if not q.strip():
        return []
    q_toks = _tokenize(q)
    df = index.df
    N = max(1, index.total_docs)
    avgdl = max(1e-6, index.avg_doc_len or 1.0)
    scores = []
    for doc in index.documents:
        if doc.length == 0:
            continue
        tf: Dict[str, int] = {}
        for t in doc.tokens:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in q_toks:
            df_t = df.get(t)
            if not df_t:
                continue
            idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0)
            f = tf.get(t, 0)
            s += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * doc.length / avgdl)))
        if s > 0:
            scores.append((s, doc.raw))
    scores.sort(key=lambda x: -x[0])
    return scores[:topk]


def _img_embed64_from_bbox(ob: Dict[str, Any], down: int = 16):
    """Downsample a bounding box to a normalized vector."""

    if Image is None:
        raise RuntimeError("Pillow is required for image search helpers")
    p = ob.get("image_path")
    if not p or not os.path.exists(p):
        return None
    import numpy as _np

    with Image.open(p) as img:
        gray = img.convert("L")
        x1, y1, x2, y2 = ob.get("bbox", [0, 0, gray.width, gray.height])
        crop = gray.crop((x1, y1, x2, y2)).resize((down, down), resample=Image.BICUBIC)
        v = _np.asarray(crop, dtype=_np.float32).reshape(-1)
    v = (v - v.mean()) / (v.std() + 1e-6)
    return v


def _cos(a, b):
    import numpy as _np

    d = float((a * b).sum())
    na = float((_np.square(a).sum()) ** 0.5)
    nb = float((_np.square(b).sum()) ** 0.5)
    return d / max(1e-6, na * nb)


def _img_search(jsonl_path: str, query_img_path: str, topk: int = 20):
    """Return cosine-ranked image matches for ``query_img_path``."""

    if Image is None:
        return []
    import numpy as _np

    with Image.open(query_img_path) as q_im:
        qv = _img_embed64_from_bbox(
            {
                "image_path": query_img_path,
                "bbox": [0, 0, q_im.size[0], q_im.size[1]],
            }
        )
    if qv is None:
        return []
    scores = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ob = json.loads(line)
            dv = _img_embed64_from_bbox(ob)
            if dv is None:
                continue
            s = _cos(qv, dv)
            scores.append((s, i, ob))
    scores.sort(key=lambda x: -x[0])
    return scores[:topk]


def _rrf_merge(listA, listB, k: int = 60, topk: int = 10) -> List[LocalSearchResult]:
    """Merge independent rank lists via reciprocal rank fusion."""

    rank: Dict[str, Dict[str, Any]] = {}

    def add_list(lst: Iterable[Any], is_img: bool = False) -> None:
        for r, tup in enumerate(lst, start=1):
            s, obj = tup[0], (tup[-1] if is_img else tup[1])
            key_bits = obj.get("bbox", []) + [obj.get("page"), obj.get("table_index")]
            key = json.dumps(key_bits)
            rank.setdefault(key, {"obj": obj, "score": 0.0})
            rank[key]["score"] += 1.0 / (k + r)

    add_list(listA, is_img=False)
    add_list(listB, is_img=True)
    merged = [
        LocalSearchResult(score=float(entry["score"]), obj=entry["obj"], source="fusion")
        for entry in rank.values()
    ]
    merged.sort(key=lambda x: -x.score)
    return merged[:topk]


def build_local_index(jsonl_path: str | os.PathLike[str], out_pkl: str | os.PathLike[str]) -> LocalIndex:
    """Build and persist the BM25 index for contextual search."""

    ix = _bm25_build(jsonl_path)
    out_path = Path(out_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ix.save(out_path)
    return ix


def load_local_index(pkl_path: str | os.PathLike[str]) -> LocalIndex:
    """Load a serialized local index with a helpful error message."""

    path = Path(pkl_path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    return LocalIndex.ensure(payload)


def ensure_local_index(
    jsonl_path: str | os.PathLike[str],
    pkl_path: str | os.PathLike[str],
    *,
    rebuild: bool = False,
    rebuild_on_stale: bool = True,
) -> Tuple[LocalIndex, bool]:
    """Return a :class:`LocalIndex` and flag when it was rebuilt."""

    jsonl = Path(jsonl_path)
    pkl = Path(pkl_path)
    rebuilt = False
    if rebuild or not pkl.exists():
        ix = build_local_index(jsonl, pkl)
        rebuilt = True
    else:
        ix = load_local_index(pkl)
        if rebuild_on_stale and ix.is_stale(jsonl):
            ix = build_local_index(jsonl, pkl)
            rebuilt = True
    return ix, rebuilt


def query_local(
    jsonl_path: str,
    pkl_path: str,
    text_query: str = "",
    image_query_path: str | None = None,
    topk: int = 10,
    *,
    autobuild: bool = True,
    rebuild: bool = False,
    rebuild_on_stale: bool = True,
    index: LocalIndex | None = None,
):
    """Query the contextual search index with optional auto-(re)builds."""

    if index is None:
        if autobuild:
            index, _ = ensure_local_index(
                jsonl_path, pkl_path, rebuild=rebuild, rebuild_on_stale=rebuild_on_stale
            )
        else:
            index = load_local_index(pkl_path)
    bm = _bm25_query(index, text_query or "", topk=topk)
    im = _img_search(jsonl_path, image_query_path, topk=topk) if image_query_path else []
    merged = _rrf_merge(bm, im, k=60, topk=topk)
    return [result.as_mapping() for result in merged]

