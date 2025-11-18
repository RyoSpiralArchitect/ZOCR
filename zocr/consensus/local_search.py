"""Local search helpers for contextual consensus outputs."""
from __future__ import annotations

import json
import os
import pickle
import re
from typing import Any, Dict, List, Sequence, Tuple, Union

try:  # pragma: no cover - optional dependency when local search is disabled
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

__all__ = [
    "_tokenize",
    "_bm25_build",
    "_bm25_query",
    "_img_embed64_from_bbox",
    "_cos",
    "_img_search",
    "_rrf_merge",
    "build_local_index",
    "query_local",
]


def _tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\-\._]+", " ", s)
    return [t for t in s.split() if t]


def _bm25_build(jsonl_path: str):
    import math

    D = []
    df: Dict[str, int] = {}
    N = 0
    avgdl = 0.0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            N += 1
            ob = json.loads(line)
            txt = ob.get("search_unit") or ob.get("text") or ""
            toks = _tokenize(txt)
            D.append({"id": N - 1, "len": len(toks), "toks": toks, "raw": ob})
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
            avgdl += len(toks)
    avgdl = avgdl / max(1, N)
    return {"D": D, "df": df, "N": N, "avgdl": avgdl}


def _bm25_query(ix, q: str, k1: float = 1.2, b: float = 0.75, topk: int = 20):
    import math

    q_toks = _tokenize(q)
    df = ix["df"]
    N = ix["N"]
    avgdl = ix["avgdl"]
    scores = []
    for doc in ix["D"]:
        dl = doc["len"]
        toks = doc["toks"]
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in q_toks:
            if t not in df:
                continue
            idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
            f = tf.get(t, 0)
            s += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(1, avgdl))))
        if s > 0:
            scores.append((s, doc))
    scores.sort(key=lambda x: -x[0])
    return scores[:topk]


def _img_embed64_from_bbox(ob: Dict[str, Any], down: int = 16):
    # downsample region to tiny vector
    if Image is None:
        raise RuntimeError("Pillow is required for image search helpers")
    p = ob.get("image_path")
    if not p or not os.path.exists(p):
        return None
    img = Image.open(p).convert("L")
    x1, y1, x2, y2 = ob.get("bbox", [0, 0, img.width, img.height])
    crop = img.crop((x1, y1, x2, y2)).resize((down, down), resample=Image.BICUBIC)
    import numpy as _np

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
    # image query: downscale query img to vector, compare cosine
    if Image is None:
        return []
    import numpy as _np

    q_im = Image.open(query_img_path)
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


def _rrf_merge(listA, listB, k: int = 60, topk: int = 10):
    # listA: [(score, doc)], listB: [(score, doc or (idx,doc))]
    rank: Dict[str, Dict[str, Any]] = {}

    def add_list(lst, is_img: bool = False):
        for r, tup in enumerate(lst, start=1):
            s, obj = tup[0], (tup[-1] if is_img else tup[1])
            key = json.dumps(obj.get("bbox", []) + [obj.get("page"), obj.get("table_index")])
            rank.setdefault(key, {"obj": obj, "score": 0.0})
            rank[key]["score"] += 1.0 / (k + r)

    add_list(listA, is_img=False)
    add_list(listB, is_img=True)
    merged = list(rank.values())
    merged.sort(key=lambda x: -x["score"])
    return merged[:topk]


def build_local_index(jsonl_path: str, out_pkl: str):
    ix = _bm25_build(jsonl_path)
    with open(out_pkl, "wb") as f:
        pickle.dump(ix, f)
    return ix


def query_local(
    jsonl_path: str,
    pkl_path: str,
    text_query: str = "",
    image_query_path: str | None = None,
    topk: int = 10,
):
    with open(pkl_path, "rb") as f:
        ix = pickle.load(f)
    bm = _bm25_query(ix, text_query or "", topk=topk)
    im = _img_search(jsonl_path, image_query_path, topk=topk) if image_query_path else []
    merged = _rrf_merge(bm, im, k=60, topk=topk)
    return merged

