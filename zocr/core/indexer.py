"""BM25 index builder utilities."""
from __future__ import annotations

import json
import math
import pickle
from typing import Any, Dict, List, Tuple

from .._compat import optional_numpy

np = optional_numpy(__name__)

from .numba_support import HAS_NUMBA, HAS_NUMBA_PARALLEL, atomic, njit, prange
from .tokenization import tokenize_jp

__all__ = ["build_index", "_bm25_numba_score", "_bm25_py_score"]


def build_index(jsonl: str, out_pkl: str) -> Dict[str, Any]:
    docs: List[Tuple[List[int], Dict[str, Any]]] = []
    vocab: Dict[str, int] = {}
    vid = 0
    maxlen = 0
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            ob = json.loads(line)
            txt = ob.get("search_unit") or ob.get("text") or ""
            toks = tokenize_jp(txt)
            ids: List[int] = []
            for t in toks:
                if t not in vocab:
                    vocab[t] = vid
                    vid += 1
                ids.append(vocab[t])
            maxlen = max(maxlen, len(ids))
            docs.append((ids, ob))
    N = len(docs)
    pad = -1
    arr = np.full((N, maxlen), pad, dtype=np.int32)
    lengths = np.zeros(N, dtype=np.int32)
    uniq_docs: List[List[int]] = []
    uniq_maxlen = 0
    for i, (ids, _) in enumerate(docs):
        lengths[i] = len(ids)
        if ids:
            arr[i, : len(ids)] = np.array(ids, dtype=np.int32)
        seen = set()
        uniq: List[int] = []
        for tid in ids:
            if tid not in seen:
                seen.add(tid)
                uniq.append(tid)
        uniq_docs.append(uniq)
        if len(uniq) > uniq_maxlen:
            uniq_maxlen = len(uniq)

    uniq_pad = -1
    uniq_shape = max(1, uniq_maxlen) if N else 0
    arr_unique = np.full((N, uniq_shape), uniq_pad, dtype=np.int32)
    uniq_lengths = np.zeros(N, dtype=np.int32)
    for i, uniq in enumerate(uniq_docs):
        L = len(uniq)
        uniq_lengths[i] = L
        if L:
            arr_unique[i, :L] = np.array(uniq, dtype=np.int32)

    @njit(cache=True)
    def _compute_df(arr_unique, lengths, V):
        n = arr_unique.shape[0]
        df = np.zeros(V, dtype=np.int64)
        for i in range(n):
            for j in range(lengths[i]):
                tid = arr_unique[i, j]
                if tid < 0:
                    break
                df[tid] += 1
        return df

    if HAS_NUMBA_PARALLEL:
        @njit(parallel=True, cache=True)
        def _compute_df_parallel(arr_unique, lengths, V):
            n = arr_unique.shape[0]
            df = np.zeros(V, dtype=np.int64)
            for i in prange(n):
                L = lengths[i]
                for j in range(L):
                    tid = arr_unique[i, j]
                    if tid < 0:
                        break
                    atomic.add(df, tid, 1)
            return df
    else:
        _compute_df_parallel = None  # type: ignore

    V = len(vocab)
    df = None
    if HAS_NUMBA:
        try:
            if HAS_NUMBA_PARALLEL and V <= 200000 and _compute_df_parallel is not None:
                df = _compute_df_parallel(arr_unique, uniq_lengths, V)
            else:
                df = _compute_df(arr_unique, uniq_lengths, V)
        except Exception:
            df = None
    if df is None:
        df = np.zeros(V, dtype=np.int64)
        for uniq in uniq_docs:
            for tid in uniq:
                df[tid] += 1
    avgdl = float(lengths.sum()) / max(1, N)
    ix = {
        "vocab": vocab,
        "df": df,
        "avgdl": avgdl,
        "N": N,
        "lengths": lengths.tolist(),
        "docs_tokens": [d[0] for d in docs],
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(ix, f)
    return ix


@njit(cache=True)
def _bm25_numba_score(N, avgdl, df, dl, q_ids, doc_ids, k1=1.2, b=0.75):
    """Compute BM25 score for a single document.

    This implementation is designed to be compatible with numba acceleration
    while preserving the same semantics as :func:`_bm25_py_score` (including
    duplicate query term handling).
    """

    # No query tokens (the query engine uses [-1] as a sentinel in this case).
    if len(q_ids) == 0 or (len(q_ids) == 1 and q_ids[0] < 0):
        return 0.0

    K = k1 * (1 - b + b * dl / max(1.0, avgdl))
    s = 0.0

    # Compact query ids into unique ids + multiplicity to avoid redundant scans.
    max_q = len(q_ids)
    uniq_q = np.empty(max_q, dtype=np.int32)
    uniq_counts = np.empty(max_q, dtype=np.int32)
    uniq_len = 0
    for i in range(max_q):
        qid = q_ids[i]
        if qid < 0:
            continue
        found = -1
        for j in range(uniq_len):
            if uniq_q[j] == qid:
                found = j
                break
        if found >= 0:
            uniq_counts[found] += 1
        else:
            uniq_q[uniq_len] = qid
            uniq_counts[uniq_len] = 1
            uniq_len += 1

    for u in range(uniq_len):
        qid = uniq_q[u]
        if qid < 0 or df[qid] <= 0:
            continue
        tf = 0
        for j in range(len(doc_ids)):
            tid = doc_ids[j]
            if tid < 0:
                break
            if tid == qid:
                tf += 1
        if tf <= 0:
            continue
        idf = math.log((N - df[qid] + 0.5) / (df[qid] + 0.5) + 1.0)
        s += (idf * ((tf * (k1 + 1)) / (tf + K))) * float(uniq_counts[u])

    return s


def _bm25_py_score(N, avgdl, df, dl, q_ids, doc_ids, k1=1.2, b=0.75):
    from collections import Counter

    K = k1 * (1 - b + b * dl / max(1.0, avgdl))
    tf = Counter([tid for tid in doc_ids if tid >= 0])
    qtf = Counter([qid for qid in q_ids if qid >= 0])
    s = 0.0
    for q, q_count in qtf.items():
        if q < 0 or df[q] == 0:
            continue
        idf = math.log((N - df[q] + 0.5) / (df[q] + 0.5) + 1.0)
        f = tf.get(q, 0)
        if f:
            s += float(q_count) * idf * ((f * (k1 + 1)) / (f + K))
    return s
