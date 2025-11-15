#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
理念 / Vision / Vision

[JA]
- 単一核で責務を折り畳み、画素→構造→文脈→検索→監視を一枚の可視仕様に落とす。
- 乱数・環境・バージョン・入出力を指紋化し、再現性を約束する。
- p95 や Hit@K、失敗率、Views を最初から計測し続け、観測可能性を設計する。
- registry で各段を開口し、壊さずに他者の知を差し込めるようにする。
- これは精度の一点突破ではなく、同じ説明が何度でも再現される系を目指す。

[EN]
- Fold every responsibility into a single core so pixels→structure→context→search→monitoring share one inspectable spec.
- Fingerprint randomness, environment, versions, and inputs/outputs to guarantee reproducibility.
- Instrument p95, Hit@K, failure rate, and Views from the start to keep the system observable.
- Expose each stage via a registry so new knowledge can be inserted without breaking the pipeline.
- The ambition is not a one-off accuracy spike but a system whose explanations can be replayed endlessly.

[FR]
- Plier toutes les responsabilités dans un seul noyau afin que pixels→structure→contexte→recherche→surveillance partagent une spécification lisible.
- Empreinter l’aléatoire, l’environnement, les versions et les entrées/sorties pour garantir la reproductibilité.
- Instrumenter p95, Hit@K, taux d’échec et Views dès le départ pour garder le système observable.
- Ouvrir chaque étape via un registre afin d’injecter de nouvelles connaissances sans briser la chaîne.
- L’objectif n’est pas un pic de précision isolé mais un système dont les explications se rejouent à l’infini.
"""

# Auto-generated single-file bundle
# Generated: 2025-11-12T15:43:19.121504+00:00
# Purpose : Merge upstream (onefile consensus OCR) -> pipe (orchestrator) -> downstream (core augment/index/query) -> watchdog (monitor)
#           into a self-contained Python script without omitting any source code.
# How it works:
#  - We embed the original three modules as strings and materialize them as in-memory modules via `types.ModuleType` + `exec`.
#  - Their original `if __name__ == "__main__":` blocks remain inert (since `__name__` is the module name),
#    so behavior is preserved without side-effects during bundling.
#  - The orchestrator (`zocr_pipeline_allinone`) keeps its CLI. Running this file calls its `main()`.
#  - Imports like `from zocr_onefile_consensus import ...` and `from zocr_multidomain_core import ...` continue to work
#    because we register those names in `sys.modules`.
#
# Layout:
#   [Upstream]   zocr_onefile_consensus  -> OCR / table reconstruction / contextual export helpers
#   [Downstream] zocr_multidomain_core   -> augment / index / query / monitor
#   [Pipe]       zocr_pipeline_allinone  -> orchestrates the end-to-end flow (+ watchdog/monitor hooks)
#
# Notes:
#  - The embedded sources are verbatim copies of your originals; no simplification or pruning.
#  - You can still `import zocr_allinone_merged_plus` from Python and access the three submodules via
#      sys.modules['zocr_onefile_consensus'], sys.modules['zocr_multidomain_core'], sys.modules['zocr_pipeline_allinone'].


import sys, types, os, tempfile, pathlib, unicodedata, math

_BUNDLE_DIR = os.path.join(tempfile.gettempdir(), "zocr_bundle_runtime")
os.makedirs(_BUNDLE_DIR, exist_ok=True)
if _BUNDLE_DIR not in sys.path:
    sys.path.insert(0, _BUNDLE_DIR)

def _materialize_module(name: str, source: str):
    # Write to a real file to please tooling (e.g., numba caching, inspect)
    filename = os.path.join(_BUNDLE_DIR, name + ".py")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(source)
    except Exception:
        # Fallback to in-memory only
        filename = name + ".py"
    mod = types.ModuleType(name)
    mod.__dict__['__file__'] = filename
    sys.modules[name] = mod
    code = compile(source, filename, 'exec')
    exec(code, mod.__dict__)
    return mod


# ---------------- [Upstream] onefile consensus OCR ----------------
_SRC_ZOCR_ONEFILE_CONSENSUS = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z‑OCR one‑file (Consensus)
- RLE CC + prefix-sum dilation (OpenCV不要)
- DP‑means columns + D² smoothing (λ scheduling)
- 列モード吸着 (行クラスタ/段) で λ 自動補正
- rowspan 昇格 = 確率合議 (p_iou × p_base × p_empty ≥ τ)
- unlabeled 目的関数に rows/cols の過不足率を追加
- 低信頼セルだけ Views 再生成 (microscope/xray) — stub

Deps: numpy, pillow  (pdftoppm があれば PDF もOK)
"""

from __future__ import annotations
import os, sys, io, json, argparse, tempfile, shutil, subprocess, time, math, re, hashlib, contextlib, bisect, atexit, difflib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Set, Mapping, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict, OrderedDict, deque

try:
    import numpy as np
except Exception:
    np = None

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageChops, ImageEnhance, ImageStat
try:
    import pytesseract  # type: ignore
    from pytesseract import Output as _PYTESS_OUTPUT  # type: ignore
    from pytesseract import TesseractError as _PYTESS_ERR  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore
    _PYTESS_OUTPUT = None  # type: ignore
    _PYTESS_ERR = None  # type: ignore
from html.parser import HTMLParser

_OCR_BACKEND_CACHE: Dict[str, Callable[["Image.Image"], Tuple[str, float]]] = {}
_OCR_BACKEND_WARNED: Set[str] = set()
_EASYOCR_READER_CACHE: Dict[Tuple[Tuple[str, ...], bool], Any] = {}

_TOY_SELF_CORRECTION_STACK: List[Dict[str, Any]] = []


def _normalize_self_correction_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    if not isinstance(config, dict):
        return normalized
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            normalized[key] = value
        elif isinstance(value, (list, tuple)):
            normalized[key] = [v for v in value]
        elif isinstance(value, dict):
            normalized[key] = {k: v for k, v in value.items()}
    return normalized


def push_toy_self_correction(config: Optional[Dict[str, Any]]) -> None:
    _TOY_SELF_CORRECTION_STACK.append(_normalize_self_correction_config(config))


def pop_toy_self_correction() -> None:
    if _TOY_SELF_CORRECTION_STACK:
        _TOY_SELF_CORRECTION_STACK.pop()


def current_toy_self_correction() -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for cfg in _TOY_SELF_CORRECTION_STACK:
        for key, value in cfg.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                combined = dict(merged[key])  # type: ignore[index]
                combined.update(value)
                merged[key] = combined
            else:
                merged[key] = value
    return merged


@contextlib.contextmanager
def toy_self_correction_scope(config: Optional[Dict[str, Any]]):
    push_toy_self_correction(config)
    try:
        yield
    finally:
        pop_toy_self_correction()

_thomas = None
try:
    from zocr.core import zocr_core as _core_mod  # type: ignore
except Exception:
    _core_mod = None
else:
    _thomas = getattr(_core_mod, "thomas_tridiag", None)

if _thomas is None:
    try:
        from zocr_multidomain_core import thomas_tridiag as _thomas  # type: ignore
    except Exception:
        _thomas = None

if __name__.startswith("zocr."):
    sys.modules.setdefault("zocr_onefile_consensus", sys.modules[__name__])

# ----------------- Utils -----------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _toy_memory_series_paths(path: str) -> Tuple[str, str]:
    base_dir = os.environ.get("ZOCR_TOY_MEMORY_SERIES")
    if base_dir:
        base_dir = os.path.abspath(base_dir)
    else:
        root = os.path.dirname(path) if path else ""
        base_dir = os.path.join(root or ".", "toy_memory_versioned")
    return base_dir, os.path.join(base_dir, "summary.json")


def _series_tail(history: Sequence[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """Return a light-weight tail of the toy memory history.

    The trimmed representation keeps the most recent ``limit`` epochs with just the
    statistics that operators tend to inspect when validating learning progress.
    """

    tail: List[Dict[str, Any]] = []
    if limit <= 0:
        return tail
    for entry in list(history)[-limit:]:
        if not isinstance(entry, dict):
            continue
        trimmed: Dict[str, Any] = {
            "epoch": entry.get("epoch"),
            "written_at": entry.get("written_at"),
            "delta_prev": entry.get("delta_prev"),
            "recognition": entry.get("recognition"),
        }
        snapshot = entry.get("snapshot")
        if isinstance(snapshot, dict):
            trimmed["snapshot"] = {
                "glyph_variants": snapshot.get("glyph_variants"),
                "avg_variants_per_char": snapshot.get("avg_variants_per_char"),
                "ngram_observations": snapshot.get("ngram_observations"),
                "avg_ngram_branching": snapshot.get("avg_ngram_branching"),
            }
        tail.append(trimmed)
    return tail


def _toy_memory_series_stats(history: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics describing the versioned toy memory history."""

    stats: Dict[str, Any] = {"epochs": len(history)}
    if not history:
        return stats
    latest = history[-1]
    try:
        stats["latest_epoch"] = int(latest.get("epoch")) if latest.get("epoch") is not None else None
    except Exception:
        stats["latest_epoch"] = latest.get("epoch")
    snapshot = latest.get("snapshot") if isinstance(latest, dict) else None
    if isinstance(snapshot, dict):
        stats["latest_variants"] = float(snapshot.get("glyph_variants") or 0.0)
        stats["latest_ngram_observations"] = float(snapshot.get("ngram_observations") or 0.0)
        stats["latest_avg_variants_per_char"] = float(snapshot.get("avg_variants_per_char") or 0.0)
        stats["latest_avg_ngram_branching"] = float(snapshot.get("avg_ngram_branching") or 0.0)
    variant_deltas: List[float] = []
    surprisal_ratios: List[float] = []
    runtime_improvements: List[float] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        delta_prev = entry.get("delta_prev")
        if isinstance(delta_prev, dict):
            try:
                variant_deltas.append(float(delta_prev.get("glyph_variants") or 0.0))
            except Exception:
                variant_deltas.append(0.0)
        recognition = entry.get("recognition")
        if isinstance(recognition, dict):
            try:
                surprisal_ratios.append(float(recognition.get("high_surprisal_cells") or 0.0) / float(recognition.get("cells") or 1.0))
            except Exception:
                pass
            try:
                runtime_improvements.append(float(recognition.get("runtime_replay_improved") or 0.0))
            except Exception:
                runtime_improvements.append(0.0)
    if variant_deltas:
        stats["variant_delta_total"] = float(sum(variant_deltas))
        stats["variant_delta_avg"] = float(sum(variant_deltas) / len(variant_deltas))
        recent_window = max(1, min(6, len(variant_deltas)))
        stats["variant_delta_recent_avg"] = float(sum(variant_deltas[-recent_window:]) / recent_window)
        stagnant = 0
        for delta in reversed(variant_deltas):
            if delta <= 0:
                stagnant += 1
            else:
                break
        stats["stagnant_epochs"] = int(stagnant)
    if surprisal_ratios:
        stats["recent_high_surprisal_ratio"] = float(sum(surprisal_ratios[-3:]) / min(3, len(surprisal_ratios)))
    if runtime_improvements:
        stats["recent_runtime_replay_improved"] = float(sum(runtime_improvements[-3:]))
        stats["runtime_replay_total"] = float(sum(runtime_improvements))
    return stats


def _load_toy_memory_series_payload(path: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    version_dir, summary_path = _toy_memory_series_paths(path)
    info: Dict[str, Any] = {
        "version_dir": version_dir,
        "summary_path": summary_path,
    }
    if not os.path.exists(summary_path):
        return None, info
    try:
        with open(summary_path, "r", encoding="utf-8") as fr:
            summary_payload = json.load(fr)
    except Exception as exc:
        info["error"] = f"summary_load_failed: {type(exc).__name__}: {exc}"
        return None, info
    latest_rel = summary_payload.get("latest_path")
    if not latest_rel:
        info["summary"] = summary_payload
        return None, info
    epoch_path = latest_rel if os.path.isabs(latest_rel) else os.path.join(version_dir, latest_rel)
    info["epoch_path"] = epoch_path
    info["latest_epoch"] = summary_payload.get("latest_epoch")
    info["summary"] = summary_payload
    history = summary_payload.get("history") if isinstance(summary_payload, dict) else None
    if isinstance(history, list):
        info["stats"] = _toy_memory_series_stats(history)
        info["history_tail"] = _series_tail(history)
    if not os.path.exists(epoch_path):
        info["error"] = "epoch_missing"
        info["summary"] = summary_payload
        return None, info
    try:
        with open(epoch_path, "r", encoding="utf-8") as fr:
            payload = json.load(fr)
    except Exception as exc:
        info["error"] = f"epoch_load_failed: {type(exc).__name__}: {exc}"
        return None, info
    info["summary"] = summary_payload
    return payload, info

def clamp(x, lo, hi): return lo if x<lo else (hi if x>hi else x)

# ----------------- Binarization -----------------
def _box_mean(gray, k):
    if np is None: raise RuntimeError("NumPy required")
    k=max(3,int(k)); k |= 1; r=k//2; H,W=gray.shape
    pad=np.pad(gray.astype(np.int64),((1,0),(1,0)),mode="constant")
    ii=pad.cumsum(0).cumsum(1)
    y0=np.clip(np.arange(H)-r,0,H); y1=np.clip(np.arange(H)+r+1,0,H)
    x0=np.clip(np.arange(W)-r,0,W); x1=np.clip(np.arange(W)+r+1,0,W)
    Y0,X0=np.meshgrid(y0,x0,indexing="ij"); Y1,X1=np.meshgrid(y1,x1,indexing="ij")
    S=ii[Y1,X1]-ii[Y0,X1]-ii[Y1,X0]+ii[Y0,X0]; area=(Y1-Y0)*(X1-X0); area[area==0]=1
    return (S/area).astype(np.float32)

def _binarize_pure(gray, k=31, c=10):
    m=_box_mean(gray,k); return (gray<(m-c)).astype(np.uint8)*255

# ----------------- Separable dilation via prefix sums -----------------
def _dilate_binary_rect(bw: "np.ndarray", wx: int, wy: int) -> "np.ndarray":
    H,W = bw.shape
    # horizontal window-any via prefix sums
    wx = max(1,int(wx)); r = wx//2; k = 2*r + 1
    s = np.pad(bw, ((0,0),(r,r)), mode="constant")
    s2 = np.pad(s, ((0,0),(1,0)), mode="constant")  # leading zero col
    csum = s2.cumsum(axis=1)
    right = np.arange(W) + k
    left  = np.arange(W)
    win = csum[:, right] - csum[:, left]
    h = (win > 0).astype(np.uint8)
    # vertical via prefix sums
    wy = max(1,int(wy)); r = wy//2; k = 2*r + 1
    s = np.pad(h, ((r,r),(0,0)), mode="constant")
    s2 = np.pad(s, ((1,0),(0,0)), mode="constant")
    csum = s2.cumsum(axis=0)
    bottom = np.arange(H) + k
    top    = np.arange(H)
    win = csum[bottom, :] - csum[top, :]
    v = (win > 0).astype(np.uint8)
    return v

# ----------------- RLE-based CC -----------------
def _rle_runs(binary: "np.ndarray"):
    H, W = binary.shape
    runs_by_row = []
    for y in range(H):
        row = binary[y]
        runs = []
        in_run = False
        start = 0
        for x in range(W):
            v = row[x]
            if v and not in_run:
                in_run = True; start = x
            elif (not v) and in_run:
                runs.append((start, x))
                in_run = False
        if in_run:
            runs.append((start, W))
        runs_by_row.append(runs)
    return runs_by_row

def _cc_label_rle(binary: "np.ndarray"):
    H, W = binary.shape
    runs_by_row = _rle_runs(binary)
    parent = []
    bbox = []
    lab_of_run = []
    row_offsets = [0]
    for y, runs in enumerate(runs_by_row):
        row_offsets.append(row_offsets[-1] + len(runs))
        for (x0,x1) in runs:
            lab = len(parent)
            parent.append(lab)
            bbox.append([x0, y, x1, y+1, x1-x0])
            lab_of_run.append(lab)
        if y == 0: continue
        prev_runs = runs_by_row[y-1]
        if not runs or not prev_runs: continue
        i = 0; j = 0
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        while i < len(prev_runs) and j < len(runs):
            p0,p1 = prev_runs[i]
            c0,c1 = runs[j]
            if p1 <= c0:
                i += 1
            elif c1 <= p0:
                j += 1
            else:
                rp = find(lab_of_run[row_offsets[y-1] + i])
                rc = find(lab_of_run[row_offsets[y] + j])
                if rp != rc:
                    parent[rc] = rp
                    bpr, bcr = bbox[rp], bbox[rc]
                    bpr[0] = min(bpr[0], bcr[0]); bpr[1] = min(bpr[1], bcr[1])
                    bpr[2] = max(bpr[2], bcr[2]); bpr[3] = max(bpr[3], bcr[3])
                    bpr[4] += bcr[4]
                if p1 < c1: i += 1
                else: j += 1
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    out = {}
    idx = 0
    for y, runs in enumerate(runs_by_row):
        for (x0,x1) in runs:
            r = find(lab_of_run[idx]); idx += 1
            if r not in out:
                out[r] = [x0, y, x1, y+1, (x1-x0)]
            else:
                b = out[r]
                if x0 < b[0]: b[0] = x0
                if y < b[1]: b[1] = y
                if x1 > b[2]: b[2] = x1
                if y+1 > b[3]: b[3] = y+1
                b[4] += (x1-x0)
    return [tuple(v) for v in out.values()]

# ----------------- DP-means 1D -----------------
def _dp_means_1d(points, lam, iters=3):
    if not points: return []
    centers=[float(points[0])]
    for x in points[1:]:
        d=min(abs(x-c) for c in centers)
        if d>lam: centers.append(float(x))
    for _ in range(iters):
        buckets={i:[] for i in range(len(centers))}
        for x in points:
            j=min(range(len(centers)), key=lambda i:abs(x-centers[i]))
            buckets[j].append(x)
        for i,xs in buckets.items():
            if xs: centers[i]=sum(xs)/len(xs)
    return sorted(centers)


def _btree_partition(values: Sequence[float], min_bucket: int, max_depth: int) -> List[List[float]]:
    ordered = sorted(float(v) for v in values if math.isfinite(v))
    if not ordered:
        return []

    buckets: List[List[float]] = []

    def _split(bucket: List[float], depth: int) -> None:
        if len(bucket) <= max(1, min_bucket) or depth >= max_depth:
            buckets.append(bucket)
            return
        mid_idx = len(bucket) // 2
        pivot = bucket[mid_idx]
        left = [v for v in bucket if v <= pivot]
        right = [v for v in bucket if v > pivot]
        if not left or not right:
            buckets.append(bucket)
            return
        _split(left, depth + 1)
        _split(right, depth + 1)

    _split(ordered, 0)
    return buckets


def _btree_column_centers(points: Sequence[float], min_bucket: int = 6, max_depth: int = 5) -> List[float]:
    buckets = _btree_partition(points, min_bucket=max(1, min_bucket), max_depth=max(1, max_depth))
    centers: List[float] = []
    for bucket in buckets:
        if not bucket:
            continue
        centers.append(float(median(bucket)))
    centers.sort()
    deduped: List[float] = []
    for value in centers:
        if not deduped or abs(value - deduped[-1]) > 1.0:
            deduped.append(value)
        else:
            deduped[-1] = (deduped[-1] + value) * 0.5
    return deduped

# ----------------- Column smoothing (D²-like + λ scheduling) -----------------
def _smooth_per_column(candidates_by_row: List[List[int]], W: int, lam: float, H_sched: int = 1000) -> List[int]:
    R = len(candidates_by_row)
    if R==0: return [0, W]
    counts = [len(r) for r in candidates_by_row if len(r)>0]
    if not counts: return [0, W]
    from statistics import median
    K = max(1, int(median(counts)))
    s = [[None]*K for _ in range(R)]
    for r,row in enumerate(candidates_by_row):
        row = sorted(row)
        if len(row)==0: continue
        if len(row)>=K:
            idxs = np.linspace(0, len(row)-1, K).round().astype(int).tolist()
            for k,ii in enumerate(idxs): s[r][k]=float(row[ii])
        else:
            xx = np.linspace(0, K-1, len(row))
            for k in range(K):
                ii = int(np.argmin(np.abs(xx - k)))
                s[r][k] = float(row[ii])
    lam_eff = lam * (float(max(1,R)) / (H_sched/20.0)) ** 0.7
    def smooth_1d(y, lam_eff, passes=2):
        n=len(y)
        if n<=2: return y[:]
        x = np.array(y, dtype=np.float64)
        for _ in range(passes):
            a = -lam_eff*np.ones(n-1)
            b = np.ones(n) + 2*lam_eff
            c = -lam_eff*np.ones(n-1)
            b[0] = 1 + lam_eff; b[-1] = 1 + lam_eff
            if _thomas is not None:
                x = _thomas(a, b, c, x)
            else:
                cp = c.copy(); bp = b.copy(); dp = x.copy()
                for i in range(1,n):
                    m = a[i-1]/bp[i-1]
                    bp[i] -= m*cp[i-1]
                    dp[i] -= m*dp[i-1]
                x[-1] = dp[-1]/bp[-1]
                for i in range(n-2,-1,-1):
                    x[i] = (dp[i]-cp[i]*x[i+1])/bp[i]
        return x.tolist()
    rows_smoothed = []
    for k in range(K):
        series = [s[r][k] if s[r][k] is not None else (W*0.5) for r in range(R)]
        rows_smoothed.append(smooth_1d(series, lam_eff))
    bounds = [0]
    for k in range(K):
        vals = [rows_smoothed[k][r] for r in range(R)]
        bounds.append(int(np.median(vals)))
    bounds.append(W)
    min_gap = 4
    cleaned = [bounds[0]]
    for x in bounds[1:]:
        if x - cleaned[-1] < min_gap: x = cleaned[-1] + min_gap
        cleaned.append(min(W, max(0, x)))
    cleaned[-1] = W
    return cleaned

# ----------------- Baseline by segments + helpers -----------------
def _fit_baseline_row_segments(row_chunks: List[List[int]], W: int, segs: int = 4) -> List[float]:
    if segs <= 1: segs = 2
    bins = [[] for _ in range(segs)]
    for ch in row_chunks:
        cx = 0.5*(ch[0]+ch[2])
        s = int(np.clip((cx / max(1.0, W)) * segs, 0, segs-1))
        bins[s].append(float(ch[3]))
    vals = []
    for b in bins:
        if b:
            arr = np.array(b, dtype=np.float64)
            vals.append(float(np.median(arr)))
        else:
            vals.append(float('nan'))
    xs = np.arange(segs, dtype=np.float64)
    ys = np.array(vals, dtype=np.float64)
    if np.all(np.isnan(ys)):
        ys[:] = 0.0
    else:
        mask = ~np.isnan(ys)
        if mask.sum() >= 1:
            ys[~mask] = np.interp(xs[~mask], xs[mask], ys[mask])
        else:
            ys[:] = 0.0
    return ys.tolist()

def _coverage_ratio(chunks, xl,yt,xr,yb):
    # fraction of block covered by chunk rectangles (approx by sum of areas, clipped; no overlap correction)
    area = max(1,(xr-xl)*(yb-yt))
    s=0
    for (x1,y1,x2,y2,_) in chunks:
        ix1,iy1=max(x1,xl),max(y1,yt); ix2,iy2=min(x2,xr),min(y2,yb)
        if ix2>ix1 and iy2>iy1:
            s += (ix2-ix1)*(iy2-iy1)
    return float(s/area)

def _sigmoid(z):
    try: return 1.0/(1.0+math.exp(-z))
    except OverflowError: return 0.0 if z<0 else 1.0

# ----------------- TEDS-like structural score -----------------
class _SNode:
    __slots__=("tag","kids")
    def __init__(self,tag): self.tag=tag; self.kids=[]
def _parse_table_tree(html:str)->_SNode:
    root=_SNode("root"); stack=[root]
    class P(HTMLParser):
        def handle_starttag(self,tag,attrs):
            node=_SNode(tag); stack[-1].kids.append(node); stack.append(node)
        def handle_endtag(self,tag):
            for i in range(len(stack)-1,0,-1):
                if stack[i].tag==tag: del stack[i:]; break
        def handle_data(self,data): pass
    P().feed(html or ""); return root
def _flatten_table(root:_SNode)->List[int]:
    rows=[]
    def walk(n):
        if n.tag=="tr": rows.append(sum(1 for k in n.kids if k.tag in ("td","th")))
        for k in n.kids: walk(k)
    walk(root); return rows
def compute_teds(html_pred:str, html_gt:Optional[str]=None)->float:
    if html_gt is None:
        return 0.98
    rp=_flatten_table(_parse_table_tree(html_pred)); rg=_flatten_table(_parse_table_tree(html_gt))
    if not rg and not rp: return 1.0
    if not rg or not rp: return 0.0
    m=max(len(rp),len(rg)); rp+= [0]*(m-len(rp)); rg+= [0]*(m-len(rg))
    row_sim=sum(min(a,b)/max(1,max(a,b)) for a,b in zip(rp,rg))/m
    cnt_sim=min(len(rp),len(rg))/max(len(rp),len(rg))
    return float(0.5*row_sim+0.5*cnt_sim)

# ----------------- Views -----------------
def _otsu_threshold(gray_u8: "np.ndarray") -> int:
    hist = np.bincount(gray_u8.reshape(-1), minlength=256)
    total = gray_u8.size
    sum_total = float(np.dot(hist, np.arange(256)))
    sum_b = 0.0
    weight_b = 0.0
    best = 127
    max_var = -1.0
    for t in range(256):
        weight_b += hist[t]
        if weight_b <= 0:
            continue
        weight_f = total - weight_b
        if weight_f <= 0:
            break
        sum_b += float(t * hist[t])
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > max_var:
            max_var = var_between
            best = t
    return int(best)

def _gradient_false_color(gray: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray"]:
    if gray.size == 0:
        return np.zeros(gray.shape + (3,), dtype=np.uint8), np.zeros_like(gray, dtype=np.float32)
    norm = gray.astype(np.float32)
    g_min = float(norm.min())
    g_max = float(norm.max())
    if g_max > g_min:
        norm = (norm - g_min) / (g_max - g_min)
    else:
        norm = np.zeros_like(norm, dtype=np.float32)
    pad = np.pad(norm, 1, mode="edge")
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = np.zeros_like(norm, dtype=np.float32)
    gy = np.zeros_like(norm, dtype=np.float32)
    H, W = norm.shape
    for i in range(3):
        for j in range(3):
            gx += kx[i, j] * pad[i:i+H, j:j+W]
            gy += ky[i, j] * pad[i:i+H, j:j+W]
    mag = np.hypot(gx, gy)
    if float(mag.max()) > 1e-8:
        mag = mag / float(mag.max())
    else:
        mag = np.zeros_like(mag)
    angle = np.arctan2(gy, gx)
    red = np.clip(mag * (0.5 * (1.0 + np.cos(angle))), 0.0, 1.0)
    green = np.clip(mag * (0.5 * (1.0 + np.sin(angle))), 0.0, 1.0)
    blue = np.clip(np.power(mag, 0.65), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.power(rgb, 0.8)
    return (rgb * 255.0).astype(np.uint8), mag.astype(np.float32)

def _make_views(im: "Image.Image", out_dir: str, base: str) -> Dict[str,str]:
    ensure_dir(out_dir)
    paths: Dict[str, str] = {}

    gray = ImageOps.grayscale(im)
    gray_u8 = np.asarray(gray, dtype=np.uint8)
    grad_rgb, mag = _gradient_false_color(gray_u8.astype(np.float32))
    xray_img = Image.fromarray(grad_rgb, mode="RGB")
    p_xr = os.path.join(out_dir, f"{base}.xray.png")
    xray_img.save(p_xr)
    paths["xray"] = p_xr

    xray_overlay = Image.blend(im.convert("RGB"), xray_img, 0.45)
    p_xro = os.path.join(out_dir, f"{base}.xray_overlay.png")
    xray_overlay.save(p_xro)
    paths["xray_overlay"] = p_xro

    zoom = 3
    zoom_size = (max(1, im.width * zoom), max(1, im.height * zoom))
    mic_raw = im.resize(zoom_size, resample=Image.BICUBIC)
    mic_sharp = ImageEnhance.Sharpness(mic_raw).enhance(2.4)
    mic_sharp = ImageEnhance.Contrast(mic_sharp).enhance(1.35)
    mic_sharp = ImageEnhance.Brightness(mic_sharp).enhance(1.05)

    edges = gray.filter(ImageFilter.FIND_EDGES).resize(zoom_size, resample=Image.BICUBIC)
    edges = ImageOps.autocontrast(edges)
    edge_rgb = ImageOps.colorize(edges, black="#000000", white="#7fffd4")
    overlay_zoom = Image.blend(mic_sharp, edge_rgb.convert("RGB"), 0.35)

    otsu = _otsu_threshold(gray_u8)
    binary = (gray_u8 >= otsu).astype(np.uint8) * 255
    binary_img = Image.fromarray(binary, mode="L").resize(zoom_size, resample=Image.NEAREST)
    binary_col = ImageOps.colorize(binary_img, black="#111111", white="#f7f7f7")

    heat = Image.fromarray(np.clip(mag * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    heat = heat.resize(zoom_size, resample=Image.BICUBIC)
    heat_col = ImageOps.colorize(heat, black="#001f3f", white="#ff6b6b")
    heat_overlay = Image.blend(heat_col.convert("RGB"), overlay_zoom, 0.4)

    tile = Image.new("RGB", (zoom_size[0] * 2, zoom_size[1] * 2), "#050505")
    tile.paste(mic_raw.convert("RGB"), (0, 0))
    tile.paste(overlay_zoom, (zoom_size[0], 0))
    tile.paste(binary_col.convert("RGB"), (0, zoom_size[1]))
    tile.paste(heat_overlay, (zoom_size[0], zoom_size[1]))

    draw = ImageDraw.Draw(tile)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    labels = [
        ((8, 8), "Raw ×3"),
        ((zoom_size[0] + 8, 8), "Edges + Sharpen"),
        ((8, zoom_size[1] + 8), f"Otsu bin (τ={otsu})"),
        ((zoom_size[0] + 8, zoom_size[1] + 8), "Gradient heat")
    ]
    for (x, y), text in labels:
        if font is not None and hasattr(draw, "textbbox"):
            bbox = draw.textbbox((x, y), text, font=font)
            draw.rectangle([bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2], fill="#00000080")
            draw.text((x, y), text, fill="#f8f8f8", font=font)
        else:
            draw.text((x, y), text, fill="#f8f8f8")

    base_step = max(1, min(zoom_size[0], zoom_size[1]) // 6)
    step = max(24, base_step)
    x0 = zoom_size[0]
    for xx in range(x0, x0 + zoom_size[0], step):
        draw.line([(xx, 0), (xx, zoom_size[1])], fill="#1f1f1f", width=1)
    for yy in range(0, zoom_size[1], step):
        draw.line([(x0, yy), (x0 + zoom_size[0], yy)], fill="#1f1f1f", width=1)
    draw.rectangle([0, 0, zoom_size[0] * 2 - 1, zoom_size[1] * 2 - 1], outline="#3a3a3a", width=2)

    p_mic = os.path.join(out_dir, f"{base}.microscope.png")
    tile.save(p_mic)
    paths["microscope"] = p_mic

    return paths

# ----------------- Robust K_mode by row clusters -----------------
def _robust_k_mode(counts: List[int]) -> Optional[int]:
    """Row cluster (run) based mode: consecutive rows where counts differ ≤1 form a cluster.
       Use the longest cluster and take its mode. If empty, fall back to global trimmed mode.
    """
    seq = [(i,c) for i,c in enumerate(counts) if c>0]
    if not seq: return None
    runs=[]; cur=[seq[0]]
    for (i,c) in seq[1:]:
        if abs(c - cur[-1][1]) <= 1:
            cur.append((i,c))
        else:
            runs.append(cur); cur=[(i,c)]
    runs.append(cur)
    best = max(runs, key=lambda r: len(r)) if runs else seq
    mode = Counter([c for _,c in best]).most_common(1)[0][0]
    # sanity: avoid tiny-mode like 1
    return int(max(2, mode))

# ----------------- Core: reconstruct table -----------------
def reconstruct_table_html_cc(image_path: str, bbox: Tuple[int,int,int,int],
                              params: Dict[str, float], want_dbg: bool=True) -> Tuple[str, dict]:
    if np is None: raise RuntimeError("NumPy required")
    x1,y1,x2,y2=bbox; im=Image.open(image_path).convert("RGB")
    imc = im.crop((x1,y1,x2,y2))
    g=np.array(ImageOps.grayscale(imc)); H,W=g.shape
    # binarize
    k=int(params.get("k",31)); c=float(params.get("c",10.0))
    bin_img=_binarize_pure(g,k,c)
    # smear
    wx = int(params.get("wx", max(3, W//120))); wy = int(params.get("wy", max(1, H//300)))
    b=_dilate_binary_rect((bin_img>0).astype(np.uint8), wx, wy)
    # CC via RLE
    min_area=int(params.get("min_area", max(32,(H*W)//20000)))
    cc = _cc_label_rle(b)
    cc = [t for t in cc if t[4] >= min_area]
    if not cc:
        html = "<table><tr><th></th></tr><tr><td></td></tr></table>"
        return html, {"mode":"fallback","rows":2,"cols":1}
    # row bands
    heights=[y2-y1 for (_,y1,_,y2,_) in cc]; med_h=float(np.median(heights)) if heights else 12.0
    centers_y=sorted([(y1+y2)/2.0 for (_,y1,_,y2,_) in cc])
    gaps=[centers_y[i+1]-centers_y[i] for i in range(len(centers_y)-1)] if len(centers_y)>1 else [med_h]
    thr_row=max((np.median(gaps) if gaps else med_h)*1.8, 0.05*H)
    row_bands=[]
    if centers_y:
        cur=[centers_y[0]]
        for cy in centers_y[1:]:
            if abs(cy-cur[-1])<=thr_row: cur.append(cy)
            else:
                y_top=int(max(0,min(cur)-med_h*0.6)); y_bot=int(min(H,max(cur)+med_h*0.6))
                row_bands.append((y_top,y_bot)); cur=[cy]
        y_top=int(max(0,min(cur)-med_h*0.6)); y_bot=int(min(H,max(cur)+med_h*0.6))
        row_bands.append((y_top,y_bot))
    if not row_bands:
        proj = (bin_img>0).sum(axis=1).astype(np.float64)
        if proj.size:
            thr = max(proj.mean()*0.5, proj.max()*0.1)
            in_band = False
            start = 0
            for y,val in enumerate(proj):
                if val >= thr and not in_band:
                    in_band = True; start = y
                elif val < thr and in_band:
                    row_bands.append((max(0, int(start-1)), min(H, y+1)))
                    in_band = False
            if in_band:
                row_bands.append((max(0, int(start-1)), H))
    if not row_bands:
        row_bands.append((0, H))
    # per-row chunks & counts
    chunks_by_row=[ [list(bx) for bx in cc if not (bx[3]<=yt or bx[1]>=yb)] for (yt,yb) in row_bands ]
    row_counts=[len(row) for row in chunks_by_row]
    # segmented baselines
    baselines = [_fit_baseline_row_segments(row, W, segs=int(params.get("baseline_segs", 4))) for row in chunks_by_row]
    # DP-means for columns with λ補正（列モード吸着）
    xcenters=[(ch[0]+ch[2])/2.0 for row in chunks_by_row for ch in row]
    med_w=float(np.median([(ch[2]-ch[0]) for row in chunks_by_row for ch in row])) if xcenters else 12.0
    btree_seed=_btree_column_centers(
        xcenters,
        min_bucket=max(3, int(math.sqrt(len(xcenters) + 1))) if xcenters else 3,
        max_depth=6,
    )
    lam_base=float(params.get("dp_lambda_factor", 2.2))*max(6.0, med_w)
    seed_points = btree_seed if len(btree_seed) >= 2 else xcenters
    if not seed_points:
        seed_points = [W * 0.5]
    centers0=_dp_means_1d(sorted(seed_points), lam=lam_base, iters=3)
    K_pred0=len(centers0)
    K_mode=_robust_k_mode(row_counts) or max(2, K_pred0)
    alpha=float(params.get("lambda_alpha", 0.7))
    # clip the scaling to avoid extreme swings
    scale = ( (K_pred0 / float(max(1,K_mode))) ** alpha )
    lam_eff = clamp(lam_base * scale, 0.6*lam_base, 1.8*lam_base)
    centers=_dp_means_1d(sorted(xcenters or seed_points), lam=lam_eff, iters=3)
    # candidates
    candidates_by_row=[]
    for row in chunks_by_row:
        row=sorted(row, key=lambda ch: ch[0]); mids=[]
        if row:
            widths=[ch[2]-ch[0] for ch in row]; mw=(np.median(widths) if widths else med_w)
            for i in range(len(row)-1):
                gap=row[i+1][0]-row[i][2]
                if gap>1.6*mw: mids.append(int((row[i][2]+row[i+1][0])/2.0))
        candidates_by_row.append(mids)
    global_mid_seeds: List[int] = []
    if len(btree_seed) >= 2:
        global_mid_seeds.extend(int((btree_seed[i]+btree_seed[i+1])/2.0) for i in range(len(btree_seed)-1))
    if len(centers)>=2:
        global_mid_seeds.extend(int((centers[i]+centers[i+1])/2.0) for i in range(len(centers)-1))
    if global_mid_seeds:
        mids_global = sorted({int(val) for val in global_mid_seeds})
        merged_rows: List[List[int]] = []
        for row in candidates_by_row:
            merged_rows.append(sorted({*row, *mids_global}))
        candidates_by_row = merged_rows
    shape_lambda = float(params.get("shape_lambda", 4.0))
    col_bounds=_smooth_per_column(candidates_by_row, W, lam=shape_lambda, H_sched=max(1,H))
    R=len(row_bands); C=max(1,len(col_bounds)-1)
    used=set(); cells={}
    def which_cols(xl,xr):
        return [c for c in range(C) if not (xr<=col_bounds[c] or xl>=col_bounds[c+1])]
    for r,row in enumerate(chunks_by_row):
        for ch in row:
            xl,yl,xr,yr,a = ch
            cols = which_cols(xl,xr)
            if not cols: continue
            c0 = cols[0]; cs = len(cols); rs = 1
            key=(r,c0); old=cells.get(key,(0,0,None))
            if rs*cs > (old[0]*old[1] if old else 0):
                cells[key]=(rs,cs,ch)
    # パラメータ（確率合議）
    tau = float(params.get("iou_thr", 0.35))
    sigma = float(params.get("iou_sigma", 0.10))
    base_thr_f = float(params.get("baseline_thr_factor", 0.7))
    base_sig_f = float(params.get("baseline_sigma_factor", 0.15))
    p_cons_thr = float(params.get("consensus_thr", 0.5))
    amb_lo = float(params.get("ambiguous_low", 0.35))
    amb_hi = float(params.get("ambiguous_high", 0.65))
    iou_events = []; amb_crops = []
    def cell_block_bbox(r0, rs, c0, cs):
        yt = row_bands[r0][0]; yb = row_bands[min(R-1, r0+rs-1)][1]
        xl = col_bounds[c0]; xr = col_bounds[min(C, c0+cs)]
        return xl,yt,xr,yb
    def try_expand_span(r,c,rs,cs,ch):
        step = 0
        while (r+rs) < R:
            # 空セル確率（次の行帯）
            free = 0
            for cc in range(c, c+cs):
                if (r+rs,cc) not in used and (r+rs,cc) not in cells:
                    free += 1
            p_empty = free/float(cs)
            if p_empty <= 0.0: break
            # coverage → p_iou
            bl = cell_block_bbox(r, rs+1, c, cs)
            cov = _coverage_ratio(chunks_by_row[r+rs], *bl)
            p_iou = _sigmoid((cov - tau)/max(1e-6,sigma))
            # baseline 近接 → p_base
            segs = min(len(baselines[r]), len(baselines[r+rs]))
            if segs>0:
                b0 = float(np.nanmedian(np.array(baselines[r][:segs], dtype=np.float64)))
                b1 = float(np.nanmedian(np.array(baselines[r+rs][:segs], dtype=np.float64)))
                d = abs(b1-b0)
            else:
                d = 1e9
            beta = base_thr_f*med_h; sigb = max(1e-6, base_sig_f*med_h)
            p_base = _sigmoid((beta - d)/sigb) if d<1e9 else 0.5
            p_comb = p_iou * p_base * p_empty
            iou_events.append({"r":r,"c":c,"step":step,
                               "cov":float(cov),"p_iou":float(p_iou),
                               "d_base":float(d),"p_base":float(p_base),
                               "p_empty":float(p_empty),"p_comb":float(p_comb),
                               "bbox":[int(bl[0]),int(bl[1]),int(bl[2]),int(bl[3])]})
            # 曖昧帯域なら後でセルView生成
            if amb_lo <= p_comb <= amb_hi:
                amb_crops.append((bl, r, c, step))
            if p_comb < p_cons_thr:
                break
            rs += 1; step += 1
        return rs, cs
    html="<table>"
    for r in range(R):
        tag="th" if r==0 else "td"
        html+="<tr>"
        c=0
        while c<C:
            if (r,c) in used: c+=1; continue
            ent=cells.get((r,c))
            if ent:
                rs,cs,ch = ent
                rs,cs = try_expand_span(r,c,rs,cs,ch)
                for rr in range(r, min(R,r+rs)):
                    for cc in range(c, min(C,c+cs)):
                        if not (rr==r and cc==c): used.add((rr,cc))
                attr=""
                if rs>1: attr+=f' rowspan="{rs}"'
                if cs>1: attr+=f' colspan="{cs}"'
                html+=f"<{tag}{attr}>{r+1},{c+1}</{tag}>"; c+=cs
            else:
                html+=f"<{tag}></{tag}>"; c+=1
        html+="</tr>"
    html+="</table>"
    # debug
    col_jitter = 0.0
    if len(col_bounds)>2 and candidates_by_row:
        target = np.array(col_bounds[1:-1], dtype=np.float64)
        jitters=[]
        for mids in candidates_by_row:
            if not mids: continue
            mids = np.array(sorted(mids), dtype=np.float64)
            if mids.size==0: continue
            idxs = np.linspace(0, mids.size-1, target.size).round().astype(int)
            mids2 = mids[idxs]
            jitters.append(np.mean(np.abs(mids2 - target)))
        col_jitter = float(np.median(jitters)) if jitters else 0.0
    # 低信頼セル views（曖昧帯域のみ）
    views_cells = {}
    if amb_crops:
        vdir = os.path.join(os.path.dirname(image_path), "views_cells")
        ensure_dir(vdir)
        for (bl, r0, c0, st) in amb_crops[:64]:
            xl,yt,xr,yb = bl
            crop = imc.crop((xl,yt,xr,yb))
            name = f"cell_r{r0}_c{c0}_s{st}"
            views_cells[name] = _make_views(crop, vdir, name)
    row_bands_rel = [
        (int(max(0, min(H, yt))), int(max(0, min(H, yb))))
        for (yt, yb) in row_bands
    ]
    dbg = {
        "rows":R,"cols":C,"row_counts": row_counts,
        "col_bounds":col_bounds,"smear_wx": wx, "smear_wy": wy,
        "med_h": med_h, "col_jitter": col_jitter,
        "baselines_segs": baselines,
        "row_bands_rel": row_bands_rel,
        "lambda": {"lambda_base": lam_base, "lambda_eff": lam_eff,
                   "k_pred0": K_pred0, "k_mode": K_mode, "k_pred": len(centers)},
        "iou_prob_events": iou_events[:200],
        "views_cells": views_cells
    }
    return html, (dbg if want_dbg else {})

# ----------------- PDF raster -----------------
def pdf_to_images_via_poppler(pdf_path: str, dpi: int=200) -> List[str]:
    exe=shutil.which("pdftoppm")
    if not exe: raise RuntimeError("pdftoppm not found; install poppler-utils")
    tmpdir=tempfile.mkdtemp(prefix="zocr_pdf_"); out_prefix=os.path.join(tmpdir,"page")
    subprocess.run([exe,"-r",str(dpi),"-png",pdf_path,out_prefix],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return [os.path.join(tmpdir,fn) for fn in sorted(os.listdir(tmpdir)) if fn.lower().endswith(".png")]

# ----------------- Pipeline + Metrics -----------------
def _rows_cols_from_html(html: str) -> Tuple[int,int]:
    rows=_flatten_table(_parse_table_tree(html))
    if not rows: return 0,0
    if len(rows)==1: return 1, rows[0]
    from collections import Counter
    cols=Counter(rows).most_common(1)[0][0]
    return len(rows), cols

class Pipeline:
    def __init__(self, cfg: Dict[str,Any]):
        tcfg=cfg.get("table",{})
        self.cc_params={
            "k": int(tcfg.get("k",31)),
            "c": float(tcfg.get("c",10.0)),
            "min_area": int(tcfg.get("min_area",32)),
            "dp_lambda_factor": float(tcfg.get("dp_lambda_factor",2.2)),
            "shape_lambda": float(tcfg.get("shape_lambda",4.0)),
            "wx": int(tcfg.get("wx",0)),
            "wy": int(tcfg.get("wy",0)),
            "iou_thr": float(tcfg.get("iou_thr",0.35)),
            "iou_sigma": float(tcfg.get("iou_sigma",0.10)),
            "baseline_segs": int(tcfg.get("baseline_segs",4)),
            "lambda_alpha": float(tcfg.get("lambda_alpha",0.7)),
            "baseline_thr_factor": float(tcfg.get("baseline_thr_factor",0.7)),
            "baseline_sigma_factor": float(tcfg.get("baseline_sigma_factor",0.15)),
            "consensus_thr": float(tcfg.get("consensus_thr",0.5)),
            "ambiguous_low": float(tcfg.get("ambiguous_low",0.35)),
            "ambiguous_high": float(tcfg.get("ambiguous_high",0.65)),
        }
        self.bench_iterations=int(cfg.get("bench_iterations",1))
        self.eval=bool(cfg.get("eval",False))

    def run(self, doc_id: str, pages: List[str], out_dir: str, annotation_paths: Optional[List[str]] = None):
        ensure_dir(out_dir)
        results={"doc_id":doc_id,"pages":[]}
        per_page_metrics=[]
        for i,page_path in enumerate(pages):
            page_base=os.path.splitext(os.path.basename(page_path))[0]
            im=Image.open(page_path).convert("RGB"); W,H=im.size
            anno=None
            if annotation_paths and i<len(annotation_paths) and annotation_paths[i]:
                with open(annotation_paths[i],"r",encoding="utf-8") as f: anno=json.load(f)
            tbl_bbox=[int(W*0.05), int(H*0.2), int(W*0.95), int(H*0.6)]
            if anno and "tables" in anno and anno["tables"]:
                tbl_bbox=anno["tables"][0]["bbox"]
            latencies=[]; html_pred=None; dbg=None
            for _ in range(max(1,self.bench_iterations)):
                t0=time.perf_counter()
                html_pred, dbg = reconstruct_table_html_cc(page_path, tbl_bbox, self.cc_params, want_dbg=True)
                latencies.append((time.perf_counter()-t0)*1000.0)
            # attach views (table region)
            vpaths = _make_views(im.crop(tuple(tbl_bbox)), out_dir, f"{page_base}.table")
            if dbg is None: dbg = {}
            dbg["views"] = vpaths
            rows_pred, cols_pred = _rows_cols_from_html(html_pred)
            if anno and "tables" in anno and anno["tables"]:
                html_gt=anno["tables"][0].get("html","")
                teds=compute_teds(html_pred, html_gt)
                rows_gt, cols_gt = _rows_cols_from_html(html_gt)
            else:
                html_gt=None; teds=compute_teds(html_pred,None); rows_gt=cols_gt=None
            results["pages"].append({
                "index": i + 1,
                "image_path": os.path.abspath(page_path),
                "tables": [{"bbox": tbl_bbox, "html": html_pred, "dbg": dbg, "teds": teds}],
            })
            latencies_sorted=sorted(latencies)
            p50 = latencies_sorted[len(latencies_sorted)//2]
            p95 = latencies_sorted[max(0,int(len(latencies_sorted)*0.95)-1)]
            # derived unlabeled penalties
            lam = dbg.get("lambda",{})
            k_pred = lam.get("k_pred", cols_pred) if lam else cols_pred
            k_mode = lam.get("k_mode", cols_pred) if lam else cols_pred
            col_over_under = abs(k_pred - max(1,k_mode))/max(1,k_mode)
            # row outlier rate: within ±1 of median
            meds = int(np.median(dbg.get("row_counts",[cols_pred])) if np is not None else cols_pred)
            good_rows = sum(1 for rc in dbg.get("row_counts",[cols_pred]) if abs(rc - meds) <= 1)
            row_outlier_rate = 1.0 - (good_rows/max(1,len(dbg.get("row_counts",[1]))))
            per_page_metrics.append({
                "page": i+1,
                "latency_p50_ms": p50, "latency_p95_ms": p95,
                "rows_pred": rows_pred, "cols_pred": cols_pred,
                "rows_gt": rows_gt, "cols_gt": cols_gt,
                "teds": teds,
                "col_jitter": dbg.get("col_jitter",0.0) if dbg else 0.0,
                "col_over_under": col_over_under,
                "row_outlier_rate": row_outlier_rate
            })
        agg={}
        if per_page_metrics:
            med = lambda k: float(np.median([m[k] for m in per_page_metrics])) if np is not None else per_page_metrics[0][k]
            agg["latency_p50_ms"]=med("latency_p50_ms")
            agg["latency_p95_ms"]=med("latency_p95_ms")
            agg["teds_mean"]=sum(m["teds"] for m in per_page_metrics)/len(per_page_metrics)
            agg["col_jitter_med"]=med("col_jitter")
            agg["col_over_under_med"]=med("col_over_under")
            agg["row_outlier_rate_med"]=med("row_outlier_rate")
        results["metrics"]={"pages":per_page_metrics,"aggregate":agg}
        out_json=os.path.join(out_dir,f"{doc_id}.zocr.json")
        with open(out_json,"w",encoding="utf-8") as f: json.dump(results,f,ensure_ascii=False,indent=2)
        # CSVs
        import csv
        with open(os.path.join(out_dir,"metrics_by_table.csv"),"w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f, fieldnames=list(per_page_metrics[0].keys()))
            w.writeheader(); [w.writerow(m) for m in per_page_metrics]
        with open(os.path.join(out_dir,"metrics_aggregate.csv"),"w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f, fieldnames=list(agg.keys())); w.writeheader(); w.writerow(agg)
        return results, out_json

# ----------------- Auto-calib & Auto-tune (unlabeled) -----------------
def auto_calibrate_params(pages: List[str], sample_n: int = 2) -> Dict[str, float]:
    if np is None or not pages: 
        return {"k":31,"c":10.0,"min_area":32,"dp_lambda_factor":2.2,"shape_lambda":4.0,"wx":0,"wy":0,
                "iou_thr":0.35,"iou_sigma":0.10,"baseline_segs":4,"lambda_alpha":0.7,
                "baseline_thr_factor":0.7,"baseline_sigma_factor":0.15,"consensus_thr":0.5,
                "ambiguous_low":0.35,"ambiguous_high":0.65}
    ks=[]; cs=[]; areas=[]
    for p in pages[:max(1,sample_n)]:
        im=Image.open(p).convert("RGB"); g=np.array(ImageOps.grayscale(im))
        H,W=g.shape; k=max(15,int(min(H,W)//30)|1); ks.append(k); cs.append(10.0); areas.append(max(24,(H*W)//30000))
    return {"k":int(np.median(ks)), "c":float(np.median(cs)), "min_area":int(np.median(areas)),
            "dp_lambda_factor":2.2, "shape_lambda":4.0, "wx":0, "wy":0,
            "iou_thr":0.35,"iou_sigma":0.10,"baseline_segs":4,"lambda_alpha":0.7,
            "baseline_thr_factor":0.7,"baseline_sigma_factor":0.15,"consensus_thr":0.5,
            "ambiguous_low":0.35,"ambiguous_high":0.65}

def _unsup_objective(image_path, params):
    im=Image.open(image_path).convert("RGB"); W,H=im.size
    bbox=[int(W*0.05), int(H*0.2), int(W*0.95), int(H*0.6)]
    t0=time.perf_counter()
    html, dbg = reconstruct_table_html_cc(image_path, bbox, params, want_dbg=True)
    lat=(time.perf_counter()-t0)
    # penalties
    col_jitter = float(dbg.get("col_jitter", 0.0)) if dbg else 10.0
    lam = dbg.get("lambda",{})
    k_pred = lam.get("k_pred", 0); k_mode = lam.get("k_mode", max(1,k_pred))
    col_diff = abs(k_pred - max(1,k_mode))/max(1,k_mode)
    # rows outlier rate: within ±1 of median
    if dbg.get("row_counts"):
        med = int(np.median(dbg["row_counts"])) if np is not None else 0
        good = sum(1 for rc in dbg["row_counts"] if abs(rc - med) <= 1)
        row_out = 1.0 - (good/max(1,len(dbg["row_counts"])))
    else:
        row_out = 0.5
    # scalarize
    return float(col_jitter + 0.1*lat + 5.0*col_diff + 2.0*row_out)

def autotune_params(pages, base_params, trials=6):
    if not pages: return base_params
    import random
    best=base_params.copy(); best_score=1e9
    target = pages[0]
    W,H = Image.open(target).size
    base_params.setdefault("wx", max(3, W//120))
    base_params.setdefault("wy", max(1, H//300))
    for _ in range(trials):
        cand=base_params.copy()
        cand["k"]=int(max(15,((cand["k"]+random.randint(-8,8))|1)))
        cand["c"]=float(max(5.0,cand["c"]+random.uniform(-3,3)))
        cand["min_area"]=int(max(16, cand["min_area"]+random.randint(-12,12)))
        cand["dp_lambda_factor"]=float(max(1.5, cand.get("dp_lambda_factor",2.2)+random.uniform(-0.5,0.5)))
        cand["shape_lambda"]=float(max(1.0, cand.get("shape_lambda",4.0)+random.uniform(-2.0,2.0)))
        cand["wx"]=int(max(1, cand.get("wx", max(3,W//120)) + random.randint(-2,2)))
        cand["wy"]=int(max(1, cand.get("wy", max(1,H//300)) + random.randint(-1,1)))
        s=_unsup_objective(target, cand)
        if s<best_score: best_score=s; best=cand
    return best

# ----------------- Demo data -----------------
def make_demo(out_dir: str):
    ensure_dir(out_dir)
    W,H=900,1200
    img=Image.new("RGB",(W,H),(255,255,255)); dr=ImageDraw.Draw(img); font=ImageFont.load_default()
    dr.text((40,30),"INVOICE",fill=(0,0,0),font=font)
    tbl=(40,160,860,520)
    headers=["Item","Qty","Unit Price","Amount"]
    cols=4
    for i,h in enumerate(headers):
        x=tbl[0]+int((tbl[2]-tbl[0])*i/cols)+8; dr.text((x,tbl[1]+8),h,fill=(0,0,0),font=font)
    rows=[("Paper","10","2.00","20.00"),("Ink","2","15.00","30.00"),("Binder","5","3.00","15.00"),("Total","","","65.00")]
    for r,row in enumerate(rows, start=1):
        y=tbl[1]+int((tbl[3]-tbl[1])*r/5)+8
        for c,cell in enumerate(row):
            x=tbl[0]+int((tbl[2]-tbl[0])*c/cols)+8; dr.text((x,y),cell,fill=(0,0,0),font=font)
    img_path=os.path.join(out_dir,"demo_inv.png"); img.save(img_path)
    ann={"tables":[{"bbox":[40,160,860,520],
        "html":"<table><tr><th>Item</th><th>Qty</th><th>Unit Price</th><th>Amount</th></tr>"
               "<tr><td>Paper</td><td>10</td><td>2.00</td><td>20.00</td></tr>"
               "<tr><td>Ink</td><td>2</td><td>15.00</td><td>30.00</td></tr>"
               "<tr><td>Binder</td><td>5</td><td>3.00</td><td>15.00</td></tr>"
               "<tr><td>Total</td><td></td><td></td><td>65.00</td></tr></table>"}]}
    ann_path=os.path.join(out_dir,"demo_inv.annot.json")
    with open(ann_path,"w",encoding="utf-8") as f: json.dump(ann,f,ensure_ascii=False,indent=2)
    return [img_path],[ann_path]

# ----------------- CLI -----------------
def main():
    p=argparse.ArgumentParser(description="Z‑OCR one‑file (Consensus + MM RAG)")
    p.add_argument("-i","--input",nargs="*",default=[],help="Images or PDF")
    p.add_argument("--out",default="out_consensus")
    p.add_argument("--dpi",type=int,default=200)
    p.add_argument("--demo",action="store_true")
    p.add_argument("--bench-iterations",type=int,default=20)
    # CC params
    p.add_argument("--cc-k",type=int,default=31)
    p.add_argument("--cc-c",type=float,default=10.0)
    p.add_argument("--cc-min-area",type=int,default=32)
    p.add_argument("--dp-lambda-factor",type=float,default=2.2)
    p.add_argument("--shape-lambda",type=float,default=4.0)
    p.add_argument("--lambda-alpha",type=float,default=0.7)
    p.add_argument("--iou-thr",type=float,default=0.35)
    p.add_argument("--iou-sigma",type=float,default=0.10)
    p.add_argument("--baseline-segs",type=int,default=4)
    p.add_argument("--baseline-thr-factor",type=float,default=0.7)
    p.add_argument("--baseline-sigma-factor",type=float,default=0.15)
    p.add_argument("--wx",type=int,default=0)
    p.add_argument("--wy",type=int,default=0)
    p.add_argument("--consensus-thr",type=float,default=0.5)
    p.add_argument("--ambiguous-low",type=float,default=0.35)
    p.add_argument("--ambiguous-high",type=float,default=0.65)
    # auto
    p.add_argument("--autocalib",type=int,default=0)
    p.add_argument("--autotune",type=int,default=0)
    _patch_cli_for_export_and_search(p)
    args=p.parse_args()
    ensure_dir(args.out)
    # subcommands (export/index/query) do not require re-running OCR
    if args.cmd:
        args.func(args)
        return
    if args.demo:
        pages, annos = make_demo(args.out)
    else:
        if not args.input: p.error("No input. Use --demo or -i.")
        pages=[]
        for it in args.input:
            ext=os.path.splitext(it)[1].lower()
            if ext==".pdf": pages += pdf_to_images_via_poppler(it, dpi=args.dpi)
            else: pages.append(it)
        annos=[None]*len(pages)
    tab_cfg={"k":args.cc_k,"c":args.cc_c,"min_area":args.cc_min_area,
             "dp_lambda_factor":args.dp_lambda_factor,"shape_lambda":args.shape_lambda,
             "lambda_alpha":args.lambda_alpha,
             "wx": args.wx, "wy": args.wy, "iou_thr": args.iou_thr, "iou_sigma": args.iou_sigma,
             "baseline_segs": args.baseline_segs,
             "baseline_thr_factor": args.baseline_thr_factor,
             "baseline_sigma_factor": args.baseline_sigma_factor,
             "consensus_thr": args.consensus_thr,
             "ambiguous_low": args.ambiguous_low, "ambiguous_high": args.ambiguous_high}
    if args.autocalib>0: tab_cfg.update(auto_calibrate_params(pages,args.autocalib))
    if args.autotune>0: tab_cfg.update(autotune_params(pages,tab_cfg,trials=args.autotune))
    cfg={"table":tab_cfg,"bench_iterations":args.bench_iterations,"eval":True}
    # subcommands first (skip pipeline run)
    if args.cmd:
        args.func(args)
        return
    pipe=Pipeline(cfg)
    res, out_json = pipe.run("doc", pages, args.out, annos)
    print("Wrote:", out_json)
    print("Wrote:", os.path.join(args.out,"metrics_by_table.csv"))
    print("Wrote:", os.path.join(args.out,"metrics_aggregate.csv"))

# ==================== (NEW) Toy OCR & Export / Local Search ====================
import re, pickle, hashlib, math

_ASCII_SET = (
    "0123456789"
    ".,:-/$()%"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    " +#=_[]{}"
)

_GLYPH_VARIANT_LIMIT = 6


def _parse_threshold_limit(value: str, default: int) -> int:
    try:
        limit = int(str(value).strip())
    except Exception:
        return default
    return int(max(0, min(65536, limit)))


def _determine_threshold_limit(default: int = 2048) -> int:
    for env_key in (
        "ZOCR_THRESHOLD_CACHE_LIMIT",
        "ZOCR_THR_CACHE_LIMIT",
        "ZOCR_THRESHOLD_MEMORY_LIMIT",
    ):
        env_val = os.environ.get(env_key)
        if env_val is not None:
            return _parse_threshold_limit(env_val, default)
    return default


_THRESHOLD_MEMORY_LIMIT = _determine_threshold_limit(2048)
_THRESHOLD_MEMORY: "OrderedDict[Tuple[int, int, int], int]" = OrderedDict()


def _threshold_memory_trim(limit: int) -> None:
    if limit <= 0:
        if _THRESHOLD_MEMORY:
            _THRESHOLD_MEMORY.clear()
        return
    if len(_THRESHOLD_MEMORY) <= limit:
        return
    try:
        while len(_THRESHOLD_MEMORY) > limit:
            _THRESHOLD_MEMORY.popitem(last=False)
    except Exception:
        while len(_THRESHOLD_MEMORY) > limit:
            _THRESHOLD_MEMORY.popitem()


def _threshold_memory_lookup(key: Tuple[int, int, int]) -> Optional[int]:
    if _THRESHOLD_MEMORY_LIMIT <= 0:
        return None
    if key in _THRESHOLD_MEMORY:
        try:
            _THRESHOLD_MEMORY.move_to_end(key)
        except Exception:
            pass
        return int(_THRESHOLD_MEMORY[key])
    return None


def _threshold_memory_store(key: Tuple[int, int, int], value: int) -> None:
    if _THRESHOLD_MEMORY_LIMIT <= 0:
        if _THRESHOLD_MEMORY:
            _THRESHOLD_MEMORY.clear()
        return
    try:
        _THRESHOLD_MEMORY[key] = int(value)
        _THRESHOLD_MEMORY.move_to_end(key)
    except Exception:
        _THRESHOLD_MEMORY[key] = int(value)
    _threshold_memory_trim(_THRESHOLD_MEMORY_LIMIT)


def _glyph_signature(arr: "np.ndarray") -> Optional[Tuple[int, int, str]]:
    try:
        if arr.ndim != 2:
            return None
        h, w = arr.shape
        if h <= 0 or w <= 0:
            return None
        digest = hashlib.sha1(arr.tobytes()).hexdigest()
        return (int(h), int(w), digest)
    except Exception:
        return None


def _glyph_runtime_trim(limit: int) -> None:
    if limit <= 0:
        if _GLYPH_RUNTIME_CACHE:
            _GLYPH_RUNTIME_CACHE.clear()
        return
    while len(_GLYPH_RUNTIME_CACHE) > limit:
        _GLYPH_RUNTIME_CACHE.popitem(last=False)


def _glyph_pending_trim(limit: int) -> None:
    if limit <= 0:
        if _GLYPH_RUNTIME_PENDING:
            _GLYPH_RUNTIME_PENDING.clear()
        return
    while len(_GLYPH_RUNTIME_PENDING) > limit:
        _GLYPH_RUNTIME_PENDING.popleft()


def _glyph_runtime_lookup(sig: Optional[Tuple[int, int, str]]) -> Optional[Tuple[str, float]]:
    if sig is None:
        _GLYPH_RUNTIME_STATS["cache_miss"] += 1.0
        return None
    rec = _GLYPH_RUNTIME_CACHE.get(sig)
    if rec is None:
        _GLYPH_RUNTIME_STATS["cache_miss"] += 1.0
        return None
    try:
        _GLYPH_RUNTIME_CACHE.move_to_end(sig)
    except Exception:
        pass
    _GLYPH_RUNTIME_STATS["cache_hit"] += 1.0
    text = rec.get("text")
    conf = rec.get("confidence")
    if isinstance(text, str) and isinstance(conf, (int, float)):
        return text, float(conf)
    return None


def _glyph_runtime_store(sig: Optional[Tuple[int, int, str]], text: str, conf: float) -> None:
    if sig is None:
        return
    if not isinstance(text, str):
        return
    _GLYPH_RUNTIME_CACHE[sig] = {"text": text, "confidence": float(conf)}
    try:
        _GLYPH_RUNTIME_CACHE.move_to_end(sig)
    except Exception:
        pass
    _glyph_runtime_trim(_GLYPH_RUNTIME_CACHE_LIMIT)
    _GLYPH_RUNTIME_STATS["cache_size"] = float(len(_GLYPH_RUNTIME_CACHE))


def _glyph_pending_enqueue(sig: Optional[Tuple[int, int, str]], arr: "np.ndarray", baseline_conf: float) -> None:
    if sig is None:
        return
    try:
        arr_u8 = np.asarray(arr, dtype=np.uint8)
    except Exception:
        return
    if arr_u8.size == 0:
        return
    _GLYPH_RUNTIME_PENDING.append({
        "sig": sig,
        "arr": arr_u8.copy(),
        "confidence": float(baseline_conf),
    })
    _GLYPH_RUNTIME_STATS["pending_records"] = float(len(_GLYPH_RUNTIME_PENDING))
    if len(_GLYPH_RUNTIME_PENDING) > _GLYPH_RUNTIME_PENDING_LIMIT:
        _glyph_pending_trim(_GLYPH_RUNTIME_PENDING_LIMIT)


def _glyph_runtime_replay(limit: int = 8) -> int:
    if not _GLYPH_RUNTIME_PENDING:
        return 0
    processed = 0
    improved = 0
    survivors: "deque[Dict[str, Any]]" = deque()
    while _GLYPH_RUNTIME_PENDING and processed < max(1, limit):
        rec = _GLYPH_RUNTIME_PENDING.popleft()
        processed += 1
        sig = rec.get("sig")
        arr = rec.get("arr")
        baseline_conf = float(rec.get("confidence", 0.0))
        if sig is None or arr is None:
            continue
        try:
            img = Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="L")
        except Exception:
            continue
        ch, conf = _match_glyph(img, _GLYPH_ATLAS)
        if ch and ch != "?" and conf >= baseline_conf + 0.05:
            _glyph_runtime_store(sig, ch, conf)
            improved += 1
        else:
            survivors.append(rec)
    if survivors:
        survivors.extend(_GLYPH_RUNTIME_PENDING)
        _GLYPH_RUNTIME_PENDING.clear()
        _GLYPH_RUNTIME_PENDING.extend(survivors)
    _glyph_pending_trim(_GLYPH_RUNTIME_PENDING_LIMIT)
    _GLYPH_RUNTIME_STATS["replay_attempts"] += float(processed)
    _GLYPH_RUNTIME_STATS["replay_improved"] += float(improved)
    _GLYPH_RUNTIME_STATS["pending_records"] = float(len(_GLYPH_RUNTIME_PENDING))
    return improved

_NGRAM_COUNTS: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
_NGRAM_TOTALS: Dict[str, float] = defaultdict(float)
_NGRAM_DECAY_ALPHA = float(os.environ.get("ZOCR_NGRAM_EMA_ALPHA", "0.05") or 0.0)
try:
    _NGRAM_SURPRISAL_REVIEW_THRESHOLD = float(
        os.environ.get("ZOCR_SURPRISAL_REVIEW_THRESHOLD", "1.65") or 0.0
    )
except Exception:
    _NGRAM_SURPRISAL_REVIEW_THRESHOLD = 0.0
if not (_NGRAM_SURPRISAL_REVIEW_THRESHOLD > 0.0):
    _NGRAM_SURPRISAL_REVIEW_THRESHOLD = 0.0

_GLYPH_RUNTIME_CACHE_LIMIT = int(os.environ.get("ZOCR_GLYPH_CACHE_LIMIT", "384") or 0)
_GLYPH_RUNTIME_PENDING_LIMIT = int(os.environ.get("ZOCR_GLYPH_PENDING_LIMIT", "256") or 0)
_GLYPH_RUNTIME_CACHE: "OrderedDict[Tuple[int, int, str], Dict[str, Any]]" = OrderedDict()
_GLYPH_RUNTIME_PENDING: "deque[Dict[str, Any]]" = deque()
_GLYPH_RUNTIME_STATS: Dict[str, float] = defaultdict(float)


def _blank_recognition_stats() -> Dict[str, Any]:
    return {
        "cells": 0,
        "characters": 0,
        "conf_sum": 0.0,
        "coherence_sum": 0.0,
        "surprisal_sum": 0.0,
        "low_conf_cells": 0,
        "high_surprisal_cells": 0,
        "lexical_quality_sum": 0.0,
        "garbled_cells": 0,
        "examples": [],
    }


_TOY_RECOGNITION_STATS: Dict[str, Any] = _blank_recognition_stats()


def reset_toy_recognition_stats() -> None:
    """Clear per-run toy OCR recognition diagnostics."""

    global _TOY_RECOGNITION_STATS
    _TOY_RECOGNITION_STATS = _blank_recognition_stats()
    _GLYPH_RUNTIME_STATS.clear()
    _GLYPH_RUNTIME_STATS["cache_size"] = float(len(_GLYPH_RUNTIME_CACHE))
    _GLYPH_RUNTIME_STATS["pending_records"] = float(len(_GLYPH_RUNTIME_PENDING))


def _record_toy_recognition(
    text: str,
    conf: float,
    coherence: float,
    surprisal: float,
    lexical_quality: float = 0.0,
) -> None:
    stats = _TOY_RECOGNITION_STATS
    stats["cells"] = int(stats.get("cells", 0) + 1)
    stats["characters"] = int(stats.get("characters", 0) + len(text))
    stats["conf_sum"] = float(stats.get("conf_sum", 0.0) + float(conf))
    stats["coherence_sum"] = float(stats.get("coherence_sum", 0.0) + float(coherence))
    stats["surprisal_sum"] = float(stats.get("surprisal_sum", 0.0) + float(surprisal))
    stats["lexical_quality_sum"] = float(
        stats.get("lexical_quality_sum", 0.0) + float(max(0.0, min(1.5, lexical_quality)))
    )
    if conf < 0.6:
        stats["low_conf_cells"] = int(stats.get("low_conf_cells", 0) + 1)
    if surprisal > 1.6:
        stats["high_surprisal_cells"] = int(stats.get("high_surprisal_cells", 0) + 1)
        examples: List[Dict[str, Any]] = stats.setdefault("examples", [])  # type: ignore[assignment]
        if len(examples) < 12:
            examples.append({
                "text": text,
                "confidence": float(conf),
                "coherence": float(coherence),
                "surprisal": float(surprisal),
                "length": len(text),
            })
    if lexical_quality and lexical_quality < 0.55:
        stats["garbled_cells"] = int(stats.get("garbled_cells", 0) + 1)


def toy_recognition_stats(reset: bool = False) -> Dict[str, Any]:
    """Return aggregate diagnostics gathered during toy OCR recognition."""

    stats = _TOY_RECOGNITION_STATS
    cells = int(stats.get("cells", 0))
    result: Dict[str, Any] = {
        "cells": cells,
        "characters": int(stats.get("characters", 0)),
        "avg_confidence": float(stats.get("conf_sum", 0.0) / cells) if cells else 0.0,
        "avg_coherence": float(stats.get("coherence_sum", 0.0) / cells) if cells else 0.0,
        "avg_surprisal": float(stats.get("surprisal_sum", 0.0) / cells) if cells else 0.0,
        "avg_lexical_quality": float(stats.get("lexical_quality_sum", 0.0) / cells) if cells else 0.0,
        "low_conf_cells": int(stats.get("low_conf_cells", 0)),
        "high_surprisal_cells": int(stats.get("high_surprisal_cells", 0)),
        "garbled_cells": int(stats.get("garbled_cells", 0)),
        "examples": [
            {
                "text": ex.get("text"),
                "confidence": float(ex.get("confidence", 0.0)),
                "coherence": float(ex.get("coherence", 0.0)),
                "surprisal": float(ex.get("surprisal", 0.0)),
                "length": int(ex.get("length", 0)),
            }
            for ex in stats.get("examples", [])[:12]
        ],
    }
    result["runtime_cache_hits"] = int(_GLYPH_RUNTIME_STATS.get("cache_hit", 0))
    result["runtime_cache_misses"] = int(_GLYPH_RUNTIME_STATS.get("cache_miss", 0))
    result["runtime_cache_size"] = int(len(_GLYPH_RUNTIME_CACHE))
    result["runtime_pending"] = int(len(_GLYPH_RUNTIME_PENDING))
    result["runtime_replay_attempts"] = int(_GLYPH_RUNTIME_STATS.get("replay_attempts", 0))
    result["runtime_replay_improved"] = int(_GLYPH_RUNTIME_STATS.get("replay_improved", 0))
    result["learned_variants"] = int(_GLYPH_RUNTIME_STATS.get("learned_variants", 0))
    result["lexical_penalties"] = int(_GLYPH_RUNTIME_STATS.get("lexical_penalty", 0))
    result["tess_dictionary_boosts"] = int(_GLYPH_RUNTIME_STATS.get("tess_dictionary_boosts", 0))
    result["tess_unknown_hits"] = int(_GLYPH_RUNTIME_STATS.get("tess_unknown_hits", 0))
    result["tess_bigram_penalties"] = int(_GLYPH_RUNTIME_STATS.get("tess_bigram_penalties", 0))
    result["baseline_splits"] = int(_GLYPH_RUNTIME_STATS.get("baseline_splits", 0))
    result["template_matches"] = int(_GLYPH_RUNTIME_STATS.get("template_matches", 0))
    result["template_observed"] = int(_GLYPH_RUNTIME_STATS.get("template_observed", 0))
    result["template_cache_hits"] = int(_GLYPH_RUNTIME_STATS.get("template_cache_hits", 0))
    result["template_cache_misses"] = int(_GLYPH_RUNTIME_STATS.get("template_cache_misses", 0))
    result["template_cache_variants"] = int(_GLYPH_RUNTIME_STATS.get("template_cache_variants", 0))
    result["template_cache_saved"] = int(_GLYPH_RUNTIME_STATS.get("template_cache_saved", 0))
    result["template_cache_loaded"] = int(_GLYPH_RUNTIME_STATS.get("template_cache_loaded", 0))
    result["template_cache_errors"] = int(_GLYPH_RUNTIME_STATS.get("template_cache_errors", 0))
    result["template_override_conf"] = int(_GLYPH_RUNTIME_STATS.get("template_override_conf", 0))
    result["template_override_quality"] = int(_GLYPH_RUNTIME_STATS.get("template_override_quality", 0))
    result["template_override_missing"] = int(_GLYPH_RUNTIME_STATS.get("template_override_missing", 0))
    if reset:
        reset_toy_recognition_stats()
    return result
_AMBIGUOUS_CHAR_MAP: Dict[str, Tuple[str, ...]] = {
    "0": ("O", "D"),
    "O": ("0",),
    "1": ("I", "l"),
    "I": ("1", "l"),
    "l": ("1", "I"),
    "5": ("S",),
    "S": ("5",),
    "2": ("Z",),
    "Z": ("2",),
    "8": ("B",),
    "B": ("8",),
    "6": ("G",),
    "G": ("6",),
    "7": ("T",),
    "T": ("7",),
}

def _compute_glyph_features_from_array(arr: "np.ndarray") -> Dict[str, float]:
    arr_f = np.asarray(arr, dtype=np.float32)
    if arr_f.size == 0:
        return {"aspect": 1.0, "density": 0.0, "symmetry": 0.0, "style_var": 0.0, "count": 0}
    if arr_f.max() > 1.5:
        arr_f = arr_f / 255.0
    h, w = arr_f.shape
    aspect = float(w) / float(h or 1)
    density = float(arr_f.mean())
    flipped = np.flip(arr_f, axis=1) if arr_f.ndim == 2 else arr_f
    symmetry = 1.0 - float(np.mean(np.abs(arr_f - flipped))) if arr_f.size else 0.0
    if arr_f.ndim == 2:
        row_profile = arr_f.mean(axis=1)
        col_profile = arr_f.mean(axis=0)
        style_var = float(np.var(row_profile) + np.var(col_profile))
    else:
        style_var = 0.0
    return {"aspect": aspect, "density": density, "symmetry": symmetry, "style_var": style_var, "count": 1}

def _render_glyphs(font=None, size=16):
    # PIL's default bitmap font via ImageFont.load_default() matches our demo
    f = ImageFont.load_default() if font is None else font
    atlas = {}
    feats = {}
    for ch in _ASCII_SET:
        # Render on tight canvas
        img = Image.new("L", (size*2, size*2), 0)
        dr = ImageDraw.Draw(img)
        dr.text((2,2), ch, fill=255, font=f)
        # crop to bbox
        bbox = img.getbbox() or (0,0,1,1)
        crop = img.crop(bbox)
        atlas[ch] = [crop]
        feats[ch] = _compute_glyph_features_from_array(np.asarray(crop, dtype=np.float32))
    return atlas, feats

_GLYPH_ATLAS, _GLYPH_FEATS = _render_glyphs()


def _toy_memory_snapshot_internal() -> Dict[str, Any]:
    glyph_chars = 0
    glyph_variants = 0
    for variants in _GLYPH_ATLAS.values():
        if variants:
            glyph_chars += 1
            glyph_variants += len(variants)
    ngram_contexts = 0
    ngram_transitions = 0
    for mapping in _NGRAM_COUNTS.values():
        if mapping:
            ngram_contexts += 1
            ngram_transitions += len(mapping)
    ngram_observations = float(sum(_NGRAM_TOTALS.values()))
    snapshot: Dict[str, Any] = {
        "glyph_chars": int(glyph_chars),
        "glyph_variants": int(glyph_variants),
        "avg_variants_per_char": float(glyph_variants / glyph_chars) if glyph_chars else 0.0,
        "ngram_contexts": int(ngram_contexts),
        "ngram_transitions": int(ngram_transitions),
        "ngram_observations": ngram_observations,
        "avg_ngram_branching": float(ngram_transitions / ngram_contexts) if ngram_contexts else 0.0,
        "runtime_cache": int(len(_GLYPH_RUNTIME_CACHE)),
        "runtime_pending": int(len(_GLYPH_RUNTIME_PENDING)),
    }
    return snapshot


def toy_memory_snapshot() -> Dict[str, Any]:
    """Return aggregate statistics describing the current toy OCR memory."""

    return dict(_toy_memory_snapshot_internal())


def toy_memory_delta(before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if before is None:
        before = {}
    if after is None:
        after = {}
    delta: Dict[str, float] = {}
    keys = set(k for k, v in after.items() if isinstance(v, (int, float))) | set(
        k for k, v in before.items() if isinstance(v, (int, float))
    )
    for key in sorted(keys):
        a = after.get(key)
        b = before.get(key)
        if isinstance(a, (int, float)) or isinstance(b, (int, float)):
            aval = float(a) if isinstance(a, (int, float)) else 0.0
            bval = float(b) if isinstance(b, (int, float)) else 0.0
            delta[key] = aval - bval
    return delta


def _toy_memory_payload(limit_ngram: int = 48) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "version": 1,
        "glyph_atlas": {},
        "glyph_feats": {},
        "ngram_counts": {},
        "ngram_totals": {},
    }
    for ch, imgs in _GLYPH_ATLAS.items():
        serialised: List[List[List[int]]] = []
        for img in imgs[:_GLYPH_VARIANT_LIMIT]:
            try:
                arr = np.asarray(img, dtype=np.uint8)
                serialised.append(arr.tolist())
            except Exception:
                continue
        if serialised:
            payload["glyph_atlas"][ch] = serialised
    for ch, feats in _GLYPH_FEATS.items():
        if not isinstance(feats, dict):
            continue
        safe = {
            "aspect": float(feats.get("aspect", 1.0)),
            "density": float(feats.get("density", 0.0)),
            "symmetry": float(feats.get("symmetry", 0.0)),
            "style_var": float(feats.get("style_var", 0.0)),
            "count": int(feats.get("count", 1)),
        }
        payload["glyph_feats"][ch] = safe
    for prev, mapping in _NGRAM_COUNTS.items():
        if not mapping:
            continue
        items = sorted(mapping.items(), key=lambda kv: kv[1], reverse=True)
        trimmed = items[:limit_ngram]
        payload["ngram_counts"][prev] = {ch: float(count) for ch, count in trimmed if count}
    payload["ngram_totals"] = {prev: float(total) for prev, total in _NGRAM_TOTALS.items() if total}
    return payload


def _write_toy_memory_series(path: str, payload: Dict[str, Any], snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    version_dir, summary_path = _toy_memory_series_paths(path)
    try:
        ensure_dir(version_dir)
    except Exception as exc:
        return {"error": f"series_dir_failed: {type(exc).__name__}: {exc}"}
    summary_payload: Dict[str, Any] = {}
    try:
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as fr:
                loaded = json.load(fr)
                if isinstance(loaded, dict):
                    summary_payload = loaded
    except Exception as exc:
        summary_payload = {"error": f"summary_parse_failed: {type(exc).__name__}: {exc}"}
    latest_epoch = int(summary_payload.get("latest_epoch") or 0)
    epoch = latest_epoch + 1
    epoch_name = f"epoch_{epoch:06d}.json"
    epoch_path = os.path.join(version_dir, epoch_name)
    try:
        with open(epoch_path, "w", encoding="utf-8") as fw:
            json.dump(payload, fw, ensure_ascii=False)
    except Exception as exc:
        return {"error": f"epoch_write_failed: {type(exc).__name__}: {exc}"}
    prev_snapshot = summary_payload.get("latest_snapshot") if isinstance(summary_payload, dict) else None
    delta_prev = toy_memory_delta(prev_snapshot, snapshot)
    recognition = toy_recognition_stats(reset=False)
    timestamp = time.time()
    entry = {
        "epoch": epoch,
        "path": epoch_name,
        "written_at": timestamp,
        "snapshot": snapshot,
        "delta_prev": delta_prev,
        "recognition": recognition,
    }
    history: List[Dict[str, Any]] = []
    if isinstance(summary_payload, dict):
        hist = summary_payload.get("history")
        if isinstance(hist, list):
            history.extend(hist)
    history.append(entry)
    history_limit = 48
    if len(history) > history_limit:
        history = history[-history_limit:]
    stats = _toy_memory_series_stats(history)
    summary_payload = {
        "latest_epoch": epoch,
        "latest_path": epoch_name,
        "updated_at": timestamp,
        "latest_snapshot": snapshot,
        "history": history,
        "stats": stats,
        "tail": _series_tail(history),
    }
    try:
        with open(summary_path, "w", encoding="utf-8") as fw:
            json.dump(summary_payload, fw, ensure_ascii=False, indent=2)
    except Exception as exc:
        return {
            "epoch": epoch,
            "path": epoch_path,
            "delta_prev": delta_prev,
            "recognition": recognition,
            "error": f"summary_write_failed: {type(exc).__name__}: {exc}",
        }
    return {
        "epoch": epoch,
        "path": epoch_path,
        "delta_prev": delta_prev,
        "recognition": recognition,
        "summary": summary_path,
        "stats": stats,
        "tail": _series_tail(history),
    }


def save_toy_memory(path: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": path,
        "saved": False,
        "snapshot": toy_memory_snapshot(),
    }
    if not path:
        return info
    try:
        payload = _toy_memory_payload()
        ensure_dir(os.path.dirname(path) or ".")
        with open(path, "w", encoding="utf-8") as fw:
            json.dump(payload, fw, ensure_ascii=False)
        info["saved"] = True
        try:
            info["bytes"] = os.path.getsize(path)
        except Exception:
            pass
        series_meta = _write_toy_memory_series(path, payload, info["snapshot"])
        if series_meta:
            info["series"] = series_meta
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    info["snapshot"] = toy_memory_snapshot()
    return info


def load_toy_memory(path: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": path,
        "loaded": False,
        "changed": False,
        "snapshot_before": toy_memory_snapshot(),
    }
    payload: Optional[Dict[str, Any]] = None
    source_path: Optional[str] = None
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fr:
                payload = json.load(fr)
            source_path = path
        except Exception as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
            payload = None
    if payload is None and path:
        series_payload, series_meta = _load_toy_memory_series_payload(path)
        if series_meta:
            info.setdefault("series", series_meta)
        if series_payload is not None:
            payload = series_payload
            source_path = series_meta.get("epoch_path") if isinstance(series_meta, dict) else None
    if payload is None:
        return info
    changed = False
    try:
        glyph_payload = payload.get("glyph_atlas", {})
        if isinstance(glyph_payload, dict):
            for ch, arrs in glyph_payload.items():
                if not isinstance(arrs, list):
                    continue
                atlas_list: List[Image.Image] = []
                for arr in arrs[:_GLYPH_VARIANT_LIMIT]:
                    try:
                        np_arr = np.asarray(arr, dtype=np.uint8)
                        if np_arr.ndim != 2:
                            continue
                        atlas_list.append(Image.fromarray(np_arr, mode="L"))
                    except Exception:
                        continue
                if atlas_list:
                    _GLYPH_ATLAS[ch] = atlas_list
                    changed = True
        feats_payload = payload.get("glyph_feats", {})
        if isinstance(feats_payload, dict):
            for ch, feats in feats_payload.items():
                if not isinstance(feats, dict):
                    continue
                safe = {
                    "aspect": float(feats.get("aspect", 1.0)),
                    "density": float(feats.get("density", 0.0)),
                    "symmetry": float(feats.get("symmetry", 0.0)),
                    "style_var": float(feats.get("style_var", 0.0)),
                    "count": int(feats.get("count", len(_GLYPH_ATLAS.get(ch, [])) or 1)),
                }
                _GLYPH_FEATS[ch] = safe
        counts_payload = payload.get("ngram_counts", {})
        if isinstance(counts_payload, dict):
            for prev, mapping in counts_payload.items():
                if not isinstance(mapping, dict):
                    continue
                target = _NGRAM_COUNTS.setdefault(prev, defaultdict(float))
                for ch, val in mapping.items():
                    try:
                        new_val = float(val)
                    except Exception:
                        continue
                    if new_val <= 0:
                        continue
                    target[ch] = max(target.get(ch, 0.0), new_val)
                    changed = True
        totals_payload = payload.get("ngram_totals", {})
        if isinstance(totals_payload, dict):
            for prev, total in totals_payload.items():
                try:
                    current = float(_NGRAM_TOTALS.get(prev, 0.0))
                    new_total = float(total)
                    if new_total > current:
                        _NGRAM_TOTALS[prev] = new_total
                        changed = True
                except Exception:
                    continue
        info["loaded"] = True
        info["changed"] = bool(changed)
        info["snapshot_after"] = toy_memory_snapshot()
        info["delta"] = toy_memory_delta(info.get("snapshot_before"), info.get("snapshot_after"))
        if source_path:
            info["source_path"] = source_path
            if source_path != path:
                try:
                    series_info = info.setdefault("series", {})
                    series_info["epoch_bytes"] = os.path.getsize(source_path)
                except Exception:
                    pass
        try:
            info["bytes"] = os.path.getsize(path)
        except Exception:
            if source_path and os.path.exists(source_path):
                try:
                    info["bytes"] = os.path.getsize(source_path)
                except Exception:
                    pass
        return info
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["snapshot_after"] = toy_memory_snapshot()
        info["delta"] = toy_memory_delta(info.get("snapshot_before"), info.get("snapshot_after"))
        return info

def _blend_glyph_features(ch: str, feats: Dict[str, float]) -> None:
    if not feats:
        return
    cur = _GLYPH_FEATS.setdefault(
        ch,
        {
            "aspect": feats.get("aspect", 1.0),
            "density": feats.get("density", 0.0),
            "symmetry": feats.get("symmetry", 0.0),
            "style_var": feats.get("style_var", 0.0),
            "count": feats.get("count", 1) or 1,
        },
    )
    count = max(1, int(cur.get("count", 1)))
    new_count = min(_GLYPH_VARIANT_LIMIT, count + 1)
    alpha = 1.0 / float(min(count + 1, _GLYPH_VARIANT_LIMIT))
    for key in ("aspect", "density", "symmetry", "style_var"):
        current_val = cur.get(key, feats.get(key, 0.0))
        target_val = feats.get(key, current_val)
        cur[key] = current_val + (target_val - current_val) * alpha
    cur["count"] = new_count

def _adapt_glyph(ch: str, img: "Image.Image") -> None:
    if not ch or img is None:
        return
    try:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
    except Exception:
        return
    if arr.size == 0:
        return
    atlas_list = _GLYPH_ATLAS.setdefault(ch, [])
    # avoid duplicates
    for tpl in atlas_list:
        try:
            if np.array_equal(np.asarray(tpl, dtype=np.uint8), arr):
                return
        except Exception:
            continue
    atlas_list.append(Image.fromarray(arr, mode="L"))
    if len(atlas_list) > _GLYPH_VARIANT_LIMIT:
        atlas_list.pop(0)
    feats = _compute_glyph_features_from_array(arr)
    _blend_glyph_features(ch, feats)
    _GLYPH_RUNTIME_STATS["learned_variants"] += 1.0
    if _GLYPH_RUNTIME_PENDING:
        _glyph_runtime_replay()

def _generate_contextual_variants(text: str) -> Set[str]:
    variants: Set[str] = set()
    if not text:
        return variants
    chars = list(text)
    L = len(chars)
    for i in range(L):
        alts = _AMBIGUOUS_CHAR_MAP.get(chars[i])
        if not alts:
            continue
        for alt in alts:
            variant = chars[:]
            variant[i] = alt
            variants.add("".join(variant))
    # second pass for two-position swaps (limited to keep combinatorics small)
    limited_positions = [i for i, ch in enumerate(chars) if _AMBIGUOUS_CHAR_MAP.get(ch)]
    if len(limited_positions) >= 2:
        for i in limited_positions[:3]:
            for j in limited_positions[:3]:
                if j <= i:
                    continue
                for alt_i in _AMBIGUOUS_CHAR_MAP.get(chars[i], ()):  # type: ignore[arg-type]
                    for alt_j in _AMBIGUOUS_CHAR_MAP.get(chars[j], ()):  # type: ignore[arg-type]
                        variant = chars[:]
                        variant[i] = alt_i
                        variant[j] = alt_j
                        variants.add("".join(variant))
    return variants

def _normalize_digit_like(text: str) -> str:
    repl = text.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
    repl = repl.replace("S", "5").replace("s", "5").replace("B", "8").replace("b", "6")
    repl = repl.replace("G", "6").replace("Z", "2")
    return repl

def _looks_like_numeric_token(text: str) -> bool:
    return bool(re.fullmatch(r"[0-9OIlSsbB]{2,}", text or ""))

def _looks_like_upper_token(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9]{3,}", text or ""))


_TOY_ALLOWED_SYMBOLS = set("-_.:/%$¥,+#&()[]{}\\")

@dataclass
class _TessLiteModel:
    glyphs: Set[str]
    dictionary: Set[str]
    bigrams: Dict[str, Dict[str, float]]
    ambiguous: Dict[str, Set[str]]
    char_categories: Dict[str, str]
    source_signature: str


_TESSLITE_MODEL: Optional[_TessLiteModel] = None
_TESSLITE_MODEL_SIG: Optional[str] = None


def _tesslite_env_signature() -> str:
    paths = [
        os.environ.get("ZOCR_TESS_UNICHARSET", ""),
        os.environ.get("ZOCR_TESS_WORDLIST", ""),
        os.environ.get("ZOCR_TESS_BIGRAM_JSON", ""),
    ]
    stats: List[str] = []
    for path in paths:
        if not path:
            stats.append("-")
            continue
        try:
            st = os.stat(path)
            stats.append(f"{path}:{int(st.st_mtime)}:{st.st_size}")
        except Exception:
            stats.append(f"{path}:missing")
    return "|".join(stats)


def _decode_unichar_token(token: str) -> str:
    tok = token.strip().strip('"')
    if not tok:
        return ""
    if tok == "NULL":
        return ""
    if tok.startswith("\\"):
        try:
            return tok.encode("utf-8").decode("unicode_escape")
        except Exception:
            pass
    return tok


def _load_unicharset(path: str) -> Tuple[Set[str], Dict[str, Set[str]], Dict[str, str]]:
    glyphs: Set[str] = set()
    ambiguous: Dict[str, Set[str]] = defaultdict(set)
    categories: Dict[str, str] = {}
    if not path:
        return glyphs, ambiguous, categories
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                parts = raw.split()
                if not parts:
                    continue
                ch = _decode_unichar_token(parts[0])
                if not ch:
                    continue
                glyphs.add(ch)
                if len(parts) >= 2:
                    categories[ch] = parts[1]
                if len(parts) >= 4 and parts[3] not in {"0", "NULL"}:
                    ambiguous[ch].add(parts[3])
    except Exception:
        return set(), defaultdict(set), {}
    return glyphs, ambiguous, categories


def _load_wordlist(path: str) -> Set[str]:
    words: Set[str] = set()
    if not path:
        return words
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                token = line.strip()
                if not token or token.startswith("#"):
                    continue
                words.add(token)
    except Exception:
        return set()
    return words


def _load_bigram_json(path: str) -> Dict[str, Dict[str, float]]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    bigrams: Dict[str, Dict[str, float]] = {}
    if isinstance(payload, dict):
        for prev, mapping in payload.items():
            if not isinstance(mapping, dict):
                continue
            table: Dict[str, float] = {}
            for ch, weight in mapping.items():
                try:
                    table[str(ch)] = float(weight)
                except Exception:
                    continue
            if table:
                bigrams[str(prev)] = table
    return bigrams


def _build_bigrams_from_words(words: Set[str]) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    totals: Dict[str, float] = defaultdict(float)
    for word in words:
        prev = "\0"
        for ch in word:
            counts[prev][ch] += 1.0
            totals[prev] += 1.0
            prev = ch
        counts[prev]["\0"] += 1.0
        totals[prev] += 1.0
    table: Dict[str, Dict[str, float]] = {}
    for prev, mapping in counts.items():
        total = max(1.0, totals.get(prev, 1.0))
        scale: Dict[str, float] = {}
        for ch, cnt in mapping.items():
            scale[ch] = float(cnt / total)
        if scale:
            table[prev] = scale
    return table


def _get_tesslite_model() -> Optional[_TessLiteModel]:
    global _TESSLITE_MODEL, _TESSLITE_MODEL_SIG
    signature = _tesslite_env_signature()
    if not signature.strip("|-"):
        _TESSLITE_MODEL = None
        _TESSLITE_MODEL_SIG = None
        return None
    if _TESSLITE_MODEL is not None and signature == _TESSLITE_MODEL_SIG:
        return _TESSLITE_MODEL
    unichar_path = os.environ.get("ZOCR_TESS_UNICHARSET", "")
    word_path = os.environ.get("ZOCR_TESS_WORDLIST", "")
    bigram_path = os.environ.get("ZOCR_TESS_BIGRAM_JSON", "")
    glyphs, ambiguous, categories = _load_unicharset(unichar_path)
    dictionary = _load_wordlist(word_path)
    bigrams = _load_bigram_json(bigram_path)
    if not bigrams and dictionary:
        bigrams = _build_bigrams_from_words(dictionary)
    model = _TessLiteModel(
        glyphs=glyphs,
        dictionary=dictionary,
        bigrams=bigrams,
        ambiguous=ambiguous,
        char_categories=categories,
        source_signature=signature,
    )
    _TESSLITE_MODEL = model
    _TESSLITE_MODEL_SIG = signature
    return model


def _tesslite_unknown_ratio(model: _TessLiteModel, text: str) -> float:
    if not model.glyphs:
        return 0.0
    total = 0
    missing = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if ch not in model.glyphs:
            missing += 1
    if total == 0:
        return 0.0
    return float(missing) / float(total)


def _tesslite_dictionary_boost(model: _TessLiteModel, text: str) -> float:
    token = text.strip()
    if not token:
        return 0.0
    if token in model.dictionary:
        return 0.15
    lower = token.lower()
    if lower in model.dictionary:
        return 0.1
    return 0.0


def _tesslite_bigram_gate(model: _TessLiteModel, text: str) -> float:
    if not model.bigrams or len(text) < 2:
        return 1.0
    penalties = 0
    transitions = 0
    prev = "\0"
    for ch in text:
        table = model.bigrams.get(prev)
        score = table.get(ch, 0.0) if table else 0.0
        if table is not None:
            transitions += 1
            if score < 0.02:
                penalties += 1
        prev = ch
    if transitions == 0:
        return 1.0
    ratio = penalties / float(transitions)
    return float(max(0.4, 1.0 - 0.6 * ratio))

_TOY_CHAR_CATEGORY_MAP = {
    "digit": set("0123456789０１２３４５６７８９"),
    "latin_upper": set("ABCDEFGHIJKLMNOPQRSTUVWXYZＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"),
    "latin_lower": set("abcdefghijklmnopqrstuvwxyzａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"),
    "kana": set(),
    "symbol": _TOY_ALLOWED_SYMBOLS | set({'+', '−', 'ー', '―', '~'}),
}
_TOY_TRANSITION_PRIORS = {
    "digit": {"digit": 1.0, "symbol": 0.95, "latin_upper": 0.7, "latin_lower": 0.7, "kana": 0.65, "kanji": 0.65},
    "latin_upper": {"latin_upper": 1.0, "latin_lower": 0.95, "digit": 0.8, "symbol": 0.75, "kana": 0.55, "kanji": 0.55},
    "latin_lower": {"latin_lower": 1.0, "latin_upper": 0.9, "digit": 0.7, "symbol": 0.75, "kana": 0.5, "kanji": 0.5},
    "kanji": {"kanji": 1.0, "kana": 0.9, "digit": 0.65, "symbol": 0.7, "latin_upper": 0.55, "latin_lower": 0.55},
    "kana": {"kana": 1.0, "kanji": 0.92, "digit": 0.68, "symbol": 0.72, "latin_upper": 0.58, "latin_lower": 0.58},
}

def _toy_char_category(ch: str) -> str:
    if not ch:
        return "other"
    code = ord(ch)
    if ch in _TOY_CHAR_CATEGORY_MAP["digit"] or 0xFF10 <= code <= 0xFF19:
        return "digit"
    if ch in _TOY_CHAR_CATEGORY_MAP["latin_upper"]:
        return "latin_upper"
    if ch in _TOY_CHAR_CATEGORY_MAP["latin_lower"]:
        return "latin_lower"
    if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF or 0xFF66 <= code <= 0xFF9D:
        return "kana"
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return "kanji"
    if ch in _TOY_CHAR_CATEGORY_MAP["symbol"]:
        return "symbol"
    if ch.isspace():
        return "space"
    return "other"

def _toy_transition_gate(text: str) -> Tuple[float, str]:
    prev_cat: Optional[str] = None
    penalty = 0
    transitions = 0
    for ch in text:
        if ch.isspace():
            continue
        cat = _toy_char_category(ch)
        if prev_cat is None:
            prev_cat = cat
            continue
        transitions += 1
        priors = _TOY_TRANSITION_PRIORS.get(prev_cat)
        score = priors.get(cat, 0.5) if priors else 0.5
        if score < 0.6:
            penalty += 1
        prev_cat = cat
    if transitions == 0:
        return 1.0, ""
    ratio = penalty / float(transitions)
    gate = max(0.55, 1.0 - 0.45 * ratio)
    reason = f"penalty:{penalty}/{transitions}"
    return float(gate), reason
_TOY_NOISE_CHARS = set("?`~^|'\"¤")
_TOY_GARBLED_PATTERN = re.compile(r"[?]{2,}|[|\\/_]{3,}|={2,}|\bfax\b", re.IGNORECASE)
_TOY_NUMERIC_TOKEN_RE = re.compile(r"^[+-]?(?:[¥$]\s*)?[0-9][0-9,]*(?:\.[0-9]+)?%?$")
_TOY_CANONICAL_TOKEN_HINTS: Dict[str, float] = {
    "item": 0.35,
    "items": 0.28,
    "qty": 0.30,
    "quantity": 0.28,
    "unitprice": 0.32,
    "price": 0.22,
    "amount": 0.35,
    "total": 0.30,
    "subtotal": 0.22,
    "tax": 0.18,
    "taxtotal": 0.2,
    "estimate": 0.2,
    "invoice": 0.18,
    "description": 0.2,
    "unit": 0.15,
    "remarks": 0.18,
    "date": 0.18,
    "duedate": 0.24,
    "servicedate": 0.24,
    "shipdate": 0.18,
    "payment": 0.2,
    "discount": 0.2,
    "project": 0.18,
    "code": 0.15,
    "client": 0.2,
}

_LEXICAL_CONF_CANONICAL_HINTS: Dict[str, str] = {
    "item": "item",
    "items": "item",
    "description": "item",
    "qty": "qty",
    "quantity": "qty",
    "unit": "qty",
    "unitprice": "unit_price",
    "price": "unit_price",
    "amount": "amount",
    "subtotal": "subtotal",
    "total": "total",
    "taxtotal": "tax",
    "tax": "tax",
}

_LEXICAL_CONF_JP_HINTS: Dict[str, str] = {
    "見積金額": "amount",
    "御見積金額": "amount",
    "数量": "qty",
    "数量合計": "qty",
    "単価": "unit_price",
    "金額": "amount",
    "小計": "subtotal",
    "合計": "total",
    "総計": "total",
    "消費税": "tax",
    "税込": "total",
    "税抜": "subtotal",
}

_JP_TOKEN_STRIP_RE = re.compile(r"[\s\u3000・／/\\\-＿ー－〜~、。\.\[\](){}<>：:；;]+")
_JP_HONORIFIC_PREFIXES = ("御", "お", "ご")
_JP_JIS_MARKERS = set("様殿各位行宛先｣『』【】")
_JP_TOKEN_SUFFIX_HINTS: Dict[str, float] = {
    "金額": 0.22,
    "見積金額": 0.3,
    "御見積金額": 0.3,
    "御見積書": 0.32,
    "見積書": 0.32,
    "合計": 0.26,
    "小計": 0.22,
    "数量": 0.22,
    "単価": 0.22,
    "金額(税込)": 0.25,
    "金額(税抜)": 0.24,
    "税抜金額": 0.24,
    "税込金額": 0.25,
    "税額": 0.22,
    "消費税": 0.24,
    "納期": 0.2,
    "有効期限": 0.22,
    "見積日": 0.22,
    "発行日": 0.2,
    "支払期日": 0.22,
}
_TOY_JP_TOKEN_HINTS: Dict[str, float] = {
    "見積書": 0.35,
    "御見積書": 0.35,
    "見積金額": 0.34,
    "御見積金額": 0.34,
    "金額": 0.26,
    "合計": 0.3,
    "総計": 0.28,
    "小計": 0.25,
    "消費税": 0.25,
    "税込": 0.2,
    "税抜": 0.2,
    "税率": 0.18,
    "数量": 0.25,
    "品名": 0.22,
    "単価": 0.25,
    "金額(税込)": 0.28,
    "金額(税抜)": 0.28,
    "単価(税込)": 0.23,
    "単価(税抜)": 0.23,
    "摘要": 0.22,
    "備考": 0.18,
    "見積日": 0.23,
    "有効期限": 0.24,
    "納期": 0.23,
    "発行日": 0.2,
    "支払期日": 0.23,
    "請求日": 0.2,
    "お支払期限": 0.24,
}


def _canonicalize_toy_token(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", (text or "").lower())


def _japanese_script_profile(text: str) -> Dict[str, float]:
    counts = {
        "kanji": 0,
        "hiragana": 0,
        "katakana": 0,
        "latin": 0,
        "digit": 0,
        "other": 0,
    }
    total = len(text or "")
    for ch in text or "":
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            counts["kanji"] += 1
        elif 0x3040 <= code <= 0x309F:
            counts["hiragana"] += 1
        elif 0x30A0 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF or 0xFF66 <= code <= 0xFF9D:
            counts["katakana"] += 1
        elif 0x0030 <= code <= 0x0039 or 0xFF10 <= code <= 0xFF19:
            counts["digit"] += 1
        elif 0x0041 <= code <= 0x005A or 0x0061 <= code <= 0x007A or 0xFF21 <= code <= 0xFF3A or 0xFF41 <= code <= 0xFF5A:
            counts["latin"] += 1
        else:
            counts["other"] += 1
    jp_chars = counts["kanji"] + counts["hiragana"] + counts["katakana"]
    profile: Dict[str, float] = {k: float(v) for k, v in counts.items()}
    profile["jp_chars"] = float(jp_chars)
    profile["total_chars"] = float(total)
    profile["jp_ratio"] = float(jp_chars) / float(total or 1)
    profile["kanji_ratio"] = float(counts["kanji"]) / float(total or 1)
    profile["kana_ratio"] = float(counts["hiragana"] + counts["katakana"]) / float(total or 1)
    profile["latin_ratio"] = float(counts["latin"]) / float(total or 1)
    return profile


def _normalize_japanese_token(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = _JP_TOKEN_STRIP_RE.sub("", normalized)
    for prefix in _JP_HONORIFIC_PREFIXES:
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            normalized = normalized[len(prefix) :]
    if normalized and normalized[0] in _JP_JIS_MARKERS:
        normalized = normalized[1:]
    return normalized


def _japanese_token_hint(text: str, profile: Dict[str, float]) -> Tuple[float, str]:
    normalized = _normalize_japanese_token(text)
    if not normalized:
        return 0.0, ""
    boost = _TOY_JP_TOKEN_HINTS.get(normalized, 0.0)
    reason = "dict" if boost else ""
    if not boost:
        for suffix, bonus in _JP_TOKEN_SUFFIX_HINTS.items():
            if normalized.endswith(suffix) and len(normalized) >= len(suffix) + 1:
                boost = max(boost, bonus)
                reason = f"suffix:{suffix}"
                break
    if not boost and profile.get("kana_ratio", 0.0) > 0.6 and profile.get("kanji_ratio", 0.0) < 0.1:
        boost = 0.05
        reason = "kana_balance"
    elif not boost and profile.get("kanji_ratio", 0.0) >= 0.45:
        boost = 0.05
        reason = "kanji_balance"
    return float(min(0.5, boost)), reason


def _toy_text_quality(text: str) -> Tuple[float, Dict[str, float]]:
    normalized = (text or "").strip()
    if not normalized:
        return 0.0, {"reason": "empty"}
    length = len(normalized)
    allowed = 0
    ascii_like = 0
    weird = 0
    jp_profile = _japanese_script_profile(normalized)
    for ch in normalized:
        if ord(ch) < 128:
            ascii_like += 1
        if ch.isalnum() or ch in _TOY_ALLOWED_SYMBOLS:
            allowed += 1
        elif ch in _TOY_NOISE_CHARS or ord(ch) >= 0x2500:
            weird += 1
    allowed_ratio = float(allowed) / float(length or 1)
    weird_ratio = float(weird) / float(length or 1)
    base = 0.4 + 0.6 * allowed_ratio
    ascii_penalty = ascii_like < max(1, length - 1) and not _TOY_NUMERIC_TOKEN_RE.match(normalized)
    if ascii_penalty and jp_profile.get("jp_ratio", 0.0) > 0.35:
        ascii_penalty = False
    if ascii_penalty:
        base *= 0.9
    if weird_ratio > 0:
        base *= max(0.35, 1.0 - 0.45 * weird_ratio)
    canonical = _canonicalize_toy_token(normalized)
    lex_boost = 1.0
    if canonical:
        lex_boost += _TOY_CANONICAL_TOKEN_HINTS.get(canonical, 0.0)
    if _TOY_NUMERIC_TOKEN_RE.match(normalized):
        lex_boost = max(lex_boost, 1.05)
    elif len(canonical) <= 1 and canonical not in {"x", "y", "z", "a"}:
        base *= 0.9
    jp_boost, jp_reason = _japanese_token_hint(normalized, jp_profile)
    if jp_boost:
        lex_boost += jp_boost
    transition_gate, transition_reason = _toy_transition_gate(normalized)
    if transition_gate < 1.0:
        base *= transition_gate
    tess_model = _get_tesslite_model()
    tess_unknown = 0.0
    tess_gate = 1.0
    tess_dict_boost = 0.0
    if tess_model:
        tess_unknown = _tesslite_unknown_ratio(tess_model, normalized)
        if tess_unknown > 0:
            base *= max(0.35, 1.0 - 0.55 * tess_unknown)
            _GLYPH_RUNTIME_STATS["tess_unknown_hits"] += 1.0
        tess_gate = _tesslite_bigram_gate(tess_model, normalized)
        if tess_gate < 1.0:
            base *= tess_gate
            _GLYPH_RUNTIME_STATS["tess_bigram_penalties"] += 1.0
        tess_dict_boost = _tesslite_dictionary_boost(tess_model, normalized)
        if tess_dict_boost > 0:
            lex_boost += tess_dict_boost
            _GLYPH_RUNTIME_STATS["tess_dictionary_boosts"] += 1.0
    if _TOY_GARBLED_PATTERN.search(normalized):
        base *= 0.6
    quality = float(max(0.2, min(1.6, base * lex_boost)))
    diag = {
        "canonical": canonical,
        "allowed_ratio": allowed_ratio,
        "weird_ratio": weird_ratio,
        "lex_boost": lex_boost,
        "jp_ratio": jp_profile.get("jp_ratio", 0.0),
        "jp_hint": jp_boost,
        "jp_hint_reason": jp_reason,
        "transition_gate": transition_gate,
        "transition_reason": transition_reason,
        "tess_unknown_ratio": tess_unknown,
        "tess_bigram_gate": tess_gate,
        "tess_dict_boost": tess_dict_boost,
    }
    return quality, diag

def _ngram_probability(prev: str, ch: str) -> float:
    counts = _NGRAM_COUNTS.get(prev)
    totals = _NGRAM_TOTALS.get(prev, 0)
    if counts and totals:
        vocab = max(len(counts), 8)
        denom = float(totals + vocab)
        return float((counts.get(ch, 0) + 1.0) / denom)
    if counts:
        vocab = max(len(counts), 8)
        return float((counts.get(ch, 0) + 1.0) / float(vocab * 2))
    vocab = max(len(_ASCII_SET), 8)
    return float(1.0 / vocab)


def _ngram_coherence(text: str) -> float:
    if len(text) < 2 or not _NGRAM_TOTALS:
        return 0.0
    prev = "\0"
    total_pairs = 0
    log_sum = 0.0
    for ch in text:
        prob = _ngram_probability(prev, ch)
        if prob > 0:
            log_sum += math.log(prob)
            total_pairs += 1
        prev = ch
    if total_pairs == 0:
        return 0.0
    return float(math.exp(log_sum / total_pairs))


def _ngram_surprisal(text: str) -> float:
    if not text or not _NGRAM_TOTALS:
        return 0.0
    prev = "\0"
    total_pairs = 0
    accum = 0.0
    for ch in text:
        prob = max(_ngram_probability(prev, ch), 1e-9)
        accum += -math.log(prob, 2)
        total_pairs += 1
        prev = ch
    if total_pairs == 0:
        return 0.0
    return float(accum / total_pairs)

def _score_candidate_with_context(text: str, base_conf: float) -> float:
    score = max(0.0, min(1.0, base_conf))
    if not text:
        return score
    numeric_like = _looks_like_numeric_token(text)
    upper_like = _looks_like_upper_token(text)
    if numeric_like:
        normalized = _normalize_digit_like(text)
        if normalized.isdigit():
            score = max(score, min(1.0, base_conf + 0.08))
    if upper_like and text.replace("0", "O").isalpha():
        score = max(score, min(1.0, base_conf + 0.05))
    coherence = _ngram_coherence(text)
    if coherence > 0:
        score = max(score, min(1.0, base_conf * (0.85 + 0.15 * coherence) + 0.05 * coherence))
    surprisal = _ngram_surprisal(text)
    if surprisal > 0:
        penalty = max(0.0, min(0.2, (surprisal - 1.2) * 0.12))
        if penalty:
            score = max(0.0, score - penalty)
    return min(1.0, score)

def _contextual_rerank_candidates(candidates: Dict[str, float]) -> Tuple[str, float]:
    if not candidates:
        return "", 0.0
    # base choice by confidence
    base_text, base_conf = max(candidates.items(), key=lambda kv: kv[1])
    enriched: Dict[str, float] = dict(candidates)
    for text, conf in list(enriched.items()):
        variants = _generate_contextual_variants(text)
        if _looks_like_numeric_token(text):
            variants.add(_normalize_digit_like(text))
        for v in variants:
            if not v or v == text:
                continue
            if v not in enriched:
                enriched[v] = max(0.0, min(1.0, conf * 0.92))
    best_text, best_score = base_text, _score_candidate_with_context(base_text, enriched[base_text])
    for text, conf in enriched.items():
        score = _score_candidate_with_context(text, conf)
        if score > best_score or (abs(score - best_score) <= 1e-6 and conf > enriched.get(best_text, 0.0)):
            best_text, best_score = text, score
    return best_text, float(best_score)

def _update_ngram_model(text: str) -> None:
    if not text:
        return
    prev = "\0"
    decay = float(_NGRAM_DECAY_ALPHA)
    decay = 0.0 if math.isnan(decay) else max(0.0, min(0.95, decay))
    for ch in text:
        mapping = _NGRAM_COUNTS[prev]
        if decay > 0.0 and mapping:
            factor = 1.0 - decay
            to_delete: List[str] = []
            for key, val in list(mapping.items()):
                new_val = float(val) * factor
                if new_val < 1e-4:
                    to_delete.append(key)
                else:
                    mapping[key] = new_val
            for key in to_delete:
                mapping.pop(key, None)
        mapping[ch] = float(mapping.get(ch, 0.0) + 1.0)
        total = float(sum(mapping.values())) if mapping else 0.0
        _NGRAM_TOTALS[prev] = total
        prev = ch

def _self_augment_views(arr: "np.ndarray", best_bw: Optional["np.ndarray"]) -> List[Tuple["np.ndarray", Dict[str, Any]]]:
    variants: List[Tuple["np.ndarray", Dict[str, Any]]] = []
    cfg = current_toy_self_correction()
    try:
        gray_img = Image.fromarray(arr.astype(np.uint8)) if arr is not None else None
    except Exception:
        gray_img = None
    if best_bw is not None:
        try:
            bw_img = Image.fromarray(best_bw.astype(np.uint8))
            filter_sizes: List[int] = [3, 5]
            if cfg and isinstance(cfg.get("augment_filter_sizes"), (list, tuple)):
                for candidate in cfg.get("augment_filter_sizes", []):  # type: ignore[assignment]
                    try:
                        filt_size = int(round(float(candidate)))
                    except Exception:
                        continue
                    if filt_size % 2 == 0 or filt_size <= 0:
                        continue
                    filter_sizes.append(filt_size)
            max_filter = int(cfg.get("max_filter_size", 9)) if isinstance(cfg, dict) else 9
            norm_sizes = sorted({s for s in filter_sizes if 1 <= s <= max_filter})
            if not norm_sizes:
                norm_sizes = [3, 5]
            for size in norm_sizes:
                try:
                    variants.append((np.asarray(bw_img.filter(ImageFilter.MaxFilter(size)), dtype=np.uint8), {"type": "augment_max", "size": size}))
                    variants.append((np.asarray(bw_img.filter(ImageFilter.MinFilter(size)), dtype=np.uint8), {"type": "augment_min", "size": size}))
                except Exception:
                    continue
            if cfg and cfg.get("augment_invert"):
                try:
                    inverted = 255 - np.asarray(bw_img, dtype=np.uint8)
                    variants.append((inverted, {"type": "augment_invert"}))
                except Exception:
                    pass
        except Exception:
            pass
    if gray_img is not None:
        fill = int(float(np.median(arr))) if arr.size else 0
        rotations = [-3, -1, 1, 3]
        if cfg and isinstance(cfg.get("extra_rotations"), (list, tuple)):
            limit = int(cfg.get("rotation_limit", 12))
            for candidate in cfg.get("extra_rotations", []):  # type: ignore[assignment]
                try:
                    rot = int(round(float(candidate)))
                except Exception:
                    continue
                if -limit <= rot <= limit:
                    rotations.append(rot)
        rotations = sorted({int(r) for r in rotations if -30 <= int(r) <= 30}) or [-3, -1, 1, 3]
        if cfg and cfg.get("include_zero_rotation"):
            rotations.append(0)
            rotations = sorted({int(r) for r in rotations if -30 <= int(r) <= 30}) or [-3, -1, 1, 3]
        for angle in rotations:
            try:
                rotated = gray_img.rotate(angle, resample=Image.BILINEAR, fillcolor=fill)
                rot_arr = np.asarray(rotated, dtype=np.uint8)
                thr = _otsu_threshold_toy(rot_arr)
                bw = (rot_arr < thr).astype(np.uint8) * 255
                variants.append((bw, {"type": "augment_rotate", "angle": angle, "thr": thr}))
            except Exception:
                continue
    return variants

_TOTAL_LABEL_HINTS = [
    "total", "grandtotal", "amountdue", "balancedue", "totaldue", "subtotal",
    "totaltax", "totaltaxes", "totals", "dueamount", "amountpayable",
    "合計", "総計", "総額", "小計", "税込合計", "税込総額", "請求額", "ご請求額", "支払金額", "合算", "合計金額"
]
_TOTAL_PREFIXES = ["total", "subtotal", "balance", "amountdue", "dueamount", "grandtotal", "amountpayable", "合計", "小計", "総額", "請求"]
_NUMERIC_RX = re.compile(r"[+\-]?\d[\d,]*(?:\.\d+)?")
_NUMERIC_SANITIZE_RX = re.compile(r"[^0-9.+-]")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except Exception:
        return default


_PYTESS_TIMEOUT = max(0.0, _env_float("ZOCR_PYTESS_TIMEOUT", 3.5))
_PYTESS_NICE = _env_int("ZOCR_PYTESS_NICE", None)


def _pytesseract_env_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if _PYTESS_TIMEOUT > 0.0:
        kwargs["timeout"] = _PYTESS_TIMEOUT
    if _PYTESS_NICE is not None:
        kwargs["nice"] = _PYTESS_NICE
    return kwargs


_PYTESS_ENV_KWARGS = _pytesseract_env_kwargs()
_PYTESS_ENV_SUPPORTED = True
_PYTESS_TIMEOUT_WARNED = False


def _pytesseract_allowed() -> bool:
    raw = os.environ.get("ZOCR_ALLOW_PYTESSERACT")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _note_pytesseract_exception(label: str, exc: Exception) -> None:
    global _PYTESS_TIMEOUT_WARNED
    if exc is None:
        return
    msg = str(exc).lower()
    if "time" in msg and "out" in msg:
        if not _PYTESS_TIMEOUT_WARNED:
            limit = f"{_PYTESS_TIMEOUT:.1f}s" if _PYTESS_TIMEOUT > 0 else "configured limit"
            print(f"[WARN] pytesseract {label} timed out ({limit}); continuing", flush=True)
            _PYTESS_TIMEOUT_WARNED = True


def _pytesseract_call(label: str, func: Callable[..., Any], *args, **kwargs):
    global _PYTESS_ENV_SUPPORTED
    extras = _PYTESS_ENV_KWARGS if _PYTESS_ENV_SUPPORTED else {}
    if extras:
        call_kwargs = dict(kwargs)
        call_kwargs.update(extras)
    else:
        call_kwargs = kwargs
    try:
        return func(*args, **call_kwargs)
    except TypeError as exc:
        if extras:
            _PYTESS_ENV_SUPPORTED = False
            return _pytesseract_call(label, func, *args, **kwargs)
        _note_pytesseract_exception(label, exc)
        return None
    except Exception as exc:  # pragma: no cover - best effort guard
        _note_pytesseract_exception(label, exc)
        return None


# --- Toy OCR knobs ----------------------------------------------------------
_TOY_SWEEPS = max(1, int(os.environ.get("ZOCR_TOY_SWEEPS", "5")))
_FORCE_NUMERIC = _env_flag("ZOCR_FORCE_NUMERIC", True)
_LAST_EXPORT_STATS: Dict[str, Any] = {}


@dataclass
class MotionPriorCfg:
    enabled: bool = False
    sigma_px: float = 8.0
    cutoff_sigma: float = 2.5
    accept_ratio: float = 0.5


@dataclass
class _BlankSkipConfig:
    enabled: bool = False
    dark_threshold: int = 210
    min_dark_pixels: int = 8
    min_dark_ratio: float = 0.002
    min_area: int = 36


@dataclass
class _ConfidenceBoostConfig:
    enabled: bool = True
    target: float = 0.82
    min_input: float = 0.35


@dataclass
class _LexicalBoostConfig(_ConfidenceBoostConfig):
    min_quality: float = 0.85


@dataclass
class _ExportSweepTracker:
    prev_y: Dict[Tuple[str, int, int], List[float]] = field(default_factory=dict)

    def get(self, doc_id: str, page_index: int, table_index: int) -> Optional[List[float]]:
        return self.prev_y.get((doc_id, page_index, table_index))

    def put(
        self,
        doc_id: str,
        page_index: int,
        table_index: int,
        y_keys: Sequence[float],
    ) -> None:
        self.prev_y[(doc_id, page_index, table_index)] = [float(y) for y in y_keys]


_EXPORT_SWEEP_TRACKER = _ExportSweepTracker()


def _motion_prior_cfg_from_env() -> MotionPriorCfg:
    cfg = MotionPriorCfg(enabled=_env_flag("ZOCR_EXPORT_MOTION_PRIOR", False))
    if cfg.enabled:
        sigma = _env_float("ZOCR_EXPORT_MOTION_SIGMA", cfg.sigma_px)
        cutoff = _env_float("ZOCR_EXPORT_MOTION_CUTOFF", cfg.cutoff_sigma)
        accept = _env_float("ZOCR_EXPORT_MOTION_ACCEPT", cfg.accept_ratio)
        cfg.sigma_px = max(0.5, float(sigma))
        cfg.cutoff_sigma = max(0.1, float(cutoff))
        cfg.accept_ratio = float(max(0.0, min(1.0, accept)))
    return cfg


def _blank_skip_cfg_from_env() -> _BlankSkipConfig:
    cfg = _BlankSkipConfig(enabled=_env_flag("ZOCR_EXPORT_SKIP_BLANK", True))
    if not cfg.enabled:
        return cfg
    thr = _env_int("ZOCR_EXPORT_BLANK_THRESHOLD", cfg.dark_threshold)
    if thr is not None:
        cfg.dark_threshold = int(max(1, min(255, thr)))
    min_px = _env_int("ZOCR_EXPORT_BLANK_MIN_PIXELS", cfg.min_dark_pixels)
    if min_px is not None:
        cfg.min_dark_pixels = int(max(1, min_px))
    min_ratio = _env_float("ZOCR_EXPORT_BLANK_MIN_RATIO", cfg.min_dark_ratio)
    if isinstance(min_ratio, (int, float)):
        cfg.min_dark_ratio = float(max(0.0, min(0.1, min_ratio)))
    min_area = _env_int("ZOCR_EXPORT_BLANK_MIN_AREA", cfg.min_area)
    if min_area is not None:
        cfg.min_area = int(max(1, min_area))
    return cfg


def _confidence_boost_cfg_from_env() -> _ConfidenceBoostConfig:
    cfg = _ConfidenceBoostConfig()
    cfg.enabled = _env_flag("ZOCR_CONF_BOOST_NUMERIC", True)
    cfg.target = float(max(0.0, min(1.0, _env_float("ZOCR_CONF_BOOST_TARGET", cfg.target))))
    cfg.min_input = float(max(0.0, min(cfg.target, _env_float("ZOCR_CONF_BOOST_MIN_INPUT", cfg.min_input))))
    return cfg


def _lexical_boost_cfg_from_env() -> _LexicalBoostConfig:
    cfg = _LexicalBoostConfig()
    cfg.enabled = _env_flag("ZOCR_CONF_BOOST_LEXICAL", True)
    cfg.target = float(max(0.0, min(1.0, _env_float("ZOCR_CONF_BOOST_LEXICAL_TARGET", cfg.target))))
    cfg.min_input = float(
        max(0.0, min(cfg.target, _env_float("ZOCR_CONF_BOOST_LEXICAL_MIN_INPUT", cfg.min_input)))
    )
    cfg.min_quality = float(
        max(0.0, min(1.5, _env_float("ZOCR_CONF_BOOST_LEXICAL_MIN_QUALITY", cfg.min_quality)))
    )
    return cfg


def _apply_confidence_boost(
    conf: Optional[float], cfg: _ConfidenceBoostConfig
) -> Tuple[Optional[float], float]:
    if conf is None or not cfg.enabled:
        return conf, 0.0
    base = _normalize_confidence(conf)
    if base >= cfg.target or base < cfg.min_input:
        return base, 0.0
    boosted = float(max(base, cfg.target))
    return boosted, boosted - base


def _confidence_boost_reason(payload: Optional[Dict[str, Any]]) -> str:
    if not payload:
        return ""
    for key in ("amount", "subtotal", "tax_amount", "unit_price", "qty", "tax_rate"):
        if key in payload:
            return key
    return ""


def _lexical_confidence_reason(
    text: Optional[str],
    lexical_diag: Dict[str, Any],
    min_quality: float,
    quality: float,
) -> str:
    if not text or quality < min_quality:
        return ""
    canonical = (lexical_diag.get("canonical") or "").strip()
    if canonical:
        hint = _LEXICAL_CONF_CANONICAL_HINTS.get(canonical)
        if hint:
            return f"lexical:{hint}"
    normalized_jp = _normalize_japanese_token(text)
    if normalized_jp:
        hint = _LEXICAL_CONF_JP_HINTS.get(normalized_jp)
        if hint:
            return f"lexical:{hint}"
    jp_reason = lexical_diag.get("jp_hint_reason")
    if lexical_diag.get("jp_hint") and jp_reason:
        return f"lexical:jp_{jp_reason}"
    return ""


def _should_skip_blank_crop(img: "Image.Image", cfg: _BlankSkipConfig) -> bool:
    if not cfg.enabled:
        return False
    try:
        w, h = img.size
    except Exception:
        return False
    area = max(1, int(w) * int(h))
    if area < cfg.min_area:
        return False
    try:
        gray = img.convert("L")
        hist = gray.histogram()
    except Exception:
        return False
    if not hist:
        return False
    thr = max(1, min(256, int(cfg.dark_threshold)))
    dark = int(sum(hist[:thr]))
    if dark < cfg.min_dark_pixels:
        return True
    ratio = dark / float(area)
    if ratio <= cfg.min_dark_ratio:
        return True
    return False


def _row_band_midpoints(row_bands: Sequence[Tuple[int, int]]) -> List[float]:
    mids: List[float] = []
    for top, bottom in row_bands:
        try:
            mid = (float(top) + float(bottom)) * 0.5
        except Exception:
            continue
        if math.isfinite(mid):
            mids.append(mid)
    return mids


def _reseed_row_bands_with_prior(
    prev_keys: Sequence[float],
    row_bands: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float]]]:
    if not prev_keys or not row_bands:
        return list(row_bands), []
    mids = _row_band_midpoints(row_bands)
    if not mids:
        return list(row_bands), []
    ordered = sorted(((mid, idx) for idx, mid in enumerate(mids)), key=lambda item: item[0])
    ordered_mids = [mid for mid, _ in ordered]
    used = [False] * len(ordered)
    matches: List[Tuple[float, float]] = []
    reseed_indices: List[int] = []
    for prev_val in prev_keys:
        pos = bisect.bisect_left(ordered_mids, prev_val)
        best_idx = None
        best_dist = float("inf")
        for offset in (pos - 1, pos, pos + 1):
            if 0 <= offset < len(ordered) and not used[offset]:
                cand_mid, cand_idx = ordered[offset]
                dist = abs(cand_mid - prev_val)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = offset
        if best_idx is None:
            continue
        used[best_idx] = True
        cand_mid, cand_idx = ordered[best_idx]
        matches.append((cand_mid, float(prev_val)))
        reseed_indices.append(cand_idx)
    for offset, (_, idx) in enumerate(ordered):
        if not used[offset]:
            reseed_indices.append(idx)
    reseeded = [row_bands[idx] for idx in reseed_indices]
    return reseeded, matches


def _apply_motion_prior_to_bands(
    prev_keys: Optional[Sequence[float]],
    row_bands: List[Tuple[int, int]],
    cfg: MotionPriorCfg,
) -> Tuple[List[Tuple[int, int]], bool]:
    if not cfg.enabled or not prev_keys or not row_bands:
        return row_bands, False
    reseeded, matches = _reseed_row_bands_with_prior(prev_keys, row_bands)
    if not reseeded:
        return row_bands, False
    sigma = max(1e-3, float(cfg.sigma_px))
    cutoff = float(cfg.cutoff_sigma) * sigma
    inside = 0
    total = len(matches)
    for cand_mid, prev_val in matches:
        if abs(cand_mid - prev_val) <= cutoff:
            inside += 1
    actual_ratio = inside / float(total or 1)
    if actual_ratio >= cfg.accept_ratio:
        return reseeded, True
    return row_bands, False


def toy_runtime_config() -> Dict[str, Any]:
    """Return the currently active toy OCR runtime knobs."""

    return {
        "threshold_sweeps": int(_TOY_SWEEPS),
        "force_numeric": bool(_FORCE_NUMERIC),
    }


def last_export_stats() -> Dict[str, Any]:
    """Return metrics captured during the most recent contextual export."""

    return dict(_LAST_EXPORT_STATS)


def configure_toy_runtime(
    *, sweeps: Optional[int] = None, force_numeric: Optional[bool] = None
) -> Dict[str, Any]:
    """Update toy OCR runtime knobs at runtime."""

    updates: Dict[str, Any] = {}
    global _TOY_SWEEPS, _FORCE_NUMERIC
    if sweeps is not None:
        try:
            new_sweeps = max(1, int(sweeps))
        except Exception:
            new_sweeps = _TOY_SWEEPS
        if new_sweeps != _TOY_SWEEPS:
            _TOY_SWEEPS = new_sweeps
            updates["threshold_sweeps"] = new_sweeps
    if force_numeric is not None:
        new_flag = bool(force_numeric)
        if new_flag != _FORCE_NUMERIC:
            _FORCE_NUMERIC = new_flag
            updates["force_numeric"] = new_flag
    return updates


_NUMERIC_HEADER_KIND = [
    ("qty", re.compile(r"(数量|数|個|台数|件数|口数|本数|点数|qty|q'?ty|quantity)", re.I)),
    ("unit_price", re.compile(r"(単価|unit\s*price|price|unit\s*cost)", re.I)),
    (
        "amount",
        re.compile(
            r"(金額|見積金額|御見積金額|御見積合計|合計金額|合計|総計|計|税込|税別|小計|amount|total|subtotal|balance)",
            re.I,
        ),
    ),
    ("tax_rate", re.compile(r"(税率|消費税率|税%|tax%|tax(\s*rate)?|vat|gst)", re.I)),
]

_NUMERIC_HEADER_INFERRED_LAST = 0


def _canonicalize_header_label(label: str) -> str:
    text = unicodedata.normalize("NFKC", str(label or ""))
    text = text.strip().lower()
    text = text.replace("：", ":").replace("　", " ")
    return re.sub(r"\s+", " ", text)


def _header_variants_for_numeric(label: str) -> List[str]:
    base = _canonicalize_header_label(label)
    variants: List[str] = []
    if not base:
        return variants

    def _add(val: str) -> None:
        val = val.strip()
        if val and val not in variants:
            variants.append(val)

    _add(base)
    _add(base.replace(" ", ""))
    honorific = base.lstrip("御お")
    _add(honorific)
    no_brackets = re.sub(r"[\(（\[［【].*?[\)）\]］】]", "", base)
    no_brackets = re.sub(r"\s+", " ", no_brackets).strip()
    _add(no_brackets)
    simplified = re.sub(r"[\-:：／/\\()（）\[\]{}<>«»《》【】「」『』]", "", base)
    simplified = re.sub(r"\s+", " ", simplified).strip()
    _add(simplified)
    _add(simplified.replace(" ", ""))
    for token in re.split(r"[/｜\|・,、]", base):
        token = token.strip()
        if not token:
            continue
        _add(token)
        _add(token.replace(" ", ""))
    return variants


def _normalize_numeric_bits(text: str, kind: str) -> Tuple[str, Optional[float], Optional[str]]:
    work = (text or "").strip()
    if not work:
        return "", None, None
    work = work.replace("，", ",").replace("．", ".")
    neg = False
    if work.endswith("円"):
        work = work[:-1]
    if work.startswith("円"):
        work = work[1:]
    if work.startswith("(") and work.endswith(")"):
        neg = True
        work = work[1:-1]
    work = work.strip()
    percent = "%" in work
    work = work.replace("%", "")
    cleaned = _NUMERIC_SANITIZE_RX.sub("", work)
    if neg and cleaned and not cleaned.startswith("-"):
        cleaned = "-" + cleaned
    if not cleaned:
        return "", None, percent
    try:
        value = float(cleaned)
    except Exception:
        value = None
    normalized = cleaned
    if kind == "qty" and value is not None:
        qty_val = int(round(value))
        value = float(qty_val)
        normalized = str(qty_val)
    if percent and normalized:
        normalized = f"{normalized}%"
    return normalized, value, percent if percent else None


def _normalize_numeric_text(text: str, kind: str) -> str:
    normalized, _, _ = _normalize_numeric_bits(text, kind)
    return normalized or (text or "")


def _coerce_numeric_filters(kind: Optional[str], text: str) -> Tuple[str, Dict[str, Any]]:
    if not kind:
        return text, {}
    normalized, value, percent = _normalize_numeric_bits(text, kind)
    payload: Dict[str, Any] = {}
    result_text = normalized or (text or "")
    if value is None:
        return result_text, payload
    if kind == "qty":
        payload["qty"] = int(round(value))
        result_text = str(payload["qty"])
    elif kind in ("amount", "subtotal", "tax_amount", "unit_price"):
        payload[kind] = float(value)
    elif kind == "tax_rate":
        rate_val = float(value)
        if percent or abs(rate_val) > 1.0:
            rate_val = rate_val / 100.0
        payload["tax_rate"] = rate_val
    return result_text, payload


def _infer_numeric_kinds_from_values(
    grid_text: Sequence[Sequence[str]],
    kinds: List[Optional[str]],
) -> List[Optional[str]]:
    global _NUMERIC_HEADER_INFERRED_LAST
    inferred = 0
    if not grid_text:
        _NUMERIC_HEADER_INFERRED_LAST = 0
        return kinds
    num_cols = max(len(row) for row in grid_text)
    if len(kinds) < num_cols:
        kinds.extend([None] * (num_cols - len(kinds)))
    currency_rx = re.compile(r"[¥￥円＄$元]")
    for c in range(num_cols):
        if kinds[c]:
            continue
        total = 0
        numeric_hits = 0
        currency_hits = 0
        decimals = 0
        values: List[float] = []
        for r in range(1, len(grid_text)):
            row = grid_text[r]
            if c >= len(row):
                continue
            txt = str(row[c] or "").strip()
            if not txt:
                continue
            total += 1
            if currency_rx.search(txt):
                currency_hits += 1
            cleaned = txt.replace("，", ",").replace("．", ".")
            cleaned = cleaned.replace(",", "")
            match = re.search(r"[+\-]?\d+(?:\.\d+)?", cleaned)
            if not match:
                continue
            numeric_hits += 1
            token = match.group(0)
            try:
                val = float(token)
            except ValueError:
                continue
            values.append(abs(val))
            if "." in token:
                decimals += 1
        if total < 2:
            continue
        ratio = numeric_hits / float(total)
        if ratio < 0.65:
            continue
        avg_val = sum(values) / float(len(values)) if values else 0.0
        if currency_hits >= max(1, math.ceil(total * 0.4)) or avg_val >= 1000:
            new_kind = "total" if c >= num_cols - 1 else "amount"
        elif decimals > 0 and avg_val >= 1:
            new_kind = "unit_price"
        else:
            new_kind = "qty"
        kinds[c] = new_kind
        inferred += 1
    _NUMERIC_HEADER_INFERRED_LAST = inferred
    return kinds


def _numeric_header_kinds(
    headers: Sequence[str],
    grid_text: Optional[Sequence[Sequence[str]]] = None,
) -> List[Optional[str]]:
    global _NUMERIC_HEADER_INFERRED_LAST
    kinds: List[Optional[str]] = []
    if not _FORCE_NUMERIC:
        _NUMERIC_HEADER_INFERRED_LAST = 0
        return kinds
    if headers:
        for header in headers:
            kind: Optional[str] = None
            for variant in _header_variants_for_numeric(header) or [""]:
                for key, rx in _NUMERIC_HEADER_KIND:
                    if variant and rx.search(variant):
                        kind = key
                        break
                if kind:
                    break
            kinds.append(kind)
    if grid_text:
        kinds = _infer_numeric_kinds_from_values(grid_text, kinds)
    else:
        _NUMERIC_HEADER_INFERRED_LAST = 0
    return kinds


def _enforce_numeric_by_headers(headers: Sequence[str], grid_text: Sequence[Sequence[str]]) -> None:
    if not _FORCE_NUMERIC:
        return
    kinds = _numeric_header_kinds(headers, grid_text)
    if not kinds:
        return
    for r in range(1, len(grid_text)):
        row = list(grid_text[r])
        changed = False
        for c, kind in enumerate(kinds):
            if not kind or c >= len(row):
                continue
            normalized = _normalize_numeric_text(str(row[c] or ""), kind)
            if normalized != row[c]:
                row[c] = normalized
                changed = True
        if changed and isinstance(grid_text[r], list):
            grid_text[r][:] = row


_DATE_FULLWIDTH = str.maketrans(
    {
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
        "－": "-",
        "．": ".",
        "／": "/",
        "年": "年",
        "月": "月",
        "日": "日",
    }
)

_DATE_ROLE_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    (
        "due",
        (
            "due date",
            "payment due",
            "due",
            "支払期限",
            "支払期日",
            "期日",
            "納期",
        ),
    ),
    (
        "issue",
        (
            "invoice date",
            "issue date",
            "billing date",
            "請求日",
            "発行日",
            "作成日",
        ),
    ),
    (
        "service",
        (
            "service date",
            "ship date",
            "delivery date",
            "利用日",
            "作業日",
        ),
    ),
]

_DATE_RX_SLASH = re.compile(
    r"(?P<year>19\d{2}|20\d{2})[./-](?P<month>0?[1-9]|1[0-2])(?:[./-](?P<day>0?[1-9]|[12]\d|3[01]))?"
)
_DATE_RX_KANJI = re.compile(
    r"(?P<year>19\d{2}|20\d{2})年(?P<month>0?[1-9]|1[0-2])月(?:(?P<day>0?[1-9]|[12]\d|3[01])日)?"
)
_DATE_RX_ERA = re.compile(
    r"(?P<era>令和|平成|昭和)(?P<erayear>元|\d{1,2})年(?P<month>0?[1-9]|1[0-2])月(?:(?P<day>0?[1-9]|[12]\d|3[01])日)?"
)
_DATE_RX_COMPACT = re.compile(
    r"(?P<year>19\d{2}|20\d{2})(?P<month>0?[1-9]|1[0-2])(?P<day>0?[1-9]|[12]\d|3[01])"
)
_JP_ERA_BASE = {"令和": 2018, "平成": 1988, "昭和": 1925}


def _date_role_from_header(header: Optional[str]) -> Optional[str]:
    if not header:
        return None
    base = header.strip().lower()
    if not base:
        return None
    for role, tokens in _DATE_ROLE_RULES:
        for token in tokens:
            if token in base:
                return role
    if "date" in base:
        return "date"
    return None


def _date_header_roles(headers: Sequence[str]) -> List[Optional[str]]:
    roles: List[Optional[str]] = []
    if not headers:
        return roles
    for header in headers:
        roles.append(_date_role_from_header(header))
    return roles


def _infer_date_role_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    base = text.strip().lower()
    if not base:
        return None
    for role, tokens in _DATE_ROLE_RULES:
        for token in tokens:
            if token in base:
                return role
    if "date" in base:
        return "date"
    return None


def _normalize_japanese_era_year(era: str, era_year: str) -> Optional[int]:
    base = _JP_ERA_BASE.get(era)
    if base is None:
        return None
    token = era_year.strip()
    if token == "元":
        offset = 1
    else:
        try:
            offset = int(token)
        except Exception:
            return None
    if offset <= 0:
        return None
    return base + offset


def _normalize_date_value(text: str) -> Optional[Tuple[str, str]]:
    if not text:
        return None
    norm = str(text).strip()
    if not norm:
        return None
    norm = norm.translate(_DATE_FULLWIDTH)
    compact = norm.replace(" ", "").replace("　", "")

    def _format(year: int, month: int, day: Optional[int]) -> Optional[Tuple[str, str]]:
        if year < 1900 or year > 2100:
            return None
        if not 1 <= month <= 12:
            return None
        precision = "ym"
        value = f"{year:04d}-{month:02d}"
        if day is not None:
            if not 1 <= day <= 31:
                return None
            precision = "ymd"
            value = f"{value}-{day:02d}"
        return value, precision

    for rx in (_DATE_RX_SLASH, _DATE_RX_KANJI):
        m = rx.search(compact)
        if not m:
            continue
        try:
            year = int(m.group("year"))
            month = int(m.group("month"))
            day = int(m.group("day")) if m.group("day") else None
        except Exception:
            continue
        formatted = _format(year, month, day)
        if formatted:
            return formatted

    m_era = _DATE_RX_ERA.search(compact)
    if m_era:
        year = _normalize_japanese_era_year(m_era.group("era"), m_era.group("erayear"))
        try:
            month = int(m_era.group("month"))
            day = int(m_era.group("day")) if m_era.group("day") else None
        except Exception:
            month = None
            day = None
        if year and month:
            formatted = _format(year, month, day)
            if formatted:
                return formatted

    m_compact = _DATE_RX_COMPACT.search(compact)
    if m_compact:
        try:
            year = int(m_compact.group("year"))
            month = int(m_compact.group("month"))
            day = int(m_compact.group("day"))
        except Exception:
            year = month = day = None
        if year and month and day:
            formatted = _format(year, month, day)
            if formatted:
                return formatted
    return None


def _extract_date_filters(text: str, header_role: Optional[str]) -> Optional[Dict[str, Any]]:
    normalized = _normalize_date_value(text)
    if not normalized:
        return None
    value, precision = normalized
    role = header_role or _infer_date_role_from_text(text)
    payload: Dict[str, Any] = {
        "date": value,
        "date_precision": precision,
    }
    if role == "due":
        payload["due_date"] = value
    elif role == "issue":
        payload["issue_date"] = value
    elif role == "service":
        payload["service_date"] = value
    if role:
        payload["date_role"] = role
    return payload


_NUMERIC_COLUMN_CHARSETS: Dict[str, str] = {
    "qty": "0123456789",
    "unit_price": "0123456789.-",
    "amount": "0123456789.-",
    "subtotal": "0123456789.-",
    "tax_amount": "0123456789.-",
    "tax_rate": "0123456789.%",
}

_ITEM_QTY_SCHEMA_COLUMNS: List[Dict[str, Any]] = [
    {
        "key": "item",
        "pattern": re.compile(r"(item|品目|description|desc|details)", re.I),
        "title": "Item",
        "normalizer": None,
    },
    {
        "key": "qty",
        "pattern": re.compile(r"(qty|数量|quantity|q'?ty|pcs?|units?)", re.I),
        "title": "Qty",
        "normalizer": "qty",
    },
    {
        "key": "unit_price",
        "pattern": re.compile(r"(unit\s*(price|cost)|単価)", re.I),
        "title": "Unit Price",
        "normalizer": "currency",
    },
    {
        "key": "amount",
        "pattern": re.compile(r"(amount|line\s*total|line\s*amount|金額|total)", re.I),
        "title": "Amount",
        "normalizer": "currency",
    },
]


def _schema_normalize_header_token(text: Optional[str]) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).strip().lower()
    if not normalized:
        return ""
    pieces: List[str] = []
    for ch in normalized:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            pieces.append(ch)
    return "".join(pieces)


def _match_item_qty_schema(headers: Sequence[str]) -> Optional[List[int]]:
    if not headers or len(headers) < len(_ITEM_QTY_SCHEMA_COLUMNS):
        return None
    normalized = [_schema_normalize_header_token(h) for h in headers]
    selected: List[int] = []
    used: Set[int] = set()
    for column in _ITEM_QTY_SCHEMA_COLUMNS:
        pattern = column.get("pattern")
        best_idx = None
        for idx in range(len(normalized) - 1, -1, -1):
            if idx in used:
                continue
            token = normalized[idx]
            if not token or pattern is None:
                continue
            if pattern.search(token):
                best_idx = idx
                break
        if best_idx is None:
            return None
        used.add(best_idx)
        selected.append(best_idx)
    if any(selected[i] >= selected[i + 1] for i in range(len(selected) - 1)):
        return None
    return selected


def _column_numeric_profiles(grid_text: Sequence[Sequence[str]]) -> List[Dict[str, float]]:
    if not grid_text:
        return []
    num_cols = max(len(row) for row in grid_text)
    if num_cols <= 0:
        return []
    profiles: List[Dict[str, float]] = []
    currency_tokens = ("¥", "￥", "円", "$", "＄", "元")
    for col in range(num_cols):
        total = 0
        numeric_hits = 0
        currency_hits = 0
        decimal_hits = 0
        text_hits = 0
        for row in list(grid_text)[1:]:
            if col >= len(row):
                continue
            cell = str(row[col] or "").strip()
            if not cell:
                continue
            total += 1
            if any(token in cell for token in currency_tokens):
                currency_hits += 1
            if _NUMERIC_RX.search(cell.replace("，", ",").replace("．", ".")):
                numeric_hits += 1
                if "." in cell or "．" in cell:
                    decimal_hits += 1
            elif any(ch.isalpha() for ch in cell):
                text_hits += 1
        denom = float(max(1, total))
        profiles.append(
            {
                "samples": float(total),
                "numeric_ratio": float(numeric_hits) / denom,
                "currency_ratio": float(currency_hits) / denom,
                "decimal_ratio": float(decimal_hits) / denom,
                "text_ratio": float(text_hits) / denom,
            }
        )
    return profiles


def _approximate_item_qty_schema(grid_text: Sequence[Sequence[str]]) -> Optional[List[int]]:
    if not grid_text:
        return None
    num_cols = max(len(row) for row in grid_text)
    if num_cols < len(_ITEM_QTY_SCHEMA_COLUMNS):
        return None
    profiles = _column_numeric_profiles(grid_text)
    if not profiles or len(profiles) < len(_ITEM_QTY_SCHEMA_COLUMNS):
        return None

    def _best_index(indices: Sequence[int], key: Callable[[int], Tuple]) -> Optional[int]:
        valid = [idx for idx in indices if 0 <= idx < len(profiles)]
        if not valid:
            return None
        return max(valid, key=key)

    amount_idx = _best_index(
        range(num_cols),
        lambda idx: (
            profiles[idx]["currency_ratio"],
            profiles[idx]["numeric_ratio"],
            idx,
        ),
    )
    if amount_idx is None:
        return None
    unit_price_candidates = [idx for idx in range(amount_idx)]
    unit_price_idx = _best_index(
        unit_price_candidates,
        lambda idx: (
            profiles[idx]["currency_ratio"],
            profiles[idx]["decimal_ratio"],
            profiles[idx]["numeric_ratio"],
            idx,
        ),
    )
    if unit_price_idx is None:
        return None
    qty_candidates = [idx for idx in range(unit_price_idx)]
    qty_idx = _best_index(
        qty_candidates,
        lambda idx: (
            profiles[idx]["numeric_ratio"],
            1.0 - min(1.0, profiles[idx]["decimal_ratio"]),
            1.0 - min(1.0, profiles[idx]["currency_ratio"]),
            -idx,
        ),
    )
    if qty_idx is None:
        return None
    item_candidates = [idx for idx in range(qty_idx)]
    item_idx = _best_index(
        item_candidates,
        lambda idx: (
            profiles[idx]["text_ratio"],
            1.0 - min(1.0, profiles[idx]["numeric_ratio"]),
            -idx,
        ),
    )
    if item_idx is None:
        return None
    indices = [item_idx, qty_idx, unit_price_idx, amount_idx]
    if any(indices[i] >= indices[i + 1] for i in range(len(indices) - 1)):
        return None
    return indices


def _schema_normalize_qty_value(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    body = str(text).strip()
    if not body:
        return None
    compact = body.replace(",", "")
    if not re.fullmatch(r"[+\-]?\d+(?:\.\d+)?", compact):
        return None
    if "." in compact:
        compact = compact.rstrip("0").rstrip(".") or "0"
    return compact


def _schema_normalize_currency_value(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    match = _NUMERIC_RX.search(str(text))
    if not match:
        return None
    token = match.group(0)
    if token.endswith("%"):
        return None
    return token.replace(",", "")


_SCHEMA_NORMALIZERS: Dict[Optional[str], Callable[[Optional[str]], Optional[str]]] = {
    None: lambda txt: txt if txt else None,
    "qty": _schema_normalize_qty_value,
    "currency": _schema_normalize_currency_value,
}


def _schema_pick_from_noise(
    noise_pool: List[Tuple[str, float]], normalizer: Callable[[Optional[str]], Optional[str]]
) -> Optional[Tuple[str, float]]:
    if not noise_pool:
        return None
    for idx, (text, conf) in enumerate(list(noise_pool)):
        normalized = normalizer(text)
        if normalized:
            noise_pool.pop(idx)
            return normalized, conf
    return None


def _rectify_item_qty_amount_schema(
    grid_text: List[List[str]],
    grid_conf: List[List[float]],
    col_bounds: List[int],
) -> Optional[Tuple[List[List[str]], List[List[float]], List[int], Dict[str, Any]]]:
    if not grid_text or not grid_text[0]:
        return None
    strategy = "header"
    match = _match_item_qty_schema(grid_text[0])
    if not match:
        match = _approximate_item_qty_schema(grid_text)
        if not match:
            return None
        strategy = "heuristic"
    if any(idx + 1 >= len(col_bounds) for idx in match):
        return None
    noise_cols = [idx for idx in range(len(grid_text[0])) if idx not in match]
    width = len(match)
    new_text: List[List[str]] = []
    new_conf: List[List[float]] = []
    rows_adjusted = 0
    cells_salvaged = 0
    cells_cleared = 0

    def _extract(row: Sequence[str], conf: Sequence[float], idx: int) -> Tuple[str, float]:
        txt = row[idx] if idx < len(row) else ""
        cf = conf[idx] if idx < len(conf) else 0.0
        return txt, cf

    for r, row in enumerate(grid_text):
        conf_row = grid_conf[r] if r < len(grid_conf) else [0.0] * len(row)
        new_row: List[str] = []
        new_conf_row: List[float] = []
        for idx in match:
            txt, cf = _extract(row, conf_row, idx)
            new_row.append(txt)
            new_conf_row.append(cf)
        row_changed = False
        if r == 0:
            header_titles = [col["title"] for col in _ITEM_QTY_SCHEMA_COLUMNS]
            if new_row != header_titles:
                new_row = header_titles
                row_changed = True
        else:
            noise_pool = []
            for idx in sorted(noise_cols, reverse=True):
                txt, cf = _extract(row, conf_row, idx)
                if not txt:
                    continue
                noise_pool.append((txt, cf))
            for pos, col_def in enumerate(_ITEM_QTY_SCHEMA_COLUMNS):
                normalizer_key = col_def.get("normalizer")
                normalizer = _SCHEMA_NORMALIZERS.get(normalizer_key) or (lambda val: val)
                normalized = normalizer(new_row[pos]) if normalizer_key else (new_row[pos] or "")
                if normalizer_key and normalized:
                    if normalized != new_row[pos]:
                        new_row[pos] = normalized
                        row_changed = True
                        cells_salvaged += 1
                    continue
                if normalizer_key:
                    candidate = _schema_pick_from_noise(noise_pool, normalizer)
                    if candidate:
                        value, cf = candidate
                        new_row[pos] = value
                        new_conf_row[pos] = max(new_conf_row[pos], cf)
                        row_changed = True
                        cells_salvaged += 1
                        continue
                    if new_row[pos]:
                        new_row[pos] = ""
                        row_changed = True
                        cells_cleared += 1
        if row_changed:
            rows_adjusted += 1
        if len(new_row) < width:
            new_row.extend([""] * (width - len(new_row)))
            new_conf_row.extend([0.0] * (width - len(new_conf_row)))
        new_text.append(new_row)
        new_conf.append(new_conf_row)

    new_bounds: List[int] = []
    for idx in match:
        new_bounds.append(int(col_bounds[idx]))
    new_bounds.append(int(col_bounds[match[-1] + 1]))
    stats = {
        "noise_columns": len(noise_cols),
        "rows_adjusted": rows_adjusted,
        "cells_salvaged": cells_salvaged,
        "cells_cleared": cells_cleared,
        "strategy": strategy,
        "columns": [int(idx) for idx in match],
    }
    return new_text, new_conf, new_bounds, stats


def _column_charset_hints(headers: Sequence[str]) -> List[Optional[str]]:
    if not headers or not _FORCE_NUMERIC:
        return []
    kinds = _numeric_header_kinds(headers)
    if not kinds:
        return []
    hints: List[Optional[str]] = []
    for kind in kinds:
        hints.append(_NUMERIC_COLUMN_CHARSETS.get(kind) if kind else None)
    return hints

_AMBIGUOUS_CHAR_MAP: Dict[str, Tuple[str, ...]] = {
    "?": ("7", "1", "2"),
    "I": ("1", "l"),
    "l": ("1",),
    "|": ("1",),
    "O": ("0",),
    "o": ("0",),
    "S": ("5",),
    "s": ("5",),
    "$": ("5",),
    "B": ("8",),
    "b": ("6",),
    "g": ("9",),
    "Z": ("2",),
    "z": ("2",),
}


def _ambiguous_variants(text: Optional[str]) -> List[str]:
    if not text:
        return []
    candidates: Set[str] = set()
    chars = list(text)
    for idx, ch in enumerate(chars):
        repls = _AMBIGUOUS_CHAR_MAP.get(ch)
        if not repls:
            continue
        for repl in repls:
            if repl == ch:
                continue
            mutated = chars[:]
            mutated[idx] = repl
            candidate = "".join(mutated)
            if candidate != text:
                candidates.add(candidate)
    return sorted(candidates)


def _pytesseract_variants(img: "Image.Image") -> List[Tuple[str, float, str]]:
    if not _pytesseract_allowed() or pytesseract is None or _PYTESS_OUTPUT is None:
        return []
    variants: List[Tuple[str, float, str]] = []
    config = "--psm 6"
    data = _pytesseract_call(
        "image_to_data",
        pytesseract.image_to_data,  # type: ignore[arg-type]
        img,
        output_type=_PYTESS_OUTPUT.DICT,
        config=config,
    )
    if isinstance(data, dict):
        words: List[str] = []
        confs: List[float] = []
        texts = data.get("text", [])
        confidences = data.get("conf", [])
        for raw_txt, raw_conf in zip(texts, confidences):
            txt = (raw_txt or "").strip()
            if txt:
                words.append(txt)
            try:
                c = float(raw_conf)
            except Exception:
                c = -1.0
            if c >= 0.0:
                confs.append(c / 100.0)
        if words:
            joined = " ".join(words)
            conf_val = max(confs) if confs else 0.6
            variants.append((joined, float(max(0.0, min(1.0, conf_val))), "engine:pytesseract_data"))
    raw = _pytesseract_call(
        "image_to_string",
        pytesseract.image_to_string,  # type: ignore[arg-type]
        img,
        config=config,
    )
    if raw:
        txt = raw.strip()
        if txt:
            conf_guess = 0.62
            variants.append((txt, conf_guess, "engine:pytesseract"))
    return variants


def _synthetic_external_variants(img: "Image.Image") -> List[Tuple[str, float, str]]:
    variants: List[Tuple[str, float, str]] = []
    try:
        base = ImageOps.autocontrast(img.convert("L"))
    except Exception:
        try:
            base = img.convert("L")
        except Exception:
            return variants

    work_gray = ImageOps.expand(base, border=1, fill=255)
    try:
        work_rgb = work_gray.convert("RGB")
    except Exception:
        work_rgb = work_gray
    seen: Set[Tuple[str, str]] = set()

    def _emit_variant(im: "Image.Image", label: str, boost: float = 0.0) -> None:
        if im is None:
            return
        try:
            rgb = im.convert("RGB")
        except Exception:
            rgb = im
        try:
            text, conf = toy_ocr_text_from_cell(rgb)
        except Exception:
            return
        if not text:
            return
        key = (text, label)
        if key in seen:
            return
        seen.add(key)
        conf_adj = float(max(0.0, min(1.0, conf + boost)))
        variants.append((text, conf_adj, f"engine:faux_tess/{label}"))

    def _emit_manual(text: str, conf: float, label: str, boost: float = 0.0) -> None:
        if not text:
            return
        key = (text, label)
        if key in seen:
            return
        seen.add(key)
        conf_adj = float(max(0.0, min(1.0, (conf or 0.0) + boost)))
        variants.append((text, conf_adj, f"engine:faux_tess/{label}"))

    _emit_variant(work_rgb, "autocontrast", 0.03)
    _emit_variant(ImageOps.equalize(work_rgb), "equalize")
    _emit_variant(ImageOps.expand(work_rgb, border=2, fill=255), "pad2", 0.03)
    try:
        _emit_variant(ImageEnhance.Sharpness(work_rgb).enhance(1.6), "sharp_1.6", 0.03)
        _emit_variant(ImageEnhance.Sharpness(work_rgb).enhance(2.2), "sharp_2.2", 0.04)
    except Exception:
        pass
    for filt, label in ((ImageFilter.MedianFilter(5), "median5"),
                        (ImageFilter.ModeFilter(size=3), "mode3"),
                        (ImageFilter.DETAIL, "detail"),
                        (ImageFilter.SMOOTH_MORE, "smooth")):
        try:
            _emit_variant(work_rgb.filter(filt), label, 0.0)
        except Exception:
            continue
    try:
        _emit_variant(work_rgb.filter(ImageFilter.UnsharpMask(radius=2, percent=180)), "unsharp", 0.05)
    except Exception:
        pass
    for filt, label in ((ImageFilter.MaxFilter(3), "max3"),
                        (ImageFilter.MinFilter(3), "min3")):
        try:
            _emit_variant(work_rgb.filter(filt), label, 0.01)
        except Exception:
            continue
    try:
        _emit_variant(ImageOps.invert(work_rgb), "invert", 0.0)
        _emit_variant(ImageOps.posterize(work_rgb, 3), "posterize3", 0.0)
    except Exception:
        pass
    for scale in (1.25, 1.5, 1.75, 2.0):
        try:
            w = max(1, int(round(work_rgb.width * scale)))
            h = max(1, int(round(work_rgb.height * scale)))
            resized = work_rgb.resize((w, h), resample=Image.BICUBIC)
        except Exception:
            continue
        _emit_variant(resized, f"scale{scale:.2f}", 0.02)

    if np is not None:
        try:
            arr = np.asarray(work_gray, dtype=np.uint8)
        except Exception:
            arr = None
        if arr is not None and arr.size:
            height, width = arr.shape
            mu = float(arr.mean())
            sigma = float(arr.std())
            thresholds: Set[int] = set()
            if sigma > 1.0:
                for coeff in (0.8, 0.5, 0.25, -0.25):
                    thresholds.add(int(np.clip(mu - sigma * coeff, 16, 240)))
            try:
                percentiles = [float(np.percentile(arr, p)) for p in (10, 20, 35, 50, 70)]
            except Exception:
                percentiles = []
            for val in percentiles:
                thresholds.add(int(np.clip(val, 12, 244)))
            thresholds.add(int(np.clip(mu, 16, 240)))
            thresholds = {t for t in thresholds if 0 <= t <= 255}

            def _segment_words(mask: "np.ndarray", tag: str) -> None:
                if mask.ndim != 2:
                    return
                ink_cols = mask.sum(axis=0)
                gap_thr = max(1, int(mask.shape[0] * 0.12))
                segments: List[Tuple[int, int]] = []
                start: Optional[int] = None
                for x, val in enumerate(ink_cols):
                    if val > gap_thr:
                        if start is None:
                            start = max(0, x - 1)
                    elif start is not None:
                        segments.append((start, min(mask.shape[1], x + 1)))
                        start = None
                if start is not None:
                    segments.append((start, mask.shape[1]))
                segments = [seg for seg in segments if seg[1] - seg[0] > 2]
                if len(segments) <= 1:
                    return
                pieces: List[str] = []
                confs: List[float] = []
                for x0, x1 in segments[:8]:
                    try:
                        seg_crop = work_rgb.crop((x0, 0, x1, work_rgb.height))
                    except Exception:
                        continue
                    txt_seg, conf_seg = toy_ocr_text_from_cell(seg_crop)
                    if txt_seg:
                        pieces.append(txt_seg.strip())
                        confs.append(conf_seg)
                joined = " ".join([t for t in pieces if t])
                if not joined:
                    return
                conf_est = 0.0
                if confs:
                    conf_est = max(confs)
                    conf_est = max(conf_est, sum(confs) / max(1, len(confs)) + 0.04)
                _emit_manual(joined, conf_est, f"segment_{tag}")

            for thr in sorted(thresholds):
                try:
                    mask = (arr <= thr).astype(np.uint8)
                except Exception:
                    continue
                if not mask.any():
                    continue
                try:
                    bw = Image.fromarray((1 - mask) * 255, mode="L")
                except Exception:
                    continue
                _emit_variant(bw, f"threshold_{thr:03d}", 0.05)
                try:
                    inv = ImageOps.invert(bw)
                except Exception:
                    inv = None
                if inv is not None:
                    _emit_variant(inv, f"threshold_inv_{thr:03d}", 0.03)
                _segment_words(mask, f"thr{thr:03d}")

            if height > 4:
                proj = width // 4
                if proj > 0:
                    mask_rows = (arr < int(mu)).astype(np.uint8)
                    row_sum = mask_rows.sum(axis=1)
                    segments_row: List[Tuple[int, int]] = []
                    start_row: Optional[int] = None
                    for idx, val in enumerate(row_sum):
                        if val > proj:
                            if start_row is None:
                                start_row = max(0, idx - 1)
                        elif start_row is not None:
                            segments_row.append((start_row, min(height, idx + 1)))
                            start_row = None
                    if start_row is not None:
                        segments_row.append((start_row, height))
                    if len(segments_row) > 1:
                        texts: List[str] = []
                        confidences: List[float] = []
                        for y0, y1 in segments_row[:4]:
                            try:
                                strip = work_rgb.crop((0, y0, work_rgb.width, y1))
                            except Exception:
                                continue
                            txt, conf = toy_ocr_text_from_cell(strip)
                            if txt:
                                texts.append(txt.strip())
                                confidences.append(conf)
                        if texts:
                            joined = " ".join(t for t in texts if t)
                            if joined:
                                conf_guess = float(max(confidences) if confidences else 0.0)
                                conf_avg = float(sum(confidences) / max(1, len(confidences)))
                                conf_val = float(max(conf_guess, conf_avg + 0.05))
                                _emit_manual(joined, conf_val, "strip_join")

    return variants


def _collect_external_ocr_variants(img: "Image.Image") -> List[Tuple[str, float, str]]:
    variants: List[Tuple[str, float, str]] = []
    if not _pytesseract_allowed():
        return _synthetic_external_variants(img)
    tess_variants = _pytesseract_variants(img)
    variants.extend(tess_variants)
    if pytesseract is not None and tess_variants:
        try:
            inverted = ImageOps.invert(img.convert("RGB"))
        except Exception:
            inverted = None
        if inverted is not None:
            for txt, conf, transform in _pytesseract_variants(inverted):
                variants.append((txt, conf, f"{transform}+invert"))
    if pytesseract is None or not variants:
        variants.extend(_synthetic_external_variants(img))
    return variants

def _normalize_total_token(text: str) -> str:
    base = (text or "").lower()
    base = re.sub(r"[\s:_\-]+", "", base)
    return base

def _is_total_row(cells: List[str]) -> bool:
    joined = _normalize_total_token(" ".join(cells))
    if any(tok in joined for tok in _TOTAL_LABEL_HINTS):
        return True
    for cell in cells[:3]:
        norm = _normalize_total_token(cell)
        if not norm:
            continue
        if norm in _TOTAL_LABEL_HINTS:
            return True
        if any(norm.startswith(pref) for pref in _TOTAL_PREFIXES):
            return True
    return False


def _numeric_token_value(token: str) -> Optional[float]:
    if not token:
        return None
    body = token.replace(",", "")
    if body.endswith("%"):
        return None
    try:
        return float(body)
    except Exception:
        return None


def _relocate_total_amount(
    row: List[str], conf_row: List[float], target_idx: int
) -> bool:
    if target_idx < 0 or target_idx >= len(row):
        return False
    existing = _NUMERIC_RX.search(row[target_idx] or "")
    if existing and not (existing.group(0).endswith("%")):
        return False
    best_idx = None
    best_token = None
    best_score: Optional[Tuple[float, int]] = None
    for idx, cell in enumerate(row):
        if idx == target_idx:
            continue
        match = _NUMERIC_RX.search(cell or "")
        if not match:
            continue
        token = match.group(0)
        if token.endswith("%"):
            continue
        value = _numeric_token_value(token)
        magnitude = abs(value) if value is not None else 0.0
        score = (magnitude, idx)
        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx
            best_token = token
    if best_idx is None or best_token is None:
        return False
    row[target_idx] = best_token
    if best_idx < len(conf_row):
        conf_row[target_idx] = max(conf_row[target_idx], conf_row[best_idx])
    return True

def _resize_keep_ar(im, w, h):
    im = im.convert("L")
    iw, ih = im.size
    scale = min(max(1, w-2)/max(1, iw), max(1, h-2)/max(1, ih))
    tw, th = max(1, int(round(iw*scale))), max(1, int(round(ih*scale)))
    imr = im.resize((tw, th), resample=Image.BILINEAR)
    out = Image.new("L", (w,h), 0)
    out.paste(imr, ((w-tw)//2,(h-th)//2))
    return out


_TOKEN_TEMPLATE_SIZE = (96, 28)
_TOKEN_TEMPLATE_MAX_VARIANTS = 8
_TOKEN_TEMPLATE_PRESETS = [
    "item",
    "qty",
    "unit price",
    "amount",
    "total",
    "subtotal",
    "tax",
    "due",
    "date",
    "見積金額",
    "御見積金額",
    "数量",
    "単価",
    "金額",
    "小計",
    "合計",
]


def _load_template_font(size: int = 18) -> "ImageFont.ImageFont":
    font_path = os.environ.get("ZOCR_TEMPLATE_FONT")
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


_TOKEN_TEMPLATE_FONT = _load_template_font()


def _render_template_bitmap(token: str) -> Optional["Image.Image"]:
    text = (token or "").strip()
    if not text:
        return None
    font = _TOKEN_TEMPLATE_FONT
    try:
        bbox = font.getbbox(text)
    except Exception:
        bbox = None
    if bbox:
        width = max(12, bbox[2] - bbox[0] + 6)
        height = max(12, bbox[3] - bbox[1] + 6)
    else:
        width = max(12, len(text) * 8)
        height = 24
    img = Image.new("L", (width, height), 0)
    dr = ImageDraw.Draw(img)
    try:
        dr.text((3, 2), text, fill=255, font=font)
    except Exception:
        return None
    if not img.getbbox():
        return None
    return img


def _normalize_template_bitmap(arr: Any) -> Optional["np.ndarray"]:
    import numpy as _np

    try:
        img = Image.fromarray(_np.asarray(arr, dtype=_np.uint8), mode="L")
    except Exception:
        return None
    resized = _resize_keep_ar(img, _TOKEN_TEMPLATE_SIZE[0], _TOKEN_TEMPLATE_SIZE[1])
    arr_f = _np.asarray(resized, dtype=_np.float32)
    if arr_f.size == 0:
        return None
    std = float(arr_f.std())
    if std < 1e-3:
        return None
    normed = (arr_f - float(arr_f.mean())) / (std + 1e-3)
    return normed


def _init_token_template_library() -> Dict[str, deque]:
    library: Dict[str, deque] = {}
    for token in _TOKEN_TEMPLATE_PRESETS:
        bmp = _render_template_bitmap(token)
        if bmp is None:
            continue
        norm = _normalize_template_bitmap(bmp)
        if norm is None:
            continue
        dq = library.setdefault(token, deque(maxlen=_TOKEN_TEMPLATE_MAX_VARIANTS))
        dq.append(norm)
    return library


_TOKEN_TEMPLATE_LIBRARY: Dict[str, deque] = _init_token_template_library()

_TEMPLATE_CACHE_STATE: Dict[str, Any] = {
    "loaded_path": None,
    "loaded": False,
    "autosave": False,
    "dirty": False,
}


def _template_cache_path() -> Optional[str]:
    raw = os.environ.get("ZOCR_TEMPLATE_CACHE")
    if not raw:
        return None
    raw = raw.strip()
    if not raw or raw.lower() in {"0", "false", "none"}:
        return None
    return os.path.abspath(raw)


def _update_template_variant_stats() -> None:
    total = 0
    for dq in _TOKEN_TEMPLATE_LIBRARY.values():
        if dq:
            total += len(dq)
    _GLYPH_RUNTIME_STATS["template_cache_variants"] = float(total)


def _ensure_template_cache_autosave() -> None:
    if _TEMPLATE_CACHE_STATE.get("autosave"):
        return
    if not _template_cache_path():
        return
    atexit.register(_autosave_template_cache)
    _TEMPLATE_CACHE_STATE["autosave"] = True


def _ensure_template_cache_loaded() -> None:
    path = _template_cache_path()
    if not path:
        return
    state_path = _TEMPLATE_CACHE_STATE.get("loaded_path")
    if _TEMPLATE_CACHE_STATE.get("loaded") and state_path == path:
        return
    _load_token_template_cache(path)


def _load_token_template_cache(path: str) -> None:
    if np is None:
        return
    try:
        with open(path, "r", encoding="utf-8") as fr:
            payload = json.load(fr)
    except FileNotFoundError:
        _TEMPLATE_CACHE_STATE["loaded_path"] = path
        _TEMPLATE_CACHE_STATE["loaded"] = True
        _TEMPLATE_CACHE_STATE["dirty"] = False
        _ensure_template_cache_autosave()
        return
    except Exception:
        _GLYPH_RUNTIME_STATS["template_cache_errors"] += 1.0
        return
    restored = 0
    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if isinstance(tokens, dict):
        for token, entry in tokens.items():
            variants = []
            if isinstance(entry, dict):
                seq = entry.get("variants")
                if isinstance(seq, list):
                    variants = seq
            elif isinstance(entry, list):
                variants = entry
            for variant in variants:
                if not isinstance(variant, dict):
                    continue
                shape = variant.get("shape")
                data = variant.get("data")
                if not shape or not data:
                    continue
                try:
                    arr = np.asarray(data, dtype=np.float32)
                    dims = tuple(int(v) for v in shape)
                    if len(dims) != 2 or arr.size != (dims[0] * dims[1]):
                        continue
                    arr = arr.reshape(dims)
                except Exception:
                    continue
                dq = _TOKEN_TEMPLATE_LIBRARY.get(token)
                if dq is None or dq.maxlen != _TOKEN_TEMPLATE_MAX_VARIANTS:
                    dq = deque(dq or [], maxlen=_TOKEN_TEMPLATE_MAX_VARIANTS)
                    _TOKEN_TEMPLATE_LIBRARY[token] = dq
                dq.append(arr)
                restored += 1
    _TEMPLATE_CACHE_STATE["loaded_path"] = path
    _TEMPLATE_CACHE_STATE["loaded"] = True
    _TEMPLATE_CACHE_STATE["dirty"] = False
    _ensure_template_cache_autosave()
    _update_template_variant_stats()
    _GLYPH_RUNTIME_STATS["template_cache_loaded"] = float(restored)


def _serialize_template_variants(dq: "deque[Any]") -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for tpl in dq:
        try:
            arr = np.asarray(tpl, dtype=np.float32)
        except Exception:
            continue
        if arr.size == 0:
            continue
        payload = {
            "shape": [int(dim) for dim in arr.shape],
            "data": [round(float(val), 4) for val in arr.flatten().tolist()],
        }
        serialized.append(payload)
    return serialized


def _save_token_template_cache(path: str) -> None:
    if np is None:
        return
    tokens: Dict[str, Any] = {}
    total = 0
    for token, dq in _TOKEN_TEMPLATE_LIBRARY.items():
        if not dq:
            continue
        variants = _serialize_template_variants(dq)
        if not variants:
            continue
        tokens[token] = {"variants": variants}
        total += len(variants)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"version": 1, "tokens": tokens, "updated_at": time.time()}
    try:
        with open(path, "w", encoding="utf-8") as fw:
            json.dump(payload, fw, ensure_ascii=False)
    except Exception:
        _GLYPH_RUNTIME_STATS["template_cache_errors"] += 1.0
        return
    _TEMPLATE_CACHE_STATE["loaded_path"] = path
    _TEMPLATE_CACHE_STATE["loaded"] = True
    _TEMPLATE_CACHE_STATE["dirty"] = False
    _GLYPH_RUNTIME_STATS["template_cache_saved"] = float(total)
    _update_template_variant_stats()


def _autosave_template_cache() -> None:
    if not _TEMPLATE_CACHE_STATE.get("dirty"):
        return
    path = _template_cache_path()
    if not path:
        return
    _save_token_template_cache(path)


def _observe_token_template(token: str, arr: Any) -> None:
    if not token:
        return
    _ensure_template_cache_loaded()
    _ensure_template_cache_autosave()
    norm = _normalize_template_bitmap(arr)
    if norm is None:
        return
    dq = _TOKEN_TEMPLATE_LIBRARY.get(token)
    if dq is None or dq.maxlen != _TOKEN_TEMPLATE_MAX_VARIANTS:
        dq = deque(dq or [], maxlen=_TOKEN_TEMPLATE_MAX_VARIANTS)
        _TOKEN_TEMPLATE_LIBRARY[token] = dq
    dq.append(norm)
    _TEMPLATE_CACHE_STATE["dirty"] = True
    _update_template_variant_stats()


def _match_token_template_from_cache(arr: Any) -> Tuple[str, float]:
    import numpy as _np

    _ensure_template_cache_loaded()
    _ensure_template_cache_autosave()
    norm = _normalize_template_bitmap(arr)
    if norm is None:
        return "", 0.0
    best_token = ""
    best_score = -1.0
    _GLYPH_RUNTIME_STATS["template_cache_checks"] += 1.0
    for token, variants in _TOKEN_TEMPLATE_LIBRARY.items():
        if not variants:
            continue
        for tpl in variants:
            if tpl.shape != norm.shape:
                continue
            score = float((norm * tpl).mean())
            if score > best_score:
                best_score = score
                best_token = token
    if best_score < 0.35:
        _GLYPH_RUNTIME_STATS["template_cache_misses"] += 1.0
        return "", 0.0
    conf = float(max(0.0, min(1.0, (best_score + 1.0) / 2.0)))
    _GLYPH_RUNTIME_STATS["template_cache_hits"] += 1.0
    return best_token, conf


def _normalized_string_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    try:
        matcher = difflib.SequenceMatcher(a=a, b=b)
        ratio = matcher.ratio()
    except Exception:
        return 1.0 if a != b else 0.0
    return float(max(0.0, min(1.0, 1.0 - ratio)))


def _template_override_thresholds() -> Tuple[float, float, float, float]:
    def _get(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    strict_delta = _get("ZOCR_TEMPLATE_OVERRIDE_DELTA", 0.05)
    min_quality = _get("ZOCR_TEMPLATE_OVERRIDE_MIN_QUALITY", 0.6)
    flex_delta = _get("ZOCR_TEMPLATE_OVERRIDE_FLEX", 0.02)
    min_diff = _get("ZOCR_TEMPLATE_OVERRIDE_MIN_DIFF", 0.3)
    return float(strict_delta), float(min_quality), float(flex_delta), float(min_diff)


def _decide_template_override(
    final_text: str,
    final_effective_conf: float,
    final_quality: float,
    template_text: str,
    template_conf: float,
) -> str:
    if not template_text:
        return ""
    strict_delta, min_quality, flex_delta, min_diff = _template_override_thresholds()
    lexical = float(final_quality)
    if (not lexical) and final_text:
        lexical = _toy_text_quality(final_text)[0]
    if not final_text:
        return "missing"
    if template_conf >= final_effective_conf + strict_delta:
        return "conf"
    if lexical < min_quality and template_conf + flex_delta >= final_effective_conf:
        distance = _normalized_string_distance(final_text, template_text)
        if distance >= min_diff:
            return "quality"
    return ""

def _shift_normed(arr: "np.ndarray", dx: int, dy: int):
    if dx == 0 and dy == 0:
        return arr
    out = np.roll(arr, shift=(dy, dx), axis=(0, 1))
    if dy > 0:
        out[:dy, :] = 0
    elif dy < 0:
        out[dy:, :] = 0
    if dx > 0:
        out[:, :dx] = 0
    elif dx < 0:
        out[:, dx:] = 0
    return out

def _match_glyph(cell_bin, atlas, allowed_chars: Optional[Sequence[str]] = None):
    # try best correlation over atlas with light shift tolerance and feature penalties
    cw, ch = cell_bin.size
    import numpy as _np
    cell_arr = _np.asarray(cell_bin, dtype=_np.float32)
    if cell_arr.size == 0:
        return "", 0.0
    cell_norm = (cell_arr - cell_arr.mean()) / (cell_arr.std() + 1e-6)
    cell_density = float((cell_arr > 0).mean())
    cell_aspect = float(cw) / float(ch or 1)
    cell_scaled = cell_arr / 255.0 if cell_arr.max() > 1.5 else cell_arr
    if cell_scaled.ndim == 2 and cell_scaled.size:
        row_profile = cell_scaled.mean(axis=1)
        col_profile = cell_scaled.mean(axis=0)
        cell_style = float(_np.var(row_profile) + _np.var(col_profile))
    else:
        cell_style = 0.0
    allowed: Optional[Set[str]] = None
    if allowed_chars:
        allowed = {str(ch) for ch in allowed_chars if str(ch)}
    best_ch, best_score = "", -1.0
    for ch_key, tpl in atlas.items():
        if allowed is not None and ch_key not in allowed:
            continue
        tpl_list = tpl if isinstance(tpl, (list, tuple)) else [tpl]
        variant_best = -1.0
        for glyph_img in tpl_list:
            t = _resize_keep_ar(glyph_img, cw, ch)
            b = _np.asarray(t, dtype=_np.float32)
            if b.size == 0:
                continue
            b_norm = (b - b.mean()) / (b.std() + 1e-6)
            score = -1.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    shifted = _shift_normed(b_norm, dx, dy)
                    cand = float((cell_norm * shifted).mean())
                    if cand > score:
                        score = cand
            if score > variant_best:
                variant_best = score
        if variant_best < -0.5:
            continue
        feats = _GLYPH_FEATS.get(ch_key, {})
        glyph_aspect = feats.get("aspect", 1.0) or 1.0
        glyph_density = feats.get("density", 0.5)
        glyph_sym = feats.get("symmetry", 0.0)
        glyph_style = feats.get("style_var", 0.0)
        aspect_penalty = math.exp(-abs(math.log((cell_aspect + 1e-3)/(glyph_aspect + 1e-3))) * 0.75)
        density_penalty = 1.0 - min(0.4, abs(cell_density - glyph_density) * 1.6)
        if glyph_sym > 0.5:
            sym_cell = float(1.0 - _np.mean(_np.abs(cell_arr - _np.flip(cell_arr, axis=1))) / 255.0)
            symmetry_penalty = 0.8 + 0.2 * max(0.0, sym_cell)
        else:
            symmetry_penalty = 1.0
        style_penalty = 1.0 - min(0.35, abs(cell_style - glyph_style) * 0.8)
        variant_best *= aspect_penalty * max(0.4, density_penalty) * symmetry_penalty * max(0.45, style_penalty)
        if variant_best > best_score:
            best_score = variant_best
            best_ch = ch_key
    conf = (best_score + 1.0) / 2.0
    return (best_ch if conf >= 0.52 else "?"), float(conf)

def _otsu_threshold_toy(arr):
    import numpy as _np
    hist = _np.bincount(arr.reshape(-1), minlength=256).astype(_np.float64)
    total = hist.sum()
    if total <= 0:
        return int(_np.median(arr))
    sum_total = _np.dot(hist, _np.arange(256, dtype=_np.float64))
    weight_bg = 0.0
    sum_bg = 0.0
    var_max = -1.0
    threshold = int(_np.median(arr))
    for i in range(256):
        weight_bg += hist[i]
        if weight_bg <= 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg <= 0:
            break
        sum_bg += hist[i] * i
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = i
    return int(threshold)


def _component_projection_splits(arr: "np.ndarray", axis: int, gap_ratio: float, min_span_ratio: float) -> List[Tuple[int, int, int, int]]:
    if arr.size == 0:
        return []
    work = (arr > 0).astype(np.uint8)
    height, width = work.shape[:2]
    if axis == 1:
        density = work.sum(axis=0)
        span = width
        orth = height
    else:
        density = work.sum(axis=1)
        span = height
        orth = width
    gap_thr = max(1, int(round(gap_ratio * max(1, orth))))
    min_span = max(1, int(round(min_span_ratio * max(1, span))))
    segments: List[Tuple[int, int, int, int]] = []
    idx = 0
    while idx < span:
        while idx < span and density[idx] <= gap_thr:
            idx += 1
        if idx >= span:
            break
        start = idx
        while idx < span and density[idx] > gap_thr:
            idx += 1
        if idx - start >= min_span:
            if axis == 1:
                segments.append((start, 0, idx, height))
            else:
                segments.append((0, start, width, idx))
    return segments

@dataclass
class _BaselineStats:
    baseline: float
    xheight: float
    ascender: float
    descender: float


def _estimate_baseline_stats(boxes: Sequence[Tuple[int, int, int, int, float]]) -> Optional[_BaselineStats]:
    if not boxes:
        return None
    bottoms: List[float] = []
    heights: List[float] = []
    ascenders: List[float] = []
    descenders: List[float] = []
    for _, y1, _, y2, _ in boxes:
        h = float(max(1, y2 - y1))
        heights.append(h)
        bottoms.append(float(y2))
    if not heights or not bottoms:
        return None
    baseline = float(median(bottoms))
    xheight = float(max(4.0, median(heights)))
    for _, y1, _, y2, _ in boxes:
        bottom = float(y2)
        top = float(y1)
        ascenders.append(max(0.0, baseline - top))
        descenders.append(max(0.0, bottom - baseline))
    asc = float(max(3.0, median(ascenders) if ascenders else xheight))
    desc = float(max(1.0, median(descenders) if descenders else xheight * 0.2))
    return _BaselineStats(baseline=baseline, xheight=xheight, ascender=asc, descender=desc)


def _segment_component_bbox(
    bw: "np.ndarray", bbox: Tuple[int, int, int, int, float], baseline: Optional[_BaselineStats] = None
) -> List[Tuple[int, int, int, int, float]]:
    x1, y1, x2, y2, _ = bbox
    sub = bw[y1:y2, x1:x2]
    if sub.size == 0:
        return []
    try:
        arr = np.asarray(sub, dtype=np.uint8)
    except Exception:
        arr = sub
    height = max(1, arr.shape[0])
    width = max(1, arr.shape[1])
    segments: List[Tuple[int, int, int, int]] = []
    wide_ratio = 1.4
    tall_ratio = 1.8
    gap_ratio = 0.08
    if baseline:
        wide_ratio = 1.2
        tall_ratio = 1.5
        gap_ratio = 0.06
    if width > height * wide_ratio:
        segments = _component_projection_splits(arr, axis=1, gap_ratio=gap_ratio, min_span_ratio=0.22)
    if not segments:
        tall_gate = height > width * tall_ratio
        if baseline:
            tall_gate = height > baseline.xheight * 1.65
        if tall_gate:
            segments = _component_projection_splits(arr, axis=0, gap_ratio=gap_ratio, min_span_ratio=0.28)
    boxes: List[Tuple[int, int, int, int, float]] = []
    if segments:
        for sx1, sy1, sx2, sy2 in segments:
            gx1, gy1 = x1 + sx1, y1 + sy1
            gx2, gy2 = x1 + sx2, y1 + sy2
            area = float(max(1, (gx2 - gx1) * (gy2 - gy1)))
            boxes.append((gx1, gy1, gx2, gy2, area))
        _GLYPH_RUNTIME_STATS["baseline_splits"] += float(len(segments))
    return boxes

def _refine_component_segments(
    bw: "np.ndarray", bbox: Tuple[int, int, int, int, float], baseline: Optional[_BaselineStats] = None
) -> List[Tuple[int, int, int, int, float]]:
    queue = [bbox]
    output: List[Tuple[int, int, int, int, float]] = []
    steps = 0
    while queue and steps < 32:
        current = queue.pop()
        children = _segment_component_bbox(bw, current, baseline=baseline)
        if children:
            queue.extend(children)
        else:
            output.append(current)
        steps += 1
    return output

def _text_from_binary(bw, allowed_chars: Optional[Sequence[str]] = None):
    cc = _cc_label_rle(bw)
    cc = [b for b in cc if (b[2]-b[0])*(b[3]-b[1]) >= 10]
    if not cc:
        return "", 0.0
    baseline_stats = _estimate_baseline_stats(cc)
    refined: List[Tuple[int, int, int, int, float]] = []
    for bbox in cc:
        refined.extend(_refine_component_segments(bw, bbox, baseline=baseline_stats))
    if not refined:
        refined = cc
    refined.sort(key=lambda b: b[0])
    atlas = _GLYPH_ATLAS
    text = []
    scores = []
    for (x1, y1, x2, y2, _) in refined:
        sub = bw[y1:y2, x1:x2]
        if sub.size == 0:
            continue
        try:
            arr = np.asarray(sub, dtype=np.uint8)
        except Exception:
            arr = sub
        sig = _glyph_signature(arr)
        cached = _glyph_runtime_lookup(sig)
        if cached is not None:
            ch, sc = cached
        else:
            try:
                patch = Image.fromarray(arr, mode="L")
            except Exception:
                patch = Image.fromarray(sub)
            ch, sc = _match_glyph(patch, atlas, allowed_chars=allowed_chars)
            _glyph_runtime_store(sig, ch, sc)
            if not ch or ch == "?" or sc < 0.6:
                _glyph_pending_enqueue(sig, arr, sc)
        if ch and ch != "?" and sc > 0.6:
            try:
                patch_img = Image.fromarray(arr, mode="L")
            except Exception:
                patch_img = None
            if patch_img is not None:
                _adapt_glyph(ch, patch_img)
        text.append(ch)
        scores.append(sc)
    if not text:
        return "", 0.0
    raw = np.asarray(scores, dtype=np.float64)
    base = (raw + 1.0) * 0.5
    mean = base.mean() if base.size else 0.0
    spread = base.std() if base.size else 0.0
    adj = (mean - 0.55) / (0.12 + spread * 0.5)
    conf = 1.0 / (1.0 + math.exp(-adj)) if base.size else 0.0
    return "".join(text), float(conf)


def _shadow_correct_array(arr: "np.ndarray") -> "np.ndarray":
    if arr.size == 0:
        return arr
    try:
        img = Image.fromarray(arr, mode="L")
    except Exception:
        img = Image.fromarray(arr)
    try:
        blur = img.filter(ImageFilter.GaussianBlur(radius=3))
    except Exception:
        blur = img.filter(ImageFilter.BoxBlur(2))
    low = np.asarray(blur, dtype=np.float32)
    src = arr.astype(np.float32)
    normalized = src - low + 128.0
    normalized -= normalized.min()
    peak = normalized.max() or 1.0
    normalized = (normalized / peak) * 255.0
    return normalized.astype(np.uint8)

def toy_ocr_text_from_cell(
    crop_img: "Image.Image", bin_k: int = 15, allowed_chars: Optional[Sequence[str]] = None
) -> Tuple[str, float]:
    """Very small OCR to work with the demo font. Returns (text, confidence)."""
    import numpy as _np
    g = ImageOps.autocontrast(crop_img.convert("L"))
    g = g.filter(ImageFilter.MedianFilter(3))
    arr = _np.asarray(g, dtype=_np.uint8)
    cfg = current_toy_self_correction()
    if isinstance(cfg.get("bin_k_override"), (int, float)):
        try:
            bin_k = max(3, int(round(float(cfg["bin_k_override"]))))
        except Exception:
            pass
    elif isinstance(cfg.get("bin_k_scale"), (int, float)):
        try:
            scale = float(cfg.get("bin_k_scale"))
            bin_k = max(3, int(round(bin_k * scale)))
        except Exception:
            pass
    if arr.size == 0:
        return "", 0.0
    arr_f = arr.astype(_np.float32)
    try:
        gy, gx = _np.gradient(arr_f)
        edge_mag = _np.hypot(gx, gy)
        thr = edge_mag.mean() + edge_mag.std()
        mask = edge_mag > thr
    except Exception:
        mask = None
    if mask is not None and mask.any():
        edge_vals = arr[mask]
        bright_ratio = float(_np.mean(edge_vals > arr.mean())) if edge_vals.size else 0.0
        if bright_ratio > 0.6:
            arr = 255 - arr
            arr_f = arr.astype(_np.float32)
            g = Image.fromarray(arr, mode="L")
    thr_med = int(_np.clip(_np.median(arr), 48, 208))
    candidates: List[Tuple[np.ndarray, Dict[str, Any]]] = []
    seen_hashes: Set[Tuple[int, int, str]] = set()

    def _add_candidate(bw_arr: Any, meta: Dict[str, Any]) -> bool:
        try:
            arr_u8 = _np.asarray(bw_arr, dtype=_np.uint8)
        except Exception:
            return False
        if arr_u8.size == 0:
            return False
        digest = hashlib.sha1(arr_u8.tobytes()).hexdigest()
        sig = (int(arr_u8.shape[0]), int(arr_u8.shape[1]), digest)
        if sig in seen_hashes:
            return False
        seen_hashes.add(sig)
        candidates.append((arr_u8.copy(), meta))
        return True

    key = (int(arr.shape[0]), int(arr.shape[1]), int(float(arr.mean()) // 16))
    bw_med = (arr < thr_med).astype(_np.uint8) * 255
    _add_candidate(bw_med, {"type": "median", "thr": thr_med})
    thr_mem = _threshold_memory_lookup(key)
    if thr_mem is not None:
        _add_candidate((arr < thr_mem).astype(_np.uint8) * 255, {"type": "memory", "thr": thr_mem})
    try:
        blur = _np.asarray(Image.fromarray(arr).filter(ImageFilter.BoxBlur(max(1, bin_k//2))), dtype=_np.float32)
        adapt_dark = (arr + 8 < blur).astype(_np.uint8) * 255
        adapt_light = (arr - 8 > blur).astype(_np.uint8) * 255
        _add_candidate(adapt_dark, {"type": "adaptive_dark"})
        _add_candidate(adapt_light, {"type": "adaptive_light"})
    except Exception:
        pass
    thr_otsu = _otsu_threshold_toy(arr)
    _add_candidate((arr < thr_otsu).astype(_np.uint8) * 255, {"type": "otsu", "thr": thr_otsu})
    _add_candidate((arr > thr_otsu).astype(_np.uint8) * 255, {"type": "otsu_inv", "thr": thr_otsu})
    threshold_expand = int(cfg.get("threshold_expand", 0)) if isinstance(cfg, dict) else 0
    sweeps = int(cfg.get("threshold_sweeps", 0)) if isinstance(cfg, dict) else 0
    if sweeps <= 0:
        sweeps = _TOY_SWEEPS
    thr_min = max(16, thr_med - 60 - threshold_expand)
    thr_max = min(240, thr_med + 70 + threshold_expand)
    if thr_max <= thr_min:
        thr_max = min(240, thr_min + 6)
    sweep_values = [int(x) for x in _np.linspace(thr_min, thr_max, num=max(1, sweeps))]
    for thr_candidate in sweep_values:
        thr_val = int(_np.clip(thr_candidate, 16, 240))
        _add_candidate((arr < thr_val).astype(_np.uint8) * 255, {"type": "sweep_global", "thr": thr_val})
    spread_base = int(max(3, arr.std()))
    extra_spread = int(cfg.get("extra_local_spread", 0)) if isinstance(cfg, dict) else 0
    spread = max(3, min(24, spread_base + max(0, extra_spread)))
    local_offsets: List[int] = [-spread, 0, spread]
    if isinstance(cfg, dict) and isinstance(cfg.get("local_sweep_offsets"), (list, tuple)):
        for candidate in cfg.get("local_sweep_offsets", []):  # type: ignore[assignment]
            try:
                offset = int(round(float(candidate)))
            except Exception:
                continue
            if offset not in local_offsets:
                local_offsets.append(offset)
    for delta in local_offsets:
        thr_val = int(_np.clip(thr_med + delta, 16, 240))
        _add_candidate((arr < thr_val).astype(_np.uint8) * 255, {"type": "sweep_local", "thr": thr_val})

    try:
        shadow = _shadow_correct_array(arr)
        thr_shadow = _otsu_threshold_toy(shadow)
        _add_candidate((shadow < thr_shadow).astype(_np.uint8) * 255, {"type": "shadow_dark", "thr": thr_shadow})
        _add_candidate((shadow > thr_shadow).astype(_np.uint8) * 255, {"type": "shadow_light", "thr": thr_shadow})
    except Exception:
        pass

    candidate_scores: Dict[str, float] = {}
    candidate_actual_conf: Dict[str, float] = {}
    candidate_quality: Dict[str, float] = {}
    best_text, best_conf = "", 0.0
    best_meta: Optional[Dict[str, Any]] = None
    best_bw: Optional[np.ndarray] = None
    best_effective_conf = 0.0

    def _evaluate_from(idx: int) -> int:
        nonlocal best_text, best_conf, best_meta, best_bw, best_effective_conf
        total = len(candidates)
        for i in range(idx, total):
            bw, meta = candidates[i]
            text, conf = _text_from_binary(bw, allowed_chars=allowed_chars)
            if text:
                lexical_quality, _ = _toy_text_quality(text)
                candidate_quality[text] = max(candidate_quality.get(text, 0.0), float(lexical_quality))
                candidate_actual_conf[text] = max(candidate_actual_conf.get(text, 0.0), float(conf))
                effective_conf = float(conf) * float(lexical_quality or 1.0)
                effective_conf = float(max(0.0, min(1.5, effective_conf)))
                prev = candidate_scores.get(text)
                if prev is None or effective_conf > prev:
                    candidate_scores[text] = effective_conf
                if lexical_quality < 0.6:
                    _GLYPH_RUNTIME_STATS["lexical_penalty"] += 1.0
            else:
                lexical_quality = 0.0
                effective_conf = float(conf)
            if text and effective_conf > best_effective_conf:
                best_text, best_conf, best_meta, best_bw = text, float(conf), meta, bw
                best_effective_conf = effective_conf
        return total

    _evaluate_from(0)
    target_conf = float(cfg.get("target_confidence", 0.5)) if isinstance(cfg, dict) else 0.5
    force_augment = bool(cfg.get("force_augment")) if isinstance(cfg, dict) else False
    extra_augment_passes = int(cfg.get("extra_augment_passes", 0)) if isinstance(cfg, dict) else 0
    extra_augment_passes = max(0, extra_augment_passes)
    need_augment = force_augment or (best_conf < target_conf)
    augment_cycles = max(1, 1 + extra_augment_passes) if need_augment else extra_augment_passes
    for _ in range(augment_cycles):
        start_len = len(candidates)
        for bw_aug, meta_aug in _self_augment_views(arr, best_bw):
            _add_candidate(bw_aug, meta_aug)
        _evaluate_from(start_len)
        if not force_augment and best_conf >= target_conf:
            break

    if best_meta and best_meta.get("thr") is not None:
        _threshold_memory_store(key, int(best_meta["thr"]))

    final_text, final_conf = best_text, best_conf
    final_effective_conf = best_effective_conf
    final_quality = candidate_quality.get(final_text, 0.0)
    if candidate_scores:
        reranked_text, reranked_conf = _contextual_rerank_candidates(candidate_scores)
        if reranked_text:
            reranked_actual = candidate_actual_conf.get(reranked_text, reranked_conf)
            reranked_quality = candidate_quality.get(reranked_text, 0.0)
            if reranked_quality <= 0.0 and reranked_text:
                reranked_quality, _ = _toy_text_quality(reranked_text)
            if (
                reranked_conf > final_effective_conf + 1e-6
                or (
                    abs(reranked_conf - final_effective_conf) <= 1e-6
                    and reranked_actual > final_conf
                )
            ):
                final_text = reranked_text
                final_conf = float(reranked_actual)
                final_quality = reranked_quality
                final_effective_conf = reranked_conf
            elif not final_text:
                final_text = reranked_text
                final_conf = float(reranked_actual)
                final_quality = reranked_quality
                final_effective_conf = reranked_conf

    ref_bitmap = best_bw
    if ref_bitmap is None and candidates:
        ref_bitmap = candidates[0][0]
    template_text, template_conf = ("", 0.0)
    if ref_bitmap is not None:
        template_text, template_conf = _match_token_template_from_cache(ref_bitmap)
    override_reason = _decide_template_override(
        final_text, final_effective_conf, final_quality, template_text, template_conf
    )
    if override_reason:
        final_text = template_text
        final_conf = max(final_conf, template_conf)
        final_effective_conf = max(final_effective_conf, template_conf)
        final_quality = max(final_quality, 0.95)
        _GLYPH_RUNTIME_STATS["template_matches"] += 1.0
        _GLYPH_RUNTIME_STATS[f"template_override_{override_reason}"] += 1.0

    if final_text:
        coherence = _ngram_coherence(final_text)
        surprisal = _ngram_surprisal(final_text)
        if final_quality <= 0.0:
            final_quality, _ = _toy_text_quality(final_text)
        bounded_conf = float(max(0.0, min(1.0, final_conf)))
        _record_toy_recognition(
            final_text,
            bounded_conf,
            coherence,
            surprisal,
            lexical_quality=float(max(0.0, min(1.6, final_quality))),
        )
        _update_ngram_model(final_text)
        if ref_bitmap is not None:
            _observe_token_template(final_text, ref_bitmap)
            _GLYPH_RUNTIME_STATS["template_observed"] += 1.0
        return final_text, bounded_conf
    return "", 0.0


def _normalize_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except Exception:
        conf = 0.0
    if not math.isfinite(conf):
        conf = 0.0
    return float(max(0.0, min(1.0, conf)))


def _resolve_ocr_backend(name: Optional[str]) -> Callable[["Image.Image"], Tuple[str, float]]:
    normalized = (name or "toy").strip()
    cache_key = normalized.lower() or "toy"
    cached = _OCR_BACKEND_CACHE.get(cache_key)
    if cached is not None:
        return cached

    engine_name, _, engine_extra = normalized.partition(":")
    engine_key = (engine_name or "toy").strip().lower() or "toy"
    extra = engine_extra.strip()

    runner: Callable[["Image.Image"], Tuple[str, float]]

    if engine_key == "fast":

        def _fast_runner(img: "Image.Image") -> Tuple[str, float]:
            return "", 0.0

        runner = _fast_runner
    elif engine_key in ("toy", "mock", "demo"):
        runner = toy_ocr_text_from_cell
    elif engine_key.startswith("tess"):
        if not _pytesseract_allowed():
            if cache_key not in _OCR_BACKEND_WARNED:
                print("[WARN] pytesseract disabled by ZOCR_ALLOW_PYTESSERACT=0; using toy OCR", flush=True)
                _OCR_BACKEND_WARNED.add(cache_key)
            runner = toy_ocr_text_from_cell
        elif pytesseract is None or _PYTESS_OUTPUT is None:
            if cache_key not in _OCR_BACKEND_WARNED:
                print("[WARN] pytesseract not available; falling back to toy OCR")
                _OCR_BACKEND_WARNED.add(cache_key)
            runner = toy_ocr_text_from_cell
        else:
            lang = extra or os.environ.get("ZOCR_TESS_LANG", os.environ.get("TESS_LANG", "jpn+eng"))
            psm = os.environ.get("ZOCR_TESS_PSM", "6")
            oem = os.environ.get("ZOCR_TESS_OEM")
            config_parts = []
            if psm:
                config_parts.append(f"--psm {psm}")
            if oem:
                config_parts.append(f"--oem {oem}")
            tess_config = " ".join(config_parts)

            def _tesseract_runner(img: "Image.Image") -> Tuple[str, float]:
                try:
                    target = img if getattr(img, "mode", "") in ("L", "RGB") else img.convert("RGB")
                except Exception:
                    target = img
                data = _pytesseract_call(
                    "image_to_data",
                    pytesseract.image_to_data,  # type: ignore[arg-type]
                    target,
                    lang=lang or None,
                    config=tess_config or "",
                    output_type=_PYTESS_OUTPUT.DICT,
                )
                words: List[str] = []
                confs: List[float] = []
                if isinstance(data, dict):
                    texts = data.get("text") or []
                    conf_vals = data.get("conf") or []
                    for raw_txt, raw_conf in zip(texts, conf_vals):
                        txt = str(raw_txt or "").strip()
                        if not txt:
                            continue
                        try:
                            conf_val = float(raw_conf)
                        except Exception:
                            conf_val = -1.0
                        if conf_val < 0:
                            continue
                        words.append(txt)
                        confs.append(max(0.0, min(100.0, conf_val)))
                if not words:
                    raw = _pytesseract_call(
                        "image_to_string",
                        pytesseract.image_to_string,  # type: ignore[arg-type]
                        target,
                        lang=lang or None,
                        config=tess_config or "",
                    )
                    text_raw = (raw or "").strip()
                    if text_raw:
                        words.append(text_raw)
                        confs.append(62.0)
                if not words:
                    return "", 0.0
                conf_avg = (sum(confs) / len(confs)) / 100.0 if confs else 0.0
                return " ".join(words), _normalize_confidence(conf_avg)

            runner = _tesseract_runner
    elif engine_key.startswith("easy"):
        if np is None:
            if cache_key not in _OCR_BACKEND_WARNED:
                print("[WARN] easyocr requested but numpy is unavailable; falling back to toy OCR")
                _OCR_BACKEND_WARNED.add(cache_key)
            runner = toy_ocr_text_from_cell
        else:
            try:
                import easyocr  # type: ignore
            except Exception:
                if cache_key not in _OCR_BACKEND_WARNED:
                    print("[WARN] easyocr not available; falling back to toy OCR")
                    _OCR_BACKEND_WARNED.add(cache_key)
                runner = toy_ocr_text_from_cell
            else:
                langs_raw = extra or os.environ.get("ZOCR_EASYOCR_LANGS", "ja,en")
                langs = [tok.strip() for tok in langs_raw.split(",") if tok.strip()]
                if not langs:
                    langs = ["ja", "en"]
                gpu_flag = (os.environ.get("ZOCR_EASYOCR_GPU", "").strip().lower())
                gpu = gpu_flag not in ("0", "false", "no", "off")
                reader_key = (tuple(langs), gpu)
                reader = _EASYOCR_READER_CACHE.get(reader_key)
                if reader is None:
                    reader = easyocr.Reader(langs, gpu=gpu)
                    _EASYOCR_READER_CACHE[reader_key] = reader

                def _easyocr_runner(img: "Image.Image") -> Tuple[str, float]:
                    try:
                        target = img if getattr(img, "mode", "") == "RGB" else img.convert("RGB")
                        arr = np.asarray(target)
                    except Exception:
                        return "", 0.0
                    try:
                        results = reader.readtext(arr, detail=1)
                    except Exception:
                        return "", 0.0
                    texts: List[str] = []
                    confs: List[float] = []
                    for item in results:
                        if not isinstance(item, (list, tuple)):
                            continue
                        text_val = item[1] if len(item) > 1 else ""
                        conf_val = item[2] if len(item) > 2 else 0.0
                        txt = str(text_val or "").strip()
                        if not txt:
                            continue
                        texts.append(txt)
                        try:
                            conf_float = float(conf_val)
                        except Exception:
                            conf_float = 0.0
                        confs.append(max(0.0, min(1.0, conf_float)))
                    if not texts:
                        return "", 0.0
                    conf_avg = sum(confs) / len(confs) if confs else 0.0
                    return " ".join(texts), _normalize_confidence(conf_avg)

                runner = _easyocr_runner
    else:
        if cache_key not in _OCR_BACKEND_WARNED:
            print(f"[WARN] Unknown OCR engine '{name}', falling back to toy OCR")
            _OCR_BACKEND_WARNED.add(cache_key)
        runner = toy_ocr_text_from_cell

    _OCR_BACKEND_CACHE[cache_key] = runner
    return runner

def _keywords_from_row(row_cells: List[str]) -> List[str]:
    kws = set()
    rx_num = re.compile(r"[+\-]?\d[\d,]*(\.\d+)?")
    rx_date = re.compile(r"\b(20\d{2}|19\d{2})[/-](0?[1-9]|1[0-2])([/-](0?[1-9]|[12][0-9]|3[01]))?\b")
    for t in row_cells:
        if not t: continue
        for m in rx_num.findall(t): kws.add(m[0] if isinstance(m, tuple) else m)
        for m in rx_date.findall(t): kws.add("-".join([x for x in m if x]))
        if any(sym in t for sym in ["$", "¥", "円"]): kws.add("currency")
    return sorted(kws)[:12]

def _context_line_from_row(headers: List[str], row: List[str]) -> str:
    if headers and len(headers)==len(row):
        pairs = [f"{h.strip()}={row[i].strip()}" for i,h in enumerate(headers)]
        return " | ".join(pairs)
    else:
        return " | ".join([x.strip() for x in row if x.strip()])

def _conceptual_tags(text: str, headers: List[str], row: List[str]) -> List[str]:
    tags: Set[str] = set()
    t = (text or "").strip()
    if not t:
        return []
    if any(ch.isalpha() for ch in t):
        tags.add("alpha")
    if any(ch.isdigit() for ch in t):
        tags.add("digit")
    if re.search(r"\d{4}[-/](0?[1-9]|1[0-2])", t):
        tags.add("date")
    if re.search(r"[+\-]?\d+[,.]\d+", t):
        tags.add("decimal")
    if re.search(r"[€$¥円]", t):
        tags.add("currency_symbol")
    if t.isupper() and len(t) > 1:
        tags.add("upper")
    if t.islower() and len(t) > 1:
        tags.add("lower")
    if t.replace(" ", "").isdigit():
        tags.add("integer")
    normalized = re.sub(r"[^0-9A-Za-z]+", " ", t).strip().lower()
    if normalized:
        tags.add(f"lex:{normalized[:20]}")
    for h in headers:
        if not h:
            continue
        h_norm = h.strip().lower()
        if not h_norm:
            continue
        if any(tok in h_norm for tok in ("total", "amount", "balance")):
            tags.add("header:total")
        if "date" in h_norm:
            tags.add("header:date")
        if any(tok in h_norm for tok in ("tax", "vat")):
            tags.add("header:tax")
    if row:
        joined = " ".join([c for c in row if c])
        if len(joined) > 6 and joined == joined.upper():
            tags.add("row:upper_span")
        if re.search(r"\bsubtotal\b", joined.lower()):
            tags.add("row:subtotal")
    return sorted(tags)


def _hypothesize_from_text(text: str, headers: List[str], concepts: List[str]) -> List[Dict[str, Any]]:
    if not text:
        text = ""
    hyps: List[Dict[str, Any]] = []
    base = text.strip()
    if base:
        hyps.append({"text": base, "strategy": "observed", "score": 0.6})
    compact = re.sub(r"\s+", "", base)
    if compact and compact != base:
        hyps.append({"text": compact, "strategy": "compact", "score": 0.45})
    digits = re.sub(r"[^0-9]", "", base)
    if digits and digits != compact:
        hyps.append({"text": digits, "strategy": "digits_only", "score": 0.4})
    if base and base.upper() != base:
        hyps.append({"text": base.upper(), "strategy": "upper", "score": 0.35})
    if base and base.lower() != base:
        hyps.append({"text": base.lower(), "strategy": "lower", "score": 0.35})
    if "header:date" in concepts and digits:
        y, rest = digits[:4], digits[4:]
        if len(rest) >= 4:
            hyps.append({"text": f"{y}-{rest[:2]}-{rest[2:4]}", "strategy": "date_template", "score": 0.5})
    if "decimal" in concepts and digits:
        if len(digits) > 2:
            hyps.append({"text": f"{digits[:-2]}.{digits[-2:]}", "strategy": "decimal_guess", "score": 0.55})
    dedup: Dict[str, Dict[str, Any]] = {}
    for hyp in hyps:
        key = hyp.get("text") or ""
        if key not in dedup or dedup[key].get("score", 0) < hyp.get("score", 0):
            dedup[key] = hyp
    return sorted(dedup.values(), key=lambda h: -float(h.get("score", 0.0)))[:6]


def export_jsonl_with_ocr(doc_json_path: str,
                          source_images: Union[str, Sequence[str], Mapping[int, str]],
                          out_jsonl_path: str,
                          ocr_engine: str = "toy", contextual: bool = True,
                          ocr_min_conf: float = 0.58) -> int:
    with open(doc_json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    page_lookup: Dict[int, str] = {}
    default_image_path: Optional[str]
    if isinstance(source_images, str):
        default_image_path = source_images
    elif isinstance(source_images, Mapping):
        for key, value in source_images.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, str):
                page_lookup[idx] = value
        default_image_path = page_lookup.get(min(page_lookup.keys())) if page_lookup else None
    else:
        seq = [p for p in source_images if isinstance(p, str)]
        page_lookup = {i: path for i, path in enumerate(seq)}
        default_image_path = seq[0] if seq else None
    image_cache: Dict[str, Image.Image] = {}
    ocr_runner = _resolve_ocr_backend(ocr_engine)

    motion_cfg = _motion_prior_cfg_from_env()
    sweep_tracker = _EXPORT_SWEEP_TRACKER
    motion_applied = 0
    motion_rejected = 0

    progress_flag = os.environ.get("ZOCR_EXPORT_PROGRESS", "0").strip().lower()
    log_progress = progress_flag not in {"", "0", "false", "no"}

    def _parse_env_int(name: str, default: int, minimum: int = 0) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        return max(minimum, value)

    log_every = max(1, _parse_env_int("ZOCR_EXPORT_LOG_EVERY", 200, 1))
    flush_every = max(0, _parse_env_int("ZOCR_EXPORT_FLUSH_EVERY", 200, 0))
    max_cells = _parse_env_int("ZOCR_EXPORT_MAX_CELLS", 0, 0)
    cells_done = 0
    t0 = time.time()
    stop_due_to_limit = False
    guard_ms = _parse_env_int("ZOCR_EXPORT_GUARD_MS", 0, 0)
    guard_timeouts = 0
    blank_cfg = _blank_skip_cfg_from_env()
    blank_skipped = 0
    confidence_boost_cfg = _confidence_boost_cfg_from_env()
    lexical_boost_cfg = _lexical_boost_cfg_from_env()
    confidence_boost_applied = 0
    confidence_boost_delta = 0.0
    confidence_boost_reasons: Dict[str, int] = defaultdict(int)
    lexical_boost_applied = 0
    lexical_boost_delta = 0.0
    lexical_boost_reasons: Dict[str, int] = defaultdict(int)
    lexical_quality_sum = 0.0
    lexical_quality_samples = 0
    schema_tables = 0
    schema_noise_columns = 0
    schema_rows_adjusted = 0
    schema_cells_salvaged = 0
    schema_cells_cleared = 0
    total_rows_seen = 0
    total_rows_reflowed = 0
    total_rows_ocr_attempts = 0
    total_rows_ocr_success = 0

    doc_identifier = str(
        doc.get("doc_id")
        or doc.get("document_id")
        or doc.get("id")
        or os.path.splitext(os.path.basename(doc_json_path))[0]
    )

    pages = doc.get("pages") if isinstance(doc, dict) else None
    if not isinstance(pages, list):
        pages = []

    def _candidate_paths(path: Optional[str], index: Optional[int]) -> List[str]:
        ordered: List[str] = []
        seen: Set[str] = set()
        for cand in (path, page_lookup.get(index) if index is not None else None, default_image_path):
            if isinstance(cand, str) and cand and cand not in seen:
                seen.add(cand)
                ordered.append(cand)
        return ordered

    def _load_page_image(path: Optional[str], index: Optional[int]) -> Optional["Image.Image"]:
        for target in _candidate_paths(path, index):
            if target in image_cache:
                return image_cache[target]
            try:
                with Image.open(target) as img:
                    loaded = img.convert("RGB")
            except Exception:
                continue
            image_cache[target] = loaded
            return loaded
        return None

    count = 0
    learning_signals: List[Dict[str, Any]] = []
    low_conf_samples = 0
    surprisal_samples = 0
    stats_start = time.time()
    pages_processed = 0
    tables_processed = 0
    total_cells = 0
    forced_cells = 0
    numeric_tables = 0
    numeric_tables_inferred = 0
    numeric_columns_total = 0
    numeric_columns_inferred = 0
    numeric_columns_by_kind: Counter = Counter()
    forced_fields: Counter = Counter()
    date_tables = 0
    date_columns_total = 0
    date_columns_by_role: Counter = Counter()
    date_cells_detected = 0
    date_cells_by_role: Counter = Counter()
    date_precision_counts: Counter = Counter()
    surprisal_threshold = (
        float(_NGRAM_SURPRISAL_REVIEW_THRESHOLD)
        if _NGRAM_SURPRISAL_REVIEW_THRESHOLD > 0.0
        else 0.0
    )
    if log_progress:
        est_cells = 0
        for p_est in pages:
            if not isinstance(p_est, dict):
                continue
            for t_est in p_est.get("tables", []) or []:
                dbg_est = t_est.get("dbg", {}) if isinstance(t_est, dict) else {}
                col_bounds = dbg_est.get("col_bounds", []) if isinstance(dbg_est, dict) else []
                cols = max(1, len(col_bounds) - 1)
                baselines = dbg_est.get("baselines_segs", []) if isinstance(dbg_est, dict) else []
                rows = max(2, len(baselines)) if baselines else 2
                row_bands_rel = dbg_est.get("row_bands_rel") if isinstance(dbg_est, dict) else None
                if isinstance(row_bands_rel, list) and row_bands_rel:
                    rows = max(rows, len(row_bands_rel))
                est_cells += max(1, rows) * max(1, cols)
        print(f"[Export] engine={ocr_engine} pages={len(pages)} ~cells={est_cells}", flush=True)

    with open(out_jsonl_path, "w", encoding="utf-8") as fw:
        records_written = 0
        for enum_idx, p in enumerate(pages):
            if stop_due_to_limit:
                break
            pages_processed += 1
            pidx = p.get("index")
            page_image_path = p.get("image_path")
            lookup_idx: Optional[int]
            if isinstance(pidx, int):
                lookup_idx = pidx
            else:
                lookup_idx = enum_idx
            page_index_int = int(lookup_idx) if lookup_idx is not None else int(enum_idx)
            page_image = _load_page_image(page_image_path, lookup_idx)
            if page_image is None:
                continue
            page_w, page_h = page_image.size
            tables = p.get("tables", []) if isinstance(p, dict) else []
            for ti, t in enumerate(tables):
                if stop_due_to_limit:
                    break
                if not isinstance(t, dict):
                    continue
                tables_processed += 1
                guard_deadline = (time.time() + guard_ms / 1000.0) if guard_ms > 0 else None
                guard_triggered = False
                prev_keys: Optional[List[float]] = None
                if motion_cfg.enabled:
                    prev_keys = sweep_tracker.get(doc_identifier, page_index_int, ti)
                x1,y1,x2,y2 = t["bbox"]
                dbg = t.get("dbg", {})
                col_bounds = dbg.get("col_bounds", [0, (x2-x1)//2, x2-x1])
                C = max(1, len(col_bounds)-1)
                baselines = list(dbg.get("baselines_segs", []) or [])
                # rows: prefer reconstruction bands if available
                row_bands = []
                rel_bands = dbg.get("row_bands_rel") or []
                if isinstance(rel_bands, list) and rel_bands:
                    H = max(1, y2 - y1)
                    for rt, rb in rel_bands:
                        try:
                            fr = float(rt)
                            to = float(rb)
                        except (TypeError, ValueError):
                            continue
                        fr = max(0.0, min(float(H), fr))
                        to = max(fr, min(float(H), to))
                        row_bands.append((int(y1 + fr), int(y1 + to)))
                if not row_bands:
                    R = max(2, len(baselines)) or 2
                    for r in range(R):
                        yt = int(y1 + (y2-y1)*r/R)
                        yb = int(y1 + (y2-y1)*(r+1)/R)
                        row_bands.append((yt, yb))
                if motion_cfg.enabled:
                    row_bands_prior, applied = _apply_motion_prior_to_bands(prev_keys, row_bands, motion_cfg)
                    if applied:
                        row_bands = row_bands_prior
                        motion_applied += 1
                    elif prev_keys:
                        motion_rejected += 1
                R = len(row_bands)
                if len(baselines) < R:
                    baselines.extend([[] for _ in range(R - len(baselines))])
                elif len(baselines) > R:
                    baselines = baselines[:R]
                table_w = max(1, x2 - x1)
                table_h = max(1, y2 - y1)
                pad_edge_x = max(2, int(round(table_w * 0.04)))
                pad_inner_x = max(1, int(round(table_w * 0.01)))
                pad_y = max(1, int(round(table_h * 0.02)))
                # OCR pass across grid
                grid_text = [["" for _ in range(C)] for __ in range(R)]
                grid_conf = [[0.0 for _ in range(C)] for __ in range(R)]
                col_charset_hints: List[Optional[str]] = []
                toy_runner = ocr_runner is toy_ocr_text_from_cell
                for r in range(R):
                    if guard_deadline and time.time() >= guard_deadline:
                        guard_triggered = True
                        break
                    for c in range(C):
                        if guard_deadline and time.time() >= guard_deadline:
                            guard_triggered = True
                            break
                        total_cells += 1
                        cx1 = x1 + col_bounds[c]
                        cx2 = x1 + col_bounds[c+1]
                        left_pad = pad_edge_x if c == 0 else pad_inner_x
                        right_pad = pad_edge_x if c == C-1 else pad_inner_x
                        cy1, cy2 = row_bands[r]
                        crop = page_image.crop((
                            max(0, cx1 - left_pad),
                            max(0, cy1 - pad_y),
                            min(page_w, cx2 + right_pad),
                            min(page_h, cy2 + pad_y)
                        ))
                        allowed_chars = None
                        if r > 0 and col_charset_hints and c < len(col_charset_hints):
                            allowed_chars = col_charset_hints[c]
                        if blank_cfg.enabled and _should_skip_blank_crop(crop, blank_cfg):
                            blank_skipped += 1
                            txt, conf = "", 0.0
                        elif allowed_chars and toy_runner:
                            txt, conf = toy_ocr_text_from_cell(crop, allowed_chars=allowed_chars)
                        else:
                            txt, conf = ocr_runner(crop)
                        if not isinstance(txt, str):
                            txt = "" if txt is None else str(txt)
                        grid_text[r][c] = txt
                        grid_conf[r][c] = _normalize_confidence(conf)
                        cells_done += 1
                        if log_progress and (cells_done % log_every == 0):
                            dt = time.time() - t0
                            print(f"[Export] {cells_done} cells ({dt:.1f}s)", flush=True)
                        if max_cells and cells_done >= max_cells:
                            stop_due_to_limit = True
                            break
                    if r == 0 and not col_charset_hints:
                        headers_sample = grid_text[0] if grid_text else []
                        col_charset_hints = _column_charset_hints(headers_sample)
                    if guard_triggered or stop_due_to_limit:
                        break
                if guard_triggered:
                    guard_timeouts += 1
                    print(
                        f"[WARN] [Export] guard timeout (page={page_index_int}, table={ti})",
                        flush=True,
                    )
                    continue
                if stop_due_to_limit:
                    break
                if motion_cfg.enabled:
                    sweep_tracker.put(
                        doc_identifier,
                        page_index_int,
                        ti,
                        _row_band_midpoints(row_bands),
                    )
                schema_adjust = _rectify_item_qty_amount_schema(grid_text, grid_conf, col_bounds)
                if schema_adjust:
                    grid_text, grid_conf, col_bounds, schema_meta = schema_adjust
                    C = max(1, len(col_bounds) - 1)
                    schema_tables += 1
                    schema_noise_columns += int(schema_meta.get("noise_columns", 0))
                    schema_rows_adjusted += int(schema_meta.get("rows_adjusted", 0))
                    schema_cells_salvaged += int(schema_meta.get("cells_salvaged", 0))
                    schema_cells_cleared += int(schema_meta.get("cells_cleared", 0))
                footer_rows: Set[int] = set()
                fallback_notes: Dict[Tuple[int, int], str] = {}
                for r in range(R):
                    if _is_total_row(grid_text[r]):
                        total_rows_seen += 1
                        footer_rows.add(r)
                        target_col = C - 1 if C > 0 else 0
                        if C > 0 and _relocate_total_amount(grid_text[r], grid_conf[r], target_col):
                            fallback_notes[(r, target_col)] = "total_realign"
                            total_rows_reflowed += 1
                        has_numeric = any(_NUMERIC_RX.search(grid_text[r][c] or "") for c in range(C))
                        if not has_numeric and C > 0:
                            total_rows_ocr_attempts += 1
                            target_col = C-1
                            cy1, cy2 = row_bands[r]
                            cx1 = x1 + col_bounds[target_col]
                            cx2 = x1 + col_bounds[target_col+1] + pad_edge_x * 2
                            crop = page_image.crop((
                                max(0, cx1 - pad_inner_x),
                                max(0, cy1 - pad_y),
                                min(page_w, cx2),
                                min(page_h, cy2 + pad_y)
                            ))
                            alt_txt, alt_conf = ocr_runner(crop)
                            m = _NUMERIC_RX.search(alt_txt or "")
                            if m:
                                grid_text[r][target_col] = m.group(0)
                                grid_conf[r][target_col] = max(
                                    grid_conf[r][target_col], _normalize_confidence(alt_conf)
                                )
                                fallback_notes[(r, target_col)] = "footer_band"
                                total_rows_reflowed += 1
                                total_rows_ocr_success += 1
                # contextual one-liners
                headers = grid_text[0] if grid_text else []
                header_fields = _numeric_header_kinds(headers, grid_text)
                inferred_columns = _NUMERIC_HEADER_INFERRED_LAST
                date_roles = _date_header_roles(headers)
                if date_roles and any(date_roles):
                    date_tables += 1
                    for role in date_roles:
                        if role:
                            date_columns_total += 1
                            date_columns_by_role[role] += 1
                if header_fields:
                    table_numeric_cols = sum(1 for kind in header_fields if kind)
                    if table_numeric_cols:
                        numeric_tables += 1
                        numeric_columns_total += table_numeric_cols
                        if inferred_columns:
                            numeric_tables_inferred += 1
                            numeric_columns_inferred += inferred_columns
                        for kind in header_fields:
                            if kind:
                                numeric_columns_by_kind[kind] += 1
                if contextual:
                    _enforce_numeric_by_headers(headers, grid_text)
                for r in range(R):
                    for c in range(C):
                        cx1 = x1 + col_bounds[c]; cx2 = x1 + col_bounds[c+1]
                        cy1, cy2 = row_bands[r]
                        txt = grid_text[r][c]
                        conf = grid_conf[r][c]
                        raw_txt = txt
                        forced_filters: Dict[str, Any] = {}
                        boost_marker = ""
                        col_kind = header_fields[c] if header_fields and c < len(header_fields) else None
                        date_role = date_roles[c] if date_roles and c < len(date_roles) else None
                        if col_kind and r > 0:
                            coerced_txt, forced_filters = _coerce_numeric_filters(col_kind, txt)
                            if coerced_txt and coerced_txt != txt:
                                txt = coerced_txt
                                grid_text[r][c] = coerced_txt
                            elif coerced_txt:
                                txt = coerced_txt
                        lexical_quality, lexical_diag = _toy_text_quality(txt)
                        lexical_quality_sum += float(lexical_quality)
                        lexical_quality_samples += 1
                        boost_reason = _confidence_boost_reason(forced_filters)
                        if boost_reason and conf is not None:
                            boosted_conf, delta = _apply_confidence_boost(conf, confidence_boost_cfg)
                            if boosted_conf is not None and delta > 0.0:
                                conf = float(boosted_conf)
                                grid_conf[r][c] = float(boosted_conf)
                                confidence_boost_applied += 1
                                confidence_boost_delta += float(delta)
                                confidence_boost_reasons[boost_reason] += 1
                                boost_marker = boost_reason
                        lexical_reason = _lexical_confidence_reason(
                            txt, lexical_diag, lexical_boost_cfg.min_quality, lexical_quality
                        )
                        if lexical_reason and conf is not None:
                            boosted_conf, delta = _apply_confidence_boost(conf, lexical_boost_cfg)
                            if boosted_conf is not None and delta > 0.0:
                                conf = float(boosted_conf)
                                grid_conf[r][c] = float(boosted_conf)
                                lexical_boost_applied += 1
                                lexical_boost_delta += float(delta)
                                lexical_boost_reasons[lexical_reason] += 1
                                boost_marker = (
                                    f"{boost_marker},{lexical_reason}" if boost_marker else lexical_reason
                                )
                        # build search/synthesis
                        row_texts = grid_text[r]
                        ctx_line = _context_line_from_row(headers, row_texts) if contextual and r>0 else txt
                        kws = _keywords_from_row(row_texts) if contextual and r>0 else []
                        low_conf = (conf is not None and conf < ocr_min_conf)
                        coherence = _ngram_coherence(txt) if txt else 0.0
                        surprisal = _ngram_surprisal(txt) if txt else 0.0
                        review_reasons: List[str] = []
                        if low_conf:
                            review_reasons.append("low_conf")
                            low_conf_samples += 1
                        high_surprisal = bool(
                            surprisal_threshold > 0.0 and surprisal >= surprisal_threshold
                        )
                        if high_surprisal:
                            review_reasons.append("high_surprisal")
                            surprisal_samples += 1
                        trace_id = f"page={pidx},table={ti},row={r},col={c}"
                        has_currency = ("currency" in kws) or any(
                            sym in (raw_txt or "") for sym in ["¥", "円", "$"]
                        )
                        filters = {
                            "has_currency": has_currency,
                            "row_index": r,
                            "col_index": c,
                            "trace_id": trace_id
                        }
                        if r in footer_rows:
                            filters["row_role"] = "footer"
                        note = fallback_notes.get((r, c))
                        if forced_filters:
                            forced_cells += 1
                            for forced_key in forced_filters:
                                forced_fields[forced_key] += 1
                            filters.update(forced_filters)
                            filters["force_numeric"] = True
                        if boost_marker:
                            filters["confidence_boost"] = boost_marker
                        if r > 0:
                            date_payload = _extract_date_filters(txt, date_role)
                        else:
                            date_payload = None
                        if date_payload:
                            filters.update(date_payload)
                            date_cells_detected += 1
                            role_key = date_payload.get("date_role") or (date_role or "unlabeled")
                            date_cells_by_role[role_key] += 1
                            precision_key = date_payload.get("date_precision") or "unknown"
                            date_precision_counts[precision_key] += 1
                        if note:
                            filters["linked"] = note
                        concepts = _conceptual_tags(txt, headers, row_texts)
                        hypotheses: List[Dict[str, Any]] = []
                        if low_conf:
                            hypotheses = _hypothesize_from_text(txt, headers, concepts)
                        needs_review = bool(review_reasons)
                        rec = {
                            "type": "cell",
                            "doc_id": doc.get("doc_id"),
                            "page": pidx, "table_index": ti, "row": r, "col": c,
                            "bbox": [int(cx1), int(cy1), int(cx2), int(cy2)],
                            "image_path": page_image_path or default_image_path,
                            "text": txt,
                            "search_unit": (txt or ctx_line),
                            "synthesis_window": ctx_line,
                            "meta": {
                                "headers": headers,
                                "keywords": kws,
                                "confidence": conf,
                                "low_conf": bool(low_conf),
                                "ngram_coherence": float(coherence),
                                "ngram_surprisal": float(surprisal),
                                "lexical_quality": float(lexical_quality),
                                "trace": trace_id,
                                "filters": filters
                            }
                        }
                        if boost_marker:
                            rec["meta"]["confidence_boost"] = boost_marker
                        if lexical_reason:
                            rec["meta"]["lexical_reason"] = lexical_reason
                        if concepts:
                            rec["meta"]["concepts"] = concepts
                        if hypotheses:
                            rec["meta"]["hypotheses"] = hypotheses
                            needs_review = True
                        if review_reasons:
                            rec["meta"]["review_reasons"] = review_reasons
                        if needs_review or hypotheses:
                            rec["meta"]["needs_review"] = True
                            signal_reasons = list(review_reasons)
                            if hypotheses and "hypothesis" not in signal_reasons:
                                signal_reasons.append("hypothesis")
                            signal = {
                                "trace_id": trace_id,
                                "page": pidx,
                                "table_index": ti,
                                "row": r,
                                "col": c,
                                "bbox": [int(cx1), int(cy1), int(cx2), int(cy2)],
                                "table_bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "image_path": page_image_path or default_image_path,
                                "row_text": row_texts,
                                "headers": headers,
                                "observed_text": txt,
                                "confidence": conf,
                                "concepts": concepts,
                                "intent": "reanalyze_cell",
                                "ngram_coherence": float(coherence),
                                "ngram_surprisal": float(surprisal),
                                "coherence": float(coherence),
                                "surprisal": float(surprisal),
                                "reasons": signal_reasons or (["hypothesis"] if hypotheses else []),
                            }
                            if hypotheses:
                                signal["hypotheses"] = hypotheses
                            learning_signals.append(signal)
                        if note:
                            rec["meta"]["fallback"] = note
                        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        records_written += 1
                        if flush_every and (records_written % flush_every == 0):
                            fw.flush()
                        count += 1
                if log_progress:
                    fw.flush()
            if stop_due_to_limit:
                break
        if log_progress:
            print(f"[Export] done: {count} records", flush=True)
    signals_path = out_jsonl_path + ".signals.json"
    learn_path = out_jsonl_path + ".learning.jsonl"
    summary_payload: Dict[str, Any] = {
        "total_records": count,
        "learning_samples": len(learning_signals),
        "learning_jsonl": learn_path if learning_signals else None,
        "low_conf_samples": int(low_conf_samples),
        "high_surprisal_samples": int(surprisal_samples),
        "low_conf_ratio": float(low_conf_samples / float(max(1, count))),
        "high_surprisal_ratio": float(surprisal_samples / float(max(1, count))),
        "review_ratio": float(len(learning_signals) / float(max(1, count))),
    }
    if surprisal_threshold > 0.0:
        summary_payload["surprisal_threshold"] = float(surprisal_threshold)
    try:
        with open(signals_path, "w", encoding="utf-8") as fw_sig:
            json.dump(summary_payload, fw_sig, ensure_ascii=False, indent=2)
    except Exception:
        pass
    if learning_signals:
        try:
            with open(learn_path, "w", encoding="utf-8") as fw_learn:
                for sig in learning_signals:
                    fw_learn.write(json.dumps(sig, ensure_ascii=False) + "\n")
        except Exception:
            pass
    elif os.path.exists(learn_path):
        try:
            os.remove(learn_path)
        except Exception:
            pass
    duration = time.time() - stats_start if stats_start else 0.0
    numeric_stats = {
        "tables": int(numeric_tables),
        "tables_with_inferred": int(numeric_tables_inferred),
        "columns": int(numeric_columns_total),
        "columns_inferred": int(numeric_columns_inferred),
        "columns_by_kind": dict(numeric_columns_by_kind),
        "forced_cells": int(forced_cells),
        "forced_fields": dict(forced_fields),
    }
    runtime_state = toy_runtime_config()
    export_stats = {
        "ocr_engine": ocr_engine,
        "records": int(count),
        "pages": int(pages_processed),
        "tables": int(tables_processed),
        "cells_total": int(total_cells),
        "duration_sec": round(duration, 3),
        "numeric": numeric_stats,
        "toy_runtime": runtime_state,
        "force_numeric": bool(_FORCE_NUMERIC),
        "flush_every": int(flush_every),
    }
    if date_cells_detected or date_columns_total:
        export_stats["date_fields"] = {
            "tables": int(date_tables),
            "columns": int(date_columns_total),
            "columns_by_role": dict(date_columns_by_role),
            "cells": int(date_cells_detected),
            "cells_by_role": dict(date_cells_by_role),
            "precision": dict(date_precision_counts),
        }
    if blank_cfg.enabled:
        export_stats["blank_skip"] = {
            "skipped": int(blank_skipped),
            "ratio": float(blank_skipped / float(max(1, total_cells))),
            "dark_threshold": int(blank_cfg.dark_threshold),
            "min_dark_ratio": float(blank_cfg.min_dark_ratio),
            "min_dark_pixels": int(blank_cfg.min_dark_pixels),
        }
    if confidence_boost_applied:
        export_stats["confidence_boost"] = {
            "cells": int(confidence_boost_applied),
            "delta_sum": round(confidence_boost_delta, 4),
            "reasons": dict(confidence_boost_reasons),
            "target": float(confidence_boost_cfg.target),
            "min_input": float(confidence_boost_cfg.min_input),
        }
    if lexical_boost_applied:
        export_stats["lexical_boost"] = {
            "cells": int(lexical_boost_applied),
            "delta_sum": round(lexical_boost_delta, 4),
            "reasons": dict(lexical_boost_reasons),
            "target": float(lexical_boost_cfg.target),
            "min_input": float(lexical_boost_cfg.min_input),
            "min_quality": float(lexical_boost_cfg.min_quality),
        }
    if lexical_quality_samples:
        export_stats["lexical_quality"] = {
            "samples": int(lexical_quality_samples),
            "mean": round(lexical_quality_sum / float(max(1, lexical_quality_samples)), 4),
        }
    if guard_ms > 0:
        export_stats["guard"] = {
            "timeout_ms": int(guard_ms),
            "timeouts": int(guard_timeouts),
        }
    if schema_tables:
        export_stats["schema_alignment"] = {
            "tables": int(schema_tables),
            "noise_columns": int(schema_noise_columns),
            "rows_adjusted": int(schema_rows_adjusted),
            "cells_salvaged": int(schema_cells_salvaged),
            "cells_cleared": int(schema_cells_cleared),
        }
    if total_rows_seen:
        export_stats["total_rows"] = {
            "rows": int(total_rows_seen),
            "reflowed": int(total_rows_reflowed),
            "ocr_attempts": int(total_rows_ocr_attempts),
            "ocr_success": int(total_rows_ocr_success),
        }
    if motion_cfg.enabled:
        export_stats["motion_prior"] = {
            "sigma_px": float(motion_cfg.sigma_px),
            "cutoff_sigma": float(motion_cfg.cutoff_sigma),
            "accept_ratio": float(motion_cfg.accept_ratio),
            "applied": int(motion_applied),
            "rejected": int(motion_rejected),
        }
    global _LAST_EXPORT_STATS
    _LAST_EXPORT_STATS = export_stats
    return count


def _prepare_reanalysis_focus(focus: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Set[str]]]]:
    if not isinstance(focus, dict):
        return None, None
    plan: Dict[str, Any] = {}
    filters: Dict[str, Set[str]] = {}

    def _collect(key: str, limit: int) -> None:
        raw = focus.get(key)
        if not isinstance(raw, (list, tuple, set)):
            return
        ordered: List[str] = []
        seen: Set[str] = set()
        for item in raw:
            text = str(item)
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
            if len(ordered) >= limit:
                break
        if ordered:
            plan[key] = ordered
            filters[key] = set(ordered)

    _collect("trace_ids", 512)
    _collect("row_keys", 128)
    _collect("table_keys", 64)
    _collect("reasons", 24)
    coverage = focus.get("coverage_ratio")
    if coverage is not None:
        try:
            plan["coverage_ratio"] = float(coverage)
        except Exception:
            pass
    if focus.get("story"):
        plan["story"] = str(focus.get("story"))
    if focus.get("source"):
        plan["source"] = str(focus.get("source"))
    limit_value = focus.get("limit")
    if isinstance(limit_value, int):
        plan["limit"] = int(limit_value)
    return (plan if plan else None), (filters if any(filters.values()) else None)


def _focus_signal_match(sig: Dict[str, Any], filters: Optional[Dict[str, Set[str]]]) -> bool:
    if not filters:
        return True
    trace = str(sig.get("trace_id") or "")
    if filters.get("trace_ids") and trace in filters.get("trace_ids", set()):
        return True
    page = sig.get("page")
    table_idx = sig.get("table_index")
    row_idx = sig.get("row")
    table_key = f"page={page};table={table_idx}"
    row_key = f"{table_key};row={row_idx}"
    if filters.get("row_keys") and row_key in filters.get("row_keys", set()):
        return True
    if filters.get("table_keys") and table_key in filters.get("table_keys", set()):
        return True
    reasons = [str(r) for r in sig.get("reasons", []) if isinstance(r, str)]
    if filters.get("reasons") and any(reason in filters.get("reasons", set()) for reason in reasons):
        return True
    return False


def reanalyze_learning_jsonl(learning_jsonl_path: str,
                             out_dir: Optional[str] = None,
                             limit: int = 64,
                             rotate: bool = True,
                             ocr_engine: str = "toy",
                             focus: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Re-run the selected OCR backend over low-confidence cells to propose improved readings."""
    summary: Dict[str, Any] = {
        "input": learning_jsonl_path,
        "limit": int(limit),
        "output_jsonl": None,
        "summary_path": None,
        "total_seen": 0,
        "processed": 0,
        "skipped": 0,
        "improved": 0,
        "regressed": 0,
        "unchanged": 0,
        "avg_confidence_delta": 0.0,
        "external_engines": {},
        "ambiguous_variants": 0,
        "fallback_variant_count": 0,
        "fallback_transform_usage": {},
        "ocr_engine": ocr_engine,
    }
    active_cfg = current_toy_self_correction()
    if active_cfg:
        summary["toy_self_correction"] = _normalize_self_correction_config(active_cfg)
    if not learning_jsonl_path or not os.path.exists(learning_jsonl_path):
        summary["error"] = "learning_jsonl_missing"
        return summary

    ocr_runner = _resolve_ocr_backend(ocr_engine)

    focus_plan, focus_filters = _prepare_reanalysis_focus(focus)
    focus_stats = {"matched": 0, "skipped": 0} if focus_filters else None
    if focus_plan:
        summary["focus_plan"] = focus_plan

    dest_dir = out_dir or os.path.dirname(learning_jsonl_path) or "."
    ensure_dir(dest_dir)
    base = os.path.basename(learning_jsonl_path)
    if base.endswith(".jsonl"):
        base = base[:-6]
    output_jsonl = os.path.join(dest_dir, f"{base}.reanalyzed.jsonl")
    summary_path = output_jsonl + ".summary.json"

    image_cache: Dict[str, Image.Image] = {}

    def _load_image(path: Optional[str]) -> Optional["Image.Image"]:
        if not path:
            return None
        if path in image_cache:
            return image_cache[path]
        try:
            with Image.open(path) as img:
                loaded = img.convert("RGB")
        except Exception:
            return None
        image_cache[path] = loaded
        return loaded

    records: List[Dict[str, Any]] = []
    delta_sum = 0.0
    engine_usage: Dict[str, int] = defaultdict(int)
    ambiguous_total = 0
    fallback_used = False
    fallback_transform_usage: Dict[str, int] = defaultdict(int)
    reason_counts: Dict[str, int] = defaultdict(int)
    surprisal_sum = 0.0
    surprisal_count = 0
    surprisal_min: Optional[float] = None
    surprisal_max: Optional[float] = None

    try:
        with open(learning_jsonl_path, "r", encoding="utf-8") as fr:
            for raw_line in fr:
                line = raw_line.strip()
                if not line:
                    continue
                if limit and limit > 0 and summary["total_seen"] >= limit:
                    break
                try:
                    sig = json.loads(line)
                except Exception:
                    summary["skipped"] += 1
                    continue
                if focus_filters and not _focus_signal_match(sig, focus_filters):
                    if focus_stats is not None:
                        focus_stats["skipped"] += 1
                    continue
                if focus_stats is not None:
                    focus_stats["matched"] += 1
                raw_reasons = sig.get("reasons")
                if isinstance(raw_reasons, list):
                    for reason in raw_reasons:
                        if isinstance(reason, str) and reason:
                            reason_counts[reason] += 1
                sig_surprisal = sig.get("ngram_surprisal", sig.get("surprisal"))
                if sig_surprisal is not None:
                    try:
                        val = float(sig_surprisal)
                    except Exception:
                        val = None
                    if val is not None and math.isfinite(val):
                        surprisal_sum += val
                        surprisal_count += 1
                        surprisal_min = val if surprisal_min is None else min(surprisal_min, val)
                        surprisal_max = val if surprisal_max is None else max(surprisal_max, val)
                summary["total_seen"] += 1
                bbox = sig.get("bbox")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    summary["skipped"] += 1
                    continue
                try:
                    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
                except Exception:
                    summary["skipped"] += 1
                    continue
                page_path = sig.get("image_path")
                page_img = _load_image(page_path)
                if page_img is None:
                    summary["skipped"] += 1
                    continue
                pw, ph = page_img.size
                x1 = max(0, min(pw, x1))
                y1 = max(0, min(ph, y1))
                x2 = max(0, min(pw, x2))
                y2 = max(0, min(ph, y2))
                if x2 <= x1 or y2 <= y1:
                    summary["skipped"] += 1
                    continue
                crop = page_img.crop((x1, y1, x2, y2))

                observed_text = sig.get("observed_text")
                observed_conf = _normalize_confidence(sig.get("confidence"))

                variants_map: Dict[str, Dict[str, Any]] = {}

                def _merge_variant(text: Any, conf: Any, transform: str) -> None:
                    nonlocal ambiguous_total, fallback_used
                    if isinstance(text, str):
                        normalized_text = text
                    elif text is None:
                        normalized_text = ""
                    else:
                        normalized_text = str(text)
                    key = normalized_text
                    conf_f = _normalize_confidence(conf)
                    created = key not in variants_map
                    if created:
                        variants_map[key] = {
                            "text": normalized_text,
                            "confidence": conf_f,
                            "transforms": [transform],
                        }
                    else:
                        rec = variants_map[key]
                        if conf_f > rec.get("confidence", 0.0):
                            rec["confidence"] = conf_f
                        transforms = rec.setdefault("transforms", [])
                        if transform not in transforms:
                            transforms.append(transform)
                    rec = variants_map[key]
                    if transform.startswith("engine:"):
                        engine_name = transform.split(":", 1)[1] if ":" in transform else "unknown"
                        engine_name = engine_name.split("+", 1)[0]
                        engine_name = engine_name.split("/", 1)[0]
                        engine_name = engine_name.split("(", 1)[0]
                        base_name = engine_name or "unknown"
                        engine_usage[base_name] += 1
                        if base_name.startswith("faux_tess"):
                            fallback_used = True
                            if transform.startswith("engine:faux_tess/"):
                                label = transform.split("/", 1)[1] if "/" in transform else ""
                                label = label.split("+", 1)[0]
                                label = label or "unknown"
                                fallback_transform_usage[label] += 1
                    elif transform.startswith("ambiguous") and created:
                        ambiguous_total += 1

                base_text_raw, base_conf_raw = ocr_runner(crop)
                base_conf = _normalize_confidence(base_conf_raw)
                _merge_variant(base_text_raw, base_conf, "base")
                if isinstance(base_text_raw, str):
                    base_text = base_text_raw
                elif base_text_raw is None:
                    base_text = ""
                else:
                    base_text = str(base_text_raw)

                try:
                    bright_enhancer = ImageEnhance.Brightness(crop)
                    txt_b1, conf_b1 = ocr_runner(bright_enhancer.enhance(1.1))
                    _merge_variant(txt_b1, conf_b1, "brightness_1.1")
                    txt_b2, conf_b2 = ocr_runner(bright_enhancer.enhance(0.9))
                    _merge_variant(txt_b2, conf_b2, "brightness_0.9")
                except Exception:
                    pass
                try:
                    contrast_enhancer = ImageEnhance.Contrast(crop)
                    txt_c, conf_c = ocr_runner(contrast_enhancer.enhance(1.2))
                    _merge_variant(txt_c, conf_c, "contrast_1.2")
                except Exception:
                    pass
                for txt_ext, conf_ext, transform_ext in _collect_external_ocr_variants(crop):
                    _merge_variant(txt_ext, conf_ext, transform_ext)
                seen_ambiguous: Set[str] = set()
                for candidate in _ambiguous_variants(observed_text) + _ambiguous_variants(base_text):
                    if not candidate or candidate in seen_ambiguous:
                        continue
                    seen_ambiguous.add(candidate)
                    conf_guess = max(observed_conf, base_conf, 0.52)
                    _merge_variant(candidate, conf_guess, "ambiguous_map")
                if rotate:
                    for angle in (-2.5, -1.0, 1.0, 2.5):
                        try:
                            rotated = crop.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
                        except Exception:
                            continue
                        text_r, conf_r = ocr_runner(rotated)
                        _merge_variant(text_r, conf_r, f"rotate_{angle:+.1f}")

                if observed_text:
                    _merge_variant(observed_text, observed_conf, "observed")
                hypotheses = sig.get("hypotheses")
                if isinstance(hypotheses, list):
                    for hyp in hypotheses:
                        if isinstance(hyp, dict) and hyp.get("text"):
                            try:
                                hyp_score = float(hyp.get("score")) if hyp.get("score") is not None else 0.0
                            except Exception:
                                hyp_score = 0.0
                            _merge_variant(hyp.get("text"), max(observed_conf, hyp_score), "hypothesis")

                variants = sorted(variants_map.values(), key=lambda rec: rec.get("confidence", 0.0), reverse=True)
                if not variants:
                    summary["skipped"] += 1
                    continue
                best = variants[0]
                best_conf = float(best.get("confidence", 0.0))
                best_text = best.get("text")
                delta = best_conf - observed_conf
                delta_sum += delta

                same_text = (best_text or "") == (observed_text or "")
                improved_flag = False
                regressed_flag = False
                if same_text:
                    if best_conf > observed_conf + 0.01:
                        improved_flag = True
                else:
                    if best_conf >= observed_conf + 0.05:
                        improved_flag = True
                    elif best_conf + 0.05 < observed_conf:
                        regressed_flag = True

                if improved_flag:
                    summary["improved"] += 1
                elif regressed_flag:
                    summary["regressed"] += 1
                else:
                    summary["unchanged"] += 1

                record = {
                    "trace_id": sig.get("trace_id"),
                    "page": sig.get("page"),
                    "table_index": sig.get("table_index"),
                    "row": sig.get("row"),
                    "col": sig.get("col"),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "image_path": page_path,
                    "observed_text": observed_text,
                    "observed_confidence": observed_conf,
                    "reanalyzed_text": best_text,
                    "reanalyzed_confidence": best_conf,
                    "confidence_delta": delta,
                    "improved": bool(improved_flag),
                    "regressed": bool(regressed_flag),
                    "variants": variants[:6],
                    "headers": sig.get("headers"),
                    "row_text": sig.get("row_text"),
                    "table_bbox": sig.get("table_bbox"),
                    "concepts": sig.get("concepts"),
                    "hypotheses": hypotheses,
                }
                records.append(record)
                summary["processed"] += 1

    except Exception as exc:
        summary["error"] = f"reanalyze_failed: {exc}"
        records = []

    summary["ambiguous_variants"] = int(ambiguous_total)
    summary["external_engines"] = {k: int(v) for k, v in sorted(engine_usage.items()) if v}
    summary["used_external_fallback"] = bool(fallback_used)
    summary["fallback_variant_count"] = int(sum(fallback_transform_usage.values()))
    if fallback_transform_usage:
        ordered = sorted(fallback_transform_usage.items(), key=lambda kv: (-kv[1], kv[0]))
        summary["fallback_transform_usage"] = {k: int(v) for k, v in ordered}
    else:
        summary["fallback_transform_usage"] = {}
    if reason_counts:
        summary["reason_counts"] = {k: int(v) for k, v in sorted(reason_counts.items())}
    if surprisal_count > 0:
        summary["input_surprisal"] = {
            "avg": float(surprisal_sum / float(max(1, surprisal_count))),
            "min": float(surprisal_min) if surprisal_min is not None else None,
            "max": float(surprisal_max) if surprisal_max is not None else None,
            "count": int(surprisal_count),
        }

    if records:
        summary["output_jsonl"] = output_jsonl
        summary["summary_path"] = summary_path
        summary["avg_confidence_delta"] = delta_sum / max(1, summary["processed"])
        with open(output_jsonl, "w", encoding="utf-8") as fw_out:
            for rec in records:
                fw_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(summary_path, "w", encoding="utf-8") as fw_sum:
            json.dump(_json_ready(summary), fw_sum, ensure_ascii=False, indent=2)
    else:
        summary["avg_confidence_delta"] = 0.0

    if focus_stats is not None:
        summary["focus_stats"] = focus_stats

    return summary


def apply_reanalysis_to_jsonl(
    contextual_jsonl_path: str,
    reanalyzed_jsonl_path: str,
    dest_jsonl_path: Optional[str] = None,
    ocr_min_conf: float = 0.58,
    surprisal_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Rewrite contextual JSONL records with improved readings from reanalysis."""
    summary: Dict[str, Any] = {
        "input_jsonl": contextual_jsonl_path,
        "reanalyzed_jsonl": reanalyzed_jsonl_path,
        "output_jsonl": dest_jsonl_path or contextual_jsonl_path,
        "written": False,
        "copied": False,
        "base_records": 0,
        "matched_records": 0,
        "applied_records": 0,
        "text_changed": 0,
        "confidence_improved": 0,
        "confidence_regressed": 0,
        "confidence_unchanged": 0,
        "avg_confidence_delta": 0.0,
        "low_conf_cleared": 0,
        "low_conf_new": 0,
        "high_surprisal_cleared": 0,
        "high_surprisal_new": 0,
    }
    if not contextual_jsonl_path or not os.path.exists(contextual_jsonl_path):
        summary["error"] = "contextual_missing"
        return summary
    if not reanalyzed_jsonl_path or not os.path.exists(reanalyzed_jsonl_path):
        summary["error"] = "reanalyzed_missing"
        return summary

    try:
        with open(reanalyzed_jsonl_path, "r", encoding="utf-8") as fr:
            updates: Dict[str, Dict[str, Any]] = {}
            for raw in fr:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                trace = rec.get("trace_id") or rec.get("meta", {}).get("trace")
                if not trace:
                    continue
                if trace in updates:
                    # keep the best confidence if duplicates appear
                    try:
                        prev_conf = float(updates[trace].get("reanalyzed_confidence") or 0.0)
                    except Exception:
                        prev_conf = 0.0
                    try:
                        new_conf = float(rec.get("reanalyzed_confidence") or 0.0)
                    except Exception:
                        new_conf = prev_conf
                    if new_conf < prev_conf:
                        continue
                updates[trace] = rec
    except Exception as exc:
        summary["error"] = f"reanalyzed_read_error: {exc}"
        return summary

    if not updates:
        summary["error"] = "reanalyzed_empty"
        return summary

    summary["matched_records"] = len(updates)

    dest = dest_jsonl_path or contextual_jsonl_path
    ensure_dir(os.path.dirname(dest) or ".")
    tmp_dest = dest + ".tmp"

    if surprisal_threshold is None:
        surprisal_threshold = (
            float(_NGRAM_SURPRISAL_REVIEW_THRESHOLD)
            if _NGRAM_SURPRISAL_REVIEW_THRESHOLD is not None
            else 0.0
        )

    applied = 0
    changed = 0
    improved = 0
    regressed = 0
    unchanged = 0
    delta_sum = 0.0
    low_conf_cleared = 0
    low_conf_new = 0
    high_surprisal_cleared = 0
    high_surprisal_new = 0
    base_records = 0

    try:
        with open(contextual_jsonl_path, "r", encoding="utf-8") as fr, \
             open(tmp_dest, "w", encoding="utf-8") as fw:
            for raw in fr:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                base_records += 1
                meta = rec.get("meta")
                if not isinstance(meta, dict):
                    meta = {}
                    rec["meta"] = meta
                trace = meta.get("trace")
                if not trace and isinstance(meta.get("filters"), dict):
                    trace = meta["filters"].get("trace_id")
                info = updates.get(trace) if trace else None
                if info:
                    applied += 1
                    try:
                        new_conf = float(info.get("reanalyzed_confidence") or 0.0)
                    except Exception:
                        new_conf = 0.0
                    try:
                        old_conf = float(meta.get("confidence") or 0.0)
                    except Exception:
                        old_conf = 0.0
                    delta = new_conf - old_conf
                    delta_sum += delta
                    if delta > 0.01:
                        improved += 1
                    elif delta < -0.01:
                        regressed += 1
                    else:
                        unchanged += 1
                    old_text = rec.get("text")
                    new_text = info.get("reanalyzed_text", old_text)
                    if new_text != old_text:
                        changed += 1
                    rec["text"] = new_text
                    if rec.get("search_unit") == old_text or not rec.get("search_unit"):
                        rec["search_unit"] = new_text
                    if rec.get("synthesis_window") == old_text or not rec.get("synthesis_window"):
                        rec["synthesis_window"] = new_text
                    meta.setdefault("reanalysis", {})
                    meta["reanalysis"] = {
                        "trace_id": trace,
                        "text_before": old_text,
                        "text_after": new_text,
                        "confidence_before": old_conf,
                        "confidence_after": new_conf,
                        "delta": delta,
                        "improved": bool(info.get("improved")),
                        "regressed": bool(info.get("regressed")),
                        "source": info.get("source"),
                    }
                    meta["confidence"] = new_conf
                    prev_low = bool(meta.get("low_conf"))
                    new_low = bool(new_conf < ocr_min_conf)
                    if prev_low and not new_low:
                        low_conf_cleared += 1
                    elif (not prev_low) and new_low:
                        low_conf_new += 1
                    meta["low_conf"] = new_low
                    coherence = _ngram_coherence(new_text) if new_text else 0.0
                    surprisal = _ngram_surprisal(new_text) if new_text else 0.0
                    meta["ngram_coherence"] = float(coherence)
                    meta["ngram_surprisal"] = float(surprisal)
                    review_reasons = meta.get("review_reasons")
                    if isinstance(review_reasons, list):
                        filtered = [
                            str(r) for r in review_reasons
                            if isinstance(r, str) and r not in {"low_conf", "high_surprisal"}
                        ]
                    else:
                        filtered = []
                    if new_low:
                        filtered.append("low_conf")
                    high_surprisal_flag = bool(surprisal_threshold and surprisal >= float(surprisal_threshold))
                    prev_high = False
                    if isinstance(review_reasons, list):
                        prev_high = any(r == "high_surprisal" for r in review_reasons if isinstance(r, str))
                    if prev_high and not high_surprisal_flag:
                        high_surprisal_cleared += 1
                    if high_surprisal_flag and not prev_high:
                        high_surprisal_new += 1
                    if high_surprisal_flag:
                        filtered.append("high_surprisal")
                    if filtered:
                        meta["review_reasons"] = filtered
                        meta["needs_review"] = True
                    else:
                        meta.pop("review_reasons", None)
                        meta["needs_review"] = bool(meta.get("hypotheses"))
                fw.write(json.dumps(_json_ready(rec), ensure_ascii=False) + "\n")
    except Exception as exc:
        summary["error"] = f"contextual_rewrite_failed: {exc}"
        try:
            if os.path.exists(tmp_dest):
                os.remove(tmp_dest)
        except Exception:
            pass
        return summary

    if not applied:
        try:
            os.remove(tmp_dest)
        except Exception:
            pass
        summary["base_records"] = base_records
        summary["applied_records"] = 0
        return summary

    try:
        os.replace(tmp_dest, dest)
    except Exception as exc:
        summary["error"] = f"contextual_replace_failed: {exc}"
        try:
            os.remove(tmp_dest)
        except Exception:
            pass
        return summary

    summary.update({
        "output_jsonl": dest,
        "written": True,
        "base_records": base_records,
        "applied_records": applied,
        "text_changed": changed,
        "confidence_improved": improved,
        "confidence_regressed": regressed,
        "confidence_unchanged": unchanged,
        "avg_confidence_delta": delta_sum / float(max(1, applied)),
        "low_conf_cleared": low_conf_cleared,
        "low_conf_new": low_conf_new,
        "high_surprisal_cleared": high_surprisal_cleared,
        "high_surprisal_new": high_surprisal_new,
    })

    def _write_signals(path: str) -> None:
        try:
            existing: Dict[str, Any]
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fr:
                    payload = json.load(fr)
                    existing = payload if isinstance(payload, dict) else {}
            else:
                existing = {}
        except Exception:
            existing = {}
        existing.setdefault("applied_reanalysis", {})
        existing["applied_reanalysis"] = {
            "applied_records": int(applied),
            "text_changed": int(changed),
            "avg_confidence_delta": float(summary["avg_confidence_delta"]),
            "low_conf_cleared": int(low_conf_cleared),
            "high_surprisal_cleared": int(high_surprisal_cleared),
        }
        with open(path, "w", encoding="utf-8") as fw:
            json.dump(_json_ready(existing), fw, ensure_ascii=False, indent=2)

    signals_src = contextual_jsonl_path + ".signals.json"
    signals_dest = dest + ".signals.json"
    try:
        if os.path.abspath(signals_dest) == os.path.abspath(signals_src):
            _write_signals(signals_dest)
        else:
            if os.path.exists(signals_src):
                shutil.copyfile(signals_src, signals_dest)
            _write_signals(signals_dest)
            summary["copied"] = True
    except Exception:
        summary.setdefault("warnings", []).append("signals_update_failed")

    return summary

# ---------- Minimal local hybrid search ----------
def _tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\-\._]+", " ", s)
    return [t for t in s.split() if t]

def _bm25_build(jsonl_path: str):
    import math
    D = []; df = {}; N=0; avgdl=0.0
    with open(jsonl_path,"r",encoding="utf-8") as f:
        for line in f:
            N+=1
            ob = json.loads(line)
            txt = ob.get("search_unit") or ob.get("text") or ""
            toks = _tokenize(txt)
            D.append({"id": N-1, "len": len(toks), "toks": toks, "raw": ob})
            for t in set(toks):
                df[t] = df.get(t,0)+1
            avgdl += len(toks)
    avgdl = avgdl / max(1,N)
    return {"D": D, "df": df, "N": N, "avgdl": avgdl}

def _bm25_query(ix, q: str, k1=1.2, b=0.75, topk=20):
    q_toks = _tokenize(q)
    df = ix["df"]; N=ix["N"]; avgdl=ix["avgdl"]
    scores = []
    for doc in ix["D"]:
        dl = doc["len"]; toks = doc["toks"]
        tf = {}
        for t in toks: tf[t] = tf.get(t,0)+1
        s=0.0
        for t in q_toks:
            if t not in df: continue
            idf = math.log( (N - df[t] + 0.5) / (df[t] + 0.5) + 1.0 )
            f = tf.get(t,0)
            s += idf * ( (f*(k1+1)) / (f + k1*(1 - b + b*dl/max(1,avgdl))) )
        if s>0: scores.append((s, doc))
    scores.sort(key=lambda x: -x[0])
    return scores[:topk]

def _img_embed64_from_bbox(ob, down=16):
    # downsample region to tiny vector
    p = ob.get("image_path")
    if not p or not os.path.exists(p): return None
    img = Image.open(p).convert("L")
    x1,y1,x2,y2 = ob.get("bbox",[0,0,img.width,img.height])
    crop = img.crop((x1,y1,x2,y2)).resize((down,down), resample=Image.BICUBIC)
    import numpy as _np
    v = _np.asarray(crop, dtype=_np.float32).reshape(-1)
    v = (v - v.mean())/(v.std()+1e-6)
    return v

def _cos(a,b):
    import numpy as _np
    d = float((a*b).sum())
    na = float((_np.square(a).sum())**0.5); nb=float((_np.square(b).sum())**0.5)
    return d/max(1e-6,na*nb)

def _img_search(jsonl_path: str, query_img_path: str, topk=20):
    # image query: downscale query img to vector, compare cosine
    import numpy as _np
    qv = _img_embed64_from_bbox({"image_path": query_img_path, "bbox":[0,0,Image.open(query_img_path).size[0], Image.open(query_img_path).size[1]]})
    if qv is None: return []
    scores=[]
    with open(jsonl_path,"r",encoding="utf-8") as f:
        for i,line in enumerate(f):
            ob = json.loads(line)
            dv = _img_embed64_from_bbox(ob)
            if dv is None: continue
            s = _cos(qv,dv)
            scores.append((s, i, ob))
    scores.sort(key=lambda x: -x[0])
    return scores[:topk]

def _rrf_merge(listA, listB, k=60, topk=10):
    # listA: [(score, doc)], listB: [(score, doc or (idx,doc))]
    rank = {}
    def add_list(lst, is_img=False):
        for r, tup in enumerate(lst, start=1):
            s, obj = tup[0], (tup[-1] if is_img else tup[1])
            key = json.dumps(obj.get("bbox", []) + [obj.get("page"), obj.get("table_index")])
            rank.setdefault(key, {"obj": obj, "score": 0.0})
            rank[key]["score"] += 1.0/(k + r)
    add_list(listA, is_img=False)
    add_list(listB, is_img=True)
    merged = list(rank.values())
    merged.sort(key=lambda x: -x["score"])
    return merged[:topk]

def build_local_index(jsonl_path: str, out_pkl: str):
    ix = _bm25_build(jsonl_path)
    with open(out_pkl, "wb") as f: pickle.dump(ix, f)
    return ix

def query_local(jsonl_path: str, pkl_path: str, text_query: str = "", image_query_path: str = None, topk=10):
    with open(pkl_path, "rb") as f: ix = pickle.load(f)
    bm = _bm25_query(ix, text_query or "", topk=topk)
    im = _img_search(jsonl_path, image_query_path, topk=topk) if image_query_path else []
    merged = _rrf_merge(bm, im, k=60, topk=topk)
    return merged

# ---- CLI modes ----
def cli_export(args):
    # Determine source image for current run
    # If --demo used earlier, try demo_inv.png; else use first input
    out_dir = args.out
    src = None
    if args.input:
        src = args.input[0]
    else:
        cand = os.path.join(out_dir, "demo_inv.png")
        src = cand if os.path.exists(cand) else None
    if not src:
        print("No source image found for export."); return
    jpath = os.path.join(out_dir, "doc.zocr.json")
    if not os.path.exists(jpath):
        print("doc.zocr.json not found in", out_dir); return
    out_jsonl = os.path.join(out_dir, "doc.contextual.jsonl")
    source_images: Union[str, Sequence[str]]
    if len(args.input) > 1:
        source_images = [p for p in args.input if isinstance(p, str)]
    else:
        source_images = src
    n = export_jsonl_with_ocr(jpath, source_images, out_jsonl, ocr_engine="toy", contextual=True)
    print("Exported", n, "records to", out_jsonl)

def cli_index(args):
    jsonl = os.path.join(args.out, "doc.contextual.jsonl")
    if not os.path.exists(jsonl):
        print("contextual JSONL not found:", jsonl); return
    pkl = os.path.join(args.out, "bm25.pkl")
    build_local_index(jsonl, pkl)
    print("Wrote local index:", pkl)

def cli_query(args):
    out_dir = args.out
    jsonl = os.path.join(out_dir, "doc.contextual.jsonl")
    pkl = os.path.join(out_dir, "bm25.pkl")
    if not (os.path.exists(jsonl) and os.path.exists(pkl)):
        print("Missing JSONL or index:", jsonl, pkl); return
    merged = query_local(jsonl, pkl, text_query=args.query or "", image_query_path=(args.image_query or None), topk=args.topk)
    # Print concise results
    for i, r in enumerate(merged, 1):
        ob = r["obj"]
        print(f"{i:2d}. score={r['score']:.4f} page={ob.get('page')} row={ob.get('row')} col={ob.get('col')} text='{(ob.get('text') or '')[:40]}' bbox={ob.get('bbox')}")

# Patch argparse: add new subcommands

def _patch_cli_for_export_and_search(parser):
    sub = parser.add_subparsers(dest="cmd")
    def add_common(sp):
        sp.add_argument("--out", type=str, default="out_consensus")
        sp.add_argument("-i","--input", nargs="*", default=[])
    sp = sub.add_parser("export", help="Export JSONL with toy OCR + contextual lines")
    add_common(sp); sp.set_defaults(func=cli_export)
    sp = sub.add_parser("index", help="Build local BM25 index from exported JSONL")
    add_common(sp); sp.set_defaults(func=cli_index)
    sp = sub.add_parser("query", help="Query local index (RRF with optional image)")
    sp.add_argument("--query", type=str, default="")
    sp.add_argument("--image-query", type=str, default="")
    sp.add_argument("--topk", type=int, default=10)
    add_common(sp); sp.set_defaults(func=cli_query)
    return parser
# ==================== /NEW ====================

if __name__=="__main__":
    main()
'''
zocr_onefile_consensus = _materialize_module('zocr_onefile_consensus', _SRC_ZOCR_ONEFILE_CONSENSUS)

# ---------------- [Downstream] multidomain core --------------------
_SRC_ZOCR_MULTIDOMAIN_CORE = r'''
# -*- coding: utf-8 -*-
"""
ZOCR Multi‑Domain Core (single file)
===================================
含むもの：
- RLE‑CC の C 実装（buildc で libzocr.so を生成）＋ Python フォールバック
- 列境界 D² λ の外出し + ページ高さ依存スケジューリング
- pHash(64bit) + 16x16 ベクトル埋め込み（各セル）
- filters の拡張：amount/date/company/address/tax_id/postal_code/phone に加え
  tax_rate / qty / unit / subtotal / tax_amount / corporate_id
- BM25（Numba 加速） + Keyword/Meta ブースト + pHash 類似の融合検索
- SQL‑RAG エクスポート（cells.csv + schema.sql）
- 監視：low_conf_rate / reprocess_rate（Viewsログ）/ reprocess_success_rate /
        Hit@K（GTセルID一致）/ p95 / tax_check_fail_rate
- ドメインプリセット（invoice|contract|delivery|estimate|receipt）でキーワードを切替

使い方（既存 ZOCR の出力 JSONL に対して）:
  python zocr_multidomain_core.py augment --jsonl out/doc.contextual.jsonl --out out/doc.mm.jsonl \
      --lambda-shape 4.5 --lambda-refheight 1000 --lambda-alpha 0.7 --org-dict org_dict.json --domain invoice
  python zocr_multidomain_core.py index   --jsonl out/doc.mm.jsonl --index out/bm25.pkl
  python zocr_multidomain_core.py query   --jsonl out/doc.mm.jsonl --index out/bm25.pkl \
      --q "合計 金額 消費税 2025 1 31" --topk 10 --image crop.png
  python zocr_multidomain_core.py sql     --jsonl out/doc.mm.jsonl --outdir out/sql --prefix invoice
  python zocr_multidomain_core.py monitor --jsonl out/doc.mm.jsonl --index out/bm25.pkl \
      --k 10 --views-log out/views.log.jsonl --gt-jsonl out/doc.gt.jsonl --out out/monitor.csv --domain invoice
  python zocr_multidomain_core.py buildc  --outdir out/lib        # RLE‑CC + POPCNT + Thomas 法

※ 依存：標準 Python + Pillow + numpy（Numba があれば自動使用。無ければフォールバック）。
"""

import os, re, csv, json, math, pickle, ctypes, tempfile, subprocess, datetime, sys, html, platform
from collections import defaultdict, Counter
from typing import List, Optional, Dict, Any, Tuple, Set
from PIL import Image, ImageOps
import numpy as np
from functools import lru_cache

# -------------------- Optional NUMBA --------------------
_HAS_NUMBA = False
try:
    from numba import njit, prange
    from numba import atomic
    _HAS_NUMBA = True
except Exception:
    def njit(*a, **k):
        def deco(f): return f
        return deco
    def prange(n):
        return range(n)
    class _AtomicStub:
        @staticmethod
        def add(arr, idx, val):
            arr[idx] += val
    atomic = _AtomicStub()

# -------------------- Optional C build ------------------
def _build_lib(outdir: Optional[str]=None):
    """
    Build libzocr.so providing:
      - hamm64(uint64_t, uint64_t) -> int
      - thomas_tridiag(int n, double* a,b,c,d, double* x) -> int
      - rle_cc(const uint8_t* img, int H, int W, int max_boxes, int* out_xyxy) -> int
        (4-neigh BFS 実装 / 1=前景,0=背景）
    """
    csrc = r"""
    #include <stdint.h>
    #include <stdlib.h>
    #include <string.h>

    #ifdef _WIN32
    #define EXP __declspec(dllexport)
    #else
    #define EXP
    #endif

    // --- POPCNT Hamming ---
    EXP int hamm64(uint64_t a, uint64_t b){
        uint64_t x = a ^ b;
        #ifdef __GNUC__
        return __builtin_popcountll(x);
        #else
        int c=0; while(x){ x &= (x-1); c++; } return c;
        #endif
    }

    // --- Thomas algorithm (tri-diagonal solver) ---
    EXP int thomas_tridiag(int n, const double* a, const double* b, const double* c,
                           const double* d, double* x){
        if(n<=0) return -1;
        double* cp = (double*)malloc(sizeof(double)*(n-1));
        double* dp = (double*)malloc(sizeof(double)*n);
        if(!cp || !dp) return -2;
        cp[0] = c[0]/b[0];
        dp[0] = d[0]/b[0];
        for(int i=1;i<n;i++){
            double denom = b[i] - a[i-1]*cp[i-1];
            if (i<n-1) cp[i] = c[i]/denom;
            dp[i] = (d[i] - a[i-1]*dp[i-1])/denom;
        }
        x[n-1] = dp[n-1];
        for(int i=n-2;i>=0;i--){
            x[i] = dp[i] - cp[i]*x[i+1];
        }
        free(cp); free(dp);
        return 0;
    }

    // --- RLE-CC (4-neigh BFS) ---
    typedef struct { int x; int y; } P;
    #define MAX_STACK_CAP 1000000
    EXP int rle_cc(const uint8_t* img, int H, int W, int max_boxes, int* out_xyxy){
        // BFS stack (iterative) to avoid recursion
        long long need = (long long)H * (long long)W;
        int max_stack = (need > MAX_STACK_CAP) ? MAX_STACK_CAP : (int)need;
        P* st = (P*)malloc(sizeof(P)*max_stack);
        if(!st) return -3;
        uint8_t* vis = (uint8_t*)calloc(H*W,1);
        if(!vis){ free(st); return -4; }

        int nb=0;
        int overflow = 0;
        for(int y=0;y<H;y++){
            for(int x=0;x<W;x++){
                int idx=y*W+x;
                if(img[idx]==0 || vis[idx]) continue;
                // new comp
                int x1=x, y1=y, x2=x, y2=y;
                int top=0;
                st[top++] = (P){x,y};
                vis[idx]=1;
                while(top>0){
                    P p = st[--top];
                    if(p.x < x1) x1=p.x;
                    if(p.x > x2) x2=p.x;
                    if(p.y < y1) y1=p.y;
                    if(p.y > y2) y2=p.y;
                    // 4-neigh
                    const int dx[4]={1,-1,0,0};
                    const int dy[4]={0,0,1,-1};
                    for(int k=0;k<4;k++){
                        int nx=p.x+dx[k], ny=p.y+dy[k];
                        if(nx>=0 && nx<W && ny>=0 && ny<H){
                            int nidx=ny*W+nx;
                            if(!vis[nidx] && img[nidx]!=0){
                                vis[nidx]=1;
                                if(top < max_stack){
                                    st[top++] = (P){nx,ny};
                                } else {
                                    overflow = 1;
                                    break;
                                }
                            }
                        }
                    }
                    if(overflow){ break; }
                }
                if(nb<max_boxes){
                    // x2,y2 を+1（半開区間）にする場合はここで調整
                    out_xyxy[nb*4+0]=x1;
                    out_xyxy[nb*4+1]=y1;
                    out_xyxy[nb*4+2]=x2+1;
                    out_xyxy[nb*4+3]=y2+1;
                }
                nb++;
            }
        }
        free(vis); free(st);
        return nb; // 実際のコンポーネント数（max_boxes を超える場合もある）
    }
    """
    try:
        tmp = tempfile.mkdtemp()
        cpath = os.path.join(tmp, "zocr.c")
        with open(cpath, "w") as f: f.write(csrc)
        outdir = outdir or tmp
        os.makedirs(outdir, exist_ok=True)
        sysname = platform.system()
        if sysname == "Windows":
            return None, None
        cc = os.environ.get("CC", "cc")
        ext = ".so"
        args = ["-O3", "-shared", "-fPIC"]
        if sysname == "Darwin":
            ext = ".dylib"
            args = ["-O3", "-dynamiclib", "-fPIC", "-undefined", "dynamic_lookup"]
        so = os.path.join(outdir, "libzocr" + ext)
        r = subprocess.run([cc, *args, cpath, "-o", so], capture_output=True)
        if r.returncode != 0:
            return None, None
        lib = ctypes.CDLL(so)
        # set signatures
        lib.hamm64.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        lib.hamm64.restype  = ctypes.c_int

        lib.thomas_tridiag.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.thomas_tridiag.restype = ctypes.c_int

        lib.rle_cc.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int)
        ]
        lib.rle_cc.restype = ctypes.c_int
        return lib, so
    except Exception:
        return None, None

_LIBC, _LIBC_PATH = _build_lib(None)

def buildc(outdir: str):
    """CLI: build C helpers explicitly."""
    lib, so = _build_lib(outdir)
    if lib:
        print("Built:", so)
    else:
        print("Build failed; Python/Numba fallbacks remain active.")

# --------------- Wrappers / Utilities ---------------
def hamm64(a:int,b:int)->int:
    if _LIBC:
        return int(_LIBC.hamm64(ctypes.c_uint64(a), ctypes.c_uint64(b)))
    x=a^b; c=0
    while x: x&=(x-1); c+=1
    return c

def thomas_tridiag(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """D² 正則化に使う三重対角ソルバ（C があれば使用）。"""
    n = b.shape[0]
    if _LIBC:
        x = np.zeros(n, dtype=np.float64)
        a_=np.ascontiguousarray(a, dtype=np.float64)
        b_=np.ascontiguousarray(b, dtype=np.float64)
        c_=np.ascontiguousarray(c, dtype=np.float64)
        d_=np.ascontiguousarray(d, dtype=np.float64)
        r = _LIBC.thomas_tridiag(
            ctypes.c_int(n),
            a_.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b_.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            c_.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            d_.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        if r==0: return x
    # fallback (numpy)
    cp = np.zeros(n-1, dtype=np.float64)
    dp = np.zeros(n, dtype=np.float64)
    cp[0] = c[0]/b[0]; dp[0] = d[0]/b[0]
    for i in range(1,n):
        denom = b[i] - a[i-1]*cp[i-1]
        if i<n-1: cp[i] = c[i]/denom
        dp[i] = (d[i] - a[i-1]*dp[i-1])/denom
    x = np.zeros(n, dtype=np.float64)
    x[-1] = dp[-1]
    for i in range(n-2,-1,-1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x

def _second_diff_tridiag(n: int, lam: float):
    """Return tri-diagonal coefficients matching the consensus D² smoothing."""
    n = int(n)
    lam = float(max(0.0, lam))
    if n <= 0:
        return np.array([]), np.array([]), np.array([])
    a = -lam * np.ones(max(0, n-1), dtype=np.float64)
    c = -lam * np.ones(max(0, n-1), dtype=np.float64)
    b = np.ones(n, dtype=np.float64) + 2.0 * lam
    if n >= 1:
        b[0] = 1.0 + lam
        b[-1] = 1.0 + lam
    return a, b, c

def cc_label_python(bw: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """Python 版 CC（フォールバック）。bwはuint8 0/1。"""
    H,W = bw.shape
    lab = np.zeros((H,W), dtype=np.int32)
    cur = 0
    boxes=[]
    for y in range(H):
        for x in range(W):
            if bw[y,x]==1 and lab[y,x]==0:
                cur+=1
                q=[(x,y)]
                lab[y,x]=cur
                x1=x2=x; y1=y2=y
                while q:
                    xx,yy = q.pop()
                    x1=min(x1,xx); x2=max(x2,xx)
                    y1=min(y1,yy); y2=max(y2,yy)
                    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx,ny=xx+dx,yy+dy
                        if 0<=nx<W and 0<=ny<H and bw[ny,nx]==1 and lab[ny,nx]==0:
                            lab[ny,nx]=cur; q.append((nx,ny))
                boxes.append((x1,y1,x2+1,y2+1))
    return boxes

def cc_label(bw: np.ndarray, max_boxes: int=65536) -> List[Tuple[int,int,int,int]]:
    """C があれば rle_cc を使う。戻りは [(x1,y1,x2,y2), ...]"""
    H,W = bw.shape
    if _LIBC is None:
        return cc_label_python(bw)
    arr = np.ascontiguousarray(bw.astype(np.uint8))
    out = np.zeros(max_boxes*4, dtype=np.int32)
    nb = _LIBC.rle_cc(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                      ctypes.c_int(H), ctypes.c_int(W),
                      ctypes.c_int(max_boxes),
                      out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    nb = int(nb)
    boxes=[]
    use = min(nb, max_boxes)
    for i in range(use):
        x1,y1,x2,y2 = out[i*4:(i+1)*4]
        boxes.append((int(x1),int(y1),int(x2),int(y2)))
    return boxes

# --------------- D² λ スケジューリング ----------------
def lambda_schedule(page_height: int, base_lambda: float, ref_height: int=1000, alpha: float=0.7) -> float:
    """λ_eff = base_lambda * (page_height/ref_height)^alpha"""
    if page_height<=0 or ref_height<=0: return base_lambda
    return float(base_lambda) * ((float(page_height)/float(ref_height))**float(alpha))

# --------------- pHash / Tiny vec ----------------------
@lru_cache(maxsize=1)
def _dct_basis_32() -> np.ndarray:
    N = 32
    x = np.arange(N, dtype=np.float64)
    k = np.arange(N, dtype=np.float64).reshape(-1, 1)
    basis = np.cos((math.pi/N)*(x+0.5)*k).astype(np.float64, copy=False)
    return np.nan_to_num(basis, copy=False)


def phash64(img: Image.Image) -> int:
    g = ImageOps.grayscale(img).resize((32,32), Image.BICUBIC)
    a = np.asarray(g, dtype=np.float64)
    a = np.nan_to_num(a, copy=False)
    basis = _dct_basis_32()
    d=basis@a@basis.T
    d=np.nan_to_num(d, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    d += 1e-9
    blk=d[:8,:8].copy(); blk[0,0]=0.0
    m=float(np.median(blk)); bits=(blk>m).astype(np.uint8).reshape(-1)
    v=0
    for i,b in enumerate(bits):
        if b: v|=(1<<i)
    return int(v)

def tiny_vec(img: Image.Image, n=16) -> np.ndarray:
    g=ImageOps.grayscale(img).resize((n,n), Image.BICUBIC)
    v=np.asarray(g, dtype=np.float32).reshape(-1)
    v=(v-v.mean())/(v.std()+1e-6); return v

# --------------- Normalization / Filters --------------
PREFS=[ "北海道","青森県","岩手県","宮城県","秋田県","山形県","福島県","茨城県","栃木県","群馬県","埼玉県","千葉県","東京都","神奈川県",
        "新潟県","富山県","石川県","福井県","山梨県","長野県","岐阜県","静岡県","愛知県","三重県","滋賀県","京都府","大阪府","兵庫県",
        "奈良県","和歌山県","鳥取県","島根県","岡山県","広島県","山口県","徳島県","香川県","愛媛県","高知県","福岡県","佐賀県",
        "長崎県","熊本県","大分県","宮崎県","鹿児島県","沖縄県" ]
CO_KW=["株式会社","有限会社","合名会社","合資会社","合同会社","Inc.","Co.","Co.,","LLC","G.K.","K.K."]
RX_AMT=re.compile(r"[¥￥$]?\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
RX_DATE=re.compile(r"(20\d{2}|19\d{2})[./\-年](0?[1-9]|1[0-2])([./\-月](0?[1-9]|[12]\d|3[01])日?)?")
RX_TAXID=re.compile(r"\bT\d{10,13}\b", re.IGNORECASE)
RX_POST=re.compile(r"\b\d{3}-\d{4}\b")
RX_PHONE=re.compile(r"\b0\d{1,3}-\d{2,4}-\d{3,4}\b")
RX_PERCENT=re.compile(r"(\d{1,2}(?:\.\d+)?)\s*%")
RX_CORP13=re.compile(r"\b\d{13}\b")  # 法人番号（13桁）

UNIT_KW=["個","式","セット","本","枚","箱","袋","台","pcs","set","kg","g","cm","m","h","時間","回"]

def norm_amount(s: str):
    m=RX_AMT.search(s or "")
    if not m: return None
    try: return int(float(m.group(1).replace(",","")))
    except: return None

def norm_date(s: str):
    m=RX_DATE.search(s or "")
    if not m: return None
    y=int(m.group(1)); mo=int(m.group(2)); d=m.group(3); dd=1
    if d:
        d=re.sub(r"[^\d]","",d)
        if d.isdigit(): dd=int(d)
    return f"{y:04d}-{mo:02d}-{dd:02d}"

def norm_company(s: str, org_dict: Optional[Dict[str,str]]=None):
    s=s or ""
    # 法人番号を先に見る
    m=RX_CORP13.search(s)
    corp_id = m.group(0) if m else None
    if corp_id and org_dict and corp_id in org_dict:
        return org_dict[corp_id].strip(), corp_id
    for kw in CO_KW:
        if kw in s:
            return s.strip(), corp_id
    return (None, corp_id)

def norm_address(s: str):
    s=s or ""
    if any(p in s for p in PREFS): return s.strip()
    if RX_POST.search(s): return s.strip()
    return None

def parse_kv_window(swin: str) -> Dict[str,str]:
    kv={}
    if not swin: return kv
    for seg in swin.split("|"):
        seg=seg.strip()
        if "=" in seg:
            k,v=seg.split("=",1); kv[k.strip()]=v.strip()
    return kv

def infer_row_fields(swin: str) -> Dict[str, Any]:
    kv=parse_kv_window(swin)
    out={}
    # tax_rate
    cand = kv.get("税率") or kv.get("tax") or kv.get("tax_rate") or ""
    m=RX_PERCENT.search(cand or swin or "")
    if m: out["tax_rate"] = float(m.group(1))/100.0
    # qty
    qstr = kv.get("数量") or kv.get("qty") or ""
    qm=re.search(r"\d+", qstr)
    if qm: out["qty"]=int(qm.group(0))
    else:
        qm=re.search(r"数量[^0-9]*?(\d+)", swin or "")
        if qm: out["qty"]=int(qm.group(1))
    # unit
    u = kv.get("単位") or kv.get("unit") or ""
    if not u:
        for w in UNIT_KW:
            if w in (qstr or "") or w in (swin or ""): u=w; break
    if u: out["unit"]=u
    # subtotal
    sub = kv.get("金額") or kv.get("小計") or kv.get("subtotal") or ""
    a = norm_amount(sub)
    if a is not None: out["subtotal"]=a
    # tax_amount
    tax_line = kv.get("消費税") or kv.get("税額") or kv.get("tax_amount") or ""
    ta = norm_amount(tax_line)
    if ta is not None: out["tax_amount"] = ta
    return out

# Domain keywords for boosts
DOMAIN_KW = {
    "invoice": [("合計",1.0),("金額",0.9),("消費税",0.8),("小計",0.6),("請求",0.4),("登録",0.3),("住所",0.3),("単価",0.3),("数量",0.3)],
    "invoice_jp_v2": [("合計",1.0),("金額",0.9),("消費税",0.8),("小計",0.6),("請求日",0.5),("発行日",0.4)],
    "invoice_en": [("invoice",1.0),("total",0.9),("amount",0.85),("tax",0.7),("due",0.5),("bill",0.4)],
    "invoice_fr": [("facture",1.0),("total",0.9),("montant",0.8),("tva",0.7),("échéance",0.5),("paiement",0.4)],
    "purchase_order": [("purchase",1.0),("order",0.95),("po",0.8),("qty",0.6),("ship",0.5),("vendor",0.4)],
    "expense": [("expense",1.0),("reimbursement",0.85),("category",0.6),("receipt",0.6),("total",0.5)],
    "timesheet": [("timesheet",1.0),("hours",0.95),("project",0.6),("rate",0.5),("overtime",0.4)],
    "shipping_notice": [("shipment",1.0),("tracking",0.9),("carrier",0.7),("delivery",0.6),("ship",0.5)],
    "medical_receipt": [("診療",1.0),("点数",0.9),("保険",0.8),("負担金",0.6),("薬剤",0.5)],
    "contract": [("契約",0.8),("署名",0.6),("印",0.5),("住所",0.3),("日付",0.3)],
    "contract_jp_v2": [("契約",0.9),("条",0.7),("締結",0.6),("甲",0.5),("乙",0.5),("印",0.4)],
    "contract_en": [("contract",1.0),("signature",0.75),("party",0.6),("term",0.6),("agreement",0.5)],
    "delivery": [("納品",1.0),("数量",0.8),("単位",0.5),("品名",0.5),("受領",0.4)],
    "delivery_jp": [("納品書",1.0),("数量",0.85),("品番",0.6),("受領",0.5),("出荷",0.4)],
    "delivery_en": [("delivery",1.0),("ship",0.85),("carrier",0.7),("qty",0.6),("item",0.5)],
    "estimate": [("見積",1.0),("単価",0.8),("小計",0.6),("有効期限",0.4)],
    "estimate_jp": [
        ("見積書", 4.0),
        ("御見積書", 3.6),
        ("見積日", 2.3),
        ("見積金額", 3.0),
        ("御見積金額", 3.0),
        ("御見積合計", 2.7),
        ("合計金額", 2.4),
        ("合計", 2.0),
        ("総計", 1.8),
        ("総額", 1.8),
        ("計", 1.5),
        ("小計", 1.6),
        ("数量", 1.2),
        ("単価", 1.2),
        ("金額", 1.2),
        ("有効期限", 2.5),
        ("お見積有効期限", 2.3),
        ("見積有効期限", 2.3),
        ("納期", 1.4),
        ("消費税", 1.5),
        ("税込", 1.2),
        ("税抜", 1.0),
    ],
    "estimate_en": [("estimate",1.0),("quote",0.9),("valid",0.6),("subtotal",0.6),("project",0.4)],
    "receipt": [("領収",1.0),("金額",0.9),("受領",0.6),("発行日",0.4),("住所",0.3)],
    "receipt_jp": [("領収書",1.0),("税込",0.8),("受領",0.6),("発行日",0.4)],
    "receipt_en": [("receipt",1.0),("paid",0.9),("total",0.75),("payment",0.6),("tax",0.5)],
    "bank_statement_en": [("statement",1.0),("account",0.9),("balance",0.8),("transaction",0.7),("debit",0.6),("credit",0.6),("bank",0.5)],
    "bank_statement_jp": [("取引明細",1.0),("口座番号",0.9),("残高",0.8),("入金",0.7),("出金",0.7),("金融機関",0.6),("支店",0.5)],
    "utility_bill_en": [("utility",1.0),("electric",0.85),("gas",0.8),("water",0.75),("kwh",0.6),("meter",0.55),("billing",0.5)],
    "utility_bill_jp": [("ご使用量",1.0),("電気",0.85),("ガス",0.8),("水道",0.75),("検針",0.6),("請求額",0.55),("契約",0.5)],
    "insurance_claim_en": [("claim",1.0),("policy",0.85),("insured",0.75),("coverage",0.65),("adjuster",0.55),("deductible",0.5)],
    "insurance_claim_jp": [("保険金請求",1.0),("被保険者",0.85),("保険証券",0.75),("事故日",0.65),("給付",0.55),("診断書",0.5)],
    "tax_form_en": [("tax",1.0),("return",0.9),("irs",0.8),("deduction",0.7),("withholding",0.6),("income",0.6)],
    "tax_form_jp": [("確定申告",1.0),("所得税",0.9),("控除",0.75),("課税",0.65),("源泉",0.6),("扶養",0.5)],
    "payslip_en": [("payslip",1.0),("payroll",0.9),("gross",0.75),("net pay",0.7),("deductions",0.65),("hours",0.55)],
    "payslip_jp": [("給与明細",1.0),("支給額",0.9),("控除",0.75),("差引支給額",0.7),("残業",0.6),("社会保険料",0.55)],
    "rental_agreement_en": [("rental",1.0),("lease",0.95),("tenant",0.75),("landlord",0.7),("premises",0.6)],
    "rental_agreement_jp": [("賃貸借",1.0),("賃料",0.9),("借主",0.75),("貸主",0.75),("物件",0.6),("契約期間",0.5)],
    "loan_statement_en": [("loan",1.0),("interest",0.9),("principal",0.85),("installment",0.7),("statement",0.6),("balance",0.6)],
    "loan_statement_jp": [("返済",1.0),("借入",0.9),("利息",0.85),("元金",0.75),("残高",0.65),("明細",0.55)],
    "travel_itinerary_en": [("itinerary",1.0),("flight",0.9),("departure",0.85),("arrival",0.85),("hotel",0.7),("booking",0.6)],
    "travel_itinerary_jp": [("旅程",1.0),("出発",0.9),("到着",0.9),("航空券",0.75),("宿泊",0.65),("予約",0.6)],
    "medical_bill_en": [("medical",1.0),("invoice",0.9),("patient",0.85),("procedure",0.7),("amount",0.6),("insurance",0.55)],
    "medical_bill_jp": [("診療",1.0),("請求",0.9),("患者",0.8),("保険",0.75),("点数",0.65),("金額",0.6)],
    "customs_declaration_en": [("customs",1.0),("declaration",0.95),("tariff",0.75),("shipment",0.7),("origin",0.6),("duty",0.55)],
    "grant_application_en": [("grant",1.0),("fund",0.9),("proposal",0.8),("budget",0.7),("milestone",0.6)],
    "boarding_pass_en": [("boarding",1.0),("flight",0.95),("seat",0.85),("gate",0.7),("departure",0.65),("passenger",0.6)]
}

DOMAIN_DEFAULTS = {
    "invoice": {"lambda_shape": 4.5, "w_kw": 0.6, "w_img": 0.3, "ocr_min_conf": 0.55},
    "invoice_jp_v2": {"lambda_shape": 4.5, "w_kw": 0.6, "w_img": 0.3, "ocr_min_conf": 0.55},
    "invoice_en": {"lambda_shape": 4.2, "w_kw": 0.55, "w_img": 0.25, "ocr_min_conf": 0.53},
    "invoice_fr": {"lambda_shape": 4.2, "w_kw": 0.55, "w_img": 0.25, "ocr_min_conf": 0.53},
    "purchase_order": {"lambda_shape": 4.0, "w_kw": 0.5, "w_img": 0.22, "ocr_min_conf": 0.60},
    "expense": {"lambda_shape": 3.8, "w_kw": 0.5, "w_img": 0.18, "ocr_min_conf": 0.60},
    "timesheet": {"lambda_shape": 3.6, "w_kw": 0.45, "w_img": 0.18, "ocr_min_conf": 0.62},
    "shipping_notice": {"lambda_shape": 4.3, "w_kw": 0.5, "w_img": 0.26, "ocr_min_conf": 0.58},
    "medical_receipt": {"lambda_shape": 5.0, "w_kw": 0.65, "w_img": 0.28, "ocr_min_conf": 0.60},
    "contract": {"lambda_shape": 4.0, "w_kw": 0.55, "w_img": 0.2, "ocr_min_conf": 0.60},
    "contract_jp_v2": {"lambda_shape": 4.1, "w_kw": 0.6, "w_img": 0.2, "ocr_min_conf": 0.60},
    "contract_en": {"lambda_shape": 4.0, "w_kw": 0.55, "w_img": 0.2, "ocr_min_conf": 0.60},
    "delivery": {"lambda_shape": 4.2, "w_kw": 0.5, "w_img": 0.25, "ocr_min_conf": 0.58},
    "delivery_jp": {"lambda_shape": 4.2, "w_kw": 0.5, "w_img": 0.25, "ocr_min_conf": 0.58},
    "delivery_en": {"lambda_shape": 4.0, "w_kw": 0.48, "w_img": 0.22, "ocr_min_conf": 0.58},
    "estimate": {"lambda_shape": 4.3, "w_kw": 0.55, "w_img": 0.25, "ocr_min_conf": 0.58},
    "estimate_jp": {"lambda_shape": 4.3, "w_kw": 0.55, "w_img": 0.25, "ocr_min_conf": 0.58},
    "estimate_en": {"lambda_shape": 4.2, "w_kw": 0.5, "w_img": 0.25, "ocr_min_conf": 0.58},
    "receipt": {"lambda_shape": 4.1, "w_kw": 0.6, "w_img": 0.2, "ocr_min_conf": 0.60},
    "receipt_jp": {"lambda_shape": 4.1, "w_kw": 0.6, "w_img": 0.2, "ocr_min_conf": 0.60},
    "receipt_en": {"lambda_shape": 4.0, "w_kw": 0.55, "w_img": 0.2, "ocr_min_conf": 0.60},
    "bank_statement_en": {"lambda_shape": 4.1, "w_kw": 0.58, "w_img": 0.24, "ocr_min_conf": 0.60},
    "bank_statement_jp": {"lambda_shape": 4.2, "w_kw": 0.6, "w_img": 0.24, "ocr_min_conf": 0.60},
    "utility_bill_en": {"lambda_shape": 3.9, "w_kw": 0.52, "w_img": 0.22, "ocr_min_conf": 0.58},
    "utility_bill_jp": {"lambda_shape": 4.0, "w_kw": 0.55, "w_img": 0.22, "ocr_min_conf": 0.58},
    "insurance_claim_en": {"lambda_shape": 4.4, "w_kw": 0.6, "w_img": 0.24, "ocr_min_conf": 0.60},
    "insurance_claim_jp": {"lambda_shape": 4.5, "w_kw": 0.62, "w_img": 0.24, "ocr_min_conf": 0.60},
    "tax_form_en": {"lambda_shape": 4.6, "w_kw": 0.63, "w_img": 0.22, "ocr_min_conf": 0.60},
    "tax_form_jp": {"lambda_shape": 4.6, "w_kw": 0.65, "w_img": 0.22, "ocr_min_conf": 0.60},
    "payslip_en": {"lambda_shape": 3.9, "w_kw": 0.58, "w_img": 0.2, "ocr_min_conf": 0.62},
    "payslip_jp": {"lambda_shape": 4.0, "w_kw": 0.6, "w_img": 0.2, "ocr_min_conf": 0.62},
    "rental_agreement_en": {"lambda_shape": 4.2, "w_kw": 0.6, "w_img": 0.22, "ocr_min_conf": 0.60},
    "rental_agreement_jp": {"lambda_shape": 4.2, "w_kw": 0.63, "w_img": 0.22, "ocr_min_conf": 0.60},
    "loan_statement_en": {"lambda_shape": 4.1, "w_kw": 0.6, "w_img": 0.24, "ocr_min_conf": 0.60},
    "loan_statement_jp": {"lambda_shape": 4.2, "w_kw": 0.62, "w_img": 0.24, "ocr_min_conf": 0.60},
    "travel_itinerary_en": {"lambda_shape": 3.8, "w_kw": 0.55, "w_img": 0.26, "ocr_min_conf": 0.58},
    "travel_itinerary_jp": {"lambda_shape": 3.9, "w_kw": 0.58, "w_img": 0.26, "ocr_min_conf": 0.58},
    "medical_bill_en": {"lambda_shape": 4.8, "w_kw": 0.64, "w_img": 0.24, "ocr_min_conf": 0.60},
    "medical_bill_jp": {"lambda_shape": 4.9, "w_kw": 0.66, "w_img": 0.24, "ocr_min_conf": 0.60},
    "customs_declaration_en": {"lambda_shape": 4.2, "w_kw": 0.58, "w_img": 0.22, "ocr_min_conf": 0.58},
    "grant_application_en": {"lambda_shape": 4.0, "w_kw": 0.6, "w_img": 0.22, "ocr_min_conf": 0.58},
    "boarding_pass_en": {"lambda_shape": 3.5, "w_kw": 0.5, "w_img": 0.3, "ocr_min_conf": 0.55}
}

for _dom_conf in DOMAIN_DEFAULTS.values():
    _dom_conf.setdefault("w_sym", 0.45)

_DOMAIN_ALIAS = {
    "invoice": "invoice_jp_v2",
    "contract": "contract_jp_v2",
    "delivery": "delivery_jp",
    "estimate": "estimate_jp",
    "receipt": "receipt_jp",
    "bank_statement": "bank_statement_en",
    "utility_bill": "utility_bill_en",
    "insurance_claim": "insurance_claim_en",
    "tax_form": "tax_form_en",
    "tax_return": "tax_form_en",
    "payslip": "payslip_en",
    "rental_agreement": "rental_agreement_en",
    "lease_contract": "rental_agreement_en",
    "loan_statement": "loan_statement_en",
    "loan_summary": "loan_statement_en",
    "travel_itinerary": "travel_itinerary_en",
    "travel_plan": "travel_itinerary_en",
    "medical_bill": "medical_bill_en",
    "medical_invoice": "medical_bill_en",
    "customs_declaration": "customs_declaration_en",
    "customs_form": "customs_declaration_en",
    "grant_application": "grant_application_en",
    "boarding_pass": "boarding_pass_en"
}

_DOMAIN_HEADER_SIGNALS: Dict[str, List[Tuple[str, float]]] = {
    "invoice_jp_v2": [
        ("請求書", 1.2),
        ("請求書番号", 0.8),
        ("合計", 0.8),
        ("消費税", 0.75),
        ("小計", 0.6),
        ("請求日", 0.55),
        ("発行日", 0.55),
    ],
    "invoice": [
        ("invoice", 1.0),
        ("total", 0.8),
        ("tax", 0.7),
        ("subtotal", 0.6),
        ("due date", 0.55),
    ],
    "invoice_en": [
        ("invoice", 1.1),
        ("total", 0.85),
        ("tax", 0.75),
        ("subtotal", 0.65),
        ("due date", 0.55),
    ],
    "invoice_fr": [
        ("facture", 1.1),
        ("total", 0.85),
        ("tva", 0.75),
        ("sous-total", 0.6),
    ],
    "estimate_jp": [
        ("見積金額", 0.9),
        ("御見積金額", 0.9),
        ("数量", 0.7),
        ("単価", 0.7),
        ("金額", 0.6),
        ("有効期限", 0.65),
        ("納期", 0.55),
    ],
}

_HEADER_CONCEPT_SIGNALS: Dict[str, List[Tuple[str, float]]] = {
    "header:total": [
        ("invoice_jp_v2", 0.6),
        ("invoice_en", 0.55),
        ("invoice", 0.55),
        ("invoice_fr", 0.5),
    ],
    "header:tax": [
        ("invoice_jp_v2", 0.6),
        ("invoice_en", 0.55),
        ("invoice", 0.55),
        ("invoice_fr", 0.45),
    ],
    "header:subtotal": [
        ("invoice_jp_v2", 0.5),
        ("invoice_en", 0.45),
        ("invoice", 0.45),
        ("invoice_fr", 0.4),
    ],
}

DOMAIN_SUGGESTED_QUERIES = {
    "invoice_jp_v2": ["合計 金額", "消費税", "支払期日", "請求先 住所"],
    "invoice_en": ["total amount", "tax amount", "due date", "billing address"],
    "invoice_fr": ["montant total", "tva", "échéance", "adresse de facturation"],
    "purchase_order": ["po number", "vendor", "ship date", "total"],
    "expense": ["employee", "category", "reimbursement", "amount"],
    "timesheet": ["hours", "project", "overtime", "approval"],
    "shipping_notice": ["tracking number", "carrier", "ship date", "items"],
    "medical_receipt": ["診療点数", "保険", "自己負担", "調剤"],
    "contract_jp_v2": ["契約期間", "甲", "乙", "締結日"],
    "contract_en": ["effective date", "party", "term", "signature"],
    "delivery_jp": ["納品日", "数量", "品番", "受領印"],
    "estimate_jp": ["御見積金額", "見積金額", "有効期限", "納期"],
    "receipt_jp": ["領収金額", "発行日", "支払方法", "住所"],
    "bank_statement_en": ["ending balance", "transaction", "deposit", "withdrawal"],
    "bank_statement_jp": ["残高", "入金", "出金", "取引日"],
    "utility_bill_en": ["meter reading", "usage", "billing period", "due date"],
    "utility_bill_jp": ["ご使用量", "検針日", "請求額", "支払期限"],
    "insurance_claim_en": ["claim number", "policy", "incident date", "payout"],
    "insurance_claim_jp": ["保険金請求", "事故日", "被保険者", "支払額"],
    "tax_form_en": ["taxable income", "deduction", "withholding", "refund"],
    "tax_form_jp": ["課税所得", "控除", "源泉徴収", "還付金"],
    "payslip_en": ["gross pay", "net pay", "deductions", "hours"],
    "payslip_jp": ["支給額", "控除", "差引支給額", "残業時間"],
    "rental_agreement_en": ["monthly rent", "lease term", "deposit", "premises"],
    "rental_agreement_jp": ["賃料", "契約期間", "敷金", "物件住所"],
    "loan_statement_en": ["principal balance", "interest paid", "payment date", "installment"],
    "loan_statement_jp": ["元金残高", "利息", "返済日", "返済額"],
    "travel_itinerary_en": ["departure time", "arrival gate", "hotel confirmation", "booking reference"],
    "travel_itinerary_jp": ["出発時刻", "到着ゲート", "宿泊先", "予約番号"],
    "medical_bill_en": ["patient", "total amount", "insurance", "procedure"],
    "medical_bill_jp": ["患者", "請求金額", "保険", "診療"],
    "customs_declaration_en": ["tariff code", "country of origin", "declared value", "duty"],
    "grant_application_en": ["project title", "requested amount", "milestone", "deliverable"],
    "boarding_pass_en": ["flight number", "seat", "gate", "boarding time"],
    "default": ["total amount", "date", "company", "reference number"]
}

DOMAIN_MONITOR_QUERIES = {
    "default": {
        "q_amount": "合計 金額 消費税 円 2023 2024 2025",
        "q_date": "請求日 発行日 2023 2024 2025",
        "q_due": "支払期日 支払期限 期日 支払日",
    },
    "contract_jp_v2": {
        "q_amount": "契約金額 代金 支払",
        "q_date": "契約日 締結日 開始日 終了日",
        "q_due": "契約期間 支払期日 締結日",
    },
    "estimate_jp": {
        "q_amount": "見積金額 御見積金額 御見積総額 合計 合計金額 総計 計 金額 税込 税抜 円",
        "q_date": "見積日 発行日 作成日 提出日 2023 2024 2025",
        "q_due": "有効期限 お見積有効期限 見積有効期限 納期 納入期限 2023 2024 2025",
    },
}


def _normalize_text(val: Optional[Any]) -> str:
    if val is None:
        return ""
    if not isinstance(val, str):
        val = str(val)
    val = val.strip()
    if not val:
        return ""
    return re.sub(r"\s+", " ", val)

def detect_domain_on_jsonl(jsonl_path: str, filename_tokens: Optional[Sequence[Any]] = None) -> Tuple[str, Dict[str, Any]]:
    scores: Dict[str, float] = {k: 0.0 for k in DOMAIN_KW.keys()}
    hits: Dict[str, int] = {k: 0 for k in DOMAIN_KW.keys()}
    token_hits: Dict[str, int] = {k: 0 for k in DOMAIN_KW.keys()}
    header_hits: Dict[str, int] = {}
    concept_hits: Dict[str, int] = {}
    token_source_hits: Dict[str, Dict[str, int]] = {}
    total_cells = 0
    header_samples: List[List[str]] = []

    def _bump_token_source(dom: str, origin: Optional[str]) -> None:
        key = (origin or "unknown").strip().lower() or "unknown"
        bucket = token_source_hits.setdefault(key, {})
        bucket[dom] = bucket.get(dom, 0) + 1

    try:
        with open(jsonl_path, "r", encoding="utf-8") as fr:
            for line in fr:
                try:
                    ob = json.loads(line)
                except Exception:
                    continue
                text_parts = [ob.get("text") or "", ob.get("synthesis_window") or ""]
                meta = ob.get("meta") or {}
                filt = meta.get("filters") or {}
                for v in filt.values():
                    if isinstance(v, str):
                        text_parts.append(v)
                joined = " ".join(text_parts)
                joined_lower = joined.lower()
                total_cells += 1
                for dom, kws in DOMAIN_KW.items():
                    score = 0.0
                    for kw, weight in kws:
                        if not kw:
                            continue
                        if kw in joined or kw.lower() in joined_lower:
                            score += float(weight)
                    if score > 0.0:
                        scores[dom] += score
                        hits[dom] += 1

                headers_val = meta.get("headers")
                header_values: List[str] = []
                if isinstance(headers_val, (list, tuple)):
                    for hv in headers_val:
                        hv_str = str(hv).strip() if hv is not None else ""
                        if hv_str:
                            header_values.append(hv_str)
                if header_values:
                    if len(header_samples) < 4:
                        header_samples.append(header_values)
                    header_joined = " ".join(header_values)
                    header_joined_lower = header_joined.lower()
                    for dom, signals in _DOMAIN_HEADER_SIGNALS.items():
                        for kw, weight in signals:
                            if not kw:
                                continue
                            kw_lower = kw.lower()
                            if kw in header_joined or kw_lower in header_joined_lower:
                                scores[dom] += float(weight)
                                hits[dom] += 1
                                header_hits[dom] = header_hits.get(dom, 0) + 1

                concepts_val = meta.get("concepts") or []
                if isinstance(concepts_val, (list, tuple)):
                    for concept in concepts_val:
                        concept_key = str(concept or "").strip()
                        if not concept_key:
                            continue
                        for dom, weight in _HEADER_CONCEPT_SIGNALS.get(concept_key, []):
                            scores[dom] += float(weight)
                            hits[dom] += 1
                            concept_hits[dom] = concept_hits.get(dom, 0) + 1
    except FileNotFoundError:
        pass

    normalized_tokens: List[Tuple[str, Optional[str]]] = []
    token_trace_detail: List[Dict[str, Any]] = []
    if filename_tokens:
        for entry in filename_tokens:
            token_raw: Any = None
            origin: Optional[str] = None
            path: Optional[str] = None
            if isinstance(entry, dict):
                token_raw = entry.get("token")
                origin = entry.get("source")
                path = entry.get("path")
            elif isinstance(entry, (list, tuple)):
                if not entry:
                    continue
                token_raw = entry[0]
                if len(entry) > 1:
                    origin = entry[1]
                if len(entry) > 2:
                    path = entry[2]
            else:
                token_raw = entry
            token_str = str(token_raw or "").strip()
            if not token_str:
                continue
            token_norm = token_str.lower()
            normalized_tokens.append((token_norm, origin if isinstance(origin, str) else None))
            token_trace_detail.append({
                "token": token_norm,
                "raw": token_str,
                "source": origin,
                "path": path,
            })

    if normalized_tokens:
        lookup: Dict[str, List[str]] = {}

        def _register(key: str, target: str) -> None:
            key = key.strip().lower()
            if not key:
                return
            lookup.setdefault(key, []).append(target)

        for dom in DOMAIN_KW.keys():
            key = dom.lower()
            _register(key, dom)
            for part in re.split(r"[^a-z0-9]+", key):
                if part:
                    _register(part, dom)

        for alias, target in _DOMAIN_ALIAS.items():
            key = alias.lower()
            _register(key, target)
            for part in re.split(r"[^a-z0-9]+", key):
                if part:
                    _register(part, target)

        for token_norm, origin in normalized_tokens:
            token_l = token_norm.strip()
            if not token_l:
                continue
            matched = False
            for dom in lookup.get(token_l, []):
                scores[dom] += 0.6
                hits[dom] += 1
                token_hits[dom] += 1
                _bump_token_source(dom, origin)
                matched = True
            if matched:
                continue
            for key, dom_list in lookup.items():
                if token_l in key and key not in lookup.get(token_l, []):
                    for dom in dom_list:
                        scores[dom] += 0.3
                        hits[dom] += 1
                        token_hits[dom] += 1
                        _bump_token_source(dom, origin)
                    break

    def _score_key(dom: str) -> Tuple[float, int]:
        return scores.get(dom, 0.0), hits.get(dom, 0)

    best_dom = "invoice_jp_v2"
    if scores:
        best_dom = max(scores.keys(), key=lambda d: (_score_key(d)[0], _score_key(d)[1]))
    resolved = _DOMAIN_ALIAS.get(best_dom, best_dom)
    score_total = sum(max(0.0, s) for s in scores.values())
    confidence = scores.get(best_dom, 0.0) / score_total if score_total > 0 else 0.0
    detail: Dict[str, Any] = {
        "scores": {k: float(v) for k, v in scores.items()},
        "hits": {k: int(v) for k, v in hits.items()},
        "token_hits": {k: int(v) for k, v in token_hits.items()},
        "total_cells": total_cells,
        "resolved": resolved,
        "raw_best": best_dom,
        "confidence": confidence,
        "filename_tokens": filename_tokens or [],
    }
    if token_trace_detail:
        detail["filename_token_trace"] = token_trace_detail
    if token_source_hits:
        detail["token_hits_by_source"] = {
            src: {dom: int(val) for dom, val in dom_map.items()}
            for src, dom_map in token_source_hits.items()
        }
    if header_hits:
        detail["header_hits"] = {k: int(v) for k, v in header_hits.items()}
    if concept_hits:
        detail["concept_hits"] = {k: int(v) for k, v in concept_hits.items()}
    if header_samples:
        detail["header_samples"] = header_samples
    return resolved, detail

# --------------- Augment (pHash + Filters + λ) ---------------
def _inject_structural_placeholders(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from collections import defaultdict

    tables: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = defaultdict(lambda: {
        "cells": [],
        "rows": defaultdict(dict),
        "template": None,
    })

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
    for key, tbl in tables.items():
        if not tbl["cells"]:
            continue
        rows = tbl["rows"]
        template = tbl["template"]
        header_row = min(rows.keys()) if rows else 0
        header_cells = rows.get(header_row, {})
        header_text = " ".join((header_cells[c].get("text") or "") for c in sorted(header_cells))
        use_jp = _contains_cjk(header_text)
        max_col = -1
        for cols in rows.values():
            for c in cols.keys():
                if c > max_col:
                    max_col = c
        header_lower = header_text.lower()
        for family in families:
            jp_variant = family[1]
            en_variant = family[0]
            present = False
            for variant in family:
                if variant.lower() in header_lower:
                    present = True
                    break
            if present:
                continue
            text_val = jp_variant if use_jp else en_variant
            replacement = None
            for col_idx in sorted(header_cells.keys()):
                cell = header_cells[col_idx]
                if _is_weak_cell(cell):
                    replacement = (col_idx, cell)
                    break
            if replacement is not None:
                col_idx, cell = replacement
                meta = dict(cell.get("meta") or {})
                filters = dict(meta.get("filters") or {})
                filters.setdefault("row_role", "header")
                meta.update({
                    "filters": filters,
                    "synthetic": True,
                    "synthetic_role": "header",
                    "low_conf": False,
                    "confidence": max(0.55, float(meta.get("confidence") or 0.0)) if isinstance(meta.get("confidence"), (int, float)) else 0.55,
                })
                cell.update({
                    "text": text_val,
                    "search_unit": text_val,
                    "synthesis_window": cell.get("synthesis_window") or text_val,
                    "meta": meta,
                })
            else:
                max_col += 1
                new_cell = _make_cell(template, header_row, max_col, text_val, "header")
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
            meta.update({
                "filters": filters,
                "synthetic": True,
                "synthetic_role": "footer",
                "low_conf": False,
                "confidence": max(0.55, float(meta.get("confidence") or 0.0)) if isinstance(meta.get("confidence"), (int, float)) else 0.55,
            })
            weak_total.update({
                "text": footer_text,
                "search_unit": footer_text,
                "synthesis_window": weak_total.get("synthesis_window") or footer_text,
                "meta": meta,
            })
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


def augment(jsonl_in: str, jsonl_out: str, lambda_shape: float=4.5, lambda_refheight: int=1000, lambda_alpha: float=0.7, org_dict_path: Optional[str]=None):
    org_dict=None
    if org_dict_path and os.path.exists(org_dict_path):
        try:
            with open(org_dict_path,"r",encoding="utf-8") as f:
                org_dict=json.load(f)
        except Exception:
            org_dict=None
    records: List[Dict[str, Any]] = []
    cur=None; img=None
    with open(jsonl_in,"r",encoding="utf-8") as fr:
        for line in fr:
            ob=json.loads(line)
            ip=ob.get("image_path"); bbox=ob.get("bbox",[0,0,0,0])
            page_h = None
            if ip and os.path.exists(ip):
                if ip!=cur:
                    img=Image.open(ip).convert("RGB"); cur=ip
                page_h = img.height
                x1,y1,x2,y2=[int(v) for v in bbox]
                x1=max(0,x1);y1=max(0,y1);x2=min(img.width,x2);y2=min(img.height,y2)
                crop=img.crop((x1,y1,x2,y2))
                try:
                    ph=phash64(crop)
                except Exception:
                    ph=0
                vec=tiny_vec(crop,16).tolist()
                ob.setdefault("meta",{}); ob["meta"]["phash64"]=ph; ob["meta"]["img16"]=vec
                if page_h:
                    ob["meta"]["lambda_shape"] = lambda_schedule(page_h, lambda_shape, lambda_refheight, lambda_alpha)
            txt=(ob.get("text") or "")+" "+(ob.get("synthesis_window") or "")
            swin=(ob.get("synthesis_window") or "")
            filt=(ob.get("meta") or {}).get("filters",{})
            filt["amount"]=filt.get("amount") or norm_amount(txt)
            filt["date"]=filt.get("date") or norm_date(txt)
            t=RX_TAXID.search(txt); filt["tax_id"]=filt.get("tax_id") or (t.group(0) if t else None)
            p=RX_POST.search(txt); filt["postal_code"]=filt.get("postal_code") or (p.group(0) if p else None)
            phn=RX_PHONE.search(txt); filt["phone"]=filt.get("phone") or (phn.group(0) if phn else None)
            comp, corp_id = norm_company(txt, org_dict)
            addr = norm_address(txt)
            if comp: filt["company"]=comp
            if corp_id: filt["corporate_id"]=corp_id
            if addr: filt["address"]=addr
            rowf = infer_row_fields(swin)
            for k,v in rowf.items():
                if filt.get(k) is None: filt[k]=v
            if filt.get("tax_amount") is None and filt.get("tax_rate") is not None and filt.get("subtotal") is not None:
                filt["tax_amount"] = int(round(float(filt["subtotal"]) * float(filt["tax_rate"])))
            ob["meta"]["filters"]=filt
            records.append(ob)

    records = _inject_structural_placeholders(records)

    with open(jsonl_out,"w",encoding="utf-8") as fw:
        for ob in records:
            fw.write(json.dumps(ob, ensure_ascii=False)+"\n")
    return len(records)

# --------------- BM25 + Fusion Search -----------------
def tokenize_jp(s: str) -> List[str]:
    s=s or ""
    toks=re.findall(r"[A-Za-z]+|\d+(?:,\d{3})*(?:\.\d+)?", s)
    jp="".join(ch for ch in s if ord(ch)>127 and not ch.isspace())
    toks += [jp[i:i+2] for i in range(len(jp)-1)]
    return [t.lower() for t in toks if t]

def build_index(jsonl: str, out_pkl: str):
    docs=[]; vocab={}; vid=0; maxlen=0
    with open(jsonl,"r",encoding="utf-8") as f:
        for line in f:
            ob=json.loads(line)
            txt=ob.get("search_unit") or ob.get("text") or ""
            toks=tokenize_jp(txt)
            ids=[]
            for t in toks:
                if t not in vocab:
                    vocab[t]=vid; vid+=1
                ids.append(vocab[t])
            maxlen=max(maxlen, len(ids))
            docs.append((ids, ob))
    V=len(vocab); N=len(docs)
    pad=-1
    arr=np.full((N, maxlen), pad, dtype=np.int32)
    lengths=np.zeros(N, dtype=np.int32)
    uniq_docs=[]
    uniq_maxlen=0
    for i,(ids,_) in enumerate(docs):
        lengths[i]=len(ids)
        if ids:
            arr[i,:len(ids)] = np.array(ids, dtype=np.int32)
        seen=set()
        uniq=[]
        for tid in ids:
            if tid not in seen:
                seen.add(tid)
                uniq.append(tid)
        uniq_docs.append(uniq)
        if len(uniq)>uniq_maxlen:
            uniq_maxlen=len(uniq)

    uniq_pad=-1
    uniq_shape=max(1, uniq_maxlen) if N else 0
    arr_unique=np.full((N, uniq_shape), uniq_pad, dtype=np.int32)
    uniq_lengths=np.zeros(N, dtype=np.int32)
    for i,uniq in enumerate(uniq_docs):
        L=len(uniq)
        uniq_lengths[i]=L
        if L:
            arr_unique[i,:L] = np.array(uniq, dtype=np.int32)

    @njit(cache=True)
    def _compute_df(arr_unique, lengths, V):
        n=arr_unique.shape[0]
        df=np.zeros(V, dtype=np.int64)
        for i in range(n):
            for j in range(lengths[i]):
                tid=arr_unique[i,j]
                if tid<0:
                    break
                df[tid]+=1
        return df

    @njit(parallel=True, cache=True)
    def _compute_df_parallel(arr_unique, lengths, V):
        n=arr_unique.shape[0]
        df=np.zeros(V, dtype=np.int64)
        for i in prange(n):
            L=lengths[i]
            for j in range(L):
                tid=arr_unique[i,j]
                if tid<0:
                    break
                atomic.add(df, tid, 1)
        return df

    df=None
    if _HAS_NUMBA:
        try:
            if V <= 200000:
                df=_compute_df_parallel(arr_unique, uniq_lengths, V)
            else:
                df=_compute_df(arr_unique, uniq_lengths, V)
        except Exception:
            df=None
    if df is None:
        df=np.zeros(V, dtype=np.int64)
        for uniq in uniq_docs:
            for tid in uniq:
                df[tid]+=1
    avgdl = float(lengths.sum())/max(1,N)
    ix={"vocab":vocab, "df":df, "avgdl":avgdl, "N":N, "lengths":lengths.tolist(), "docs_tokens":[d[0] for d in docs]}
    with open(out_pkl,"wb") as f: pickle.dump(ix,f)
    return ix

@njit(cache=True)
def _bm25_numba_score(N, avgdl, df, dl, q_ids, doc_ids, k1=1.2, b=0.75):
    s=0.0
    # simplistic tf; could be optimized further
    for i in range(len(doc_ids)):
        tid = doc_ids[i]
        if tid < 0: break
        # tf for this tid
        tf=0
        for j in range(len(doc_ids)):
            if doc_ids[j] < 0: break
            if doc_ids[j]==tid: tf+=1
        for q in q_ids:
            if q==tid and df[q]>0:
                idf = math.log((N - df[q] + 0.5)/(df[q] + 0.5) + 1.0)
                s += idf * ((tf*(k1+1))/(tf + k1*(1 - b + b*dl/max(1.0,avgdl))))
    return s

def _bm25_py_score(N, avgdl, df, dl, q_ids, doc_ids, k1=1.2, b=0.75):
    from collections import Counter
    tf = Counter([tid for tid in doc_ids if tid>=0])
    s=0.0
    for q in q_ids:
        if q<0 or df[q]==0: continue
        idf = math.log((N - df[q] + 0.5)/(df[q] + 0.5) + 1.0)
        f = tf.get(q,0)
        s += idf * ((f*(k1+1))/(f + k1*(1 - b + b*dl/max(1.0,avgdl))))
    return s

def _phash_sim(q_img_path: Optional[str], ph: int) -> float:
    if not q_img_path or not os.path.exists(q_img_path) or ph==0: return 0.0
    try:
        qi=Image.open(q_img_path).convert("RGB"); qh=phash64(qi)
    except Exception:
        return 0.0
    hd = hamm64(int(qh), int(ph))
    return 1.0 - (hd/64.0)

def _kw_meta_boost(ob: Dict[str,Any], q_toks: List[str], domain:str="invoice") -> float:
    text=((ob.get("synthesis_window") or "")+" "+(ob.get("text") or "")).lower()
    filt=(ob.get("meta") or {}).get("filters",{})
    s=0.0
    nums=[int(t.replace(",","")) for t in q_toks if re.fullmatch(r"\d+(?:,\d{3})*", t)]
    if filt.get("amount") and any(abs(filt["amount"]-n)<=5 for n in nums): s+=1.5
    if filt.get("date"):
        for d in re.findall(r"\d+", filt["date"]):
            if d in q_toks: s+=0.3
    for kw,w in DOMAIN_KW.get(domain, DOMAIN_KW["invoice"]):
        if kw in text: s+=w
    return s

def _symbolic_match_score(filters: Dict[str, Any], q_text: str, tokens: List[str]) -> float:
    if not filters or not q_text:
        return 0.0
    q_norm = _normalize_text(q_text)
    if not isinstance(tokens, list):
        tokens = list(tokens)
    digits = re.findall(r"\d+(?:[.,]\d+)?", q_text)
    score = 0.0
    seen_keys: Set[str] = set()
    for key, val in filters.items():
        if key in (None, "trace"):
            continue
        if val is None:
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
            seen_keys.add(str(key))
            score += key_hits
    if seen_keys:
        score += min(len(seen_keys), 4) * 0.5
    return float(score)


def query(index_pkl: str, jsonl: str, q_text: str="", q_image: Optional[str]=None, topk:int=10,
          w_bm25:float=1.0, w_kw:float=0.6, w_img:float=0.3, w_sym:float=0.45, domain:str="invoice"):
    with open(index_pkl,"rb") as f: ix=pickle.load(f)
    vocab=ix["vocab"]; df=np.array(ix["df"], dtype=np.int32); N=int(ix["N"]); avgdl=float(ix["avgdl"])
    raws=[]
    with open(jsonl,"r",encoding="utf-8") as fr:
        for line in fr:
            raws.append(json.loads(line))
    q_ids=[]
    toks=tokenize_jp(q_text or "")
    for t in toks:
        if t in vocab: q_ids.append(vocab[t])
    q_ids=np.array(q_ids, dtype=np.int32) if q_ids else np.array([-1], dtype=np.int32)
    results=[]
    for i, doc_ids in enumerate(ix["docs_tokens"]):
        di = np.array(doc_ids + [-1], dtype=np.int32)
        dl = len(doc_ids)
        sb = (_bm25_numba_score(N, avgdl, df, dl, q_ids, di) if _HAS_NUMBA else _bm25_py_score(N, avgdl, df, dl, q_ids, di))
        ob = raws[i]
        sk = _kw_meta_boost(ob, toks, domain)
        si = _phash_sim(q_image, (ob.get("meta") or {}).get("phash64") or 0)
        filters = ((ob.get("meta") or {}).get("filters") or {})
        sym = _symbolic_match_score(filters, q_text or "", list(toks))
        score = w_bm25*sb + w_kw*sk + w_img*si + w_sym*sym
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
    results.sort(key=lambda x:-x[0])
    return results[:topk]

# --------------- SQL Export ---------------------------
def sql_export(jsonl: str, outdir: str, prefix: str="invoice"):
    os.makedirs(outdir, exist_ok=True)
    csv_path=os.path.join(outdir, f"{prefix}_cells.csv")
    schema_path=os.path.join(outdir, f"{prefix}_schema.sql")
    cols=["doc_id","page","table_index","row","col","text","search_unit","synthesis_window",
          "amount","date","company","address","tax_id","postal_code","phone",
          "tax_rate","qty","unit","subtotal","tax_amount","corporate_id",
          "bbox_x1","bbox_y1","bbox_x2","bbox_y2","confidence","low_conf","phash64","lambda_shape","trace"]
    with open(csv_path,"w",encoding="utf-8-sig",newline="") as fw:
        wr=csv.writer(fw); wr.writerow(cols)
        with open(jsonl,"r",encoding="utf-8") as fr:
            for line in fr:
                ob=json.loads(line); meta=(ob.get("meta") or {}); filt=meta.get("filters",{})
                x1,y1,x2,y2=ob.get("bbox",[0,0,0,0])
                wr.writerow([ob.get("doc_id"),ob.get("page"),ob.get("table_index"),ob.get("row"),ob.get("col"),
                             ob.get("text"),ob.get("search_unit"),ob.get("synthesis_window"),
                             filt.get("amount"),filt.get("date"),filt.get("company"),filt.get("address"),filt.get("tax_id"),
                             filt.get("postal_code"),filt.get("phone"),filt.get("tax_rate"),filt.get("qty"),filt.get("unit"),filt.get("subtotal"),filt.get("tax_amount"),filt.get("corporate_id"),
                             x1,y1,x2,y2, meta.get("confidence"), meta.get("low_conf"), meta.get("phash64"), meta.get("lambda_shape"),
                             filt.get("trace")])
    schema=f"""
CREATE TABLE IF NOT EXISTS {prefix}_cells (
  doc_id TEXT, page INT, table_index INT, row INT, col INT,
  text TEXT, search_unit TEXT, synthesis_window TEXT,
  amount BIGINT, date TEXT, company TEXT, address TEXT, tax_id TEXT,
  postal_code TEXT, phone TEXT, tax_rate REAL, qty BIGINT, unit TEXT, subtotal BIGINT, tax_amount BIGINT, corporate_id TEXT,
  bbox_x1 INT, bbox_y1 INT, bbox_x2 INT, bbox_y2 INT,
  confidence REAL, low_conf BOOLEAN, phash64 BIGINT, lambda_shape REAL, trace TEXT
);
-- COPY {prefix}_cells FROM '{csv_path}' WITH CSV HEADER;
"""
    open(schema_path,"w",encoding="utf-8").write(schema)
    return {"csv":csv_path,"schema":schema_path}


def _build_trace(doc_id: Optional[str], page: int, table_idx: Optional[int],
                 row_idx: Optional[int], col_idx: Optional[int], bbox: List[Any]) -> Tuple[Dict[str, Any], str]:
    trace_dict = {
        "doc_id": doc_id,
        "page": int(page) if page is not None else None,
        "table_index": int(table_idx) if table_idx is not None else None,
        "row": int(row_idx) if row_idx is not None else None,
        "col": int(col_idx) if col_idx is not None else None,
        "bbox": bbox,
    }
    label_parts = [
        f"doc={doc_id if doc_id is not None else 'NA'}",
        f"page={page if page is not None else 'NA'}",
        f"table={table_idx if table_idx is not None else 'NA'}",
        f"row={row_idx if row_idx is not None else 'NA'}",
        f"col={col_idx if col_idx is not None else 'NA'}",
    ]
    trace_label = ";".join(label_parts)
    return trace_dict, trace_label


def _fact_tag(text: str, trace_label: str, lang: Optional[str]) -> str:
    payload = {
        "trace": trace_label,
        "lang": lang or "",
    }
    attrs = " ".join(
        f"{k}={html.escape(str(v), quote=True)!r}" for k, v in payload.items() if v is not None
    )
    body = html.escape(text or "", quote=False)
    return f"<fact {attrs}>{body}</fact>"


def export_rag_bundle(jsonl: str, outdir: str, domain: Optional[str]=None,
                      summary: Optional[Dict[str, Any]]=None, limit_per_section: int=40) -> Dict[str, Any]:
    """Generate a multi-view RAG bundle (cells / sections / tables / markdown)."""
    if not os.path.exists(jsonl):
        raise FileNotFoundError(jsonl)
    os.makedirs(outdir, exist_ok=True)
    cells_path = os.path.join(outdir, "cells.jsonl")
    sections_path = os.path.join(outdir, "sections.jsonl")
    tables_path = os.path.join(outdir, "tables.json")
    manifest_path = os.path.join(outdir, "manifest.json")
    markdown_path = os.path.join(outdir, "bundle.md")

    resolved = None
    if domain:
        resolved = _DOMAIN_ALIAS.get(domain, domain)
    if not resolved:
        resolved = "invoice_jp_v2"
    suggested = DOMAIN_SUGGESTED_QUERIES.get(resolved, DOMAIN_SUGGESTED_QUERIES["default"])

    doc_ids: set = set()
    languages: set = set()
    page_sections: Dict[int, Dict[str, Any]] = {}
    tables: Dict[str, Dict[str, Any]] = {}
    cells_written = 0

    with open(jsonl, "r", encoding="utf-8") as fr, open(cells_path, "w", encoding="utf-8") as fw:
        for idx, line in enumerate(fr):
            try:
                ob = json.loads(line)
            except Exception:
                continue
            doc_ids.add(ob.get("doc_id"))
            meta = (ob.get("meta") or {})
            filters = dict(meta.get("filters") or {})
            lang = meta.get("lang") or ob.get("lang")
            if isinstance(lang, str) and lang:
                languages.add(lang)
            text = ob.get("text") or ""
            normalized = _normalize_text(text)
            synth = _normalize_text(ob.get("synthesis_window") or "")
            cell_id = ob.get("cell_id") or ob.get("id") or f"cell_{idx:05d}"
            page = int(ob.get("page") or 0)
            table_idx = meta.get("table_id", ob.get("table_index"))
            row_idx = ob.get("row")
            col_idx = ob.get("col")
            bbox = ob.get("bbox") or [None, None, None, None]
            trace_dict, trace_label = _build_trace(ob.get("doc_id"), page, table_idx, row_idx, col_idx, bbox)
            filters.setdefault("trace", trace_label)
            base_meta = {k: v for k, v in meta.items() if k != "filters"}
            base_meta.setdefault("trace", trace_dict)

            embedding_pieces: List[str] = []
            for piece in (normalized, synth):
                if piece:
                    embedding_pieces.append(piece)
            for key in ("amount","date","company","address","tax_id","postal_code","phone","tax_rate",
                        "qty","unit","subtotal","tax_amount","corporate_id"):
                val = filters.get(key)
                if val is None:
                    continue
                emb_val = _normalize_text(val)
                if emb_val:
                    embedding_pieces.append(f"{key}:{emb_val}")
            seen_kw: List[str] = []
            for kw, _w in DOMAIN_KW.get(resolved, []):
                if not kw:
                    continue
                low_kw = kw.lower()
                if kw in text or kw in synth or low_kw in normalized.lower():
                    if kw not in seen_kw:
                        seen_kw.append(kw)
            embedding_hint = " | ".join(dict.fromkeys(embedding_pieces))

            cell_payload = {
                "cell_id": cell_id,
                "doc_id": ob.get("doc_id"),
                "page": page,
                "table_index": table_idx,
                "row": row_idx,
                "col": col_idx,
                "text": text,
                "normalized": normalized,
                "synthesis_window": ob.get("synthesis_window"),
                "filters": filters,
                "meta": base_meta,
                "bbox": bbox,
                "confidence": (meta.get("confidence") if isinstance(meta, dict) else None),
                "low_conf": (meta.get("low_conf") if isinstance(meta, dict) else None),
                "embedding_hint": embedding_hint,
                "domain": resolved,
                "domain_hits": seen_kw,
                "trace": trace_label,
                "trace_dict": trace_dict,
                "fact_tag": _fact_tag(text, trace_label, lang if isinstance(lang, str) else None),
            }
            fw.write(json.dumps(cell_payload, ensure_ascii=False) + "\n")
            cells_written += 1

            section = page_sections.setdefault(page, {
                "section_id": f"page-{page:03d}",
                "page": page,
                "title": f"Page {page}",
                "cells": [],
                "body": [],
                "facts": [],
            })
            section["cells"].append(cell_id)
            if normalized:
                section["body"].append(f"[{cell_id}] {normalized}")
            section["facts"].append(cell_payload["fact_tag"])

            if table_idx is not None:
                tkey = str(table_idx)
                table = tables.setdefault(tkey, {
                    "table_id": tkey,
                    "cells": [],
                    "rows": {},
                    "facts": [],
                })
                table["cells"].append(cell_id)
                table["facts"].append(cell_payload["fact_tag"])
                if row_idx is not None and col_idx is not None:
                    row_key = str(row_idx)
                    col_key = str(col_idx)
                    table.setdefault("rows", {})
                    row_bucket = table["rows"].setdefault(row_key, {})
                    row_bucket[col_key] = {
                        "cell_id": cell_id,
                        "text": text,
                        "normalized": normalized,
                        "filters": filters,
                        "trace": trace_label,
                    }

    def _sorted_numeric(keys: List[str]) -> List[str]:
        def _key(k: str):
            try:
                return (0, int(k))
            except Exception:
                return (1, k)
        return sorted(keys, key=_key)

    with open(sections_path, "w", encoding="utf-8") as fw:
        for page in sorted(page_sections.keys()):
            sec = page_sections[page]
            body_lines = sec.get("body", [])
            if limit_per_section and len(body_lines) > limit_per_section:
                body_lines = body_lines[:limit_per_section] + ["..."]
            payload = {
                "section_id": sec["section_id"],
                "title": sec["title"],
                "page": sec["page"],
                "cells": sec["cells"],
                "body": "\n".join(body_lines),
                "fact_tags": sec.get("facts", []),
            }
            fw.write(json.dumps(payload, ensure_ascii=False) + "\n")
        for table_id in _sorted_numeric(list(tables.keys())):
            table = tables[table_id]
            rows_out = []
            for row_key in _sorted_numeric(list(table.get("rows", {}).keys())):
                cols = table["rows"][row_key]
                ordered = []
                for col_key in _sorted_numeric(list(cols.keys())):
                    cell_info = cols[col_key]
                    ordered.append({
                        "cell_id": cell_info.get("cell_id"),
                        "col": col_key,
                        "text": cell_info.get("text"),
                        "normalized": cell_info.get("normalized"),
                        "filters": cell_info.get("filters"),
                        "trace": cell_info.get("trace"),
                    })
                rows_out.append({"row_index": row_key, "cells": ordered})
            payload = {
                "section_id": f"table-{table_id}",
                "title": f"Table {table_id}",
                "table_id": table_id,
                "cells": table["cells"],
                "rows": rows_out,
                "fact_tags": table.get("facts", []),
            }
            fw.write(json.dumps(payload, ensure_ascii=False) + "\n")

    tables_payload = []
    for table_id in _sorted_numeric(list(tables.keys())):
        table = tables[table_id]
        rows_out = []
        for row_key in _sorted_numeric(list(table.get("rows", {}).keys())):
            cols = table["rows"][row_key]
            ordered = []
            for col_key in _sorted_numeric(list(cols.keys())):
                cell_info = cols[col_key]
                ordered.append({
                    "cell_id": cell_info.get("cell_id"),
                    "row": row_key,
                    "col": col_key,
                    "text": cell_info.get("text"),
                    "normalized": cell_info.get("normalized"),
                    "trace": cell_info.get("trace"),
                })
            rows_out.append({"row_index": row_key, "cells": ordered})
        tables_payload.append({"table_id": table_id, "rows": rows_out, "fact_tags": table.get("facts", [])})
    with open(tables_path, "w", encoding="utf-8") as tf:
        json.dump(tables_payload, tf, ensure_ascii=False, indent=2)

    now = datetime.datetime.utcnow().isoformat() + "Z"
    languages_list = sorted(languages)
    doc_ids_list = sorted([d for d in doc_ids if d])

    manifest: Dict[str, Any] = {
        "bundle_dir": os.path.abspath(outdir),
        "generated_at": now,
        "source_jsonl": jsonl,
        "domain": domain,
        "resolved_domain": resolved,
        "cell_count": cells_written,
        "page_sections": len(page_sections),
        "table_sections": len(tables),
        "doc_ids": doc_ids_list,
        "languages": languages_list,
        "paths": {
            "cells": cells_path,
            "sections": sections_path,
            "tables": tables_path,
            "markdown": markdown_path,
        },
        "suggested_queries": suggested,
        "embedding_fields": ["text", "normalized", "synthesis_window", "filters", "meta"],
        "trace_schema": {
            "label": "doc/page/table/row/col locator",
            "format": "doc=<id>;page=<int>;table=<int|NA>;row=<int|NA>;col=<int|NA>",
            "fields": ["doc_id", "page", "table_index", "row", "col", "bbox"],
        },
        "fact_tag_example": _fact_tag("Total", "doc=DEMO;page=1;table=NA;row=NA;col=NA", None),
    }
    if summary:
        manifest["summary_snapshot"] = {
            k: summary.get(k)
            for k in ("contextual_jsonl", "mm_jsonl", "sql_csv", "sql_schema", "monitor_csv")
        }

    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)
    manifest["manifest"] = manifest_path

    md_lines = ["# Z-OCR RAG Bundle / バンドル概要", "",
                f"- Generated / 生成日時: {now}",
                f"- Domain / ドメイン: {resolved}",
                f"- Cells / セル数: {cells_written}",
                f"- Pages / ページ: {len(page_sections)}",
                f"- Tables / テーブル: {len(tables)}"]
    if doc_ids_list:
        md_lines.append("- Document IDs / 文書ID: " + ", ".join(doc_ids_list))
    if languages_list:
        md_lines.append("- Languages / 言語: " + ", ".join(languages_list))
    md_lines.append("")
    if suggested:
        md_lines.append("## Suggested queries / 推奨クエリ")
        md_lines.append("```")
        for q in suggested:
            md_lines.append(q)
        md_lines.append("```")
        md_lines.append("")

    def _append_section_preview(title: str, lines: List[str]):
        if not lines:
            return
        preview = lines
        if limit_per_section and len(preview) > limit_per_section:
            preview = preview[:limit_per_section] + ["..."]
        md_lines.append(title)
        md_lines.append("```")
        md_lines.extend(preview)
        md_lines.append("```")
        md_lines.append("")

    for page in sorted(page_sections.keys()):
        sec = page_sections[page]
        _append_section_preview(f"### Page {page} / ページ {page}", sec.get("body", []))
        _append_section_preview(f"### Page {page} facts / ページ{page} ファクトタグ", sec.get("facts", []))
    for table_id in _sorted_numeric(list(tables.keys())):
        table = tables[table_id]
        body_lines: List[str] = []
        for row_key in _sorted_numeric(list(table.get("rows", {}).keys())):
            cols = table["rows"][row_key]
            parts = []
            for col_key in _sorted_numeric(list(cols.keys())):
                cell_info = cols[col_key]
                parts.append(f"[{col_key}] {cell_info.get('normalized') or cell_info.get('text') or ''}")
            body_lines.append(f"row {row_key}: " + " | ".join(parts))
        _append_section_preview(f"### Table {table_id} / テーブル {table_id}", body_lines)
        _append_section_preview(f"### Table {table_id} facts / テーブル{table_id} ファクトタグ", table.get("facts", []))

    with open(markdown_path, "w", encoding="utf-8") as mf:
        mf.write("\n".join(md_lines))

    manifest.update({
        "cells": cells_path,
        "sections": sections_path,
        "tables_json": tables_path,
        "markdown": markdown_path,
    })
    return manifest

# --------------- Monitoring (KPI) ----------------------
def _read_views_log(views_log: str) -> Dict[str, Set]:
    """
    JSONL 形式の Views/補完ログを読む。
    戻り値: {"reprocess": set(cell_keys), "success": set(cell_keys)}
    cell_key = (doc_id,page,table_index,row,col)
    """
    R=set(); S=set()
    if not views_log or not os.path.exists(views_log): return {"reprocess":R,"success":S}
    with open(views_log,"r",encoding="utf-8") as f:
        for line in f:
            try:
                ob=json.loads(line)
                key=(ob.get("doc_id"), int(ob.get("page",0)), int(ob.get("table_index",0)), int(ob.get("row",0)), int(ob.get("col",0)))
                ev=ob.get("event")
                if ev in ("reprocess","view_reprocess","llm_completion","reocr"): R.add(key)
                if ev in ("reocr_success","llm_completion_success"): S.add(key)
            except Exception:
                continue
    return {"reprocess":R, "success":S}

def _read_views_sets(views_log: Optional[str]) -> Tuple[Set, Set]:
    if not views_log:
        return set(), set()
    try:
        logs = _read_views_log(views_log)
        return set(logs.get("reprocess", set())), set(logs.get("success", set()))
    except Exception:
        return set(), set()

def _read_gt(gt_jsonl: Optional[str]) -> Dict[str, Set]:
    G: Dict[str, Set] = {"amount": set(), "date": set()}
    if not gt_jsonl or not os.path.exists(gt_jsonl):
        return G
    with open(gt_jsonl,"r",encoding="utf-8") as f:
        for line in f:
            try:
                ob=json.loads(line)
            except Exception:
                continue
            lab=str(ob.get("label","")).lower()
            if lab not in G:
                continue
            key=(ob.get("doc_id"), int(ob.get("page",0)), int(ob.get("table_index",0)), int(ob.get("row",0)), int(ob.get("col",0)))
            G[lab].add(key)
    return G


_INVOICE_GATE_DOMAINS = {"invoice", "invoice_jp_v2", "invoice_en", "invoice_fr"}


def _evaluate_gate(
    domain: Optional[str],
    amount_score: Optional[float],
    date_score: Optional[float],
    due_score: Optional[float],
    corporate_rate: Optional[float],
    tax_fail_rate: Optional[float],
) -> Tuple[bool, str, float]:
    resolved = _DOMAIN_ALIAS.get(domain or "", domain or "")
    amt = float(amount_score) if amount_score is not None else 0.0
    dt = float(date_score) if date_score is not None else None
    due = float(due_score) if due_score is not None else None
    corp = float(corporate_rate) if corporate_rate is not None else None
    tax_fail = float(tax_fail_rate) if tax_fail_rate is not None else None
    if resolved in _INVOICE_GATE_DOMAINS:
        if dt is None:
            return False, "date missing", min(amt, 0.0)
        if amt < 0.8:
            return False, "amount below gate", amt
        if dt < 0.5:
            return False, "date below gate", dt
        if due is None:
            return False, "due missing", min(amt, dt)
        if due < 0.4:
            return False, "due below gate", due
        if corp is None or corp < 0.6:
            return False, "corporate match low", corp or 0.0
        if tax_fail is not None and tax_fail > 0.15:
            return False, "tax mismatch high", 1.0 - tax_fail
        components = [amt, dt, due, corp if corp is not None else 1.0]
        if tax_fail is not None:
            components.append(max(0.0, 1.0 - tax_fail))
        score = min(components)
        return True, "amount+date+due+corp", score
    if amount_score is None or date_score is None:
        return False, "insufficient metrics", 0.0
    mean = (float(amount_score) + float(date_score)) / 2.0
    if mean >= 0.95:
        return True, "hit_mean>=0.95", mean
    return False, "hit_mean<0.95", mean

# --------------- CLI ----------------------------------
def main():
    import argparse
    ap=argparse.ArgumentParser("ZOCR Multi‑Domain Core (all-in-one)")
    sub=ap.add_subparsers(dest="cmd")

    sp=sub.add_parser("buildc"); sp.add_argument("--outdir", default="out_lib")

    sp=sub.add_parser("augment")
    sp.add_argument("--jsonl", required=True); sp.add_argument("--out", required=True)
    sp.add_argument("--lambda-shape", type=float, default=4.5)
    sp.add_argument("--lambda-refheight", type=int, default=1000)
    sp.add_argument("--lambda-alpha", type=float, default=0.7)
    sp.add_argument("--org-dict", default=None)
    sp.add_argument("--domain", default="invoice")

    sp=sub.add_parser("index"); sp.add_argument("--jsonl", required=True); sp.add_argument("--index", required=True)

    sp=sub.add_parser("query")
    sp.add_argument("--jsonl", required=True); sp.add_argument("--index", required=True)
    sp.add_argument("--q", default=""); sp.add_argument("--image", default=None); sp.add_argument("--topk", type=int, default=10)
    sp.add_argument("--w-bm25", type=float, default=1.0); sp.add_argument("--w-kw", type=float, default=0.6); sp.add_argument("--w-img", type=float, default=0.3)
    sp.add_argument("--w-sym", type=float, default=0.45)
    sp.add_argument("--domain", default="invoice")

    sp=sub.add_parser("sql"); sp.add_argument("--jsonl", required=True); sp.add_argument("--outdir", required=True); sp.add_argument("--prefix", default="invoice")

    sp=sub.add_parser("monitor")
    sp.add_argument("--jsonl", required=True); sp.add_argument("--index", required=True)
    sp.add_argument("--k", type=int, default=10); sp.add_argument("--views-log", default=None); sp.add_argument("--gt-jsonl", default=None)
    sp.add_argument("--out", required=True); sp.add_argument("--domain", default="invoice")

    args=ap.parse_args()

    if args.cmd=="buildc":
        buildc(args.outdir); return
    if args.cmd=="augment":
        n=augment(args.jsonl, args.out, args.lambda_shape, args.lambda_refheight, args.lambda_alpha, args.org_dict)
        print(f"Augmented {n} -> {args.out} (domain={args.domain})"); return
    if args.cmd=="index":
        build_index(args.jsonl, args.index); print(f"Indexed -> {args.index}"); return
    if args.cmd=="query":
        res=query(args.index, args.jsonl, args.q, args.image, args.topk, args.w_bm25, args.w_kw, args.w_img, args.w_sym, args.domain)
        for i,(s,ob) in enumerate(res,1):
            f=(ob.get("meta") or {}).get("filters",{})
            scores=(ob.get("meta") or {}).get("retrieval_scores", {})
            print(f"{i:2d}. {s:.3f} page={ob.get('page')} r={ob.get('row')} c={ob.get('col')} sym={scores.get('symbolic',0):.2f} "
                  f"amt={f.get('amount')} date={f.get('date')} tax={f.get('tax_rate')} "
                  f"qty={f.get('qty')} unit={f.get('unit')} sub={f.get('subtotal')} tax_amt={f.get('tax_amount')} "
                  f"corp={f.get('corporate_id')} text='{(ob.get('text') or '')[:40]}'")
        return
    if args.cmd=="sql":
        p=sql_export(args.jsonl, args.outdir, args.prefix); print("SQL:", p); return
    if args.cmd=="monitor":
        row=monitor(args.jsonl, args.index, args.k, args.out,
                    views_log=args.views_log, gt_jsonl=args.gt_jsonl, domain=args.domain)
        print("Monitor:", row)
        if row.get("gate_pass"):
            print(f"GATE: PASS ({row.get('gate_reason')})")
        else:
            print(f"GATE: FAIL ({row.get('gate_reason')})")
        return

    ap.print_help()

if __name__=="__main__":
    main()

# ===================== Robust p95 + Column Smoothing Hook =====================

def _preload_index_and_raws(index_pkl: str, jsonl: str):
    """Load index + raw JSONL once for repeated timing queries."""
    with open(index_pkl,"rb") as f:
        ix=pickle.load(f)
    raws=[]
    with open(jsonl,"r",encoding="utf-8") as fr:
        for line in fr:
            raws.append(json.loads(line))
    return ix, raws

def _query_scores_preloaded(ix: Dict[str,Any], raws: List[Dict[str,Any]], q_text: str,
                            domain: Optional[str], w_kw: float, w_img: float, w_sym: float) -> float:
    """Return only the max score (top-1) to emulate typical scoring cost while reducing allocation."""
    vocab=ix["vocab"]; df=np.array(ix["df"], dtype=np.int32); N=int(ix["N"]); avgdl=float(ix["avgdl"])
    tokens = tokenize_jp(q_text or "")
    q_ids=[vocab[t] for t in tokens if t in vocab]
    if not q_ids: q_ids=[-1]
    q_ids=np.array(q_ids, dtype=np.int32)
    best=-1e9
    for i, doc_ids in enumerate(ix["docs_tokens"]):
        di=np.array(doc_ids+[-1], dtype=np.int32)
        dl=len(doc_ids)
        sb = (_bm25_numba_score(N, avgdl, df, dl, q_ids, di) if _HAS_NUMBA else _bm25_py_score(N, avgdl, df, dl, q_ids, di))
        ob=raws[i]
        sk=_kw_meta_boost(ob, tokens, domain)
        filters = ((ob.get("meta") or {}).get("filters") or {})
        sym = _symbolic_match_score(filters, q_text or "", tokens)
        # no q_image for timing; pHash sim is 0
        s=(1.0*sb + w_kw*sk + w_img*0.0 + w_sym*sym)
        if s>best: best=s
    return float(best)

def _time_queries_preloaded(ix: Dict[str,Any], raws: List[Dict[str,Any]], domain: Optional[str],
                            w_kw: float, w_img: float, w_sym: float,
                            trials: int = 60, warmup: int = 8) -> Dict[str,float]:
    """Warm-up + fixed number of trials for robust p95."""
    import time, random
    queries = {
        "invoice":        ["合計","金額","消費税","小計","請求","振込"],
        "invoice_jp_v2": ["合計","金額","消費税","小計","請求日","発行日"],
        "invoice_en":    ["invoice total", "amount due", "tax", "balance", "payment"],
        "invoice_fr":    ["facture", "montant", "tva", "total", "date"],
        "purchase_order":["purchase order", "po", "vendor", "ship", "qty"],
        "expense":       ["expense", "category", "total", "tax", "reimburse"],
        "timesheet":     ["timesheet", "hours", "project", "rate", "total"],
        "shipping_notice":["shipment", "tracking", "carrier", "delivery", "ship"],
        "medical_receipt":["診療", "点数", "保険", "負担金", "薬剤"],
        "delivery":      ["納品", "数量", "受領", "出荷", "品名"],
        "delivery_jp":   ["納品", "数量", "品番", "伝票", "受領"],
        "delivery_en":   ["delivery", "tracking", "carrier", "qty", "item"],
        "estimate":      ["見積", "単価", "小計", "有効期限"],
        "estimate_jp":   ["御見積金額", "見積金額", "有効期限", "納期"],
        "estimate_en":   ["estimate", "quote", "valid", "subtotal", "project"],
        "receipt":       ["領収", "合計", "発行日", "住所", "税込"],
        "receipt_jp":    ["領収書", "税込", "受領", "発行日", "現金"],
        "receipt_en":    ["receipt", "paid", "total", "tax", "cash"],
        "contract":      ["契約", "締結", "署名", "条", "甲"],
        "contract_jp_v2":["契約", "甲", "乙", "条", "締結日", "署名"],
        "contract_en":   ["contract", "signature", "party", "term", "agreement"],
        "rental_agreement_en": ["monthly rent", "lease", "tenant", "landlord", "deposit"],
        "rental_agreement_jp": ["賃貸借", "賃料", "借主", "貸主", "敷金"],
        "loan_statement_en": ["loan", "interest", "principal", "installment", "balance"],
        "loan_statement_jp": ["返済", "利息", "元金", "残高", "返済日"],
        "travel_itinerary_en": ["itinerary", "flight", "departure", "arrival", "hotel"],
        "travel_itinerary_jp": ["旅程", "出発", "到着", "航空券", "宿泊"],
    }
    fallback = queries["invoice_jp_v2"]
    dom_q = queries.get(domain or "invoice_jp_v2", fallback)
    # deterministic seed for reproducibility
    rnd = random.Random(0x5A17)
    lat=[]
    total=warmup+trials
    for t in range(total):
        q = " ".join(rnd.sample(dom_q, min(3,len(dom_q))))
        t0=time.perf_counter()
        _ = _query_scores_preloaded(ix, raws, q_text=q, domain=domain, w_kw=w_kw, w_img=w_img, w_sym=w_sym)
        dt=(time.perf_counter()-t0)*1000.0
        if t>=warmup:
            lat.append(dt)
    if not lat:
        return {"p50":None,"p95":None}
    lat=sorted(lat)
    p50 = lat[int(0.50*(len(lat)-1))]
    p95 = lat[int(0.95*(len(lat)-1))]
    return {"p50":float(p50), "p95":float(p95)}

# ---------- Column smoothing alignment metric (direct hook to objective) ----------

def _prepare_alignment_cache(jsonl_mm: str):
    """Precompute table matrices (left/right per row/col) once from JSONL."""
    from collections import defaultdict
    by_tbl=defaultdict(list)
    with open(jsonl_mm,"r",encoding="utf-8") as f:
        for line in f:
            ob=json.loads(line)
            key=(ob.get("doc_id"), int(ob.get("page",0)), int(ob.get("table_index",0)))
            by_tbl[key].append(ob)
    tbls=[]
    for key, cells in by_tbl.items():
        # infer dims
        max_r=max(int(c.get("row",0)) for c in cells)+1
        max_c=max(int(c.get("col",0)) for c in cells)+1
        left=[[None]*max_c for _ in range(max_r)]
        right=[[None]*max_c for _ in range(max_r)]
        # height (prefer meta.page_height; else from bbox)
        H=None; ymax=0
        for c in cells:
            meta=c.get("meta") or {}
            if H is None and meta.get("page_height"): H=int(meta["page_height"])
            x1,y1,x2,y2=c.get("bbox",[0,0,0,0])
            ymax=max(ymax, int(y2))
            r=int(c.get("row",0)); co=int(c.get("col",0))
            left[r][co]=int(x1); right[r][co]=int(x2)
        if H is None: H=int(max(1000, ymax))
        tbls.append({"H":H,"left":left,"right":right})
    return tbls

def _second_diff_energy(vec: np.ndarray) -> float:
    if vec.shape[0] < 3: return 0.0
    v=0.0
    for i in range(1, vec.shape[0]-1):
        d = vec[i+1] - 2.0*vec[i] + vec[i-1]
        v += float(d*d)
    return v / float(vec.shape[0]-2)

def metric_col_over_under_rate(jsonl_mm: str) -> float:
    meta_values: List[float] = []
    table_rows: Dict[Tuple[Any, Any, Any], Dict[Any, Set[int]]] = defaultdict(lambda: defaultdict(set))
    if not os.path.exists(jsonl_mm):
        return 1.0
    with open(jsonl_mm, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ob = json.loads(line)
            except Exception:
                continue
            meta = ob.get("meta") or {}
            val = meta.get("col_over_under")
            if isinstance(val, (int, float)):
                meta_values.append(float(val))
            key = (ob.get("doc_id"), ob.get("page"), ob.get("table_index"))
            row = ob.get("row")
            col = ob.get("col")
            if row is None or col is None:
                continue
            try:
                r = int(row)
                c = int(col)
            except Exception:
                continue
            table_rows[key][r].add(c)
    if meta_values:
        return max(0.01, float(sum(meta_values) / max(1, len(meta_values))))

    diff_sum = 0.0
    count = 0
    for rows in table_rows.values():
        lengths = [len(cols) for cols in rows.values() if cols]
        if not lengths:
            continue
        freq = Counter(lengths)
        mode_len = freq.most_common(1)[0][0]
        for L in lengths:
            count += 1
            diff_sum += abs(L - mode_len) / max(1, mode_len)
    if count == 0:
        return 1.0
    return max(0.01, diff_sum / count)

def metric_chunk_consistency(jsonl_mm: str) -> float:
    chunk_map: Dict[Tuple[Any, Any], Counter] = defaultdict(Counter)
    if not os.path.exists(jsonl_mm):
        return 1.0
    with open(jsonl_mm, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ob = json.loads(line)
            except Exception:
                continue
            meta = ob.get("meta") or {}
            chunk = (
                meta.get("chunk_id")
                or meta.get("chunk")
                or meta.get("section_id")
                or meta.get("section_key")
                or ob.get("chunk_id")
                or ob.get("chunk")
            )
            if chunk is None:
                continue
            key = (ob.get("doc_id"), ob.get("page"))
            chunk_map[key][str(chunk)] += 1
    total = 0
    aligned = 0
    for counter in chunk_map.values():
        vals = list(counter.values())
        if not vals:
            continue
        total += sum(vals)
        aligned += counter.most_common(1)[0][1]
    if total == 0:
        return 1.0
    return float(aligned) / float(total)

def metric_col_alignment_energy_cached(tbl_cache: List[Dict[str,Any]], lambda_shape: float, height_ref: float=1000.0, exp: float=0.7) -> float:
    """
    Return ratio E_after/E_before (<=1.0 is better). If no curvature, returns 1.0 (neutral).
    Uses the exact smoothing operator (D² penalized tri-diagonal) with lam_eff schedule.
    """
    num=0.0; den=0.0
    for tb in tbl_cache:
        H=tb["H"]
        lam_eff=lambda_schedule(H, lambda_shape, height_ref, exp)
        for mat in [tb["left"], tb["right"]]:
            # iterate columns
            max_r = len(mat)
            max_c = len(mat[0]) if max_r>0 else 0
            for co in range(max_c):
                arr=[mat[r][co] for r in range(max_r)]
                idx=[i for i,v in enumerate(arr) if v is not None]
                if len(idx)<3: continue
                y=np.array([arr[i] for i in idx], dtype=np.float64)
                # before energy
                e_before=_second_diff_energy(y)
                if e_before<=1e-9:
                    # perfectly straight already; count neutral
                    num+=1.0; den+=1.0
                    continue
                # smooth
                a,b,c=_second_diff_tridiag(len(y), lam_eff)
                x=thomas_tridiag(a,b,c,y)
                e_after=_second_diff_energy(x)
                num += float(e_after)
                den += float(e_before)
    if den<=0.0: return 1.0
    return float(num/den)

# ---------- Replace autotune_unlabeled with smoothing-aware + robust p95 ----------

def autotune_unlabeled(jsonl_mm: str, index_pkl: str, outdir: str, method: str="random", budget: int=30, domain_hint: Optional[str]=None, seed:int=0,
                       p95_target_ms: float=300.0, use_smoothing_metric: bool=True) -> Dict[str,Any]:
    """
    Unlabeled微調整ループ（改）:
      score = 列過不足率 × (p95/p95_target) × (1 - chunk整合度 + 0.05) × f(列アライン比)
      f(列アライン比) = 0.3 + 0.7*(E_after/E_before)  ※ <=1.0 ほど良い（1未満でスコア低減）
    """
    import random as pyrand
    np.random.seed(seed); pyrand.seed(seed)
    os.makedirs(outdir, exist_ok=True)

    domain,_ = detect_domain_on_jsonl(jsonl_mm)
    if domain_hint: domain = domain_hint
    base = DOMAIN_DEFAULTS.get(domain, DOMAIN_DEFAULTS["invoice_jp_v2"])

    # Precompute fixed metrics
    col_rate = metric_col_over_under_rate(jsonl_mm)
    chunk_c  = metric_chunk_consistency(jsonl_mm)

    # Preload for robust timing
    ix, raws = _preload_index_and_raws(index_pkl, jsonl_mm)

    # Prepare cache for smoothing metric
    tbl_cache = _prepare_alignment_cache(jsonl_mm) if use_smoothing_metric else None

    log_rows=[]
    best=None

    def _score(p95, lam_shape):
        p95n = (p95 or p95_target_ms)/max(1.0,p95_target_ms)
        if use_smoothing_metric and tbl_cache is not None:
            align_ratio = metric_col_alignment_energy_cached(tbl_cache, lam_shape, 1000.0, 0.7)  # <=1 better
            f_align = 0.3 + 0.7*float(align_ratio)
        else:
            f_align = 1.0
        return col_rate * p95n * (1.0 - chunk_c + 0.05) * f_align, f_align

    # search space
    def sample(center=None, scale=1.0):
        if center is None:
            lam = float(np.random.uniform(1.0, 6.0))
            wkw = float(np.random.uniform(0.3, 0.8))
            wimg= float(np.random.uniform(0.0, 0.5))
            wsym= float(np.random.uniform(0.3, 0.7))
            ocr = float(np.random.uniform(0.4, 0.8))
        else:
            lam = float(np.clip(np.random.normal(center["lambda_shape"], 0.5*scale), 1.0, 6.0))
            wkw = float(np.clip(np.random.normal(center["w_kw"], 0.1*scale), 0.2, 0.9))
            wimg= float(np.clip(np.random.normal(center["w_img"],0.1*scale), 0.0, 0.6))
            wsym= float(np.clip(np.random.normal(center.get("w_sym", base.get("w_sym", 0.45)), 0.08*scale), 0.2, 0.85))
            ocr = float(np.clip(np.random.normal(center["ocr_min_conf"],0.05*scale),0.3,0.9))
        return {"lambda_shape":lam,"w_kw":wkw,"w_img":wimg,"w_sym":wsym,"ocr_min_conf":ocr}

    # Stage 1: random init
    n_init = max(8, min(15, budget//2))
    for i in range(n_init):
        params = sample()
        lat = _time_queries_preloaded(ix, raws, domain, params["w_kw"], params["w_img"], params["w_sym"], trials=48, warmup=8)
        score, f_align=_score(lat["p95"], params["lambda_shape"])
        row={"iter":i,"phase":"init","domain":domain,"col_rate":col_rate,"chunk_c":chunk_c,"p95":lat["p95"],"score":score,"align_factor":f_align,**params}
        log_rows.append(row)
        if best is None or score<best["score"]:
            best=row

    # Stage 2: local refinement
    remain = max(0, budget - n_init)
    for j in range(remain):
        params = sample(center=best, scale=max(0.5, 1.5*(remain-j)/max(1,remain)))
        lat = _time_queries_preloaded(ix, raws, domain, params["w_kw"], params["w_img"], params["w_sym"], trials=48, warmup=8)
        score, f_align=_score(lat["p95"], params["lambda_shape"])
        row={"iter":n_init+j,"phase":"refine","domain":domain,"col_rate":col_rate,"chunk_c":chunk_c,"p95":lat["p95"],"score":score,"align_factor":f_align,**params}
        log_rows.append(row)
        if score<best["score"]:
            best=row

    # save log
    csv_path=os.path.join(outdir, "autotune_log.csv")
    hdr= ["iter","phase","domain","lambda_shape","w_kw","w_img","w_sym","ocr_min_conf","col_rate","chunk_c","p95","align_factor","score"]
    with open(csv_path,"w",encoding="utf-8-sig",newline="") as fw:
        wr=csv.DictWriter(fw, fieldnames=hdr); wr.writeheader()
        for r in log_rows: wr.writerow({k:r.get(k) for k in hdr})

    # update profile json
    prof_path=os.path.join(outdir,"auto_profile.json")
    try:
        prof=json.load(open(prof_path,"r",encoding="utf-8"))
    except Exception:
        prof={"domain":domain}
    prof.update({
        "domain": domain,
        "lambda_shape": float(best["lambda_shape"]),
        "w_bm25": 1.0,
        "w_kw": float(best["w_kw"]),
        "w_img": float(best["w_img"]),
        "w_sym": float(best.get("w_sym", base.get("w_sym", 0.45))),
        "ocr_min_conf": float(best["ocr_min_conf"]),
        "tune_col_rate": float(col_rate),
        "tune_chunk_c": float(chunk_c),
        "tune_p95": float(best["p95"]) if best["p95"] is not None else None,
        "tune_align_factor": float(best["align_factor"]),
        "tune_score": float(best["score"])
    })
    with open(prof_path,"w",encoding="utf-8") as fw: json.dump(prof, fw, ensure_ascii=False, indent=2)
    return {"best":best,"log_csv":csv_path,"profile_json":prof_path}

# ---------- Monitor: compute p95 when aggregator missing ----------

def _compute_p95_if_needed(jsonl: str, index_pkl: str, domain: Optional[str]) -> Optional[float]:
    try:
        ix, raws = _preload_index_and_raws(index_pkl, jsonl)
        d = domain or detect_domain_on_jsonl(jsonl)[0]
        base = DOMAIN_DEFAULTS.get(d, DOMAIN_DEFAULTS["invoice_jp_v2"])
        lat = _time_queries_preloaded(ix, raws, d, base["w_kw"], base["w_img"], base.get("w_sym", 0.45), trials=60, warmup=8)
        return float(lat["p95"]) if lat["p95"] is not None else None
    except Exception:
        return None

# Patch monitor to fallback p95
def monitor(jsonl: str, index_pkl: str, k: int, out_csv: str, views_log: Optional[str]=None, gt_jsonl: Optional[str]=None, domain: Optional[str]=None):
    total=0; low=0; corp_hits=0; corp_total=0
    lc_keys=set()
    with open(jsonl,"r",encoding="utf-8") as f:
        for line in f:
            ob=json.loads(line); total+=1
            meta=ob.get("meta") or {}; filt=meta.get("filters",{})
            if meta.get("low_conf"):
                lc_keys.add((ob.get("doc_id"), ob.get("page"), ob.get("table_index"), ob.get("row"), ob.get("col")))
                low+=1
            if filt.get("corporate_id") is not None:
                corp_total+=1
                if filt.get("company_canonical"): corp_hits+=1
    low_conf_rate = low/max(1,total)
    corporate_match_rate = (corp_hits/max(1,corp_total)) if corp_total>0 else 0.0

    S_reproc, S_success = _read_views_sets(views_log)
    reprocess_rate = len(S_reproc & lc_keys)/max(1,len(lc_keys)) if lc_keys else 0.0
    reprocess_success_rate = len(S_success & S_reproc)/max(1,len(S_reproc)) if S_reproc else 0.0

    if not os.path.exists(index_pkl): build_index(jsonl, index_pkl)
    G=_read_gt(gt_jsonl)
    def _score(label: str, q: str) -> Tuple[int, Optional[float]]:
        res = query(index_pkl, jsonl, q, None, topk=k, domain=domain)
        if not res:
            return 0, None

        rel = G.get(label) or set()
        good = 0
        hit = 0
        for _score, ob in res:
            key = (
                ob.get("doc_id"),
                ob.get("page"),
                ob.get("table_index"),
                ob.get("row"),
                ob.get("col"),
            )
            filt = (ob.get("meta") or {}).get("filters", {})
            if rel:
                if key in rel:
                    hit = 1
                    good += 1
            else:
                if label == "amount" and filt.get("amount") is not None:
                    hit = 1
                    good += 1
                elif label == "date" and filt.get("date"):
                    hit = 1
                    good += 1
                elif label == "due" and (filt.get("due_date") or filt.get("due") or filt.get("payment_due") or filt.get("deadline")):
                    hit = 1
                    good += 1
        trust = good / len(res) if res else None
        return hit, trust
    domain_key = domain or "default"
    resolved_monitor_key = _DOMAIN_ALIAS.get(domain_key, domain_key)
    monitor_cfg = (
        DOMAIN_MONITOR_QUERIES.get(resolved_monitor_key)
        or DOMAIN_MONITOR_QUERIES.get(domain_key)
        or DOMAIN_MONITOR_QUERIES["default"]
    )
    defaults_monitor = DOMAIN_MONITOR_QUERIES["default"]
    q_amount = monitor_cfg.get("q_amount") or defaults_monitor["q_amount"]
    q_date = monitor_cfg.get("q_date") or defaults_monitor["q_date"]
    q_due = monitor_cfg.get("q_due") or defaults_monitor["q_due"]
    hit_amount, trust_amount = _score("amount", q_amount)
    hit_date, trust_date = _score("date", q_date)
    hit_due, trust_due = _score("due", q_due)
    metrics = [hit_amount, hit_date]
    if hit_due is not None:
        metrics.append(hit_due)
    hit_mean = sum(metrics)/len(metrics) if metrics else 0.0
    trust_vals = [v for v in (trust_amount, trust_date, trust_due) if v is not None]
    trust_mean = sum(trust_vals)/len(trust_vals) if trust_vals else None

    # tax check fail
    tax_fail=0; tax_cov=0
    with open(jsonl,"r",encoding="utf-8") as f:
        for line in f:
            ob=json.loads(line); filt=(ob.get("meta") or {}).get("filters",{})
            if filt.get("tax_amount") is not None and filt.get("tax_amount_expected") is not None:
                tax_cov+=1
                if abs(int(filt["tax_amount"])-int(filt["tax_amount_expected"]))>1:
                    tax_fail+=1
    tax_fail_rate = (tax_fail/max(1,tax_cov)) if tax_cov>0 else 0.0

    p95=None
    agg=os.path.join(os.path.dirname(jsonl),"metrics_aggregate.csv")
    if os.path.exists(agg):
        try:
            import pandas as pd
            df=pd.read_csv(agg)
            if "latency_p95_ms" in df.columns:
                p95=float(df["latency_p95_ms"].iloc[0])
        except Exception:
            p95=None
    if p95 is None:
        p95=_compute_p95_if_needed(jsonl, index_pkl, domain)

    gate_pass, gate_reason, gate_score = _evaluate_gate(
        domain,
        hit_amount,
        hit_date,
        hit_due,
        corporate_match_rate,
        tax_fail_rate,
    )
    row={"timestamp":datetime.datetime.utcnow().isoformat()+"Z","jsonl":jsonl,"K":k,
         "domain": domain or "auto",
         "low_conf_rate":low_conf_rate,"reprocess_rate":reprocess_rate,"reprocess_success_rate":reprocess_success_rate,
         "hit_amount":hit_amount,"hit_date":hit_date,"hit_due":hit_due,"hit_mean":hit_mean,
         "tax_fail_rate":tax_fail_rate,"tax_coverage":tax_cov,
         "corporate_match_rate":corporate_match_rate,"corporate_coverage":corp_total,
         "p95_ms":p95,
         "trust_amount":trust_amount,"trust_date":trust_date,"trust_due":trust_due,"trust_mean":trust_mean,
         "gate_pass":gate_pass,"gate_reason":gate_reason,"gate_score":gate_score}
    hdr=not os.path.exists(out_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv,"a",encoding="utf-8-sig",newline="") as fw:
        wr=csv.DictWriter(fw, fieldnames=list(row.keys()))
        if hdr: wr.writeheader()
        wr.writerow(row)
    print("Monitor:", row)
    if gate_pass:
        print(f"GATE: PASS ({gate_reason})")
    else:
        print(f"GATE: FAIL ({gate_reason})")
    return row

def learn_from_monitor(monitor_csv: str, profile_json_in: Optional[str], profile_json_out: Optional[str]=None,
                       domain_hint: Optional[str]=None, ema: float=0.5) -> Dict[str, Any]:
    if profile_json_out is None:
        profile_json_out = profile_json_in

    profile: Dict[str, Any] = {}
    if profile_json_in and os.path.exists(profile_json_in):
        try:
            with open(profile_json_in, "r", encoding="utf-8") as fr:
                profile = json.load(fr)
        except Exception:
            profile = {}

    metrics: Dict[str, Any] = {}
    if monitor_csv and os.path.exists(monitor_csv):
        try:
            import csv
            with open(monitor_csv, "r", encoding="utf-8-sig", newline="") as fr:
                rows = list(csv.DictReader(fr))
            if rows:
                metrics = rows[-1]
        except Exception:
            metrics = {}

    if metrics:
        numeric_keys = [
            "low_conf_rate", "reprocess_rate", "reprocess_success_rate",
            "hit_amount", "hit_date", "hit_due", "hit_mean", "p95_ms", "tax_fail_rate",
            "tax_coverage", "corporate_match_rate", "corporate_coverage",
            "trust_amount", "trust_date", "trust_due", "trust_mean",
        ]
        for key in numeric_keys:
            if key in metrics:
                try:
                    metrics[key] = float(metrics[key]) if metrics[key] not in (None, "") else None
                except Exception:
                    metrics[key] = None
        profile.setdefault("domain", domain_hint or profile.get("domain"))
        profile["last_monitor"] = metrics

        if "ocr_min_conf" in profile and isinstance(profile["ocr_min_conf"], (int, float)):
            target = float(profile["ocr_min_conf"])
            if metrics.get("low_conf_rate") is not None:
                target = max(0.1, min(0.99, target - 0.25*(metrics["low_conf_rate"] - 0.1)))
            profile["ocr_min_conf"] = round((1.0 - ema) * float(profile["ocr_min_conf"]) + ema * target, 4)

    if profile_json_out:
        try:
            with open(profile_json_out, "w", encoding="utf-8") as fw:
                json.dump(profile, fw, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return {
        "profile_json": profile_json_out,
        "profile": profile,
        "monitor": metrics or None,
    }
'''
zocr_multidomain_core = _materialize_module('zocr_multidomain_core', _SRC_ZOCR_MULTIDOMAIN_CORE)

# ---- Compatibility shim (core may not expose learn_from_monitor) ----
if not hasattr(zocr_multidomain_core, 'learn_from_monitor'):
    def _zocr__learn_from_monitor_shim(*args, **kwargs):
        # Not used by orchestrator in this bundle; present to satisfy import.
        # If future code calls it, delegate to auto_all if present, else raise.
        if hasattr(zocr_multidomain_core, 'auto_all'):
            return zocr_multidomain_core.auto_all(*args, **kwargs)
        raise NotImplementedError('learn_from_monitor is not available in this build')
    zocr_multidomain_core.learn_from_monitor = _zocr__learn_from_monitor_shim

# --- Wire the C-backed solver into consensus once both modules are ready ---
try:
    if getattr(zocr_onefile_consensus, "_thomas", None) is None \
       and hasattr(zocr_multidomain_core, "thomas_tridiag"):
        zocr_onefile_consensus._thomas = zocr_multidomain_core.thomas_tridiag
except Exception:
    pass

# ---------------- [Pipe] all-in-one orchestrator -------------------
_SRC_ZOCR_PIPELINE_ALLINONE = r'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-in-one pipeline orchestrator:
- Calls the "Consensus" one-file OCR to produce `doc.zocr.json`
- Exports contextual JSONL (with OCR) for RAG
- Augments / indexes / monitors via the multi-domain core
- Optionally runs unlabeled tuning + metric-linked learning
- Windows-friendly (no shell tools required except optional Poppler if PDF)

Outputs are consolidated under a single outdir.
"""

import os, sys, json, time, traceback, argparse, random, platform, hashlib, subprocess, importlib, re, glob, shutil, math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set, TypedDict
try:
    from typing import Literal  # py39+
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore
from html import escape

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _np = None  # type: ignore

try:
    from ..consensus import zocr_consensus as zocr_onefile_consensus  # type: ignore
except Exception:
    import zocr_onefile_consensus  # type: ignore

try:
    from ..core import zocr_core as zocr_multidomain_core  # type: ignore
except Exception:
    import zocr_multidomain_core  # type: ignore

if __name__.startswith("zocr."):
    sys.modules.setdefault("zocr_pipeline_allinone", sys.modules[__name__])

PLUGINS = {}
def register(stage):
    def deco(fn):
        PLUGINS.setdefault(stage, []).append(fn); return fn
    return deco
def _call(stage, **kw):
    for fn in PLUGINS.get(stage, []):
        try:
            fn(**kw)
        except Exception as e:
            print(f"[PLUGIN:{stage}] {fn.__name__} -> {e}")

def _json_ready(obj: Any):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, set):
        return [_json_ready(v) for v in obj]
    if _np is not None:
        if isinstance(obj, _np.generic):  # type: ignore[attr-defined]
            return obj.item()
        if isinstance(obj, _np.ndarray):  # type: ignore[attr-defined]
            return obj.tolist()
    return obj

_STAGE_TRACE_SINK: Optional[List[Dict[str, Any]]] = None


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _set_stage_trace_sink(sink: Optional[List[Dict[str, Any]]]) -> None:
    global _STAGE_TRACE_SINK
    _STAGE_TRACE_SINK = sink


def _stage_output_preview(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        interesting = (
            "path",
            "paths",
            "count",
            "records",
            "pages",
            "tables",
            "cells",
            "reason",
            "summary",
            "output",
            "metrics",
        )
        preview: Dict[str, Any] = {}
        for key in interesting:
            if key in value:
                preview[key] = value[key]
        if preview:
            return _json_ready(preview)
        if len(value) <= 4:
            return _json_ready(value)
        return f"{len(value)} keys"
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        if not seq:
            return []
        if len(seq) <= 4 and all(isinstance(item, (bool, int, float, str)) for item in seq):
            return seq
        return f"{len(seq)} items"
    return str(type(value).__name__)


def _record_stage_trace(rec: Dict[str, Any]) -> None:
    if _STAGE_TRACE_SINK is None:
        return
    snapshot: Dict[str, Any] = {
        "name": rec.get("name"),
        "elapsed_ms": float(rec.get("elapsed_ms") or 0.0),
    }
    if rec.get("ok") is None:
        snapshot["ok"] = None
    else:
        snapshot["ok"] = bool(rec.get("ok"))
    if rec.get("error"):
        snapshot["error"] = rec.get("error")
    preview = _stage_output_preview(rec.get("out"))
    if preview is not None:
        snapshot["out"] = preview
    _STAGE_TRACE_SINK.append(snapshot)


def _summarize_stage_preview(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, int, float)):
        return str(value)
    if isinstance(value, str):
        return value if len(value) <= 80 else value[:77] + "..."
    if isinstance(value, dict):
        items: List[str] = []
        for idx, (key, val) in enumerate(value.items()):
            if idx >= 3:
                items.append("…")
                break
            items.append(f"{key}={val}")
        return ", ".join(items)
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        if not seq:
            return ""
        snippet = ", ".join(str(item) for item in seq[:3])
        if len(seq) > 3:
            snippet += ", …"
        return snippet
    return str(value)


def _print_stage_trace_console(stage_trace: List[Dict[str, Any]], stats: Optional[Dict[str, Any]] = None) -> None:
    if not stage_trace:
        return
    print("\n[Stage Trace]")
    header = f"{'Stage':<28} {'OK':<4} {'Elapsed (ms)':>12}  Details"
    print(header)
    print("-" * len(header))
    for entry in stage_trace:
        name = (entry.get("name") or "?")
        ok_val = entry.get("ok")
        status = "ok" if ok_val is True else ("fail" if ok_val is False else "…")
        elapsed = float(entry.get("elapsed_ms") or 0.0)
        detail = _summarize_stage_preview(entry.get("out"))
        if entry.get("error"):
            err = str(entry.get("error"))
            detail = f"{detail} | {err}" if detail else err
        if len(detail) > 96:
            detail = detail[:93] + "..."
        print(f"{name:<28.28} {status:<4} {elapsed:>12.1f}  {detail}")
    if stats:
        total = float(stats.get("total_elapsed_ms") or 0.0)
        fail = stats.get("failures")
        count = stats.get("count")
        print("-" * len(header))
        print(f"Total stages: {count}, failures: {fail}, elapsed: {total:.1f} ms")
        slowest = stats.get("slowest") if isinstance(stats, dict) else None
        if isinstance(slowest, dict) and slowest.get("name"):
            print(f"Slowest: {slowest.get('name')} ({slowest.get('elapsed_ms')} ms)")

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

_STOP_TOKENS = {"samples", "sample", "demo", "image", "images", "img", "scan", "page", "pages", "document", "documents", "doc"}


class IntentPayload(TypedDict, total=False):
    action: str
    priority: Literal["low", "medium", "high"]
    reason: str
    signals: Dict[str, Any]
    profile_domain: Optional[str]
    narrative: str


class MetaIntentPayload(TypedDict, total=False):
    intent_action: str
    meta_action: str
    priority: Optional[str]
    reason: Optional[str]
    story: Optional[str]
    focus_plan: Dict[str, Any]
    recommendations: List[str]
    external_inputs: Dict[str, Any]
    learning_outcome: Dict[str, Any]


_EPISODE_CONTEXT: Dict[str, Any] = {}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _discover_demo_input_targets() -> List[str]:
    """Locate real demo input directories/files to honour `--input demo`."""

    env_override = os.environ.get("ZOCR_DEMO_INPUTS")
    env_candidates = []
    if env_override:
        for segment in env_override.split(os.pathsep):
            segment = segment.strip()
            if segment:
                env_candidates.append(segment)

    here = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    search_roots = [os.getcwd(), here]
    seen_roots = set()
    uniq_roots: List[str] = []
    for root in search_roots:
        norm = os.path.abspath(root)
        if norm in seen_roots:
            continue
        seen_roots.add(norm)
        uniq_roots.append(norm)

    relative_candidates = [
        os.path.join("samples", "demo_inputs"),
        os.path.join("samples", "input_demo"),
        "demo_inputs",
        "input_demo",
    ]

    resolved: List[str] = []
    seen_paths = set()

    def _add_candidate(path: str) -> None:
        norm = os.path.abspath(path)
        if norm in seen_paths:
            return
        seen_paths.add(norm)
        if os.path.exists(norm):
            resolved.append(norm)

    for candidate in env_candidates:
        _add_candidate(candidate if os.path.isabs(candidate) else os.path.join(os.getcwd(), candidate))

    for root in uniq_roots:
        for rel in relative_candidates:
            _add_candidate(os.path.join(root, rel))

    return resolved


def _resolve_toy_memory_path(outdir: str) -> str:
    env_path = os.environ.get("ZOCR_TOY_MEMORY")
    if env_path:
        return env_path
    return os.path.join(outdir, "toy_memory.json")


def _default_toy_sweeps() -> int:
    base = getattr(zocr_onefile_consensus, "toy_runtime_config", None)
    if callable(base):
        cfg = base()
        sweeps = cfg.get("threshold_sweeps") if isinstance(cfg, dict) else None
        if isinstance(sweeps, int) and sweeps > 0:
            return sweeps
    raw = os.environ.get("ZOCR_TOY_SWEEPS")
    try:
        return max(1, int(raw)) if raw is not None else 5
    except Exception:
        return 5


def _collect_dependency_diagnostics() -> Dict[str, Any]:
    """Summarise optional dependencies so operators can self-check the environment."""
    diag: Dict[str, Any] = {}

    poppler_path = shutil.which("pdftoppm")
    diag["poppler_pdftoppm"] = {
        "status": "available" if poppler_path else "missing",
        "path": poppler_path,
        "hint": None if poppler_path else "Install poppler-utils (pdftoppm) for multi-page PDF rasterisation",
    }

    numba_enabled = bool(getattr(zocr_multidomain_core, "_HAS_NUMBA", False))
    diag["numba"] = {
        "status": "enabled" if numba_enabled else "python-fallback",
        "detail": "Numba acceleration active" if numba_enabled else "Falling back to pure Python BM25 scoring",
    }

    libc_path = getattr(zocr_multidomain_core, "_LIBC_PATH", None)
    diag["c_extensions"] = {
        "status": "loaded" if libc_path else "python-fallback",
        "path": libc_path,
        "detail": "Custom SIMD/Thomas/rle helpers" if libc_path else "Using pure Python/NumPy helpers",
    }

    numpy_version = None
    if _np is not None:
        try:
            numpy_version = getattr(_np, "__version__", None)
        except Exception:
            numpy_version = None
    diag["numpy"] = {
        "status": "available" if _np is not None else "missing",
        "version": numpy_version,
    }

    try:
        import PIL  # type: ignore

        pillow_version = getattr(PIL, "__version__", None)
    except Exception:
        pillow_version = None
    diag["pillow"] = {
        "status": "available" if pillow_version else "unknown",
        "version": pillow_version,
    }

    return diag


def _is_auto_domain(value: Optional[str]) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        norm = value.strip().lower()
        return norm in {"", "auto", "autodetect", "detect", "default"}
    return False


def _prepare_domain_hints(inputs: List[str], extra_paths: Optional[List[str]] = None) -> Dict[str, Any]:
    tokens_raw: List[str] = []
    token_trace: List[Dict[str, Any]] = []
    per_input: Dict[str, List[str]] = {}
    extra_tokens: Dict[str, List[str]] = {}

    def _ingest(raw: str, bucket: Dict[str, List[str]], source: str) -> None:
        norm = os.path.normpath(raw)
        seg_tokens: List[str] = []
        parts = norm.replace("\\", "/").split("/")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            base = os.path.splitext(part)[0]
            for tok in re.split(r"[^a-z0-9]+", base.lower()):
                if not tok or tok in _STOP_TOKENS or tok.isdigit() or len(tok) < 2:
                    continue
                tokens_raw.append(tok)
                token_trace.append({"token": tok, "source": source, "path": raw})
                seg_tokens.append(tok)
        if seg_tokens:
            bucket[raw] = seg_tokens

    for raw in inputs:
        _ingest(raw, per_input, "input")
    if extra_paths:
        for raw in extra_paths:
            _ingest(raw, extra_tokens, "page")
    unique_tokens = sorted(set(tokens_raw))
    domain_kw = getattr(zocr_multidomain_core, "DOMAIN_KW", {})
    alias_map = getattr(zocr_multidomain_core, "_DOMAIN_ALIAS", {})
    candidate_scores: Dict[str, float] = {dom: 0.0 for dom in domain_kw.keys()}
    for tok in tokens_raw:
        target = alias_map.get(tok)
        if target:
            candidate_scores.setdefault(target, 0.0)
            candidate_scores[target] += 1.2
        for dom in list(candidate_scores.keys()):
            dom_l = dom.lower()
            if tok == dom_l:
                candidate_scores[dom] += 1.0
            elif tok in dom_l.split("_"):
                candidate_scores[dom] += 0.5
    best_dom = None
    best_score = 0.0
    for dom, score in candidate_scores.items():
        if score > best_score:
            best_dom = dom
            best_score = score
    return {
        "tokens_raw": tokens_raw,
        "token_trace": token_trace,
        "tokens": unique_tokens,
        "per_input": per_input,
        "extra_paths": extra_tokens,
        "guess": best_dom,
        "best_score": best_score,
        "scores": {k: float(v) for k, v in candidate_scores.items() if v > 0.0},
    }


def _apply_domain_defaults(prof: Dict[str, Any], domain: Optional[str]) -> None:
    if not domain:
        return
    defaults = getattr(zocr_multidomain_core, "DOMAIN_DEFAULTS", {})
    alias_map = getattr(zocr_multidomain_core, "_DOMAIN_ALIAS", {})
    base = defaults.get(domain)
    if base is None:
        base = defaults.get(alias_map.get(domain, "")) if alias_map else None
    if base is None and "invoice_jp_v2" in defaults:
        base = defaults["invoice_jp_v2"]
    if base:
        for key, value in base.items():
            prof.setdefault(key, value)
    prof.setdefault("domain", domain)
    if prof.get("w_bm25") is None:
        prof["w_bm25"] = 1.0
def _read_ok_steps(outdir: str) -> set:
    path = os.path.join(outdir, "pipeline_history.jsonl")
    done = set()
    if not os.path.exists(path): return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ob = json.loads(line)
                if ob.get("ok"):
                    done.add(ob.get("name") or ob.get("step"))
            except Exception:
                pass
    return done

def _append_hist(outdir: str, rec: dict):
    rec = dict(rec)
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if _EPISODE_CONTEXT.get("id"):
        rec.setdefault("episode_id", _EPISODE_CONTEXT.get("id"))
    with open(os.path.join(outdir, "pipeline_history.jsonl"), "a", encoding="utf-8") as fw:
        fw.write(json.dumps(_json_ready(rec), ensure_ascii=False) + "\n")


def _episodes_root(outdir: str) -> str:
    return os.path.join(outdir, "episodes")


def _load_episode_index(outdir: str) -> Dict[str, Any]:
    path = os.path.join(_episodes_root(outdir), "episodes_index.json")
    if not os.path.exists(path):
        return {"episodes": []}
    try:
        with open(path, "r", encoding="utf-8") as fr:
            data = json.load(fr)
            if isinstance(data, dict) and isinstance(data.get("episodes"), list):
                return data
    except Exception:
        pass
    return {"episodes": []}


def _save_episode_index(outdir: str, payload: Dict[str, Any]) -> None:
    root = _episodes_root(outdir)
    ensure_dir(root)
    path = os.path.join(root, "episodes_index.json")
    try:
        with open(path, "w", encoding="utf-8") as fw:
            json.dump(_json_ready(payload), fw, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[WARN] episode index write failed: {exc}")


def _begin_episode(outdir: str) -> Optional[Dict[str, Any]]:
    if not outdir:
        return None
    root = _episodes_root(outdir)
    ensure_dir(root)
    index = _load_episode_index(outdir)
    episodes = index.get("episodes") if isinstance(index, dict) else []
    parent = None
    next_num = 1
    if isinstance(episodes, list) and episodes:
        try:
            last = episodes[-1]
            parent = str(last.get("id")) if last.get("id") is not None else None
        except Exception:
            parent = None
        values: List[int] = []
        for entry in episodes:
            try:
                values.append(int(str(entry.get("id")), 10))
            except Exception:
                continue
        if values:
            next_num = max(values) + 1
        elif parent:
            try:
                next_num = int(parent, 10) + 1
            except Exception:
                next_num = 1
    episode_id = f"{next_num:06d}"
    ep_dir = os.path.join(root, episode_id)
    ensure_dir(ep_dir)
    _EPISODE_CONTEXT.clear()
    _EPISODE_CONTEXT.update({"id": episode_id, "parent": parent, "path": ep_dir, "outdir": outdir})
    return {"id": episode_id, "parent": parent, "path": ep_dir}


def _finalize_episode(outdir: str, summary: Dict[str, Any]) -> None:
    info = summary.get("episode") or {}
    if not info.get("id"):
        info = _EPISODE_CONTEXT
    episode_id = str(info.get("id") or "").strip()
    if not episode_id:
        return
    ep_dir = info.get("path") or os.path.join(_episodes_root(outdir), episode_id)
    ensure_dir(ep_dir)
    artifacts: Dict[str, str] = {}

    def _rel(dest: str) -> str:
        return os.path.relpath(dest, outdir)

    def _copy_artifact(src: Optional[str], name: str) -> None:
        if not src:
            return
        abs_src = src if os.path.isabs(src) else os.path.join(outdir, src)
        if not os.path.exists(abs_src):
            return
        dest = os.path.join(ep_dir, os.path.basename(name))
        try:
            shutil.copy2(abs_src, dest)
        except Exception as exc:
            print(f"[WARN] episode artifact copy failed ({name}): {exc}")
            return
        artifacts[name] = _rel(dest)

    _copy_artifact(os.path.join(outdir, "pipeline_summary.json"), "pipeline_summary.json")
    _copy_artifact(summary.get("history"), "pipeline_history.jsonl")
    _copy_artifact(summary.get("monitor_csv"), "monitor.csv")
    _copy_artifact(summary.get("profile_json"), "auto_profile.json")
    rag_manifest = os.path.join(outdir, "rag", "manifest.json")
    if os.path.exists(rag_manifest):
        _copy_artifact(rag_manifest, "rag_manifest.json")
    _copy_artifact(summary.get("repro_signature_path"), "repro_signature.json")

    stage_trace = summary.get("stage_trace")
    if stage_trace:
        path = os.path.join(ep_dir, "stage_trace.json")
        try:
            with open(path, "w", encoding="utf-8") as fw:
                json.dump(_json_ready(stage_trace), fw, ensure_ascii=False, indent=2)
            artifacts["stage_trace.json"] = _rel(path)
        except Exception as exc:
            print(f"[WARN] episode stage trace write failed: {exc}")

    toy_delta = (summary.get("toy_memory") or {}).get("delta_run")
    if toy_delta:
        path = os.path.join(ep_dir, "toy_memory_delta.json")
        try:
            with open(path, "w", encoding="utf-8") as fw:
                json.dump(_json_ready(toy_delta), fw, ensure_ascii=False, indent=2)
            artifacts["toy_memory_delta.json"] = _rel(path)
        except Exception as exc:
            print(f"[WARN] episode toy delta write failed: {exc}")

    for key in ("learning_hotspots", "selective_reanalysis_plan", "hotspot_gallery"):
        if not summary.get(key):
            continue
        snap_path = os.path.join(ep_dir, f"{key}.json")
        try:
            with open(snap_path, "w", encoding="utf-8") as fw:
                json.dump(_json_ready(summary.get(key)), fw, ensure_ascii=False, indent=2)
            artifacts[f"{key}.json"] = _rel(snap_path)
        except Exception as exc:
            print(f"[WARN] episode {key} snapshot failed: {exc}")

    summary.setdefault("episode", {})
    summary["episode"].update({
        "id": episode_id,
        "parent": info.get("parent"),
        "path": _rel(ep_dir),
        "artifacts": artifacts,
    })

    index = _load_episode_index(outdir)
    episodes = [entry for entry in index.get("episodes", []) if entry.get("id") != episode_id]
    monitor = summary.get("monitor_row") or {}
    intent = summary.get("intent") or {}
    meta_intent = summary.get("meta_intent") or {}
    repro_sig = summary.get("repro_signature") or {}

    def _as_float(val: Any) -> Optional[float]:
        try:
            if val is None:
                return None
            return float(val)
        except Exception:
            return None

    entry = {
        "id": episode_id,
        "created_at": summary.get("generated_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "domain": summary.get("domain"),
        "inputs_hash": repro_sig.get("inputs_hash"),
        "profile_hash": repro_sig.get("profile_hash"),
        "parent": info.get("parent"),
        "hit_mean": _as_float(monitor.get("hit_mean") or monitor.get("hit_mean_gt")),
        "p95_ms": _as_float(monitor.get("p95_ms")),
        "gate_pass": monitor.get("gate_pass"),
        "gate_reason": monitor.get("gate_reason"),
        "intent_action": intent.get("action"),
        "meta_intent": meta_intent.get("meta_action"),
        "episode_path": summary["episode"]["path"],
    }
    episodes.append({k: v for k, v in entry.items() if v is not None})
    episodes.sort(key=lambda item: item.get("id"))
    _save_episode_index(outdir, {"episodes": episodes})

def _load_history(outdir: str) -> List[Dict[str, Any]]:
    path = os.path.join(outdir, "pipeline_history.jsonl")
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records

def _print_history(records: List[Dict[str, Any]], limit: Optional[int] = None) -> None:
    if limit is not None and limit > 0:
        records = records[-limit:]
    if not records:
        print("(no history)")
        return
    w_step = max(4, max(len(str(r.get("name") or r.get("step"))) for r in records))
    w_status = 7
    header = f"{'timestamp':<20}  {'step':<{w_step}}  {'status':<{w_status}}  elapsed_ms  note"
    print(header)
    print("-" * len(header))
    for rec in records:
        ts = rec.get("ts", "-")
        step = rec.get("name") or rec.get("step") or "?"
        status = "OK" if rec.get("ok") else ("FAIL" if rec.get("ok") is False else "-")
        elapsed = rec.get("elapsed_ms")
        note = rec.get("error") or ""
        if rec.get("out") and status == "OK" and not isinstance(rec["out"], (str, int, float)):
            if isinstance(rec["out"], dict) and rec["out"].get("path"):
                note = rec["out"]["path"]
        print(f"{ts:<20}  {step:<{w_step}}  {status:<{w_status}}  {elapsed!s:<10}  {note}")

def _read_summary(outdir: str) -> Dict[str, Any]:
    path = os.path.join(outdir, "pipeline_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as fr:
        return json.load(fr)

def _read_meta(outdir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(outdir, "pipeline_meta.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fr:
            return json.load(fr)
    except Exception:
        return None

def _render_value(value: Any) -> str:
    if value is None:
        return "<span class=\"muted\">–</span>"
    if isinstance(value, (dict, list)):
        try:
            formatted = json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            formatted = str(value)
        return f"<pre>{escape(formatted)}</pre>"
    if isinstance(value, float):
        return escape(f"{value:,.4g}")
    return escape(str(value))

def _render_table(data: Dict[str, Any], title: str, keys: Optional[List[str]] = None) -> str:
    if not data:
        return ""
    rows = []
    items = data.items() if keys is None else ((k, data.get(k)) for k in keys if k in data)
    for key, value in items:
        rows.append(
            f"<tr><th scope=\"row\">{escape(str(key))}</th><td>{_render_value(value)}</td></tr>"
        )
    if not rows:
        return ""
    return (
        f"<section>\n<h2>{escape(title)}</h2>\n"
        "<table class=\"kv\">\n" + "\n".join(rows) + "\n</table>\n</section>"
    )

def _render_history_table(records: List[Dict[str, Any]]) -> str:
    if not records:
        return "<p class=\"muted\">(no history recorded)</p>"
    header = "<thead><tr><th>timestamp</th><th>step</th><th>status</th><th>elapsed</th><th>note</th></tr></thead>"
    body_rows = []
    for rec in records:
        status = "ok" if rec.get("ok") else ("fail" if rec.get("ok") is False else "skip")
        cls = {
            "ok": "status-ok",
            "fail": "status-fail",
            "skip": "status-skip",
        }.get(status, "")
        elapsed = rec.get("elapsed_ms")
        if isinstance(elapsed, (int, float)):
            elapsed_s = f"{elapsed:,.1f} ms"
        else:
            elapsed_s = escape(str(elapsed)) if elapsed is not None else "–"
        note = rec.get("error") or ""
        out = rec.get("out")
        if not note and isinstance(out, dict) and out.get("path"):
            note = str(out.get("path"))
        body_rows.append(
            "<tr class=\"{cls}\"><td>{ts}</td><td>{step}</td><td><span class=\"badge {cls}\">{status}</span></td><td>{elapsed}</td><td>{note}</td></tr>".format(
                cls=cls,
                ts=escape(rec.get("ts", "–")),
                step=escape(str(rec.get("name") or rec.get("step") or "?")),
                status=escape(status.upper()),
                elapsed=elapsed_s,
                note=escape(str(note)) if note else "",
            )
        )
    return "<table class=\"history\">" + header + "<tbody>" + "".join(body_rows) + "</tbody></table>"


def _render_hotspots_section(summary: Dict[str, Any]) -> str:
    hotspots = summary.get("learning_hotspots") if isinstance(summary, dict) else None
    plan = summary.get("selective_reanalysis_plan") if isinstance(summary, dict) else None
    gallery = summary.get("hotspot_gallery") if isinstance(summary, dict) else None
    if not any([hotspots, plan, gallery]):
        return ""
    parts: List[str] = ["<section>", "<h2>ホットスポット / Hotspots</h2>"]
    if isinstance(hotspots, dict) and hotspots:
        reasons = hotspots.get("reason_counts") if isinstance(hotspots.get("reason_counts"), list) else []
        if reasons:
            parts.append("<h3>Signals</h3><ul>")
            for rec in reasons[:6]:
                if not isinstance(rec, dict):
                    continue
                label = escape(str(rec.get("reason") or "?"))
                count = escape(str(rec.get("count") or ""))
                parts.append(f"<li>{label}: {count}</li>")
            parts.append("</ul>")
        cells = hotspots.get("hot_cells") if isinstance(hotspots.get("hot_cells"), list) else []
        if cells:
            rows = ["<thead><tr><th>trace</th><th>page,row</th><th>score</th><th>reasons</th></tr></thead>"]
            body: List[str] = []
            for cell in cells[:6]:
                if not isinstance(cell, dict):
                    continue
                trace = escape(str(cell.get("trace_id") or "?"))
                loc = f"p{cell.get('page')} r{cell.get('row')}"
                score = escape(str(cell.get("score") or ""))
                reasons_txt = ", ".join(escape(str(r)) for r in cell.get("reasons", [])[:4]) if cell.get("reasons") else ""
                body.append(f"<tr><td><code>{trace}</code></td><td>{escape(loc)}</td><td>{score}</td><td>{reasons_txt}</td></tr>")
            if body:
                rows.append("<tbody>" + "".join(body) + "</tbody>")
                parts.append("<details open><summary>Top cells</summary><table class=\"history\">" + "".join(rows) + "</table></details>")
    if isinstance(plan, dict) and plan:
        parts.append(_render_table(plan, "選択的再解析計画 / Selective plan"))
    if isinstance(gallery, dict) and gallery.get("entries"):
        entries = gallery.get("entries")
        limit = min(6, len(entries)) if isinstance(entries, list) else 0
        if limit:
            parts.append("<h3>Hotspot gallery</h3>")
            parts.append("<div class=\"hotspot-gallery\">")
            for entry in entries[:limit]:
                if not isinstance(entry, dict):
                    continue
                img = entry.get("image")
                caption_bits: List[str] = []
                if entry.get("trace_id"):
                    caption_bits.append(f"trace {escape(str(entry['trace_id']))}")
                if entry.get("role"):
                    caption_bits.append(f"role {escape(str(entry['role']))}")
                if entry.get("reason_rank"):
                    caption_bits.append(f"reason #{escape(str(entry['reason_rank']))}")
                caption = " ・ ".join(caption_bits) or "cell"
                before = entry.get("before_text") or entry.get("text")
                after = entry.get("after_text")
                text_lines = []
                if before:
                    text_lines.append(f"<div class=\"muted\">before</div><div>{escape(str(before))}</div>")
                if after and after != before:
                    text_lines.append(f"<div class=\"muted\">after</div><div>{escape(str(after))}</div>")
                img_html = f"<img src=\"{escape(str(img))}\" alt=\"hotspot\">" if img else ""
                parts.append(
                    "<figure>" + img_html + f"<figcaption>{caption}</figcaption>" + "".join(text_lines) + "</figure>"
                )
            parts.append("</div>")
        if gallery.get("story"):
            parts.append(f"<p class=\"muted\"><a href=\"{escape(str(gallery['story']))}\">gallery notes</a></p>")
    parts.append("</section>")
    return "".join(parts)


def _coerce_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        if isinstance(val, str) and not val.strip():
            return None
        return float(val)
    except Exception:
        return None



_PROFILE_GUARD_KEYS = {
    "ocr_min_conf",
    "lambda_shape",
    "header_boost",
    "w_kw",
    "w_img",
    "reanalyze_target",
    "force_monitor_refresh",
    "speed_priority",
}


def _profile_guard_max_changes() -> int:
    try:
        return max(1, int(os.environ.get("ZOCR_PROFILE_MAX_CHANGES", "3")))
    except Exception:
        return 3


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"0", "false", "no", "off"}:
            return False
        if lower in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def _guard_profile_value(
    key: str,
    requested: Any,
    current: Any,
    baseline: Any,
) -> Tuple[bool, Any, Optional[str]]:
    reason: Optional[str] = None
    if key == "ocr_min_conf":
        val = _coerce_float(requested)
        if val is None:
            return False, current, "invalid_value"
        orig = _coerce_float(baseline if baseline is not None else current)
        val = max(0.3, min(0.95, val))
        if orig is not None and abs(val - orig) > 0.1:
            direction = 1.0 if val > orig else -1.0
            val = float(orig) + 0.1 * direction
            reason = "delta_clamped"
        if reason is None and (val <= 0.3 or val >= 0.95):
            reason = "clamped_range"
        return True, float(f"{val:.4f}"), reason
    if key == "lambda_shape":
        val = _coerce_float(requested)
        if val is None:
            return False, current, "invalid_value"
        orig_val = val
        val = max(2.5, min(6.0, val))
        if val != orig_val:
            reason = "clamped_range"
        return True, float(f"{val:.4f}"), reason
    if key in {"w_kw", "w_img"}:
        val = _coerce_float(requested)
        if val is None:
            return False, current, "invalid_value"
        orig_val = val
        val = max(0.2, min(6.0, val))
        if val != orig_val:
            reason = "clamped_range"
        return True, float(f"{val:.4f}"), reason
    if key == "header_boost":
        val = _coerce_float(requested)
        if val is None:
            return False, current, "invalid_value"
        orig_val = val
        val = max(0.5, min(5.0, val))
        if val != orig_val:
            reason = "clamped_range"
        return True, float(f"{val:.4f}"), reason
    if key in {"force_monitor_refresh", "speed_priority"}:
        val = _coerce_bool(requested)
        return True, val, None
    return True, requested, None


class _ProfileGuard:
    def __init__(
        self,
        baseline: Optional[Dict[str, Any]],
        *,
        max_changes: Optional[int] = None,
        keys: Optional[Set[str]] = None,
    ) -> None:
        self.baseline = json.loads(json.dumps(baseline or {}))
        self.max_changes = max_changes or _profile_guard_max_changes()
        self.keys = set(keys) if keys else set(_PROFILE_GUARD_KEYS)
        self.changed: Dict[str, List[Dict[str, Any]]] = {}
        self.blocked: Dict[str, Dict[str, Any]] = {}
        self.adjusted: Dict[str, str] = {}

    def _within_scope(self, key: str) -> bool:
        if not self.keys:
            return True
        return key in self.keys

    def apply(
        self,
        key: str,
        requested: Any,
        current: Any,
        *,
        source: Optional[str] = None,
    ) -> Tuple[bool, Any, Optional[str]]:
        if not self._within_scope(key):
            return True, requested, None
        already = key in self.changed
        if not already and len(self.changed) >= self.max_changes:
            self.blocked[key] = {
                "reason": "max_changes",
                "requested": _json_ready(requested),
                "source": source,
            }
            return False, current, "max_changes"
        allowed, final, reason = _guard_profile_value(
            key, requested, current, self.baseline.get(key)
        )
        if not allowed:
            self.blocked[key] = {
                "reason": reason or "invalid",
                "requested": _json_ready(requested),
                "source": source,
            }
            return False, current, reason
        if reason:
            self.adjusted[key] = reason
        self.changed.setdefault(key, []).append(
            {"source": source, "requested": _json_ready(requested), "applied": _json_ready(final)}
        )
        return True, final, reason

    def report(self) -> Dict[str, Any]:
        return {
            "max_changes": self.max_changes,
            "guarded_keys": sorted(self.keys),
            "applied": _json_ready(self.changed),
            "blocked": _json_ready(self.blocked),
            "adjusted": _json_ready(self.adjusted),
        }


_GATE_FAIL_ESCALATE_THRESHOLD = max(1, int(os.environ.get("ZOCR_GATE_FAIL_ESCALATE", "3")))


def _gate_fail_safety(
    profile: Optional[Dict[str, Any]],
    monitor_row: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not profile or not monitor_row:
        return None
    gate_flag = monitor_row.get("gate_pass")
    if gate_flag is None:
        return None
    gate_pass = _coerce_bool(gate_flag)
    prev_raw = profile.get("gate_fail_streak")
    try:
        prev = int(prev_raw)
    except Exception:
        prev = 0
    new_val = 0 if gate_pass else prev + 1
    info: Dict[str, Any] = {
        "gate_pass": gate_pass,
        "previous": prev,
        "value": new_val,
        "threshold": _GATE_FAIL_ESCALATE_THRESHOLD,
    }
    if not gate_pass and new_val >= _GATE_FAIL_ESCALATE_THRESHOLD:
        info["escalate"] = True
        info["recommendation"] = "escalate_to_human"
    if new_val != prev:
        profile["gate_fail_streak"] = new_val
        info["updated"] = True
    return info


def _derive_insights(summary: Dict[str, Any]) -> List[str]:
    insights: List[str] = []
    monitor = summary.get("monitor_row") or {}
    tune = summary.get("tune") or {}
    learn = summary.get("learn") or {}
    metrics = summary.get("consensus_metrics") or {}
    aggregate = metrics.get("aggregate") if isinstance(metrics, dict) else {}

    best = tune.get("best") if isinstance(tune, dict) else {}
    profile = learn.get("profile") if isinstance(learn, dict) else {}

    def pick(source: Dict[str, Any], *keys: str) -> Optional[float]:
        for key in keys:
            if isinstance(source, dict) and key in source:
                v = _coerce_float(source.get(key))
                if v is not None:
                    return v
        return None

    col_over = _coerce_float((aggregate or {}).get("col_over_under_med"))
    teds = _coerce_float((aggregate or {}).get("teds_mean"))
    row_out = _coerce_float((aggregate or {}).get("row_outlier_rate_med"))
    hit_mean = _coerce_float(monitor.get("hit_mean") or monitor.get("hit_mean_gt"))
    trust_mean = _coerce_float(monitor.get("trust_mean"))
    if col_over is not None or teds is not None or row_out is not None:
        parts: List[str] = []
        if col_over is not None:
            parts.append(f"列数一致（over/under≈{col_over:.2f}）")
        if teds is not None:
            parts.append(f"TEDS≈{teds:.2f}")
        msg = "構造は概ね取れている"
        if parts:
            msg += "：" + "、".join(parts)
        if row_out is not None:
            msg += f"。残課題はヘッダ/末尾Totalの検出で、行外れ≈{row_out:.2f}を詰めればHit@Kも上がる見込み"
        elif hit_mean is not None:
            msg += f"。Hit@K≈{hit_mean*100:.0f}% まで見えているのでヘッダ/Total補完でさらに伸ばせます"
        insights.append(msg)

    if trust_mean is not None:
        if trust_mean >= 0.98:
            insights.append(f"Trust@K≈{trust_mean*100:.0f}%：trace付きセルとシンボリック検索で幻覚率ほぼゼロ化")
        else:
            insights.append(f"Trust@K≈{trust_mean*100:.0f}%：trace/filters の補完を確認すると幻覚率をさらに抑えられます")

    gate_flag = monitor.get("gate_pass")
    gate_pass = bool(gate_flag) if isinstance(gate_flag, bool) else str(gate_flag).lower() == "true"
    gate_reason = monitor.get("gate_reason") if isinstance(monitor, dict) else None
    hit_date = _coerce_float(monitor.get("hit_date") or monitor.get("hit_date_gt"))
    if gate_pass:
        if gate_reason:
            insights.append(f"ゲートは {gate_reason} で通過。Date/TAX の期待値は運用要件に合わせて任意指定にできます")
    else:
        msg = "ゲート落ちの主因はスキーマ期待値"
        if gate_reason:
            msg = f"ゲート落ちの主因は {gate_reason}"
        if hit_date is not None:
            msg += f" (hit_date≈{hit_date:.2f})"
        msg += "。Date を任意扱いにするか、請求書タイプを明細のみ/メタ付きで分岐させると安定"
        insights.append(msg)

    weights_source = best or profile or {}
    w_kw = pick(weights_source, "w_kw")
    w_img = pick(weights_source, "w_img")
    ocr_min = pick(weights_source, "ocr_min_conf")
    lam = pick(weights_source, "lambda_shape")
    if w_kw is not None and w_img is not None:
        msg = f"現プロファイルの方向性: w_kw={w_kw:.2f} > w_img={w_img:.2f} でキーワード寄り"
        tweaks: List[str] = []
        if ocr_min is not None:
            tweaks.append(f"ocr_min_conf≈{ocr_min:.2f}")
        if lam is not None:
            tweaks.append(f"λ_shape≈{lam:.2f}")
        if tweaks:
            msg += "。ヘッダ補完を入れるなら " + " と ".join(tweaks) + " を少し下げて再走査すると早い"
        insights.append(msg)

    return insights


def _dedupe_insights_and_queries(summary: Dict[str, Any]) -> None:
    insights = summary.get("insights")
    queries = summary.get("rag_suggested_queries")
    if not insights or not queries:
        return

    def _canon(val: Any) -> Optional[str]:
        if not isinstance(val, str):
            return None
        return " ".join(val.split()).strip().lower()

    insight_keys = {c for c in (_canon(v) for v in insights) if c}
    if not insight_keys:
        return

    filtered: List[Any] = []
    for q in queries:
        canon = _canon(q)
        if canon and canon in insight_keys:
            continue
        filtered.append(q)
    summary["rag_suggested_queries"] = filtered


def _derive_rag_bundle_status(
    cell_count: Optional[int],
    table_count: Optional[int],
    page_count: Optional[int],
    doc_ids: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    def _is_positive(val: Optional[int]) -> bool:
        try:
            return int(val) > 0
        except (TypeError, ValueError):
            return False

    has_cells = _is_positive(cell_count)
    has_tables = _is_positive(table_count)
    has_pages = _is_positive(page_count)
    issues: List[str] = []
    if not has_cells:
        issues.append("no_cells")
    if has_cells and not has_tables:
        issues.append("no_tables")
    if has_cells and not has_pages:
        issues.append("no_pages")

    status: Dict[str, Any] = {
        "has_cells": has_cells,
        "has_tables": has_tables,
        "has_pages": has_pages,
    }
    if doc_ids:
        status["doc_ids"] = doc_ids
    if languages:
        status["languages"] = languages
    if issues:
        status["issues"] = issues
    return status

def _generate_report(
    outdir: str,
    dest: Optional[str] = None,
    summary: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    meta: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = 50,
) -> str:
    summary = summary or _read_summary(outdir)
    history = history or _load_history(outdir)
    meta = meta if meta is not None else _read_meta(outdir)
    if limit is not None and limit > 0 and len(history) > limit:
        history = history[-limit:]
    dest = dest or os.path.join(outdir, "pipeline_report.html")
    ensure_dir(os.path.dirname(dest) or ".")

    stats = summary.get("history_stats") or {}
    total_ms = stats.get("total_elapsed_ms")
    ok_count = stats.get("ok")
    fail_count = stats.get("fail")
    total_s = None
    if isinstance(total_ms, (int, float)):
        total_s = total_ms / 1000.0

    css = """
    body { font-family: 'Inter', 'Segoe UI', 'Hiragino Sans', sans-serif; margin: 2rem; background: #0d1117; color: #e6edf3; }
    a { color: #9cdcfe; }
    h1, h2, h3 { color: #58a6ff; }
    section { margin-bottom: 2rem; }
    table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
    th, td { border: 1px solid #30363d; padding: 0.45rem 0.6rem; vertical-align: top; }
    th { background: rgba(88, 166, 255, 0.08); text-align: left; font-weight: 600; }
    table.kv th { width: 18%; }
    pre { background: #161b22; border-radius: 8px; padding: 0.75rem; overflow-x: auto; }
    .muted { opacity: 0.65; }
    .badge { display: inline-block; padding: 0.1rem 0.6rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
    .status-ok .badge { background: rgba(63, 185, 80, 0.2); color: #3fb950; }
    .status-fail .badge { background: rgba(248, 81, 73, 0.2); color: #f85149; }
    .status-skip .badge { background: rgba(201, 148, 0, 0.2); color: #c99400; }
    details { margin-top: 1rem; }
    summary { cursor: pointer; }
    .hotspot-gallery { display: flex; flex-wrap: wrap; gap: 1rem; }
    .hotspot-gallery figure { width: 220px; background: #161b22; border: 1px solid #30363d; padding: 0.5rem; border-radius: 8px; }
    .hotspot-gallery img { max-width: 100%; border-radius: 4px; margin-bottom: 0.35rem; }
    .hotspot-gallery figcaption { font-weight: 600; margin-bottom: 0.35rem; }
    footer { margin-top: 3rem; font-size: 0.85rem; opacity: 0.7; }
    """

    meta_table = _render_table(meta or {}, "環境 / Environment / Environnement", [
        "seed",
        "python",
        "platform",
        "env",
        "versions",
    ]) if meta else "<p class=\"muted\">(no snapshot metadata — run with --snapshot)</p>"

    dep_table = ""
    deps = summary.get("dependencies") if isinstance(summary, dict) else None
    if isinstance(deps, dict) and deps:
        dep_table = _render_table(
            deps,
            "依存診断 / Dependency Check / Diagnostic",
        )

    core_table = _render_table(
        {
            "Output": summary.get("contextual_jsonl"),
            "Augmented": summary.get("mm_jsonl"),
            "Index": summary.get("index"),
            "Monitor": summary.get("monitor_csv"),
            "Profile": summary.get("profile_json"),
            "SQL CSV": summary.get("sql_csv"),
            "SQL schema": summary.get("sql_schema"),
            "Report": summary.get("report_html"),
        },
        "成果物 / Artifacts / Artefacts",
    )

    info_table = _render_table(
        {
            "inputs": summary.get("inputs"),
            "page_images": summary.get("page_images"),
            "pages": summary.get("page_count"),
            "domain": summary.get("domain"),
            "seed": summary.get("seed"),
            "resume_requested": summary.get("resume_requested"),
            "resume_applied": summary.get("resume_applied"),
            "resume_steps": summary.get("resume_steps"),
            "snapshot": summary.get("snapshot"),
            "tune_budget": summary.get("tune_budget"),
            "generated_at": summary.get("generated_at"),
        },
        "概要 / Overview / Aperçu",
    )

    plugins = summary.get("plugins") or {}
    if plugins:
        plugin_rows = []
        for stage, fns in sorted(plugins.items()):
            names = ", ".join(escape(str(fn)) for fn in fns) or "–"
            plugin_rows.append(f"<tr><th scope=\"row\">{escape(stage)}</th><td>{names}</td></tr>")
        plugin_html = (
            "<section><h2>プラグイン / Plugins / Extensions</h2><table class=\"kv\">" +
            "".join(plugin_rows) + "</table></section>"
        )
    else:
        plugin_html = "<section><h2>プラグイン / Plugins / Extensions</h2><p class=\"muted\">(no plugins registered)</p></section>"

    monitor_html = ""
    if summary.get("monitor_row"):
        monitor_html = _render_table(summary.get("monitor_row"), "モニタ / Monitor / Surveillance")
    tune_html = ""
    if summary.get("tune"):
        tune_html = _render_table(summary.get("tune"), "自動調整 / Tuning / Ajustement")
    learn_html = ""
    if summary.get("learn"):
        learn_html = _render_table(summary.get("learn"), "学習 / Learning / Apprentissage")
    hotspot_html = _render_hotspots_section(summary)

    history_html = _render_history_table(history)

    stats_text = []
    if total_ms is not None:
        stats_text.append(f"総処理時間 / Total / Total : {total_ms:,.1f} ms")
    if total_s is not None:
        stats_text.append(f"≈ {total_s:,.2f} s")
    if ok_count is not None or fail_count is not None:
        stats_text.append(
            "成否 / Status : OK={ok} / FAIL={fail}".format(
                ok=ok_count if ok_count is not None else "–",
                fail=fail_count if fail_count is not None else "–",
            )
        )
    stats_block = "<p class=\"muted\">" + " ・ ".join(stats_text) + "</p>" if stats_text else ""

    pip_html = ""
    if meta and meta.get("pip_freeze"):
        pip_lines = "\n".join(meta["pip_freeze"][:200])
        extra = ""
        if len(meta["pip_freeze"]) > 200:
            extra = f"\n… ({len(meta['pip_freeze']) - 200} more)"
        pip_html = (
            "<details><summary>pip freeze</summary><pre>" + escape(pip_lines + extra) + "</pre></details>"
        )

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>ZOCR Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>ZOCR Pipeline Report / パイプラインレポート / Rapport</h1>
  <p>outdir: <code>{escape(os.path.abspath(outdir))}</code></p>
  {stats_block}
  {info_table}
  {core_table}
  {monitor_html}
  {tune_html}
  {learn_html}
  {hotspot_html}
  {plugin_html}
  <section>
    <h2>履歴 / History / Historique</h2>
    {history_html}
  </section>
  <section>
    <h2>環境 / Environment / Environnement</h2>
    {meta_table}
    {dep_table}
    {pip_html}
  </section>
  <footer>
    Generated at {escape(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))}
  </footer>
</body>
</html>
"""

    with open(dest, "w", encoding="utf-8") as fw:
        fw.write(html)
    return dest

def _safe_step(name, fn, *a, **kw):
    t0 = time.perf_counter()
    try:
        print(f"[RUN]  {name}")
        out = fn(*a, **kw)
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[OK]   {name} ({dt:.1f} ms)")
        result = {"ok": True, "elapsed_ms": dt, "out": out, "name": name}
        _record_stage_trace(result)
        return result
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[FAIL] {name} ({dt:.1f} ms): {type(e).__name__}: {e}")
        traceback.print_exc()
        result = {"ok": False, "elapsed_ms": dt, "error": f"{type(e).__name__}: {e}", "name": name}
        _record_stage_trace(result)
        return result

def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1<<16), b""):
            h.update(b)
    return h.hexdigest()

def _write_pipeline_meta(outdir: str, seed: int):
    meta = {
        "seed": int(seed),
        "python": sys.version,
        "platform": platform.platform(),
        "env": {k:v for k,v in os.environ.items() if k in ("PYTHONHASHSEED","OMP_NUM_THREADS","MKL_NUM_THREADS")},
        "versions": {},
        "files": {}
    }
    mods = [sys.modules.get(__name__), zocr_onefile_consensus, zocr_multidomain_core]
    for mod in mods:
        if mod is None: continue
        try:
            p = mod.__file__
            meta["files"][mod.__name__] = {"path": p, "sha256": _sha256(p)}
        except Exception:
            pass
    bundle_dir = getattr(zocr_onefile_consensus, "_BUNDLE_DIR", None)
    if bundle_dir and os.path.isdir(bundle_dir):
        bundle_meta: Dict[str, Any] = {}
        for root, _dirs, files in os.walk(bundle_dir):
            for fn in sorted(f for f in files if f.endswith(".py")):
                full = os.path.join(root, fn)
                try:
                    bundle_meta[os.path.relpath(full, bundle_dir)] = {
                        "sha256": _sha256(full),
                        "size": os.path.getsize(full),
                    }
                except Exception:
                    continue
        if bundle_meta:
            meta["bundle_files"] = bundle_meta
    for name in ("numpy", "Pillow"):
        try:
            meta["versions"][name] = importlib.import_module(name).__version__
        except Exception:
            meta["versions"][name] = None
    try:
        meta["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=False, capture_output=True, text=True).stdout.strip().splitlines()
    except Exception:
        meta["pip_freeze"] = []
    with open(os.path.join(outdir, "pipeline_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _collect_pages(inputs: List[str], dpi: int) -> List[str]:
    pages: List[str] = []
    def _handle_path(path: str):
        nonlocal pages
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs.sort()
                for fn in sorted(files):
                    full = os.path.join(root, fn)
                    ext = os.path.splitext(fn)[1].lower()
                    if ext == ".pdf":
                        try:
                            pages.extend(zocr_onefile_consensus.pdf_to_images_via_poppler(full, dpi=dpi))
                        except Exception as e:
                            raise RuntimeError(f"PDF rasterization failed for {full}: {e}")
                    elif ext in _IMAGE_EXTS:
                        pages.append(full)
            return
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            try:
                pages.extend(zocr_onefile_consensus.pdf_to_images_via_poppler(path, dpi=dpi))
            except Exception as e:
                raise RuntimeError(f"PDF rasterization failed for {path}: {e}")
        elif ext in _IMAGE_EXTS or not ext:
            pages.append(path)

    for raw in inputs:
        candidates = [raw]
        if any(ch in raw for ch in "*?[]"):
            candidates = sorted(glob.glob(raw)) or [raw]
        for cand in candidates:
            if os.path.exists(cand):
                _handle_path(cand)
            else:
                pages.append(cand)
    return pages

def _load_profile(outdir: str, domain_hint: Optional[str]) -> Dict[str, Any]:
    prof_path = os.path.join(outdir, "auto_profile.json")
    try:
        with open(prof_path, "r", encoding="utf-8") as f:
            prof = json.load(f)
    except Exception:
        prof = {}
    if domain_hint and not prof.get("domain"):
        prof["domain"] = domain_hint
    return prof


def _load_export_signals(jsonl_path: str) -> Dict[str, Any]:
    signals_path = jsonl_path + ".signals.json"
    if not os.path.exists(signals_path):
        return {}
    try:
        with open(signals_path, "r", encoding="utf-8") as fr:
            data = json.load(fr)
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


def _analyze_learning_hotspots(learning_jsonl_path: Optional[str], max_samples: int = 400) -> Dict[str, Any]:
    if not learning_jsonl_path or not os.path.exists(learning_jsonl_path):
        return {}
    table_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"score": 0.0, "count": 0})
    row_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"score": 0.0, "count": 0})
    trace_scores: Dict[str, Dict[str, Any]] = {}
    reason_counts: Counter = Counter()
    total = 0
    try:
        with open(learning_jsonl_path, "r", encoding="utf-8") as fr:
            for raw in fr:
                if max_samples and total >= max_samples:
                    break
                line = raw.strip()
                if not line:
                    continue
                try:
                    sig = json.loads(line)
                except Exception:
                    continue
                total += 1
                trace = sig.get("trace_id") or sig.get("meta", {}).get("trace") if isinstance(sig.get("meta"), dict) else None
                page = sig.get("page")
                table_idx = sig.get("table_index")
                row_idx = sig.get("row")
                col_idx = sig.get("col")
                try:
                    page_int = int(page) if page is not None else None
                except Exception:
                    page_int = None
                try:
                    table_int = int(table_idx) if table_idx is not None else None
                except Exception:
                    table_int = None
                try:
                    row_int = int(row_idx) if row_idx is not None else None
                except Exception:
                    row_int = None
                try:
                    col_int = int(col_idx) if col_idx is not None else None
                except Exception:
                    col_int = None
                table_key = f"page={page_int};table={table_int}"
                row_key = f"{table_key};row={row_int}"
                if not trace:
                    trace = f"{row_key};col={col_int}"
                trace = str(trace)
                conf = _coerce_float(sig.get("confidence"))
                surprisal = _coerce_float(sig.get("ngram_surprisal") or sig.get("surprisal"))
                reasons = [str(r) for r in sig.get("reasons", []) if isinstance(r, str) and r]
                for reason in reasons:
                    reason_counts[reason] += 1
                score = 1.0
                if conf is not None:
                    score += max(0.0, 1.0 - conf)
                if surprisal is not None and surprisal > 0:
                    score += min(1.0, surprisal / 6.0)
                if "high_surprisal" in reasons:
                    score += 0.4
                if "low_conf" in reasons:
                    score += 0.3
                if sig.get("hypotheses"):
                    score += 0.2
                table_stats[table_key]["score"] += score
                table_stats[table_key]["count"] += 1
                row_stats[row_key]["score"] += score
                row_stats[row_key]["count"] += 1
                entry = trace_scores.setdefault(trace, {"score": 0.0, "count": 0, "page": page_int, "table": table_int, "row": row_int})
                entry["score"] += score
                entry["count"] += 1
                if conf is not None:
                    entry.setdefault("avg_conf", 0.0)
                    entry["avg_conf"] = ((entry.get("avg_conf") or 0.0) * (entry["count"] - 1) + conf) / max(1, entry["count"])
                if reasons:
                    existing = entry.setdefault("reasons", set())
                    for reason in reasons:
                        existing.add(reason)
                observed = sig.get("observed_text") or sig.get("text")
                if observed and "text" not in entry:
                    entry["text"] = str(observed)[:64]
    except Exception as exc:
        return {"error": str(exc), "path": learning_jsonl_path}
    if total == 0:
        return {}

    def _rank(stats: Dict[str, Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        for key, payload in stats.items():
            ranked.append({
                "key": key,
                "score": round(float(payload.get("score") or 0.0), 3),
                "count": int(payload.get("count") or 0),
            })
        ranked.sort(key=lambda item: (-item["score"], -item["count"]))
        return ranked[:limit]

    hot_tables = _rank(table_stats, 6)
    hot_rows = _rank(row_stats, 8)
    trace_rank = sorted(trace_scores.items(), key=lambda item: (-float(item[1].get("score") or 0.0), item[1].get("count", 0)))[:24]
    hot_cells: List[Dict[str, Any]] = []
    for trace, payload in trace_rank:
        cell_entry = {
            "trace_id": trace,
            "score": round(float(payload.get("score") or 0.0), 3),
            "page": payload.get("page"),
            "table": payload.get("table"),
            "row": payload.get("row"),
        }
        if payload.get("text"):
            cell_entry["text"] = payload.get("text")
        if payload.get("reasons"):
            cell_entry["reasons"] = sorted(payload["reasons"])
        hot_cells.append(cell_entry)
    reason_rank = [{"reason": name, "count": count} for name, count in reason_counts.most_common(6)]

    plan = _selective_focus_from_hotspots(trace_scores, row_stats, table_stats, reason_counts)

    result: Dict[str, Any] = {
        "total_samples": total,
        "table_hotspots": hot_tables,
        "row_hotspots": hot_rows,
        "reason_counts": reason_rank,
        "hot_cells": hot_cells,
    }
    if plan:
        result["focus_plan"] = plan
    return result


def _selective_focus_from_hotspots(
    trace_scores: Dict[str, Dict[str, Any]],
    row_stats: Dict[str, Dict[str, Any]],
    table_stats: Dict[str, Dict[str, Any]],
    reason_counts: Counter,
    max_traces: int = 96,
) -> Optional[Dict[str, Any]]:
    if not trace_scores and not row_stats and not table_stats:
        return None
    trace_order = sorted(trace_scores.items(), key=lambda item: (-float(item[1].get("score") or 0.0), item[1].get("count", 0)))
    trace_ids = [trace for trace, _ in trace_order[:max_traces] if trace]
    row_order = sorted(row_stats.items(), key=lambda item: (-float(item[1].get("score") or 0.0), -float(item[1].get("count") or 0)))
    row_keys = [row for row, _ in row_order[:16]]
    table_order = sorted(table_stats.items(), key=lambda item: (-float(item[1].get("score") or 0.0), -float(item[1].get("count") or 0)))
    table_keys = [table for table, _ in table_order[:10]]
    reasons = [name for name, _ in reason_counts.most_common(6)]
    if not trace_ids and not row_keys and not table_keys:
        return None
    total_row_score = sum(float(payload.get("score") or 0.0) for payload in row_stats.values())
    focus_row_score = sum(float(row_stats[key].get("score") or 0.0) for key in row_keys if key in row_stats)
    coverage = (focus_row_score / total_row_score) if total_row_score else None
    story_bits: List[str] = []
    if coverage is not None:
        story_bits.append(f"{coverage * 100:.1f}% of review load in {len(row_keys)} rows")
    if reasons:
        story_bits.append(f"top signal {reasons[0]}")
    if table_keys:
        story_bits.append(f"priority table {table_keys[0]}")
    story_text = "; ".join(story_bits)
    plan: Dict[str, Any] = {
        "trace_ids": trace_ids,
        "row_keys": row_keys,
        "table_keys": table_keys,
        "reasons": reasons,
        "coverage_ratio": coverage,
        "source": "learning_hotspots",
    }
    if trace_ids:
        plan["limit"] = len(trace_ids)
    if story_text:
        plan["story"] = story_text
    return {k: v for k, v in plan.items() if v not in (None, [], {})}


def _generate_hotspot_gallery(
    outdir: str,
    learning_jsonl_path: Optional[str],
    learning_hotspots: Optional[Dict[str, Any]],
    focus_plan: Optional[Dict[str, Any]],
    page_images: Optional[Dict[int, str]],
    limit: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if not learning_jsonl_path or not os.path.exists(learning_jsonl_path):
        return None
    if Image is None:
        return None
    try:
        gallery_limit = int(limit if limit is not None else os.environ.get("ZOCR_HOTSPOT_GALLERY_LIMIT", "12"))
    except Exception:
        gallery_limit = 12
    if gallery_limit <= 0:
        return None
    trace_order: List[str] = []
    cell_lookup: Dict[str, Dict[str, Any]] = {}
    reason_order: Dict[str, int] = {}
    if isinstance(learning_hotspots, dict):
        rank = learning_hotspots.get("reason_counts")
        if isinstance(rank, list):
            for idx, rec in enumerate(rank, 1):
                name = rec.get("reason") if isinstance(rec, dict) else None
                if name:
                    reason_order.setdefault(str(name), idx)
        for cell in learning_hotspots.get("hot_cells", []):
            if not isinstance(cell, dict):
                continue
            trace = str(cell.get("trace_id") or "").strip()
            if not trace:
                continue
            if trace not in trace_order:
                trace_order.append(trace)
            cell_lookup.setdefault(trace, cell)
    if isinstance(focus_plan, dict):
        for trace in focus_plan.get("trace_ids", []):
            if trace is None:
                continue
            trace_str = str(trace).strip()
            if not trace_str:
                continue
            if trace_str not in trace_order:
                trace_order.append(trace_str)
            cell_lookup.setdefault(trace_str, {"trace_id": trace_str})
    if not trace_order:
        return None
    candidate_traces = trace_order[: max(gallery_limit * 3, gallery_limit)]
    needed: Set[str] = set(candidate_traces)
    samples: Dict[str, Dict[str, Any]] = {}
    try:
        with open(learning_jsonl_path, "r", encoding="utf-8") as fr:
            for raw in fr:
                if len(samples) >= gallery_limit:
                    break
                line = raw.strip()
                if not line:
                    continue
                try:
                    sig = json.loads(line)
                except Exception:
                    continue
                trace = sig.get("trace_id") or sig.get("meta", {}).get("trace") if isinstance(sig.get("meta"), dict) else None
                if trace is None:
                    continue
                trace_str = str(trace).strip()
                if not trace_str or trace_str not in needed or trace_str in samples:
                    continue
                bbox = sig.get("bbox")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                try:
                    ints = [int(round(float(v))) for v in bbox]
                except Exception:
                    continue
                image_path = sig.get("image_path")
                page_idx = sig.get("page")
                if (not image_path) and isinstance(page_images, dict):
                    try:
                        image_path = page_images.get(int(page_idx))
                    except Exception:
                        image_path = None
                hypotheses = sig.get("hypotheses") if isinstance(sig.get("hypotheses"), list) else None
                after_text = None
                if isinstance(hypotheses, list):
                    for hypo in hypotheses:
                        if not isinstance(hypo, dict):
                            continue
                        cand = hypo.get("text") or hypo.get("candidate")
                        if cand:
                            after_text = str(cand)
                            break
                samples[trace_str] = {
                    "bbox": ints,
                    "page": page_idx,
                    "table": sig.get("table_index"),
                    "row": sig.get("row"),
                    "col": sig.get("col"),
                    "image_path": image_path,
                    "before_text": sig.get("observed_text") or sig.get("text"),
                    "after_text": after_text,
                    "reasons": [str(r) for r in sig.get("reasons", []) if isinstance(r, str)],
                }
    except Exception as exc:
        print(f"[WARN] hotspot gallery read failed: {exc}")
        return None
    if not samples:
        return None
    gallery_dir = os.path.join(outdir, "rag", "hotspots")
    try:
        ensure_dir(gallery_dir)
    except Exception as exc:
        print(f"[WARN] hotspot gallery dir failed: {exc}")
        return None
    entries: List[Dict[str, Any]] = []
    missing: Set[str] = set()
    for trace in trace_order:
        if len(entries) >= gallery_limit:
            break
        sample = samples.get(trace)
        if not sample:
            if trace in needed:
                missing.add(trace)
            continue
        img_path = sample.get("image_path")
        if not img_path or not os.path.exists(img_path):
            missing.add(trace)
            continue
        bbox = sample.get("bbox")
        if not bbox:
            missing.add(trace)
            continue
        try:
            with Image.open(img_path) as page_img:
                pw, ph = page_img.size
                x1, y1, x2, y2 = bbox
                margin = 4
                x1 = max(0, min(pw, x1 - margin))
                y1 = max(0, min(ph, y1 - margin))
                x2 = max(0, min(pw, x2 + margin))
                y2 = max(0, min(ph, y2 + margin))
                if x2 <= x1 or y2 <= y1:
                    missing.add(trace)
                    continue
                crop = page_img.crop((x1, y1, x2, y2))
                safe_trace = re.sub(r"[^A-Za-z0-9._-]", "_", trace)[:48] or "cell"
                dest = os.path.join(gallery_dir, f"{len(entries)+1:02d}_{safe_trace}.png")
                crop.save(dest)
        except Exception:
            missing.add(trace)
            continue
        cell_meta = cell_lookup.get(trace, {})
        reasons = sample.get("reasons") or cell_meta.get("reasons")
        role = None
        row_idx = sample.get("row") if sample.get("row") is not None else cell_meta.get("row")
        try:
            row_int = int(row_idx) if row_idx is not None else None
        except Exception:
            row_int = None
        if isinstance(reasons, list):
            joined = " ".join(reasons).lower()
            if "header" in joined:
                role = "header"
            elif "footer" in joined or "total" in joined:
                role = "footer"
        if role is None and row_int == 0:
            role = "header"
        if role is None and row_int is not None and row_int < 0:
            role = "footer"
        if role is None:
            role = "body"
        reason_rank = None
        if isinstance(reasons, list):
            ranks = [reason_order.get(r) for r in reasons if reason_order.get(r)]
            if ranks:
                reason_rank = min(ranks)
        entry = {
            "trace_id": trace,
            "image": os.path.relpath(dest, outdir),
            "page": sample.get("page"),
            "table": sample.get("table"),
            "row": sample.get("row"),
            "col": sample.get("col"),
            "text": sample.get("before_text") or cell_meta.get("text"),
            "role": role,
            "before_text": sample.get("before_text") or cell_meta.get("text"),
            "after_text": sample.get("after_text"),
            "reasons": reasons,
            "reason_rank": reason_rank,
            "score": cell_meta.get("score"),
        }
        entries.append({k: v for k, v in entry.items() if v not in (None, [], {})})
    if not entries:
        return None
    gallery = {
        "count": len(entries),
        "limit": gallery_limit,
        "dir": os.path.relpath(gallery_dir, outdir),
        "entries": entries,
        "source": "learning_hotspots",
    }
    if missing:
        gallery["missing_traces"] = sorted(missing)
    story_rel = _write_hotspot_gallery_story(outdir, gallery)
    if story_rel:
        gallery["story"] = story_rel
    return gallery


def _write_hotspot_gallery_story(outdir: str, gallery: Dict[str, Any]) -> Optional[str]:
    entries = gallery.get("entries") if isinstance(gallery, dict) else None
    if not entries:
        return None
    story_dir = os.path.join(outdir, "rag", "hotspots")
    try:
        ensure_dir(story_dir)
    except Exception as exc:
        print(f"[WARN] hotspot gallery story dir failed: {exc}")
        return None
    story_path = os.path.join(story_dir, "gallery.md")
    lines: List[str] = [
        "# Hotspot Gallery",
        "",
        f"Extracted {len(entries)} hotspot crops for advisor review.",
        "",
        "Each section links the cropped cell image and highlights why the pipeline flagged it.",
        "",
    ]
    for idx, entry in enumerate(entries, 1):
        trace = entry.get("trace_id") or "unknown"
        title = f"## Hotspot {idx}: trace `{trace}`"
        lines.append(title)
        bullet: List[str] = []
        for label in ("page", "table", "row", "col"):
            if entry.get(label) is not None:
                bullet.append(f"{label}={entry[label]}")
        if entry.get("role"):
            bullet.append(f"role={entry['role']}")
        if entry.get("reason_rank"):
            bullet.append(f"reason_rank={entry['reason_rank']}")
        if entry.get("text"):
            bullet.append(f"text=`{entry['text']}`")
        if entry.get("score") is not None:
            bullet.append(f"score={entry['score']}")
        if bullet:
            lines.append("- " + ", ".join(bullet))
        reasons = entry.get("reasons")
        if isinstance(reasons, list) and reasons:
            lines.append("- reasons: " + "; ".join(reasons))
        if entry.get("before_text"):
            lines.append(f"- before: `{entry['before_text']}`")
        if entry.get("after_text") and entry.get("after_text") != entry.get("before_text"):
            lines.append(f"- after: `{entry['after_text']}`")
        image_rel = entry.get("image")
        if image_rel:
            lines.append("")
            lines.append(f"![Hotspot {idx}]({image_rel})")
        lines.append("")
    try:
        with open(story_path, "w", encoding="utf-8") as fw:
            fw.write("\n".join(lines).strip() + "\n")
    except Exception as exc:
        print(f"[WARN] hotspot gallery story write failed: {exc}")
        return None
    return os.path.relpath(story_path, outdir)


def _summarize_toy_learning(
    toy_memory_delta: Optional[Dict[str, Any]],
    recognition_stats: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"signals": {}, "reasons": []}

    if toy_memory_delta:
        glyph_variants = toy_memory_delta.get("glyph_variants")
        if isinstance(glyph_variants, (int, float)):
            summary["signals"]["glyph_variants"] = float(glyph_variants)
            if glyph_variants > 0:
                summary["reasons"].append(f"learned {glyph_variants:.0f} glyph variants")
        surprisal_shift = toy_memory_delta.get("avg_surprisal")
        if isinstance(surprisal_shift, (int, float)) and abs(surprisal_shift) > 1e-6:
            direction = "dropped" if surprisal_shift < 0 else "rose"
            summary["signals"]["avg_surprisal_delta"] = float(surprisal_shift)
            summary["reasons"].append(f"surprisal {direction} by {abs(surprisal_shift):.3f}")
    stats = recognition_stats or {}
    cells = stats.get("cells") if isinstance(stats.get("cells"), (int, float)) else 0
    try:
        cells = int(cells)
    except Exception:
        cells = 0
    if cells > 0:
        low_conf = stats.get("low_conf_cells")
        high_surprisal = stats.get("high_surprisal_cells")
        try:
            low_conf_ratio = float(low_conf) / float(cells)
        except Exception:
            low_conf_ratio = None
        try:
            high_surprisal_ratio = float(high_surprisal) / float(cells)
        except Exception:
            high_surprisal_ratio = None
        if low_conf_ratio is not None:
            summary["signals"]["recognition_low_conf_ratio"] = low_conf_ratio
            if low_conf_ratio > 0.25:
                summary["reasons"].append(f"low-confidence cells at {low_conf_ratio:.2%}")
        if high_surprisal_ratio is not None:
            summary["signals"]["recognition_high_surprisal_ratio"] = high_surprisal_ratio
            if high_surprisal_ratio > 0.18:
                summary["reasons"].append(f"high surprisal at {high_surprisal_ratio:.2%}")
    runtime_gain = stats.get("runtime_replay_improved")
    if isinstance(runtime_gain, (int, float)) and runtime_gain:
        summary["signals"]["runtime_replay_improved"] = float(runtime_gain)
        summary["reasons"].append(f"runtime replay rescued {int(runtime_gain)} cells")

    if not summary["reasons"]:
        if summary["signals"]:
            summary["reasons"].append("toy OCR steady; no explicit triggers")
        else:
            return {}
    summary["narrative"] = " / ".join(summary["reasons"])
    return summary


def _intent_narrative(intent: IntentPayload) -> str:
    action = intent.get("action") or "steady"
    reason = intent.get("reason") or ""
    signals = intent.get("signals") or {}
    fragments: List[str] = []
    if reason:
        fragments.append(reason)
    key_pairs = [
        ("low_conf_ratio", "low-conf ratio"),
        ("high_surprisal_ratio", "surprisal"),
        ("recognition_low_conf_ratio", "recognition low-conf"),
        ("recognition_high_surprisal_ratio", "recognition surprisal"),
        ("p95_ms", "latency"),
    ]
    for key, label in key_pairs:
        value = signals.get(key)
        if value is None:
            continue
        try:
            val = float(value)
        except Exception:
            continue
        fragments.append(f"{label}={val:.3f}")
    narrative = f"Intent '{action}' chosen: " + ", ".join(fragments) if fragments else f"Intent '{action}' selected"
    return narrative


def _evaluate_learning_outcome(
    before_signals: Optional[Dict[str, Any]],
    after_signals: Optional[Dict[str, Any]],
    reanalysis_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not before_signals and not after_signals and not reanalysis_summary:
        return {}

    def _metric(payload: Optional[Dict[str, Any]], key: str) -> Optional[float]:
        if not payload:
            return None
        value = payload.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    outcome: Dict[str, Any] = {"before": {}, "after": {}}
    for label in ("low_conf_ratio", "high_surprisal_ratio"):
        before_value = _metric(before_signals, label)
        after_value = _metric(after_signals, label)
        if before_value is not None:
            outcome["before"][label] = before_value
        if after_value is not None:
            outcome["after"][label] = after_value
        if before_value is not None or after_value is not None:
            delta = (after_value if after_value is not None else before_value or 0.0) - (
                before_value if before_value is not None else after_value or 0.0
            )
            outcome.setdefault("delta", {})[label] = delta

    improved_cells = None
    avg_conf_delta = None
    if isinstance(reanalysis_summary, dict):
        try:
            improved_cells = int(reanalysis_summary.get("improved") or 0)
        except Exception:
            improved_cells = None
        try:
            avg_conf_delta = float(reanalysis_summary.get("avg_confidence_delta") or 0.0)
        except Exception:
            avg_conf_delta = None
    success = False
    reason: List[str] = []
    delta_low = outcome.get("delta", {}).get("low_conf_ratio") if outcome.get("delta") else None
    if delta_low is not None and delta_low < -0.02:
        success = True
        reason.append(f"low_conf_ratio improved by {abs(delta_low):.3f}")
    if isinstance(improved_cells, int) and improved_cells > 0:
        success = True
        reason.append(f"reanalyzer fixed {improved_cells} cells")
    if avg_conf_delta is not None and avg_conf_delta > 0:
        reason.append(f"avg confidence +{avg_conf_delta:.3f}")
    outcome["success"] = bool(success)
    if reason:
        outcome["reason"] = "; ".join(reason)
    if not success:
        outcome["needs_retry"] = True if reanalysis_summary else False
    outcome["reanalysis_summary"] = reanalysis_summary or None
    return outcome


def _write_advice_packet(outdir: str, summary: Dict[str, Any]) -> Optional[str]:
    payload = {
        "task": "ZOCR advisor request",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "intent": summary.get("intent"),
        "intent_story": summary.get("intent", {}).get("narrative") if isinstance(summary.get("intent"), dict) else None,
        "learning_outcome": summary.get("learning_outcome"),
        "toy_learning": summary.get("toy_memory", {}).get("learning_reason"),
        "monitor_row": summary.get("monitor_row"),
        "questions": [
            "What should the next corrective action be?",
            "Which headers or totals look unreliable?",
        ],
    }
    path = os.path.join(outdir, "advisor_prompt.json")
    try:
        with open(path, "w", encoding="utf-8") as fw:
            json.dump(_json_ready(payload), fw, ensure_ascii=False, indent=2)
        return path
    except Exception as exc:
        print(f"Advisor packet skipped: {exc}")
        return None


_ADVISOR_TEXT_HINTS = {
    "reanalyze_cells": [
        "reanalyze",
        "re-analyze",
        "reanlysis",
        "cell sweep",
        "再解析",
        "セル再解析",
    ],
    "rerun_monitor": [
        "rerun monitor",
        "monitor again",
        "monitor once more",
        "再モニタ",
        "監視をやり直し",
    ],
    "rerun_augment": [
        "rerun augment",
        "augment again",
        "再augment",
        "再度augment",
        "再増強",
    ],
}


def _canonical_advisor_action(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    norm = name.strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "reanalyze": "reanalyze_cells",
        "reanalyze_cells": "reanalyze_cells",
        "reanalyze_grid": "reanalyze_cells",
        "reanalyze_learning": "reanalyze_cells",
        "reanalyze_cells_now": "reanalyze_cells",
        "rerun_monitor": "rerun_monitor",
        "monitor_again": "rerun_monitor",
        "rerun_augment": "rerun_augment",
        "augment_again": "rerun_augment",
        "rerun_aug": "rerun_augment",
    }
    if norm in mapping:
        return mapping[norm]
    if norm.startswith("reanalyze") or "再解析" in norm:
        return "reanalyze_cells"
    if norm.startswith("monitor") or "再モニタ" in norm or "監視" in norm:
        return "rerun_monitor"
    if norm.startswith("augment") or "増強" in norm:
        return "rerun_augment"
    return norm if norm else None


def _parse_advisor_suggestions(text: Optional[str], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    actions: Set[str] = set()

    def _push(action: Optional[str]) -> None:
        if action:
            actions.add(action)

    if isinstance(payload, dict):
        raw_actions = payload.get("actions")
        if isinstance(raw_actions, (list, tuple, set)):
            for entry in raw_actions:
                if isinstance(entry, str):
                    _push(_canonical_advisor_action(entry))
        for key, value in payload.items():
            if isinstance(value, bool) and value:
                _push(_canonical_advisor_action(str(key)))
    lower_text = text.lower() if text else ""
    if lower_text:
        for action, hints in _ADVISOR_TEXT_HINTS.items():
            for hint in hints:
                if hint.lower() in lower_text:
                    actions.add(action)
                    break
    suggestions = {action: True for action in sorted(actions)}
    return {"actions": sorted(actions), "flags": suggestions}


def _ingest_advisor_response(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    info: Dict[str, Any] = {"path": path}
    if not os.path.exists(path):
        info["error"] = "not_found"
        info["status"] = "missing"
        return info
    raw_text = None
    payload = None
    try:
        with open(path, "r", encoding="utf-8") as fr:
            raw_text = fr.read()
    except Exception as exc:
        info["error"] = str(exc)
        info["status"] = "unreadable"
        return info
    if raw_text is None:
        info["status"] = "empty"
        return info
    snippet = raw_text[:4000]
    info["preview"] = snippet
    try:
        payload = json.loads(raw_text)
    except Exception:
        payload = None
    if payload is not None:
        info["payload"] = _json_ready(payload)
    parsed = _parse_advisor_suggestions(raw_text, payload if isinstance(payload, dict) else None)
    if parsed.get("actions"):
        info["actions"] = parsed["actions"]
        info["suggestions"] = parsed.get("flags")
    info["status"] = "ok"
    return info


def _git_revision() -> Optional[str]:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return rev.decode("utf-8", "ignore").strip() or None


def _fingerprint_page(path: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": path}
    try:
        st = os.stat(path)
        info["size"] = int(st.st_size)
        info["mtime"] = float(st.st_mtime)
    except Exception:
        pass
    try:
        h = hashlib.sha1()
        with open(path, "rb") as fr:
            chunk = fr.read(512 * 1024)
            h.update(chunk)
        info["sha1_head"] = h.hexdigest()
    except Exception:
        info.setdefault("sha1_head", None)
    return info


def _build_repro_signature(
    inputs: List[str],
    page_images: Dict[int, str],
    profile: Dict[str, Any],
    toy_runtime_snapshot: Optional[Dict[str, Any]],
    export_ocr_engine: str,
    toy_runtime_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    profile_digest = hashlib.sha256()
    try:
        profile_digest.update(json.dumps(profile, sort_keys=True).encode("utf-8"))
    except Exception:
        profile_digest.update(str(profile).encode("utf-8", "ignore"))
    pages_fp = [_fingerprint_page(path) for path in page_images.values() if path and os.path.exists(path)]
    inputs_digest = hashlib.sha256()
    for fp in pages_fp:
        inputs_digest.update((fp.get("path") or "").encode("utf-8", "ignore"))
        if fp.get("sha1_head"):
            inputs_digest.update(fp["sha1_head"].encode("utf-8"))
    signature = {
        "git_revision": _git_revision(),
        "inputs": inputs,
        "page_fingerprints": pages_fp,
        "profile_hash": profile_digest.hexdigest(),
        "inputs_hash": inputs_digest.hexdigest(),
        "export_ocr_engine": export_ocr_engine,
        "toy_runtime": toy_runtime_snapshot,
        "toy_runtime_overrides": toy_runtime_overrides,
    }
    return signature


def _diff_signatures(local: Dict[str, Any], foreign: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    diff: Dict[str, Any] = {}
    keys = set(local.keys()) | set(foreign.keys())
    for key in sorted(keys):
        lval = local.get(key)
        rval = foreign.get(key)
        if lval == rval:
            continue
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(lval, dict) and isinstance(rval, dict):
            sub = _diff_signatures(lval, rval, path)
            diff.update(sub)
        else:
            diff[path] = {"local": lval, "foreign": rval}
    return diff


def _write_repro_signature(
    outdir: str,
    signature: Dict[str, Any],
    ingest_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    sig_path = os.path.join(outdir, "repro_signature.json")
    ingest_result: Optional[Dict[str, Any]] = None
    try:
        with open(sig_path, "w", encoding="utf-8") as fw:
            json.dump(_json_ready(signature), fw, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"Signature write skipped: {exc}")
        sig_path = None
    if ingest_path:
        ingest_signature = ingest_path
        if os.path.exists(ingest_signature):
            try:
                with open(ingest_signature, "r", encoding="utf-8") as fr:
                    foreign = json.load(fr)
                diff = _diff_signatures(signature, foreign if isinstance(foreign, dict) else {})
                ingest_result = {
                    "path": ingest_signature,
                    "diff": diff,
                    "match": not diff,
                }
            except Exception as exc:
                ingest_result = {"path": ingest_path, "error": str(exc)}
    return sig_path, ingest_result


def _should_toy_self_correct(
    export_signals: Optional[Dict[str, Any]],
    recognition_stats: Optional[Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    details: Dict[str, Any] = {"reasons": [], "metrics": {}}
    signals = export_signals or {}

    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            f = float(value)
        except Exception:
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return f

    low_conf_ratio = _as_float(signals.get("low_conf_ratio"))
    high_surprisal_ratio = _as_float(signals.get("high_surprisal_ratio"))
    review_ratio = _as_float(signals.get("review_ratio"))
    learning_samples = _as_float(signals.get("learning_samples"))

    metrics = details["metrics"]
    if low_conf_ratio is not None:
        metrics["low_conf_ratio"] = low_conf_ratio
        if low_conf_ratio >= 0.2:
            details["reasons"].append("low_conf_ratio")
    if high_surprisal_ratio is not None:
        metrics["high_surprisal_ratio"] = high_surprisal_ratio
        if high_surprisal_ratio >= 0.14:
            details["reasons"].append("high_surprisal_ratio")
    if review_ratio is not None:
        metrics["review_ratio"] = review_ratio
        if review_ratio >= 0.25:
            details["reasons"].append("review_ratio")
    if learning_samples is not None:
        metrics["learning_samples"] = learning_samples
        if learning_samples >= 8:
            details["reasons"].append("learning_samples")

    stats = recognition_stats or {}
    cells = _as_float(stats.get("cells")) or 0.0
    if cells > 0:
        low_conf_cells = _as_float(stats.get("low_conf_cells"))
        if low_conf_cells is not None:
            recog_low_conf_ratio = low_conf_cells / cells
            metrics["recognition_low_conf_ratio"] = recog_low_conf_ratio
            if recog_low_conf_ratio >= 0.24:
                details["reasons"].append("recognition_low_conf")
        high_surprisal_cells = _as_float(stats.get("high_surprisal_cells"))
        if high_surprisal_cells is not None:
            recog_high_surprisal_ratio = high_surprisal_cells / cells
            metrics["recognition_high_surprisal_ratio"] = recog_high_surprisal_ratio
            if recog_high_surprisal_ratio >= 0.18:
                details["reasons"].append("recognition_high_surprisal")

    runtime_replay = _as_float(stats.get("runtime_replay_improved"))
    if runtime_replay is not None:
        metrics["runtime_replay_improved"] = runtime_replay
        if runtime_replay >= 3.0:
            details["reasons"].append("runtime_replay")

    severity = len(details["reasons"])
    details["severity"] = severity
    if severity:
        plan_levels: List[Dict[str, Any]] = []
        base_expand = 10
        if low_conf_ratio is not None and low_conf_ratio >= 0.28:
            base_expand += 8
        if review_ratio is not None and review_ratio >= 0.3:
            base_expand += 4
        base_step = 10
        recog_low = details["metrics"].get("recognition_low_conf_ratio") if isinstance(details.get("metrics"), dict) else None
        if isinstance(recog_low, (int, float)) and recog_low >= 0.3:
            base_step = 8
        recog_high = details["metrics"].get("recognition_high_surprisal_ratio") if isinstance(details.get("metrics"), dict) else None
        if isinstance(recog_high, (int, float)) and recog_high >= 0.18:
            base_step = 8
        fine_step = 6 if (high_surprisal_ratio is not None and high_surprisal_ratio >= 0.16) else 0
        extra_spread = 0
        runtime_replay = details["metrics"].get("runtime_replay_improved") if isinstance(details.get("metrics"), dict) else None
        if isinstance(runtime_replay, (int, float)) and runtime_replay >= 3.0:
            extra_spread = 4
        passes = min(3, max(1, severity + (1 if recog_high and recog_high >= 0.22 else 0)))
        for idx in range(passes):
            level_cfg: Dict[str, Any] = {
                "level": idx + 1,
                "threshold_expand": base_expand + idx * 6,
                "threshold_step": max(6, base_step - idx * 2),
                "target_confidence": 0.56 + 0.04 * min(idx + 1, 3),
                "extra_augment_passes": 1 + idx,
            }
            if fine_step:
                level_cfg["fine_threshold_step"] = max(3, fine_step - idx)
            if extra_spread:
                level_cfg["extra_local_spread"] = extra_spread + idx * 2
            if "high_surprisal_ratio" in details["reasons"] or "recognition_high_surprisal" in details["reasons"]:
                level_cfg.setdefault("force_augment", True)
                level_cfg.setdefault("extra_rotations", [-5, -2, 2, 5])
            if review_ratio is not None and review_ratio >= 0.32:
                level_cfg.setdefault("augment_filter_sizes", [7])
            plan_levels.append(level_cfg)
        details["plan"] = {
            "levels": plan_levels,
            "stop_on_improvement": True,
            "require_improvement": bool(review_ratio is not None and review_ratio >= 0.42),
            "severity": severity,
        }
    return (severity > 0), details


def _reanalyze_output_paths(learning_jsonl: str, outdir: str) -> Tuple[str, str]:
    base = os.path.basename(learning_jsonl)
    if base.endswith(".jsonl"):
        base = base[:-6]
    output = os.path.join(outdir, f"{base}.reanalyzed.jsonl")
    return output, output + ".summary.json"


def _apply_reanalysis_to_contextual_jsonl(
    contextual_jsonl: str,
    reanalyzed_jsonl: str,
    outdir: str,
    summary: Dict[str, Any],
    ocr_min_conf: float,
    surprisal_threshold: Optional[float] = None,
) -> str:
    if not reanalyzed_jsonl or not os.path.exists(reanalyzed_jsonl):
        return contextual_jsonl
    base_dir = os.path.dirname(contextual_jsonl) or outdir
    base_name = os.path.basename(contextual_jsonl)
    if base_name.endswith(".jsonl"):
        base_name = base_name[:-6]
    if base_name.endswith(".reanalyzed"):
        dest_path = contextual_jsonl
    else:
        dest_path = os.path.join(base_dir, f"{base_name}.reanalyzed.jsonl")
    rewrite = zocr_onefile_consensus.apply_reanalysis_to_jsonl(
        contextual_jsonl,
        reanalyzed_jsonl,
        dest_path,
        ocr_min_conf=ocr_min_conf,
        surprisal_threshold=surprisal_threshold,
    )
    if rewrite.get("written"):
        applied_entry = _json_ready(rewrite)
        summary.setdefault("reanalysis_applied", [])
        summary["reanalysis_applied"].append(applied_entry)
        if os.path.abspath(dest_path) != os.path.abspath(contextual_jsonl):
            summary.setdefault("contextual_jsonl_original", contextual_jsonl)
        new_jsonl = applied_entry.get("output_jsonl") or dest_path
        new_signals = _load_export_signals(new_jsonl)
        if new_signals:
            summary["export_signals"] = new_signals
        summary["contextual_jsonl"] = new_jsonl
        return new_jsonl
    if rewrite.get("error"):
        summary.setdefault("reanalysis_errors", []).append(str(rewrite.get("error")))
    return contextual_jsonl


def _profile_snapshot(prof: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(prof))


def _profile_diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    diff: Dict[str, Tuple[Any, Any]] = {}
    keys = set(before.keys()) | set(after.keys())
    for key in sorted(keys):
        if before.get(key) != after.get(key):
            diff[key] = (before.get(key), after.get(key))
    return diff


def _derive_intent(
    monitor_row: Optional[Dict[str, Any]],
    export_signals: Dict[str, Any],
    profile: Dict[str, Any],
    toy_memory_delta: Optional[Dict[str, Any]] = None,
    recognition_stats: Optional[Dict[str, Any]] = None,
) -> IntentPayload:
    intent: IntentPayload = {"action": "steady", "reason": "metrics within guardrails", "priority": "low"}
    hit_mean = None
    p95 = None
    if monitor_row:
        try:
            hit_mean = float(monitor_row.get("hit_mean") or monitor_row.get("hit_mean_gt"))
        except Exception:
            hit_mean = None
        try:
            p95 = float(monitor_row.get("p95_ms")) if monitor_row.get("p95_ms") is not None else None
        except Exception:
            p95 = None
    low_conf_ratio = None
    try:
        low_conf_ratio = float(export_signals.get("low_conf_ratio")) if export_signals else None
    except Exception:
        low_conf_ratio = None
    high_surprisal_ratio = None
    try:
        high_surprisal_ratio = (
            float(export_signals.get("high_surprisal_ratio")) if export_signals else None
        )
    except Exception:
        high_surprisal_ratio = None
    recog_low_conf_ratio = None
    recog_high_surprisal_ratio = None
    learned_variants = 0.0
    runtime_replay = 0.0
    if recognition_stats:
        try:
            cells = float(recognition_stats.get("cells") or 0.0)
        except Exception:
            cells = 0.0
        if cells > 0:
            try:
                recog_low_conf_ratio = float(recognition_stats.get("low_conf_cells", 0.0)) / cells
            except Exception:
                recog_low_conf_ratio = None
            try:
                recog_high_surprisal_ratio = float(recognition_stats.get("high_surprisal_cells", 0.0)) / cells
            except Exception:
                recog_high_surprisal_ratio = None
        try:
            runtime_replay = float(recognition_stats.get("runtime_replay_improved", 0.0))
        except Exception:
            runtime_replay = 0.0
    if toy_memory_delta:
        try:
            learned_variants = float(toy_memory_delta.get("glyph_variants", 0.0))
        except Exception:
            learned_variants = 0.0
    if hit_mean is None:
        intent = {"action": "recover", "reason": "monitor missing", "priority": "high"}
    elif hit_mean < 0.8:
        intent = {"action": "focus_headers", "reason": f"hit_mean={hit_mean:.3f} below 0.8", "priority": "high"}
    elif p95 is not None and p95 > 400.0:
        intent = {"action": "optimize_speed", "reason": f"p95_ms={p95:.1f} > 400", "priority": "medium"}
    elif recog_high_surprisal_ratio is not None and recog_high_surprisal_ratio > 0.18:
        intent = {
            "action": "reanalyze_cells",
            "reason": f"recognition_high_surprisal={recog_high_surprisal_ratio:.2f}",
            "priority": "high",
        }
    elif recog_low_conf_ratio is not None and recog_low_conf_ratio > 0.28:
        intent = {
            "action": "reanalyze_cells",
            "reason": f"recognition_low_conf={recog_low_conf_ratio:.2f}",
            "priority": "medium",
        }
    elif low_conf_ratio is not None and low_conf_ratio > 0.2:
        intent = {
            "action": "reanalyze_cells",
            "reason": f"low_conf_ratio={low_conf_ratio:.2f}",
            "priority": "medium",
        }
    elif high_surprisal_ratio is not None and high_surprisal_ratio > 0.12:
        intent = {
            "action": "reanalyze_cells",
            "reason": f"high_surprisal_ratio={high_surprisal_ratio:.2f}",
            "priority": "medium",
        }
    elif learned_variants > 4.0 and (
        (low_conf_ratio is not None and low_conf_ratio > 0.14)
        or (recog_low_conf_ratio is not None and recog_low_conf_ratio > 0.18)
    ):
        intent = {
            "action": "reanalyze_cells",
            "reason": f"memory_growth={learned_variants:.0f} variants without confidence relief",
            "priority": "medium",
        }
    else:
        intent = {"action": "explore_footer", "reason": "metrics nominal", "priority": "low"}
    intent["signals"] = {
        "hit_mean": hit_mean,
        "p95_ms": p95,
        "low_conf_ratio": low_conf_ratio,
        "high_surprisal_ratio": high_surprisal_ratio,
        "recognition_low_conf_ratio": recog_low_conf_ratio,
        "recognition_high_surprisal_ratio": recog_high_surprisal_ratio,
        "learned_variants": learned_variants,
        "runtime_replay_improved": runtime_replay,
        "surprisal_threshold": export_signals.get("surprisal_threshold") if export_signals else None,
        "learning_samples": export_signals.get("learning_samples") if export_signals else None,
    }
    intent["profile_domain"] = profile.get("domain")
    intent["narrative"] = _intent_narrative(intent)
    return intent


def _derive_meta_intent(
    intent: Optional[IntentPayload],
    learning_hotspots: Optional[Dict[str, Any]],
    focus_plan: Optional[Dict[str, Any]],
    rag_feedback: Optional[Dict[str, Any]] = None,
    advisor_ingest: Optional[Dict[str, Any]] = None,
    learning_outcome: Optional[Dict[str, Any]] = None,
) -> MetaIntentPayload:
    if not intent:
        return {}
    action = intent.get("action") or "steady"
    meta: MetaIntentPayload = {
        "intent_action": action,
        "meta_action": "reflect_intent",
        "priority": intent.get("priority"),
        "signals": intent.get("signals"),
    }
    story_bits: List[str] = []
    recommendations: List[str] = []
    if focus_plan:
        meta["focus_plan"] = focus_plan
    coverage = focus_plan.get("coverage_ratio") if isinstance(focus_plan, dict) else None
    top_reason = None
    reasons = learning_hotspots.get("reason_counts") if isinstance(learning_hotspots, dict) else None
    if isinstance(reasons, list) and reasons:
        top_reason = reasons[0].get("reason")
    if top_reason:
        story_bits.append(f"top signal {top_reason}")
    if coverage is not None:
        story_bits.append(f"focus covers {coverage * 100:.1f}% of review load")
    if action == "reanalyze_cells":
        meta["meta_action"] = "prioritize_hotspots" if focus_plan else "validate_reanalysis_reason"
        meta["reason"] = intent.get("reason")
        recommendations.append("rerun_selective_reanalysis")
        if not focus_plan:
            recommendations.append("collect_hotspots")
    elif action == "focus_headers":
        meta["meta_action"] = "explain_header_shift"
        meta["reason"] = intent.get("reason")
        recommendations.append("compare_header_rows")
    elif action == "optimize_speed":
        meta["meta_action"] = "speed_accuracy_tradeoff"
        meta["reason"] = intent.get("reason")
        recommendations.append("audit_latency_trace")
    elif action == "explore_footer":
        meta["meta_action"] = "validate_footer_scan"
    else:
        meta["meta_action"] = "stabilize_intent"
    if learning_outcome:
        meta["learning_outcome"] = learning_outcome
        if not learning_outcome.get("success"):
            recommendations.append("escalate_learning_loop")
            story_bits.append("learning outcome pending")
        else:
            story_bits.append("learning succeeded")
    external_inputs: Dict[str, Any] = {}
    if rag_feedback and rag_feedback.get("actions"):
        external_inputs["rag_actions"] = rag_feedback.get("actions")
        story_bits.append(f"RAG requested {', '.join(rag_feedback.get('actions', []))}")
    if advisor_ingest and advisor_ingest.get("actions"):
        external_inputs["advisor_actions"] = advisor_ingest.get("actions")
        story_bits.append(f"Advisor requested {', '.join(advisor_ingest.get('actions', []))}")
    if external_inputs:
        meta["external_inputs"] = external_inputs
    if not external_inputs:
        recommendations.append("publish_feedback_request")
    if story_bits:
        meta["story"] = "; ".join(story_bits)
    if recommendations:
        meta["recommendations"] = sorted(set(recommendations))
    return meta


def _apply_intent_to_profile(
    intent: IntentPayload,
    profile: Dict[str, Any],
    guard: Optional[_ProfileGuard] = None,
) -> Dict[str, Tuple[Any, Any]]:
    updates: Dict[str, Tuple[Any, Any]] = {}

    def _set_value(key: str, value: Any) -> bool:
        old = profile.get(key)
        new_value = value
        if guard:
            applied, final, _ = guard.apply(key, value, old, source="intent")
            if not applied:
                return False
            new_value = final
        profile[key] = new_value
        updates[key] = (old, profile.get(key))
        return True

    action = intent.get("action")
    if action == "focus_headers":
        old = profile.get("header_boost", 1.0)
        new_val = float(old) * 1.15 if isinstance(old, (int, float)) else 1.2
        _set_value("header_boost", new_val)
        targets = list(profile.get("reanalyze_target") or [])
        if "headers" not in targets:
            targets.append("headers")
            _set_value("reanalyze_target", targets)
    elif action == "optimize_speed":
        old = profile.get("lambda_shape", 4.5)
        try:
            new_val = max(2.5, float(old) * 0.9)
        except Exception:
            new_val = 3.8
        _set_value("lambda_shape", new_val)
        if not _coerce_bool(profile.get("speed_priority")):
            _set_value("speed_priority", True)
    elif action == "reanalyze_cells":
        prev = list(profile.get("reanalyze_target") or [])
        if "learning_cells" not in prev:
            prev.append("learning_cells")
            _set_value("reanalyze_target", prev)
    elif action == "recover":
        if not _coerce_bool(profile.get("force_monitor_refresh")):
            _set_value("force_monitor_refresh", True)
    return updates


def _needs_rerun_for_keys(keys: List[str]) -> Dict[str, bool]:
    rerun = {"augment": False, "monitor": False}
    for key in keys:
        if key in {"lambda_shape", "header_boost", "reanalyze_target"}:
            rerun["augment"] = True
            rerun["monitor"] = True
        elif key in {"w_kw", "w_img", "ocr_min_conf", "force_monitor_refresh"}:
            rerun["monitor"] = True
    return rerun


def _apply_rag_feedback(
    manifest_path: Optional[str],
    profile: Optional[Dict[str, Any]],
    profile_path: str,
    *,
    persist_profile: bool = True,
    guard: Optional[_ProfileGuard] = None,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {"manifest": manifest_path, "applied": []}
    if not manifest_path:
        info["status"] = "missing"
        return info
    manifest_abs = os.path.abspath(manifest_path)
    info["manifest"] = manifest_abs
    if not os.path.exists(manifest_abs):
        info["status"] = "not_found"
        return info
    try:
        with open(manifest_abs, "r", encoding="utf-8") as fr:
            payload = json.load(fr)
    except Exception as exc:
        info["status"] = "manifest_unreadable"
        info["error"] = str(exc)
        return info
    feedback = payload.get("feedback") if isinstance(payload, dict) else None
    if not isinstance(feedback, dict):
        info["status"] = "no_feedback"
        return info
    info["status"] = "ok"
    note = feedback.get("notes") or feedback.get("summary") or feedback.get("reason")
    if note:
        info["note"] = note
    overrides = feedback.get("profile_overrides") if isinstance(feedback, dict) else None
    if isinstance(overrides, dict) and overrides:
        info["overrides"] = {k: _json_ready(v) for k, v in overrides.items()}
        applied: List[str] = []
        for key, value in overrides.items():
            applied.append(key)
            if persist_profile and profile is not None:
                target_value = value
                if guard:
                    allowed, final, _ = guard.apply(
                        key, value, profile.get(key), source="rag_feedback"
                    )
                    if not allowed:
                        continue
                    target_value = final
                profile[key] = target_value
        info["applied"] = applied
        if persist_profile and applied and profile is not None:
            try:
                with open(profile_path, "w", encoding="utf-8") as fw:
                    json.dump(_json_ready(profile), fw, ensure_ascii=False, indent=2)
            except Exception as exc:
                info["error"] = str(exc)
    actions: Set[str] = set()

    def _push_action(name: Optional[str]) -> None:
        if not name:
            return
        canon = _canonical_advisor_action(name)
        if canon:
            actions.add(canon)
        else:
            actions.add(name)

    for key in ("actions", "advisor_actions", "recommended_actions"):
        block = feedback.get(key)
        if isinstance(block, (list, tuple, set)):
            for entry in block:
                if isinstance(entry, str):
                    _push_action(entry)
    for key, value in feedback.items():
        if key in {"profile_overrides", "actions", "advisor_actions", "recommended_actions", "notes", "summary", "reason"}:
            continue
        if isinstance(value, bool) and value:
            _push_action(key)
        elif isinstance(value, str) and value.lower() in {"true", "yes"}:
            _push_action(key)
    if actions:
        info["actions"] = sorted(actions)
    return info


def _feedback_observations(summary: Dict[str, Any]) -> Dict[str, Any]:
    monitor = summary.get("monitor_row") or {}
    export_signals = summary.get("export_signals") or {}
    stage_stats = summary.get("stage_stats") or {}
    observations: Dict[str, Any] = {
        "domain": summary.get("domain"),
        "domain_guess": summary.get("domain_autodetect", {})
        .get("from_inputs", {})
        .get("guess"),
        "domain_confidence": summary.get("domain_autodetect", {})
        .get("from_inputs", {})
        .get("best_score"),
        "intent_action": (summary.get("intent") or {}).get("action"),
        "intent_reason": (summary.get("intent") or {}).get("reason"),
        "intent_story": (summary.get("intent") or {}).get("narrative"),
        "low_conf_ratio": _coerce_float(export_signals.get("low_conf_ratio")),
        "high_surprisal_ratio": _coerce_float(export_signals.get("high_surprisal_ratio")),
        "hit_amount": _coerce_float(monitor.get("hit_amount") or monitor.get("hit_amount_gt")),
        "hit_date": _coerce_float(monitor.get("hit_date") or monitor.get("hit_date_gt")),
        "hit_mean": _coerce_float(monitor.get("hit_mean") or monitor.get("hit_mean_gt")),
        "gate_pass": monitor.get("gate_pass"),
        "gate_reason": monitor.get("gate_reason"),
        "p95_ms": _coerce_float(monitor.get("p95_ms")),
        "toy_runtime": summary.get("toy_runtime_config"),
        "toy_runtime_overrides": summary.get("toy_runtime_overrides"),
        "toy_sweeps": summary.get("toy_sweeps"),
        "last_export_stats": summary.get("last_export_stats"),
        "stage_stats": stage_stats,
    }
    meta_intent = summary.get("meta_intent") or {}
    if isinstance(meta_intent, dict):
        observations["meta_intent_action"] = meta_intent.get("meta_action")
        observations["meta_intent_story"] = meta_intent.get("story")
    export_cells = summary.get("last_export_stats") or {}
    if export_cells:
        observations.setdefault("export_cells", export_cells.get("cells"))
    if summary.get("learning_hotspots"):
        observations["learning_hotspots"] = summary.get("learning_hotspots")
    if summary.get("selective_reanalysis_plan"):
        observations["selective_reanalysis_plan"] = summary.get("selective_reanalysis_plan")
    if summary.get("hotspot_gallery"):
        observations["hotspot_gallery"] = summary.get("hotspot_gallery")
    recognizer = summary.get("recognition_stats") or summary.get("toy_recognition_stats")
    if recognizer:
        observations["recognition_stats"] = recognizer
    return _json_ready({k: v for k, v in observations.items() if v is not None})


def _emit_rag_feedback_request(
    outdir: str,
    summary: Dict[str, Any],
    *,
    manifest_path: Optional[str] = None,
    rag_feedback_ingest: Optional[Dict[str, Any]] = None,
    advisor_ingest: Optional[Dict[str, Any]] = None,
    rag_feedback_actions: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    rag_dir = os.path.join(outdir, "rag")
    try:
        ensure_dir(rag_dir)
    except Exception as exc:
        print(f"[WARN] feedback_request dir failed: {exc}")
        return None
    target_manifest = manifest_path or os.path.join(rag_dir, "manifest.json")
    context = _feedback_observations(summary)
    generated_at = datetime.utcnow().isoformat() + "Z"
    intent = summary.get("intent") or {}
    meta_intent = summary.get("meta_intent") or {}
    low_conf = context.get("low_conf_ratio")
    questions: List[str] = []
    if summary.get("hotspot_gallery"):
        questions.append(
            "Which hotspot traces show the clearest header/footer mistakes? Reference trace_id and suggest corrections."
        )
    if summary.get("selective_reanalysis_plan"):
        questions.append("Should we expand or shrink the selective reanalysis plan? Name the rows/tables to change.")
    if isinstance(low_conf, (int, float)):
        questions.append(
            f"Propose up to 3 profile_overrides that would reduce low_conf_ratio (current≈{low_conf:.2f})."
        )
    if intent.get("action"):
        questions.append(
            f"Does the current intent `{intent.get('action')}` still make sense? Suggest an alternative action or confirm it."
        )
    if meta_intent.get("story"):
        questions.append(
            "Summarize the meta-intent story back in 1 sentence to ensure alignment, then state the next manual check."
        )
    request_payload: Dict[str, Any] = {
        "generated_at": generated_at,
        "outdir": outdir,
        "target_manifest": target_manifest,
        "instructions": {
            "ja": "rag/manifest.json の feedback ブロックに profile_overrides/actions を追記して --resume で再実行してください。",
            "en": "Edit the manifest's feedback block (notes/profile_overrides/actions) then re-run the pipeline with --resume.",
        },
        "example_feedback": {
            "feedback": {
                "notes": "ex: high surprisal on footer, please reanalyze",
                "profile_overrides": {"ocr_min_conf": 0.52},
                "actions": ["reanalyze_cells", "rerun_monitor"],
            }
        },
        "observations": context,
        "pending_actions": rag_feedback_actions or summary.get("feedback_passes"),
        "questions": questions,
    }
    if rag_feedback_ingest:
        request_payload["current_feedback"] = _json_ready(rag_feedback_ingest)
    if advisor_ingest:
        request_payload["advisor_feedback"] = _json_ready(advisor_ingest)
    if summary.get("meta_intent"):
        request_payload["meta_intent"] = summary.get("meta_intent")
    if summary.get("learning_hotspots"):
        request_payload["learning_hotspots"] = summary.get("learning_hotspots")
    if summary.get("selective_reanalysis_plan"):
        request_payload["selective_reanalysis_plan"] = summary.get("selective_reanalysis_plan")
    if summary.get("hotspot_gallery"):
        request_payload["hotspot_gallery"] = summary.get("hotspot_gallery")
    req_json = os.path.join(rag_dir, "feedback_request.json")
    req_md = os.path.join(rag_dir, "feedback_request.md")
    try:
        with open(req_json, "w", encoding="utf-8") as fw:
            json.dump(_json_ready(request_payload), fw, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[WARN] feedback_request.json write failed: {exc}")
        return None
    lines = [
        "# RAG Feedback Request",
        f"Generated: {generated_at}",
        "",
        f"Target manifest: {target_manifest}",
        "",
        "## Observations",
    ]
    for key, value in context.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Next steps",
            "1. Open the manifest above (or copy it elsewhere).",
            "2. Add/adjust the `feedback` block with notes, profile_overrides, and actions.",
            "3. Save it and run `python -m zocr run --resume --outdir ...` (or pass --rag-feedback).",
        ]
    )
    if questions:
        lines.append("")
        lines.append("## Questions for reviewers")
        for q in questions:
            lines.append(f"- {q}")
    if rag_feedback_actions:
        lines.append("")
        lines.append("### Pending actions")
        for act in rag_feedback_actions:
            lines.append(f"- {act}")
    try:
        with open(req_md, "w", encoding="utf-8") as fw:
            fw.write("\n".join(lines) + "\n")
    except Exception as exc:
        print(f"[WARN] feedback_request.md write failed: {exc}")
    return {
        "target_manifest": target_manifest,
        "request_json": req_json,
        "request_markdown": req_md,
    }


def _append_rag_conversation_entry(outdir: str, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not outdir:
        return None
    rag_dir = os.path.join(outdir, "rag")
    try:
        ensure_dir(rag_dir)
    except Exception as exc:
        print(f"[WARN] rag conversation dir failed: {exc}")
        return None
    convo_path = os.path.join(rag_dir, "conversation.jsonl")
    record = dict(entry or {})
    record.setdefault("role", "pipeline")
    record.setdefault("kind", "note")
    record.setdefault("ts", datetime.utcnow().isoformat() + "Z")
    try:
        with open(convo_path, "a", encoding="utf-8") as fw:
            fw.write(json.dumps(_json_ready(record), ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[WARN] rag conversation append failed: {exc}")
        return None
    return {"path": convo_path, "entry": record}


def _simulate_param_shift(
    monitor_row: Optional[Dict[str, Any]],
    export_signals: Dict[str, Any],
    recognition_stats: Optional[Dict[str, Any]],
    toy_memory_delta: Optional[Dict[str, Any]],
    profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    simulations: List[Dict[str, Any]] = []
    low_conf_ratio = None
    high_surprisal_ratio = None
    if export_signals:
        try:
            low_conf_ratio = float(export_signals.get("low_conf_ratio"))
        except Exception:
            low_conf_ratio = None
        try:
            high_surprisal_ratio = float(export_signals.get("high_surprisal_ratio"))
        except Exception:
            high_surprisal_ratio = None
    recog_low_conf = None
    recog_high_surprisal = None
    cells = None
    if recognition_stats:
        try:
            cells = float(recognition_stats.get("cells") or 0.0)
        except Exception:
            cells = 0.0
        if cells and cells > 0:
            try:
                recog_low_conf = float(recognition_stats.get("low_conf_cells", 0.0)) / cells
            except Exception:
                recog_low_conf = None
            try:
                recog_high_surprisal = float(recognition_stats.get("high_surprisal_cells", 0.0)) / cells
            except Exception:
                recog_high_surprisal = None
    memory_growth = 0.0
    runtime_improved = 0.0
    if toy_memory_delta:
        try:
            memory_growth = float(toy_memory_delta.get("glyph_variants", 0.0))
        except Exception:
            memory_growth = 0.0
        try:
            runtime_improved = float(toy_memory_delta.get("runtime_replay_improved", 0.0))
        except Exception:
            runtime_improved = 0.0
    averaged_low_conf = None
    ratios = [r for r in (low_conf_ratio, recog_low_conf) if r is not None]
    if ratios:
        averaged_low_conf = sum(ratios) / float(len(ratios))
    averaged_high_surprisal = None
    ratios = [r for r in (high_surprisal_ratio, recog_high_surprisal) if r is not None]
    if ratios:
        averaged_high_surprisal = sum(ratios) / float(len(ratios))
    base_conf = None
    try:
        base_conf = float(profile.get("ocr_min_conf", 0.58))
    except Exception:
        base_conf = 0.58
    if base_conf is None or math.isnan(base_conf):
        base_conf = 0.58
    for offset in (-0.05, 0.05):
        candidate = float(min(0.95, max(0.3, base_conf + offset)))
        delta = candidate - base_conf
        predicted_low_conf = None
        if averaged_low_conf is not None:
            learning_boost = min(0.2, max(0.0, memory_growth) * 0.01 + max(0.0, runtime_improved) * 0.005)
            predicted_low_conf = averaged_low_conf + delta * 0.9
            if delta < 0:
                predicted_low_conf = max(0.0, predicted_low_conf - learning_boost)
            else:
                predicted_low_conf = min(1.0, predicted_low_conf + learning_boost * 0.5)
        predicted_high_surprisal = None
        if averaged_high_surprisal is not None:
            predicted_high_surprisal = averaged_high_surprisal + delta * 0.6
            if delta < 0:
                predicted_high_surprisal = max(0.0, predicted_high_surprisal - 0.03)
        confidence = 0.4 + min(0.5, max(0.0, memory_growth) * 0.02 + (runtime_improved * 0.01))
        simulations.append(
            {
                "type": "profile_param",
                "param": "ocr_min_conf",
                "delta": round(delta, 4),
                "candidate": round(candidate, 4),
                "confidence": round(confidence, 3),
                "predictions": {
                    "low_conf_ratio": predicted_low_conf,
                    "high_surprisal_ratio": predicted_high_surprisal,
                },
            }
        )
    base_lambda = None
    try:
        base_lambda = float(profile.get("lambda_shape", 4.5))
    except Exception:
        base_lambda = 4.5
    if base_lambda is None or math.isnan(base_lambda):
        base_lambda = 4.5
    base_p95 = None
    if monitor_row:
        try:
            base_p95 = float(monitor_row.get("p95_ms")) if monitor_row.get("p95_ms") is not None else None
        except Exception:
            base_p95 = None
    for offset in (-0.4, 0.4):
        candidate = float(min(8.0, max(2.0, base_lambda + offset)))
        delta = candidate - base_lambda
        predicted_p95 = None
        if base_p95 is not None:
            speed_factor = -delta * 18.0
            predicted_p95 = max(120.0, base_p95 + speed_factor)
        simulations.append(
            {
                "type": "profile_param",
                "param": "lambda_shape",
                "delta": round(delta, 4),
                "candidate": round(candidate, 4),
                "confidence": 0.5,
                "predictions": {
                    "p95_ms": predicted_p95,
                },
            }
        )
    averaged_low_conf = averaged_low_conf if averaged_low_conf is not None else low_conf_ratio
    if averaged_low_conf is not None:
        expected = max(0.0, averaged_low_conf - min(0.12, max(0.0, memory_growth) * 0.015))
        simulations.append(
            {
                "type": "action",
                "action": "reanalyze_cells",
                "confidence": round(0.55 + min(0.4, max(0.0, memory_growth) * 0.03), 3),
                "predictions": {
                    "low_conf_ratio": expected,
                    "runtime_replay_gain": runtime_improved + max(0.0, memory_growth) * 0.1,
                },
            }
        )
    return simulations

def _patched_run_full_pipeline(
    inputs: List[str],
    outdir: str,
    dpi: int = 200,
    domain_hint: Optional[str] = None,
    k: int = 10,
    do_tune: bool = True,
    tune_budget: int = 20,
    views_log: Optional[str] = None,
    gt_jsonl: Optional[str] = None,
    org_dict: Optional[str] = None,
    resume: bool = False,
    seed: int = 24601,
    snapshot: bool = False,
    ocr_engine: Optional[str] = None,
    toy_lite: bool = False,
    toy_sweeps: Optional[int] = None,
    force_numeric_by_header: Optional[bool] = None,
    ingest_signature: Optional[str] = None,
    advisor_response: Optional[str] = None,
    print_stage_trace: Optional[bool] = None,
    rag_feedback: Optional[str] = None,
    motion_prior: bool = False,
    motion_sigma_px: Optional[float] = None,
    motion_cutoff_sigma: Optional[float] = None,
    motion_accept_ratio: Optional[float] = None,
    export_guard_ms: Optional[int] = None,
    sweeps_fixed: Optional[int] = None,
    blank_skip: Optional[bool] = None,
    blank_threshold: Optional[int] = None,
    blank_min_pixels: Optional[int] = None,
    blank_min_ratio: Optional[float] = None,
    blank_min_area: Optional[int] = None,
    allow_pytesseract: Optional[bool] = None,
    tess_unicharset: Optional[str] = None,
    tess_wordlist: Optional[str] = None,
    tess_bigram_json: Optional[str] = None,
) -> Dict[str, Any]:
    if sweeps_fixed is not None and sweeps_fixed > 0:
        toy_sweeps = int(sweeps_fixed)
        os.environ["ZOCR_TOY_SWEEPS"] = str(toy_sweeps)
        os.environ["ZOCR_TOY_SWEEP_LIMIT"] = str(toy_sweeps)
    if motion_prior:
        os.environ["ZOCR_EXPORT_MOTION_PRIOR"] = "1"
    if motion_prior and motion_sigma_px is None:
        motion_sigma_px = 10.0
    if motion_prior and motion_cutoff_sigma is None:
        motion_cutoff_sigma = 2.5
    if motion_prior and motion_accept_ratio is None:
        motion_accept_ratio = 0.6
    if motion_sigma_px is not None:
        os.environ["ZOCR_EXPORT_MOTION_SIGMA"] = str(motion_sigma_px)
    if motion_cutoff_sigma is not None:
        os.environ["ZOCR_EXPORT_MOTION_CUTOFF"] = str(motion_cutoff_sigma)
    if motion_accept_ratio is not None:
        os.environ["ZOCR_EXPORT_MOTION_ACCEPT"] = str(motion_accept_ratio)
    if export_guard_ms is not None:
        os.environ["ZOCR_EXPORT_GUARD_MS"] = str(max(0, int(export_guard_ms)))
    if blank_skip is not None:
        os.environ["ZOCR_EXPORT_SKIP_BLANK"] = "1" if blank_skip else "0"
    if blank_threshold is not None:
        os.environ["ZOCR_EXPORT_BLANK_THRESHOLD"] = str(int(blank_threshold))
    if blank_min_pixels is not None:
        os.environ["ZOCR_EXPORT_BLANK_MIN_PIXELS"] = str(int(blank_min_pixels))
    if blank_min_ratio is not None:
        os.environ["ZOCR_EXPORT_BLANK_MIN_RATIO"] = str(float(blank_min_ratio))
    if blank_min_area is not None:
        os.environ["ZOCR_EXPORT_BLANK_MIN_AREA"] = str(int(blank_min_area))
    if allow_pytesseract is True:
        os.environ["ZOCR_ALLOW_PYTESSERACT"] = "1"
    elif allow_pytesseract is False:
        os.environ["ZOCR_ALLOW_PYTESSERACT"] = "0"
    else:
        os.environ.setdefault("ZOCR_ALLOW_PYTESSERACT", "0")
    if tess_unicharset is not None:
        if tess_unicharset:
            os.environ["ZOCR_TESS_UNICHARSET"] = tess_unicharset
        else:
            os.environ.pop("ZOCR_TESS_UNICHARSET", None)
    if tess_wordlist is not None:
        if tess_wordlist:
            os.environ["ZOCR_TESS_WORDLIST"] = tess_wordlist
        else:
            os.environ.pop("ZOCR_TESS_WORDLIST", None)
    if tess_bigram_json is not None:
        if tess_bigram_json:
            os.environ["ZOCR_TESS_BIGRAM_JSON"] = tess_bigram_json
        else:
            os.environ.pop("ZOCR_TESS_BIGRAM_JSON", None)
    ensure_dir(outdir)
    stage_trace: List[Dict[str, Any]] = []
    _set_stage_trace_sink(stage_trace)
    stage_trace_console = _env_truthy("ZOCR_STAGE_TRACE_CONSOLE", False)
    if print_stage_trace is not None:
        stage_trace_console = bool(print_stage_trace)
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if snapshot:
        _write_pipeline_meta(outdir, seed)

    ok = _read_ok_steps(outdir) if resume else set()

    toy_memory_path = _resolve_toy_memory_path(outdir)
    toy_memory_info_load = zocr_onefile_consensus.load_toy_memory(toy_memory_path)
    toy_memory_after_load = toy_memory_info_load.get("snapshot_after") or toy_memory_info_load.get("snapshot_before")
    if hasattr(zocr_onefile_consensus, "reset_toy_recognition_stats"):
        zocr_onefile_consensus.reset_toy_recognition_stats()

    demo_requested = len(inputs) == 1 and inputs[0].lower() == "demo"

    if demo_requested:
        real_demo_targets = []
        if os.path.exists(inputs[0]):
            real_demo_targets = [inputs[0]]
        else:
            real_demo_targets = _discover_demo_input_targets()

        pages = _collect_pages(real_demo_targets, dpi=dpi) if real_demo_targets else []

        filtered_pages: List[str] = []
        seen_page_paths = set()
        for page in pages:
            norm = os.path.abspath(page)
            if norm in seen_page_paths:
                continue
            if not os.path.exists(page):
                continue
            seen_page_paths.add(norm)
            filtered_pages.append(page)
        pages = filtered_pages

        if pages:
            annos = [None] * len(pages)
        else:
            pages, annos = zocr_onefile_consensus.make_demo(outdir)
    else:
        pages = _collect_pages(inputs, dpi=dpi)
        annos = [None] * len(pages)
    if not pages:
        raise RuntimeError("No input pages provided")

    page_images = {idx: page for idx, page in enumerate(pages)}

    pipe = zocr_onefile_consensus.Pipeline({"table": {}, "bench_iterations": 1, "eval": False})

    doc_json_path = os.path.join(outdir, "doc.zocr.json")
    jsonl_path = os.path.join(outdir, "doc.contextual.jsonl")
    mm_jsonl = os.path.join(outdir, "doc.mm.jsonl")
    idx_path = os.path.join(outdir, "bm25.pkl")
    mon_csv = os.path.join(outdir, "monitor.csv")
    prof_path = os.path.join(outdir, "auto_profile.json")
    prof = _load_profile(outdir, domain_hint)
    profile_guard = _ProfileGuard(prof)

    auto_demo_lite = demo_requested
    effective_toy_lite = bool(toy_lite or auto_demo_lite)
    toy_sweep_limit: Optional[int] = None
    if toy_sweeps is not None and toy_sweeps > 0:
        toy_sweep_limit = int(toy_sweeps)
    elif effective_toy_lite:
        toy_sweep_limit = _default_toy_sweeps()
    force_numeric_flag = force_numeric_by_header
    if effective_toy_lite and force_numeric_flag is None:
        force_numeric_flag = True
    if toy_sweep_limit is not None and tune_budget is not None and tune_budget > 0:
        tune_budget = min(int(tune_budget), toy_sweep_limit)

    env_ocr_engine = os.environ.get("ZOCR_OCR_ENGINE")
    effective_ocr_engine = ocr_engine or env_ocr_engine or prof.get("ocr_engine") or "toy"
    export_ocr_override = os.environ.get("ZOCR_EXPORT_OCR")
    export_ocr_engine = export_ocr_override or effective_ocr_engine

    toy_runtime_overrides: Dict[str, Any] = {}
    toy_runtime_snapshot: Optional[Dict[str, Any]] = None
    configure_runtime = getattr(zocr_onefile_consensus, "configure_toy_runtime", None)
    if callable(configure_runtime) and (toy_sweep_limit is not None or force_numeric_flag is not None):
        try:
            toy_runtime_overrides = configure_runtime(
                sweeps=toy_sweep_limit, force_numeric=force_numeric_flag
            ) or {}
        except Exception as exc:
            print(f"[WARN] Toy runtime configure failed: {exc}")
            toy_runtime_overrides = {}
    runtime_config_fn = getattr(zocr_onefile_consensus, "toy_runtime_config", None)
    if callable(runtime_config_fn):
        try:
            toy_runtime_snapshot = runtime_config_fn()
        except Exception:
            toy_runtime_snapshot = None

    advisor_ingest = _ingest_advisor_response(advisor_response)
    advisor_actions: Set[str] = set()
    if advisor_ingest.get("actions"):
        advisor_actions = {str(a) for a in advisor_ingest.get("actions") if isinstance(a, str)}

    rag_feedback_path = rag_feedback
    if not rag_feedback_path:
        default_manifest = os.path.join(outdir, "rag", "manifest.json")
        if os.path.exists(default_manifest):
            rag_feedback_path = default_manifest
    rag_feedback_ingest: Optional[Dict[str, Any]] = None
    rag_feedback_actions: Set[str] = set()
    if rag_feedback_path:
        rag_feedback_ingest = _apply_rag_feedback(
            rag_feedback_path, prof, prof_path, guard=profile_guard
        )
        if rag_feedback_ingest.get("actions"):
            rag_feedback_actions = {
                str(a) for a in rag_feedback_ingest.get("actions", []) if isinstance(a, str)
            }

    auto_demo_lite = demo_requested
    effective_toy_lite = bool(toy_lite or auto_demo_lite)
    toy_sweep_limit: Optional[int] = None
    if toy_sweeps is not None and toy_sweeps > 0:
        toy_sweep_limit = int(toy_sweeps)
    elif effective_toy_lite:
        toy_sweep_limit = _default_toy_sweeps()
    force_numeric_flag = force_numeric_by_header
    if effective_toy_lite and force_numeric_flag is None:
        force_numeric_flag = True
    if toy_sweep_limit is not None and tune_budget is not None and tune_budget > 0:
        tune_budget = min(int(tune_budget), toy_sweep_limit)

    env_ocr_engine = os.environ.get("ZOCR_OCR_ENGINE")
    effective_ocr_engine = ocr_engine or env_ocr_engine or prof.get("ocr_engine") or "toy"
    export_ocr_override = os.environ.get("ZOCR_EXPORT_OCR")
    export_ocr_engine = export_ocr_override or effective_ocr_engine

    toy_runtime_overrides: Dict[str, Any] = {}
    toy_runtime_snapshot: Optional[Dict[str, Any]] = None
    configure_runtime = getattr(zocr_onefile_consensus, "configure_toy_runtime", None)
    if callable(configure_runtime) and (toy_sweep_limit is not None or force_numeric_flag is not None):
        try:
            toy_runtime_overrides = configure_runtime(
                sweeps=toy_sweep_limit, force_numeric=force_numeric_flag
            ) or {}
        except Exception as exc:
            print(f"[WARN] Toy runtime configure failed: {exc}")
            toy_runtime_overrides = {}
    runtime_config_fn = getattr(zocr_onefile_consensus, "toy_runtime_config", None)
    if callable(runtime_config_fn):
        try:
            toy_runtime_snapshot = runtime_config_fn()
        except Exception:
            toy_runtime_snapshot = None

    advisor_ingest = _ingest_advisor_response(advisor_response)
    advisor_actions: Set[str] = set()
    if advisor_ingest.get("actions"):
        advisor_actions = {str(a) for a in advisor_ingest.get("actions") if isinstance(a, str)}

    rag_feedback_path = rag_feedback
    if not rag_feedback_path:
        default_manifest = os.path.join(outdir, "rag", "manifest.json")
        if os.path.exists(default_manifest):
            rag_feedback_path = default_manifest
    rag_feedback_ingest: Optional[Dict[str, Any]] = None
    rag_feedback_actions: Set[str] = set()
    if rag_feedback_path:
        rag_feedback_ingest = _apply_rag_feedback(rag_feedback_path, prof, prof_path)
        if rag_feedback_ingest.get("actions"):
            rag_feedback_actions = {
                str(a) for a in rag_feedback_ingest.get("actions", []) if isinstance(a, str)
            }

    summary: Dict[str, Any] = {
        "contextual_jsonl": jsonl_path,
        "mm_jsonl": mm_jsonl,
        "index": idx_path,
        "monitor_csv": mon_csv,
        "profile_json": prof_path,
        "history": os.path.join(outdir, "pipeline_history.jsonl"),
        "inputs": inputs[:],
        "page_count": len(pages),
        "page_images": page_images,
        "domain": prof.get("domain"),
        "seed": seed,
        "resume_requested": bool(resume),
        "resume_applied": bool(ok),
        "resume_steps": sorted(str(s) for s in ok if s is not None),
        "snapshot": bool(snapshot),
        "tune_budget": int(tune_budget) if tune_budget is not None else None,
        "ocr_engine": effective_ocr_engine,
        "export_ocr_engine": export_ocr_engine,
        "toy_lite": bool(effective_toy_lite),
        "toy_lite_auto": bool(auto_demo_lite),
        "toy_memory": {
            "path": toy_memory_path,
            "load": _json_ready(toy_memory_info_load),
        },
        "ingest_signature": ingest_signature,
    }

    tesslite_cfg = {
        "unicharset": os.environ.get("ZOCR_TESS_UNICHARSET") or None,
        "wordlist": os.environ.get("ZOCR_TESS_WORDLIST") or None,
        "bigram_json": os.environ.get("ZOCR_TESS_BIGRAM_JSON") or None,
    }
    if any(tesslite_cfg.values()):
        sig_fn = getattr(zocr_onefile_consensus, "_tesslite_env_signature", None)
        signature = sig_fn() if callable(sig_fn) else None
        tesslite_summary = {k: v for k, v in tesslite_cfg.items() if v}
        tesslite_summary["signature"] = signature
        tesslite_summary["enabled"] = True
        summary["tesslite"] = tesslite_summary
    else:
        summary["tesslite"] = {"enabled": False}

    episode_info = _begin_episode(outdir)
    if episode_info:
        summary["episode"] = {"id": episode_info.get("id"), "parent": episode_info.get("parent")}

    def _record_rag_conversation(entry: Dict[str, Any]) -> None:
        info = _append_rag_conversation_entry(outdir, entry)
        if not info:
            return
        convo = summary.setdefault("rag_conversation", {"path": info["path"]})
        convo["last_entry"] = _json_ready(info["entry"])

    if rag_feedback_ingest:
        summary["rag_feedback"] = _json_ready(rag_feedback_ingest)
    if rag_feedback_path:
        summary["rag_feedback_source"] = rag_feedback_path
    if rag_feedback_actions:
        summary["rag_feedback_actions"] = sorted(rag_feedback_actions)
    if rag_feedback_ingest and rag_feedback_ingest.get("status") == "ok":
        _record_rag_conversation(
            {
                "role": "rag_agent",
                "kind": "feedback",
                "source": rag_feedback_ingest.get("manifest"),
                "actions": rag_feedback_ingest.get("actions"),
                "overrides": rag_feedback_ingest.get("overrides"),
                "note": rag_feedback_ingest.get("note"),
            }
        )

    if force_numeric_flag is not None:
        summary["force_numeric_by_header"] = bool(force_numeric_flag)
    if toy_runtime_overrides:
        summary["toy_runtime_overrides"] = _json_ready(toy_runtime_overrides)
    if toy_runtime_snapshot:
        summary["toy_runtime_config"] = _json_ready(toy_runtime_snapshot)
    if toy_sweep_limit is not None:
        summary["toy_sweeps"] = int(toy_sweep_limit)
    if toy_runtime_overrides:
        summary["toy_runtime_overrides"] = _json_ready(toy_runtime_overrides)

    prof.setdefault("ocr_engine", effective_ocr_engine)

    toy_memory_run_baseline = toy_memory_after_load or zocr_onefile_consensus.toy_memory_snapshot()
    toy_memory_after_run: Optional[Dict[str, Any]] = None
    toy_memory_delta_run: Optional[Dict[str, Any]] = None
    toy_recognition_stats: Optional[Dict[str, Any]] = None

    domain_hints = _prepare_domain_hints(inputs, list(page_images.values()))
    content_conf_threshold = float(os.environ.get("ZOCR_DOMAIN_CONF_THRESHOLD", "0.25"))
    domain_auto_summary: Dict[str, Any] = {
        "provided": domain_hint,
        "from_inputs": {
            "guess": domain_hints.get("guess"),
            "best_score": float(domain_hints.get("best_score") or 0.0) if domain_hints.get("best_score") else None,
            "tokens": domain_hints.get("tokens"),
            "per_input": domain_hints.get("per_input"),
            "extra_paths": domain_hints.get("extra_paths"),
            "token_trace": domain_hints.get("token_trace"),
            "scores": domain_hints.get("scores"),
        },
        "initial_profile": prof.get("domain"),
        "content_threshold": content_conf_threshold,
    }
    selected_source: Optional[str] = None
    selected_confidence: Optional[float] = None
    if prof.get("domain") and not _is_auto_domain(prof.get("domain")):
        selected_source = "profile"
    elif domain_hint and not _is_auto_domain(domain_hint):
        prof["domain"] = domain_hint
        selected_source = "cli"
    elif _is_auto_domain(prof.get("domain")) and domain_hints.get("guess"):
        prof["domain"] = domain_hints.get("guess")
        selected_source = "inputs"
        try:
            selected_confidence = float(domain_hints.get("best_score") or 0.0)
        except Exception:
            selected_confidence = None
    summary["domain_autodetect"] = domain_auto_summary
    
    if "OCR" in ok:
        print("[SKIP] OCR (resume)")
        try:
            with open(doc_json_path, "r", encoding="utf-8") as fr:
                doc_payload = json.load(fr)
                if isinstance(doc_payload, dict):
                    summary["consensus_metrics"] = _json_ready(doc_payload.get("metrics"))
        except Exception:
            pass
    else:
        r = _safe_step("OCR", pipe.run, "doc", pages, outdir, annos)
        _append_hist(outdir, r)
        if not r.get("ok"):
            raise RuntimeError("OCR failed")
        try:
            pipe_res, doc_json_path = r.get("out", (None, doc_json_path))
            if isinstance(pipe_res, dict):
                summary["consensus_metrics"] = _json_ready(pipe_res.get("metrics"))
        except Exception:
            pass

    if "Export" in ok:
        print("[SKIP] Export JSONL (resume)")
    else:
        os.environ.setdefault("ZOCR_EXPORT_EXT_VARIANTS", "0")
        os.environ.setdefault("ZOCR_EXPORT_PROGRESS", "1")
        os.environ.setdefault("ZOCR_EXPORT_LOG_EVERY", "100")
        os.environ.setdefault("ZOCR_EXPORT_SKIP_BLANK", "1")
        if not os.environ.get("ZOCR_TEMPLATE_CACHE"):
            template_cache_path = os.path.join(outdir, "toy_template_cache.json")
            os.environ["ZOCR_TEMPLATE_CACHE"] = template_cache_path
            summary.setdefault("toy_templates", {})["cache"] = template_cache_path
        ocr_min_conf = float(prof.get("ocr_min_conf", 0.58))
        r = _safe_step(
            f"Export (engine={export_ocr_engine})",
            zocr_onefile_consensus.export_jsonl_with_ocr,
            doc_json_path,
            page_images,
            jsonl_path,
            export_ocr_engine,
            True,
            ocr_min_conf,
        )
        _append_hist(outdir, r)
        if not r.get("ok"):
            raise RuntimeError("Export failed")
        export_stats_fn = getattr(zocr_onefile_consensus, "last_export_stats", None)
        if callable(export_stats_fn):
            try:
                export_stats = export_stats_fn()
            except Exception:
                export_stats = None
            if export_stats:
                summary["export_stats"] = _json_ready(export_stats)
    _call("post_export", jsonl=jsonl_path, outdir=outdir)
    export_signals = _load_export_signals(jsonl_path)
    if export_signals:
        summary["export_signals"] = export_signals
        if export_signals.get("learning_jsonl"):
            summary["learning_jsonl"] = export_signals.get("learning_jsonl")
    export_signals_before_learning = json.loads(json.dumps(export_signals)) if export_signals else None

    reanalysis_summary: Optional[Dict[str, Any]] = None
    reanalysis_reasons_done: Set[str] = set()
    reanalysis_last_execs: List[Dict[str, Any]] = []
    learning_jsonl_path = export_signals.get("learning_jsonl") if export_signals else None
    toy_snapshot: Optional[Dict[str, Any]] = None
    if isinstance(export_ocr_engine, str) and export_ocr_engine.lower().startswith("toy"):
        if hasattr(zocr_onefile_consensus, "toy_recognition_stats"):
            try:
                toy_snapshot = zocr_onefile_consensus.toy_recognition_stats(reset=False)
            except Exception:
                toy_snapshot = None

    learning_hotspots: Optional[Dict[str, Any]] = None
    selective_focus_plan: Optional[Dict[str, Any]] = None
    hotspot_gallery: Optional[Dict[str, Any]] = None
    if learning_jsonl_path and os.path.exists(learning_jsonl_path):
        hotspots_payload = _analyze_learning_hotspots(learning_jsonl_path)
        if hotspots_payload:
            learning_hotspots = hotspots_payload
            plan = hotspots_payload.get("focus_plan") if isinstance(hotspots_payload, dict) else None
            if plan:
                selective_focus_plan = plan
            summary_hotspots = dict(hotspots_payload)
            if "focus_plan" in summary_hotspots:
                summary_hotspots.pop("focus_plan")
            if summary_hotspots:
                summary["learning_hotspots"] = _json_ready(summary_hotspots)
            if selective_focus_plan:
                summary["selective_reanalysis_plan"] = _json_ready(selective_focus_plan)
        hotspot_gallery = _generate_hotspot_gallery(
            outdir,
            learning_jsonl_path,
            learning_hotspots,
            selective_focus_plan,
            page_images,
        )
        if hotspot_gallery:
            summary["hotspot_gallery"] = _json_ready(hotspot_gallery)

    def _run_learning_reanalysis(
        step_label: str,
        reason: str,
        resume_key: Optional[str] = None,
        toy_plan: Optional[Dict[str, Any]] = None,
        focus_plan: Optional[Dict[str, Any]] = None,
    ) -> bool:
        nonlocal reanalysis_summary, jsonl_path, export_signals, reanalysis_last_execs
        if not learning_jsonl_path:
            return False
        if reason in reanalysis_reasons_done:
            return False
        re_dir = os.path.join(outdir, "reanalyze")
        ensure_dir(re_dir)
        reanalysis_last_execs = []
        plan_levels: List[Optional[Dict[str, Any]]] = []
        stop_on_improvement = True
        require_improvement = False
        if isinstance(toy_plan, dict):
            raw_levels = toy_plan.get("levels")
            if isinstance(raw_levels, list):
                for entry in raw_levels:
                    plan_levels.append(entry if isinstance(entry, dict) else None)
            stop_on_improvement = bool(toy_plan.get("stop_on_improvement", True))
            require_improvement = bool(toy_plan.get("require_improvement", False))
        if not plan_levels:
            plan_levels = [None]
        executed_runs: List[Dict[str, Any]] = []
        selected_summary: Optional[Dict[str, Any]] = None
        best_metric: Tuple[int, float] = (-1, -1.0)
        for idx, level_cfg in enumerate(plan_levels):
            pass_label = step_label if idx == 0 else f"{step_label}.pass{idx+1}"
            hist_base = resume_key or step_label
            hist_key = f"{hist_base}#{idx}" if hist_base else pass_label
            cache_summary: Optional[Dict[str, Any]] = None
            if hist_key in ok:
                print(f"[SKIP] {pass_label} (resume)")
                _, summary_path = _reanalyze_output_paths(learning_jsonl_path, re_dir)
                try:
                    with open(summary_path, "r", encoding="utf-8") as fr:
                        loaded = json.load(fr)
                        if isinstance(loaded, dict):
                            cache_summary = loaded
                except Exception:
                    cache_summary = None
            if cache_summary is None:
                try:
                    re_limit = int(prof.get("reanalyze_limit") or 64)
                except Exception:
                    re_limit = 64
                runner = zocr_onefile_consensus.reanalyze_learning_jsonl
                context_manager = getattr(zocr_onefile_consensus, "toy_self_correction_scope", None)
                if callable(context_manager) and level_cfg:
                    with context_manager(level_cfg):
                        result = _safe_step(
                            pass_label,
                            runner,
                            learning_jsonl_path,
                            re_dir,
                            re_limit,
                            ocr_engine=export_ocr_engine,
                            focus=focus_plan,
                        )
                else:
                    result = _safe_step(
                        pass_label,
                        runner,
                        learning_jsonl_path,
                        re_dir,
                        re_limit,
                        ocr_engine=export_ocr_engine,
                        focus=focus_plan,
                    )
                _append_hist(outdir, result)
                if not result.get("ok"):
                    executed_runs.append({"label": pass_label, "ok": False, "config": level_cfg})
                    continue
                out = result.get("out")
                cache_summary = out if isinstance(out, dict) else None
            if not isinstance(cache_summary, dict):
                executed_runs.append({"label": pass_label, "ok": False, "config": level_cfg})
                continue
            run_record: Dict[str, Any] = {
                "label": pass_label,
                "ok": True,
                "config": level_cfg,
                "summary": cache_summary,
            }
            executed_runs.append(run_record)
            improved = int(cache_summary.get("improved") or 0)
            avg_delta = float(cache_summary.get("avg_confidence_delta") or 0.0)
            metric = (improved, avg_delta)
            if cache_summary.get("toy_self_correction") and level_cfg:
                run_record["effective_config"] = cache_summary.get("toy_self_correction")
            if improved > 0 and stop_on_improvement:
                selected_summary = cache_summary
                break
            if metric > best_metric or selected_summary is None:
                best_metric = metric
                selected_summary = cache_summary
        reanalysis_last_execs = executed_runs
        if not isinstance(selected_summary, dict):
            return False
        reanalysis_summary = selected_summary
        reanalysis_reasons_done.add(reason)
        summary.setdefault("reanalysis_runs", []).append(
            _json_ready({"step": step_label, "reason": reason, "passes": len(executed_runs)})
        )
        summary["reanalyze_learning"] = _json_ready(selected_summary)
        output_jsonl = selected_summary.get("output_jsonl")
        if output_jsonl:
            summary["learning_reanalyzed_jsonl"] = output_jsonl
            jsonl_path = _apply_reanalysis_to_contextual_jsonl(
                jsonl_path,
                output_jsonl,
                outdir,
                summary,
                prof.get("ocr_min_conf", 0.58),
                export_signals.get("surprisal_threshold") if export_signals else None,
            )
            export_signals = summary.get("export_signals", export_signals)
        improved_total = int(selected_summary.get("improved") or 0)
        if require_improvement and improved_total <= 0:
            return False
        return True

    re_targets = {str(t) for t in (prof.get("reanalyze_target") or []) if t}
    if learning_jsonl_path and "learning_cells" in re_targets:
        _run_learning_reanalysis(
            "ReanalyzeLearning",
            "profile_reanalyze_target",
            "ReanalyzeLearning",
            focus_plan=selective_focus_plan,
        )
    toy_self_correction_details: Optional[Dict[str, Any]] = None
    toy_triggered = False
    toy_executed = False
    if learning_jsonl_path and toy_snapshot is not None:
        toy_triggered, toy_self_correction_details = _should_toy_self_correct(export_signals, toy_snapshot)
        if toy_triggered:
            plan = toy_self_correction_details.get("plan") if isinstance(toy_self_correction_details, dict) else None
            toy_executed = _run_learning_reanalysis(
                "ReanalyzeLearningAuto",
                "toy_self_correction",
                toy_plan=plan if isinstance(plan, dict) else None,
                focus_plan=selective_focus_plan,
            )
            if reanalysis_last_execs and isinstance(toy_self_correction_details, dict):
                toy_executed = True
                exec_payload: List[Dict[str, Any]] = []
                for rec in reanalysis_last_execs:
                    entry: Dict[str, Any] = {
                        "label": rec.get("label"),
                        "ok": bool(rec.get("ok")),
                    }
                    if rec.get("config") is not None:
                        entry["config"] = _json_ready(rec.get("config"))
                    if rec.get("effective_config") is not None:
                        entry["effective_config"] = _json_ready(rec.get("effective_config"))
                    summary_obj = rec.get("summary")
                    if isinstance(summary_obj, dict):
                        entry["improved"] = int(summary_obj.get("improved") or 0)
                        entry["avg_confidence_delta"] = float(summary_obj.get("avg_confidence_delta") or 0.0)
                        entry["output_jsonl"] = summary_obj.get("output_jsonl")
                    exec_payload.append(entry)
                if exec_payload:
                    toy_self_correction_details["executions"] = exec_payload
                result_info = {
                    "improved_total": int((reanalysis_summary or {}).get("improved") or 0),
                    "avg_confidence_delta": float((reanalysis_summary or {}).get("avg_confidence_delta") or 0.0),
                }
                result_info["success"] = bool(result_info["improved_total"] > 0)
                toy_self_correction_details["result"] = result_info
    if toy_self_correction_details is None and toy_snapshot is not None:
        _, toy_self_correction_details = _should_toy_self_correct(export_signals, toy_snapshot)
    if toy_self_correction_details is not None:
        summary["toy_self_correction"] = {
            "triggered": bool(toy_triggered),
            "executed": bool(toy_executed),
            "details": _json_ready(toy_self_correction_details),
        }
    learning_outcome = _evaluate_learning_outcome(
        export_signals_before_learning,
        export_signals,
        reanalysis_summary,
    )
    if learning_outcome:
        summary["learning_outcome"] = _json_ready(learning_outcome)
        if summary.get("toy_self_correction"):
            summary["toy_self_correction"].setdefault("details", {})
            details_obj = summary["toy_self_correction"].get("details")
            if isinstance(details_obj, dict):
                details_obj["learning_outcome"] = learning_outcome

    pre_augment_reanalysis_reasons = set(reanalysis_reasons_done)

    autodetect_detail: Optional[Dict[str, Any]] = None
    autodetect_error: Optional[str] = None
    if os.path.exists(jsonl_path):
        try:
            detected_domain, autodetect_detail = zocr_multidomain_core.detect_domain_on_jsonl(
                jsonl_path,
                domain_hints.get("token_trace") or domain_hints.get("tokens_raw"),
            )
        except Exception as e:
            autodetect_error = str(e)
            detected_domain = None  # type: ignore
            autodetect_detail = None
        if autodetect_detail:
            domain_auto_summary["from_content"] = autodetect_detail
            resolved = autodetect_detail.get("resolved") or detected_domain
            if resolved:
                decision: Dict[str, Any] = {"candidate": resolved}
                take = False
                conf_val = autodetect_detail.get("confidence")
                try:
                    conf_float = float(conf_val) if conf_val is not None else None
                except Exception:
                    conf_float = None
                decision["confidence"] = conf_float
                if conf_float is not None and conf_float >= content_conf_threshold:
                    take = True
                    decision["reason"] = "confidence>=threshold"
                elif conf_float is None:
                    decision["reason"] = "confidence-missing"
                else:
                    decision["reason"] = "below-threshold"
                if take:
                    prof["domain"] = resolved
                    selected_source = "content"
                    selected_confidence = conf_float
                    decision["applied"] = resolved
                else:
                    decision["kept"] = prof.get("domain")
                domain_auto_summary["content_decision"] = decision
        if autodetect_error:
            domain_auto_summary["content_error"] = autodetect_error

    if not prof.get("domain"):
        prof["domain"] = "invoice_jp_v2"
        if selected_source is None:
            selected_source = "default"
    _apply_domain_defaults(prof, prof.get("domain"))
    try:
        with open(prof_path, "w", encoding="utf-8") as pf:
            json.dump(_json_ready(prof), pf, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Profile save skipped:", e)

    domain_auto_summary["selected"] = {
        "source": selected_source,
        "domain": prof.get("domain"),
        "confidence": selected_confidence,
    }
    summary["domain_autodetect"] = domain_auto_summary
    summary["domain"] = prof.get("domain")

    if "Augment" in ok:
        print("[SKIP] Augment (resume)")
    else:
        r = _safe_step("Augment", zocr_multidomain_core.augment, jsonl_path, mm_jsonl,
                       prof.get("lambda_shape", 4.5), org_dict_path=org_dict)
        _append_hist(outdir, r)
        if not r.get("ok"):
            raise RuntimeError("Augment failed")

    if "Index" in ok:
        print("[SKIP] Index (resume)")
    else:
        r = _safe_step("Index", zocr_multidomain_core.build_index, mm_jsonl, idx_path)
        _append_hist(outdir, r)
        if not r.get("ok"):
            raise RuntimeError("Index failed")
    _call("post_index", index=idx_path, jsonl=mm_jsonl)

    profile_before_feedback = _profile_snapshot(prof)
    monitor_row = None
    if "Monitor" in ok:
        print("[SKIP] Monitor (resume)")
    else:
        r = _safe_step("Monitor", zocr_multidomain_core.monitor, mm_jsonl, idx_path, k, mon_csv,
                       views_log=views_log, gt_jsonl=gt_jsonl, domain=prof.get("domain"))
        _append_hist(outdir, r)
        if r.get("ok"):
            monitor_row = r.get("out")
    if monitor_row is None and os.path.exists(mon_csv):
        try:
            import csv
            with open(mon_csv, "r", encoding="utf-8-sig", newline="") as fr:
                rows = list(csv.DictReader(fr))
                if rows:
                    monitor_row = rows[-1]
        except Exception:
            monitor_row = None
    summary["monitor_row"] = monitor_row
    _call("post_monitor", csv=mon_csv, profile=prof)

    tune_row = None
    learn_row = None
    if do_tune:
        if "Tune" not in ok:
            r = _safe_step("Tune", zocr_multidomain_core.autotune_unlabeled, mm_jsonl, idx_path, outdir,
                           method="grid", budget=int(tune_budget), domain_hint=prof.get("domain"),
                           seed=0, p95_target_ms=300.0, use_smoothing_metric=True)
            _append_hist(outdir, r)
            if r.get("ok"):
                tune_row = r.get("out")
        if "MonitorPostTune" not in ok:
            r = _safe_step("MonitorPostTune", zocr_multidomain_core.monitor, mm_jsonl, idx_path, k, mon_csv,
                           views_log=views_log, gt_jsonl=gt_jsonl, domain=prof.get("domain"))
            _append_hist(outdir, r)
            if r.get("ok"):
                monitor_row = r.get("out") or monitor_row
        if monitor_row is None and os.path.exists(mon_csv):
            try:
                import csv
                with open(mon_csv, "r", encoding="utf-8-sig", newline="") as fr:
                    rows = list(csv.DictReader(fr))
                    if rows:
                        monitor_row = rows[-1]
            except Exception:
                monitor_row = None
        try:
            learn_row = zocr_multidomain_core.learn_from_monitor(mon_csv, prof_path, prof_path,
                                                                  prof.get("domain"), ema=0.5)
        except Exception as e:
            print("Learn-from-monitor skipped:", e)
    summary["tune"] = tune_row
    summary["learn"] = learn_row
    summary["insights"] = _derive_insights(summary)

    toy_memory_after_run = zocr_onefile_consensus.toy_memory_snapshot()
    toy_memory_delta_run = zocr_onefile_consensus.toy_memory_delta(
        toy_memory_run_baseline, toy_memory_after_run
    )
    toy_recognition_stats = zocr_onefile_consensus.toy_recognition_stats(reset=False)

    prof_after = _load_profile(outdir, prof.get("domain"))
    profile_diff = _profile_diff(profile_before_feedback, prof_after)
    intent = _derive_intent(
        summary.get("monitor_row"),
        export_signals,
        prof_after,
        toy_memory_delta_run,
        toy_recognition_stats,
    )
    summary["intent"] = intent
    meta_intent = _derive_meta_intent(
        intent,
        learning_hotspots,
        selective_focus_plan,
        rag_feedback_ingest,
        advisor_ingest,
        learning_outcome,
    )
    if meta_intent:
        summary["meta_intent"] = _json_ready(meta_intent)
    simulations = _simulate_param_shift(
        summary.get("monitor_row"),
        export_signals,
        toy_recognition_stats,
        toy_memory_delta_run,
        prof_after,
    )
    if simulations:
        summary["intent_simulations"] = _json_ready(simulations)
    intent_updates = _apply_intent_to_profile(intent, prof_after, guard=profile_guard)
    combined_updates: Dict[str, Tuple[Any, Any]] = {}
    if profile_diff:
        summary["profile_diff"] = {k: _json_ready(v) for k, v in profile_diff.items()}
        combined_updates.update(profile_diff)
    if intent_updates:
        combined_updates.update(intent_updates)
        summary["intent_applied"] = True
    if combined_updates:
        try:
            with open(prof_path, "w", encoding="utf-8") as pf:
                json.dump(_json_ready(prof_after), pf, ensure_ascii=False, indent=2)
        except Exception as e:
            print("Profile save skipped:", e)
        summary["profile_updates"] = {k: _json_ready(v) for k, v in combined_updates.items()}
    intent_runs: List[str] = []
    if intent.get("action") == "reanalyze_cells" and learning_jsonl_path:
        if _run_learning_reanalysis(
            "ReanalyzeLearningIntent",
            "intent_reanalyze",
            focus_plan=selective_focus_plan,
        ):
            intent_runs.append("reanalyze_learning")
    if intent_runs:
        summary["intent_runs"] = intent_runs
    advisor_actions_applied: List[str] = []
    advisor_runs: List[str] = []
    rag_feedback_actions_applied: List[str] = []
    rag_feedback_runs: List[str] = []
    combined_actions = set(advisor_actions)
    combined_actions.update(rag_feedback_actions)
    if learning_jsonl_path and "reanalyze_cells" in combined_actions:
        if _run_learning_reanalysis(
            "ReanalyzeLearningAdvisor",
            "advisor_reanalyze",
            focus_plan=selective_focus_plan,
        ):
            if "reanalyze_cells" in advisor_actions:
                advisor_runs.append("reanalyze_learning")
                advisor_actions_applied.append("reanalyze_cells")
            if "reanalyze_cells" in rag_feedback_actions:
                rag_feedback_runs.append("reanalyze_learning")
                rag_feedback_actions_applied.append("reanalyze_cells")
    if advisor_runs:
        summary["advisor_runs"] = advisor_runs
    if rag_feedback_runs:
        summary["rag_feedback_runs"] = rag_feedback_runs
    rerun_flags = _needs_rerun_for_keys(list(combined_updates.keys())) if combined_updates else {"augment": False, "monitor": False}
    new_reanalysis_reasons = reanalysis_reasons_done - pre_augment_reanalysis_reasons
    if new_reanalysis_reasons:
        rerun_flags["augment"] = True
        rerun_flags["monitor"] = True
        summary["reanalysis_post_augment"] = sorted(new_reanalysis_reasons)
    if "rerun_augment" in combined_actions:
        rerun_flags["augment"] = True
        rerun_flags["monitor"] = True
        if "rerun_augment" in advisor_actions:
            advisor_actions_applied.append("rerun_augment")
        if "rerun_augment" in rag_feedback_actions:
            rag_feedback_actions_applied.append("rerun_augment")
    elif "rerun_monitor" in combined_actions:
        rerun_flags["monitor"] = True
        if "rerun_monitor" in advisor_actions:
            advisor_actions_applied.append("rerun_monitor")
        if "rerun_monitor" in rag_feedback_actions:
            rag_feedback_actions_applied.append("rerun_monitor")
    summary["feedback_rerun_flags"] = rerun_flags
    feedback_passes: List[str] = []
    prof = prof_after
    if rerun_flags.get("augment"):
        r = _safe_step("AugmentIntent", zocr_multidomain_core.augment, jsonl_path, mm_jsonl,
                       prof.get("lambda_shape", 4.5), org_dict_path=org_dict)
        _append_hist(outdir, r)
        if r.get("ok"):
            feedback_passes.append("augment")
        r = _safe_step("IndexIntent", zocr_multidomain_core.build_index, mm_jsonl, idx_path)
        _append_hist(outdir, r)
        if r.get("ok"):
            feedback_passes.append("index")
    if rerun_flags.get("monitor"):
        r = _safe_step("MonitorIntent", zocr_multidomain_core.monitor, mm_jsonl, idx_path, k, mon_csv,
                       views_log=views_log, gt_jsonl=gt_jsonl, domain=prof.get("domain"))
        _append_hist(outdir, r)
        if r.get("ok"):
            monitor_row = r.get("out") or monitor_row
            feedback_passes.append("monitor")
            summary["monitor_row"] = monitor_row
    if feedback_passes:
        summary["feedback_passes"] = feedback_passes
    if advisor_actions_applied:
        summary["advisor_actions_applied"] = sorted(set(advisor_actions_applied))
    if rag_feedback_actions_applied:
        summary["rag_feedback_actions_applied"] = sorted(set(rag_feedback_actions_applied))

    safety_flags: Dict[str, Any] = {}
    gate_safety = _gate_fail_safety(prof, summary.get("monitor_row"))
    if gate_safety:
        safety_flags["gate_fail_streak"] = _json_ready(gate_safety)
        summary["gate_fail_streak"] = gate_safety.get("value")
        if gate_safety.get("updated"):
            combined_updates["gate_fail_streak"] = (
                gate_safety.get("previous"),
                gate_safety.get("value"),
            )
            try:
                with open(prof_path, "w", encoding="utf-8") as pf:
                    json.dump(_json_ready(prof), pf, ensure_ascii=False, indent=2)
            except Exception as exc:
                print("Profile save skipped (gate streak):", exc)
    try:
        sql_paths = zocr_multidomain_core.sql_export(mm_jsonl, os.path.join(outdir, "sql"),
                                                     prefix=(prof.get("domain") or "invoice"))
        summary["sql_csv"] = sql_paths.get("csv")
        summary["sql_schema"] = sql_paths.get("schema")
    except Exception as e:
        print("SQL export skipped:", e)
    _call("post_sql", sql_csv=summary.get("sql_csv"), sql_schema=summary.get("sql_schema"))

    try:
        rag_dir = os.path.join(outdir, "rag")
        rag_manifest = zocr_multidomain_core.export_rag_bundle(
            mm_jsonl,
            rag_dir,
            domain=prof.get("domain"),
            summary=summary,
        )
        summary["rag_manifest"] = rag_manifest.get("manifest")
        summary["rag_bundle"] = rag_manifest.get("bundle_dir")
        summary["rag_cells"] = rag_manifest.get("cells")
        summary["rag_sections"] = rag_manifest.get("sections")
        summary["rag_tables_json"] = rag_manifest.get("tables_json")
        summary["rag_markdown"] = rag_manifest.get("markdown")
        summary["rag_cell_count"] = rag_manifest.get("cell_count")
        summary["rag_table_count"] = rag_manifest.get("table_sections")
        summary["rag_page_count"] = rag_manifest.get("page_sections")
        summary["rag_languages"] = rag_manifest.get("languages")
        summary["rag_doc_ids"] = rag_manifest.get("doc_ids")
        summary["rag_bundle_metrics"] = {
            "cells": rag_manifest.get("cell_count"),
            "tables": rag_manifest.get("table_sections"),
            "pages": rag_manifest.get("page_sections"),
        }
        summary["rag_bundle_status"] = _derive_rag_bundle_status(
            rag_manifest.get("cell_count"),
            rag_manifest.get("table_sections"),
            rag_manifest.get("page_sections"),
            doc_ids=rag_manifest.get("doc_ids"),
            languages=rag_manifest.get("languages"),
        )
        summary["rag_suggested_queries"] = rag_manifest.get("suggested_queries")
        summary["rag_trace_schema"] = rag_manifest.get("trace_schema")
        summary["rag_fact_tag_example"] = rag_manifest.get("fact_tag_example")
        _dedupe_insights_and_queries(summary)
    except Exception as e:
        print("RAG bundle export skipped:", e)
        summary["rag_trace_schema"] = summary.get("rag_trace_schema") or None
        summary["rag_fact_tag_example"] = summary.get("rag_fact_tag_example") or None
        summary["rag_bundle_status"] = {"issues": ["export_skipped"]}
    rag_status = summary.get("rag_bundle_status")
    if isinstance(rag_status, dict) and rag_status.get("issues"):
        safety_flags.setdefault("rag_bundle", rag_status)
    if safety_flags:
        summary["safety_flags"] = safety_flags
    _call(
        "post_rag",
        manifest=summary.get("rag_manifest"),
        bundle=summary.get("rag_bundle"),
        trace_schema=summary.get("rag_trace_schema"),
        fact_tag_example=summary.get("rag_fact_tag_example"),
    )
    if summary.get("rag_manifest"):
        summary["rag_feedback_scan"] = _json_ready(
            _apply_rag_feedback(
                summary.get("rag_manifest"),
                prof,
                prof_path,
                persist_profile=False,
            )
        )

    rag_request_info = _emit_rag_feedback_request(
        outdir,
        summary,
        manifest_path=summary.get("rag_manifest") or rag_feedback_path,
        rag_feedback_ingest=rag_feedback_ingest,
        advisor_ingest=advisor_ingest,
        rag_feedback_actions=sorted(rag_feedback_actions) if rag_feedback_actions else None,
    )
    if rag_request_info:
        summary["rag_feedback_request"] = _json_ready(rag_request_info)
        _record_rag_conversation(
            {
                "role": "pipeline",
                "kind": "feedback_request",
                "path": rag_request_info.get("request_markdown") or rag_request_info.get("request_json"),
                "pending_actions": sorted(rag_feedback_actions) if rag_feedback_actions else None,
                "meta_intent": summary.get("meta_intent", {}).get("story"),
            }
        )

    if PLUGINS:
        summary["plugins"] = {stage: [getattr(fn, "__name__", str(fn)) for fn in fns]
                               for stage, fns in PLUGINS.items()}

    history_records = _load_history(outdir)
    if history_records:
        ok_count = sum(1 for r in history_records if r.get("ok"))
        fail_count = sum(1 for r in history_records if r.get("ok") is False)
        total_elapsed = sum(float(r.get("elapsed_ms") or 0.0) for r in history_records if isinstance(r.get("elapsed_ms"), (int, float)))
        summary["history_stats"] = {
            "ok": ok_count,
            "fail": fail_count,
            "total_elapsed_ms": total_elapsed,
        }
    summary["dependencies"] = _collect_dependency_diagnostics()
    summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report_path = os.path.join(outdir, "pipeline_report.html")
    summary["report_html"] = report_path

    summary.setdefault("toy_memory", {})
    summary["toy_memory"]["before_run"] = _json_ready(toy_memory_run_baseline)
    summary["toy_memory"]["after_run"] = _json_ready(toy_memory_after_run)
    summary["toy_memory"]["delta_run"] = _json_ready(toy_memory_delta_run)
    learning_story = _summarize_toy_learning(toy_memory_delta_run, toy_recognition_stats)
    if learning_story:
        summary["toy_memory"]["learning_reason"] = _json_ready(learning_story)
    if toy_recognition_stats is not None:
        summary["toy_memory"]["recognition"] = _json_ready(toy_recognition_stats)
        if hasattr(zocr_onefile_consensus, "reset_toy_recognition_stats"):
            try:
                zocr_onefile_consensus.reset_toy_recognition_stats()
            except Exception:
                pass
    elif hasattr(zocr_onefile_consensus, "toy_recognition_stats"):
        summary["toy_memory"]["recognition"] = _json_ready(
            zocr_onefile_consensus.toy_recognition_stats(reset=True)
        )

    toy_memory_saved = zocr_onefile_consensus.save_toy_memory(toy_memory_path)
    summary["toy_memory"]["save"] = _json_ready(toy_memory_saved)
    if profile_guard:
        summary["profile_guard"] = profile_guard.report()

    repro_signature = _build_repro_signature(
        inputs,
        page_images,
        prof,
        toy_runtime_snapshot,
        export_ocr_engine,
        toy_runtime_overrides,
    )
    summary["repro_signature"] = _json_ready(repro_signature)
    sig_path, ingest_info = _write_repro_signature(outdir, repro_signature, ingest_signature)
    if sig_path:
        summary["repro_signature_path"] = sig_path
    if ingest_info:
        summary["repro_ingest"] = _json_ready(ingest_info)

    advisor_path = _write_advice_packet(outdir, summary)
    if advisor_path:
        summary["advisor_prompt"] = advisor_path
    if advisor_ingest:
        summary["advisor_ingest"] = _json_ready(advisor_ingest)
        if advisor_ingest.get("status") == "ok":
            preview = advisor_ingest.get("preview") or ""
            _record_rag_conversation(
                {
                    "role": "advisor",
                    "kind": "response",
                    "source": advisor_ingest.get("path"),
                    "actions": advisor_ingest.get("actions"),
                    "note": preview[:400],
                }
            )

    if stage_trace:
        total_ms = sum(float(entry.get("elapsed_ms") or 0.0) for entry in stage_trace)
        failures = sum(1 for entry in stage_trace if entry.get("ok") is False)
        slowest = max(stage_trace, key=lambda e: float(e.get("elapsed_ms") or 0.0)) if stage_trace else None
        summary["stage_trace"] = _json_ready(stage_trace)
        summary["stage_stats"] = {
            "count": len(stage_trace),
            "failures": failures,
            "total_elapsed_ms": total_ms,
            "slowest": {"name": slowest.get("name"), "elapsed_ms": slowest.get("elapsed_ms")} if slowest else None,
        }
        if stage_trace_console:
            _print_stage_trace_console(stage_trace, summary.get("stage_stats"))

    _finalize_episode(outdir, summary)

    repro_signature = _build_repro_signature(
        inputs,
        page_images,
        prof,
        toy_runtime_snapshot,
        export_ocr_engine,
        toy_runtime_overrides,
    )
    summary["repro_signature"] = _json_ready(repro_signature)
    sig_path, ingest_info = _write_repro_signature(outdir, repro_signature, ingest_signature)
    if sig_path:
        summary["repro_signature_path"] = sig_path
    if ingest_info:
        summary["repro_ingest"] = _json_ready(ingest_info)

    advisor_path = _write_advice_packet(outdir, summary)
    if advisor_path:
        summary["advisor_prompt"] = advisor_path
    if advisor_ingest:
        summary["advisor_ingest"] = _json_ready(advisor_ingest)
        if advisor_ingest.get("status") == "ok":
            preview = advisor_ingest.get("preview") or ""
            _record_rag_conversation(
                {
                    "role": "advisor",
                    "kind": "response",
                    "source": advisor_ingest.get("path"),
                    "actions": advisor_ingest.get("actions"),
                    "note": preview[:400],
                }
            )

    if stage_trace:
        total_ms = sum(float(entry.get("elapsed_ms") or 0.0) for entry in stage_trace)
        failures = sum(1 for entry in stage_trace if entry.get("ok") is False)
        slowest = max(stage_trace, key=lambda e: float(e.get("elapsed_ms") or 0.0)) if stage_trace else None
        summary["stage_trace"] = _json_ready(stage_trace)
        summary["stage_stats"] = {
            "count": len(stage_trace),
            "failures": failures,
            "total_elapsed_ms": total_ms,
            "slowest": {"name": slowest.get("name"), "elapsed_ms": slowest.get("elapsed_ms")} if slowest else None,
        }
        if stage_trace_console:
            _print_stage_trace_console(stage_trace, summary.get("stage_stats"))

    with open(os.path.join(outdir, "pipeline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(_json_ready(summary), f, ensure_ascii=False, indent=2)
    try:
        _generate_report(outdir, dest=report_path, summary=summary, history=history_records, meta=_read_meta(outdir))
    except Exception as e:
        print("Report generation skipped:", e)
    _set_stage_trace_sink(None)
    return summary

run_full_pipeline = _patched_run_full_pipeline

# ---------------- CLI ----------------
def main():
    argv = sys.argv[1:]
    if argv and argv[0] in {"history", "summary", "plugins", "report", "diagnose"}:
        cmd = argv[0]
        rest = argv[1:]
        if cmd == "history":
            hp = argparse.ArgumentParser("ZOCR pipeline history")
            hp.add_argument("--outdir", default="out_allinone")
            hp.add_argument("--limit", type=int, default=20, help="show only the latest N records; 0 for all")
            hp.add_argument("--full", action="store_true", help="ignore --limit and show all records")
            hargs = hp.parse_args(rest)
            recs = _load_history(hargs.outdir)
            _print_history(recs, None if hargs.full or hargs.limit <= 0 else hargs.limit)
            return
        if cmd == "summary":
            sp = argparse.ArgumentParser("ZOCR pipeline summary")
            sp.add_argument("--outdir", default="out_allinone")
            sp.add_argument("--keys", nargs="*", default=[], help="optional keys to filter the summary output")
            sargs = sp.parse_args(rest)
            try:
                data = _read_summary(sargs.outdir)
            except FileNotFoundError:
                print("No summary found at", os.path.join(sargs.outdir, "pipeline_summary.json"))
                sys.exit(1)
            if sargs.keys:
                data = {k: data.get(k) for k in sargs.keys}
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return
        if cmd == "plugins":
            pp = argparse.ArgumentParser("ZOCR plugin registry")
            pp.add_argument("--stage", default=None, help="filter by stage name")
            pargs = pp.parse_args(rest)
            if not PLUGINS:
                print("(no plugins registered)")
                return
            stages = [pargs.stage] if pargs.stage else sorted(PLUGINS.keys())
            for stage in stages:
                fns = PLUGINS.get(stage, [])
                print(f"[{stage}] {len(fns)} plugin(s)")
                for fn in fns:
                    print(" -", getattr(fn, "__name__", repr(fn)))
            return
        if cmd == "report":
            rp = argparse.ArgumentParser("ZOCR pipeline report")
            rp.add_argument("--outdir", default="out_allinone")
            rp.add_argument("--dest", default=None, help="optional destination HTML path")
            rp.add_argument("--limit", type=int, default=50, help="history rows to include (0 = all)")
            rp.add_argument("--open", action="store_true", help="open the generated report in a browser")
            rargs = rp.parse_args(rest)
            limit = None if rargs.limit <= 0 else rargs.limit
            path = _generate_report(rargs.outdir, dest=rargs.dest, limit=limit)
            print("Report written to", path)
            if rargs.open:
                try:
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(path)}")
                except Exception as e:
                    print("Browser open failed:", e)
            return
        if cmd == "diagnose":
            dp = argparse.ArgumentParser("ZOCR dependency diagnostics")
            dp.add_argument("--json", action="store_true", help="emit structured JSON instead of a table")
            dargs = dp.parse_args(rest)
            diag = _collect_dependency_diagnostics()
            if dargs.json:
                print(json.dumps(_json_ready(diag), ensure_ascii=False, indent=2))
            else:
                print("Dependency check:")
                for key in sorted(diag.keys()):
                    info = diag[key]
                    status = info.get("status") if isinstance(info, dict) else None
                    print(f" - {key}: {status or info}")
                    if isinstance(info, dict):
                        for sub_key in ("path", "version", "detail", "hint"):
                            if info.get(sub_key):
                                print(f"     {sub_key}: {info[sub_key]}")
            return

    if argv and argv[0] == "run":
        argv = argv[1:]

    ap = argparse.ArgumentParser("ZOCR All-in-one Orchestrator")
    ap.add_argument("-i","--input", nargs="+", default=["demo"], help="images or PDFs; use 'demo' for synthetic invoice")
    ap.add_argument("--outdir", default="out_allinone")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--tune-budget", type=int, default=20)
    ap.add_argument("--views-log", default=None)
    ap.add_argument("--gt-jsonl", default=None)
    ap.add_argument("--org-dict", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=24601)
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument(
        "--toy-lite",
        action="store_true",
        help="Clamp toy OCR sweeps and force numeric columns for faster demo-style runs",
    )
    ap.add_argument(
        "--toy-sweeps",
        type=int,
        default=None,
        help="Upper bound for toy OCR threshold sweeps (defaults to env/auto)",
    )
    ap.add_argument(
        "--force-numeric-by-header",
        action="store_true",
        help="Normalize numeric columns according to header heuristics",
    )
    ap.add_argument(
        "--motion-prior",
        action="store_true",
        help="Enable motion prior seeding between export sweeps",
    )
    ap.add_argument(
        "--motion-sigma-px",
        type=float,
        default=None,
        help="Motion prior std-dev in pixels (default: 10 when enabled)",
    )
    ap.add_argument(
        "--motion-cutoff-sigma",
        type=float,
        default=None,
        help="Reject motion priors when deviation exceeds this multiple of sigma",
    )
    ap.add_argument(
        "--motion-accept-ratio",
        type=float,
        default=None,
        help="Minimum inlier ratio required to accept motion prior reseeding",
    )
    ap.add_argument(
        "--export-guard-ms",
        type=int,
        default=15000,
        help="Abort per-table export loops after this many milliseconds",
    )
    ap.add_argument(
        "--sweeps-fixed",
        type=int,
        default=None,
        help="Force toy OCR threshold sweeps to a fixed count",
    )
    ap.add_argument(
        "--blank-skip",
        dest="blank_skip",
        action="store_true",
        default=None,
        help="Enable blank-cell skip heuristic during export",
    )
    ap.add_argument(
        "--no-blank-skip",
        dest="blank_skip",
        action="store_false",
        help="Disable blank-cell skipping",
    )
    ap.add_argument(
        "--blank-threshold",
        type=int,
        default=None,
        help="Grayscale threshold (0-255) for blank detection",
    )
    ap.add_argument(
        "--blank-min-pixels",
        type=int,
        default=None,
        help="Minimum dark pixel count required to avoid blank skip",
    )
    ap.add_argument(
        "--blank-min-ratio",
        type=float,
        default=None,
        help="Minimum dark pixel ratio required to avoid blank skip",
    )
    ap.add_argument(
        "--blank-min-area",
        type=int,
        default=None,
        help="Minimum crop area required before blank skip applies",
    )
    ap.add_argument(
        "--allow-pytesseract",
        dest="allow_pytesseract",
        action="store_true",
        default=None,
        help="Opt back into spawning pytesseract during export",
    )
    ap.add_argument(
        "--no-allow-pytesseract",
        dest="allow_pytesseract",
        action="store_false",
        default=None,
        help="Force-disable pytesseract even if the environment opts in",
    )
    ap.add_argument(
        "--tess-unicharset",
        default=None,
        help="Path to a Tesseract-style unicharset file for toy lexical gating",
    )
    ap.add_argument(
        "--tess-wordlist",
        default=None,
        help="Optional newline-delimited dictionary that boosts toy OCR tokens",
    )
    ap.add_argument(
        "--tess-bigram-json",
        default=None,
        help="JSON bigram table that penalizes unlikely glyph transitions",
    )
    ap.add_argument(
        "--print-stage-trace",
        action="store_true",
        help="Print the stage timing table after the run",
    )
    ap.add_argument(
        "--ocr-engine",
        default=None,
        help="OCR backend to use (e.g. toy, tesseract, easyocr). Overrides ZOCR_OCR_ENGINE.",
    )
    ap.add_argument(
        "--ingest-signature",
        default=None,
        help="Optional reproducibility signature JSON to compare against",
    )
    ap.add_argument(
        "--advisor-response",
        default=None,
        help="Path to JSON/text advisor feedback to ingest before reruns",
    )
    ap.add_argument(
        "--rag-feedback",
        default=None,
        help="Optional rag/manifest.json to ingest feedback/profile overrides from",
    )
    args = ap.parse_args(argv)

    ensure_dir(args.outdir)
    toy_sweeps = args.toy_sweeps
    if toy_sweeps is not None and toy_sweeps <= 0:
        toy_sweeps = None
    force_numeric_flag = True if args.force_numeric_by_header else None
    try:
        res = _patched_run_full_pipeline(
            inputs=args.input,
            outdir=args.outdir,
            dpi=args.dpi,
            domain_hint=args.domain,
            k=args.k,
            do_tune=(not args.no_tune),
            tune_budget=args.tune_budget,
            views_log=args.views_log,
            gt_jsonl=args.gt_jsonl,
            org_dict=args.org_dict,
            resume=args.resume,
            seed=args.seed,
            snapshot=args.snapshot,
            ocr_engine=args.ocr_engine,
            toy_lite=args.toy_lite,
            toy_sweeps=toy_sweeps,
            force_numeric_by_header=force_numeric_flag,
            ingest_signature=args.ingest_signature,
            advisor_response=args.advisor_response,
            print_stage_trace=args.print_stage_trace,
            rag_feedback=args.rag_feedback,
            motion_prior=args.motion_prior,
            motion_sigma_px=args.motion_sigma_px,
            motion_cutoff_sigma=args.motion_cutoff_sigma,
            motion_accept_ratio=args.motion_accept_ratio,
            export_guard_ms=args.export_guard_ms,
            sweeps_fixed=args.sweeps_fixed,
            blank_skip=args.blank_skip,
            blank_threshold=args.blank_threshold,
            blank_min_pixels=args.blank_min_pixels,
            blank_min_ratio=args.blank_min_ratio,
            blank_min_area=args.blank_min_area,
            allow_pytesseract=args.allow_pytesseract,
            tess_unicharset=args.tess_unicharset,
            tess_wordlist=args.tess_wordlist,
            tess_bigram_json=args.tess_bigram_json,
        )
        print("\n[SUCCESS] Summary written:", os.path.join(args.outdir, "pipeline_summary.json"))
        print(json.dumps(res, ensure_ascii=False, indent=2))
    except Exception as e:
        print("\n💀 Pipeline crashed:", e)
        sys.exit(1)

if __name__=="__main__":
    main()
'''
zocr_pipeline_allinone = _materialize_module('zocr_pipeline_allinone', _SRC_ZOCR_PIPELINE_ALLINONE)

# ----------------------------- Entry Point ----------------------------
def main():
    """Entrypoint that forwards to the orchestrator's CLI."""
    if hasattr(zocr_pipeline_allinone, 'main'):
        return zocr_pipeline_allinone.main()
    # Fallback: expose a minimal callable if `main` is missing
    if hasattr(zocr_pipeline_allinone, 'run_full_pipeline'):
        import argparse
        ap = argparse.ArgumentParser('ZOCR merged runner')
        ap.add_argument('-i','--input', nargs='+', default=['demo'])
        ap.add_argument('--outdir', default='out_allinone')
        ap.add_argument('--dpi', type=int, default=200)
        ap.add_argument('--domain', default=None)
        ap.add_argument('--k', type=int, default=10)
        ap.add_argument('--no-tune', action='store_true')
        ap.add_argument('--tune-budget', type=int, default=20)
        ap.add_argument('--views-log', default=None)
        ap.add_argument('--gt-jsonl', default=None)
        ap.add_argument('--org-dict', default=None)
        args = ap.parse_args()
        return zocr_pipeline_allinone.run_full_pipeline(
            inputs=args.input, outdir=args.outdir, dpi=args.dpi,
            domain_hint=args.domain, k=args.k, do_tune=(not args.no_tune),
            tune_budget=args.tune_budget, views_log=args.views_log,
            gt_jsonl=args.gt_jsonl, org_dict=args.org_dict
        )

if __name__ == '__main__':
    main()
