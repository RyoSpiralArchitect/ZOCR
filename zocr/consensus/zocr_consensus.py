#!/usr/bin/env python3
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
import os, sys, io, json, argparse, tempfile, shutil, subprocess, time, math, re, hashlib, contextlib, bisect, unicodedata, atexit, difflib
from statistics import median
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Set, Mapping, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict, OrderedDict, deque

try:  # pragma: no cover - optional built-in tesslite tables
    from ..resources import tesslite_defaults as _tesslite_defaults  # type: ignore
except Exception:  # pragma: no cover - fallback when package data is unavailable
    _tesslite_defaults = None  # type: ignore

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
    """Push a new toy OCR self-correction configuration onto the active stack."""

    _TOY_SELF_CORRECTION_STACK.append(_normalize_self_correction_config(config))


def pop_toy_self_correction() -> None:
    """Pop the most recent toy OCR self-correction configuration."""

    if _TOY_SELF_CORRECTION_STACK:
        _TOY_SELF_CORRECTION_STACK.pop()


def current_toy_self_correction() -> Dict[str, Any]:
    """Return the merged toy OCR self-correction configuration."""

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
    """Context manager helper to apply a temporary toy self-correction config."""

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


def _estimate_slant_slope(binary: "np.ndarray") -> float:
    if np is None:
        return 0.0
    try:
        coords = np.argwhere(np.asarray(binary, dtype=np.uint8) > 0)
    except Exception:
        return 0.0
    if coords.size == 0:
        return 0.0
    y = coords[:, 0].astype(np.float64)
    x = coords[:, 1].astype(np.float64)
    if y.size < 128:
        return 0.0
    y_centered = y - float(np.mean(y))
    x_centered = x - float(np.mean(x))
    denom = float(np.dot(y_centered, y_centered))
    if denom <= 1e-6:
        return 0.0
    slope = float(np.dot(x_centered, y_centered) / denom)
    return float(max(-0.5, min(0.5, slope)))


def _shear_rows_binary(binary01: "np.ndarray", slope: float) -> "np.ndarray":
    arr = (np.asarray(binary01, dtype=np.uint8) > 0).astype(np.uint8)
    H, W = arr.shape
    out = np.zeros_like(arr)
    center = H / 2.0
    max_shift = int(min(W * 0.25, abs(slope) * H * 1.2 + 2))
    for y in range(H):
        shift = int(round((y - center) * slope))
        shift = max(-max_shift, min(max_shift, shift))
        row = arr[y]
        if shift > 0:
            out[y, shift:] = np.maximum(out[y, shift:], row[: W - shift])
        elif shift < 0:
            out[y, : W + shift] = np.maximum(out[y, : W + shift], row[-shift:])
        else:
            out[y] = np.maximum(out[y], row)
    return out


def _suppress_diagonal_bridges(binary01: "np.ndarray") -> "np.ndarray":
    arr = (np.asarray(binary01, dtype=np.uint8) > 0).astype(np.uint8)
    if arr.size == 0:
        return arr
    padded = np.pad(arr, 1, mode="constant")
    left = padded[1:-1, :-2] > 0
    right = padded[1:-1, 2:] > 0
    up = padded[:-2, 1:-1] > 0
    down = padded[2:, 1:-1] > 0
    diag = (
        padded[:-2, :-2]
        | padded[:-2, 2:]
        | padded[2:, :-2]
        | padded[2:, 2:]
    ) > 0
    horiz = left | right
    vert = up | down
    mask = (diag & (~horiz) & (~vert))
    trimmed = arr.copy()
    trimmed[mask] = 0
    return trimmed


def _apply_italic_guard(binary: "np.ndarray") -> "np.ndarray":
    if np is None:
        return binary
    arr = (np.asarray(binary, dtype=np.uint8) > 0).astype(np.uint8)
    arr = _suppress_diagonal_bridges(arr)
    slope = _estimate_slant_slope(arr)
    if abs(slope) < 0.08:
        return (arr * 255).astype(np.uint8)
    deskewed = _shear_rows_binary(arr, -slope)
    combined = np.maximum(arr, deskewed)
    combined = _suppress_diagonal_bridges(combined)
    return (combined * 255).astype(np.uint8)

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
    """Recursively partition column centers using a B-tree like split rule."""

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
    """Return stable column centers derived from a B-tree style recursive sort."""

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


def _vertical_vote_boundaries(
    binary: "np.ndarray", max_candidates: int = 16, min_gap: int = 6
) -> List[int]:
    """Return low-ink column boundaries via a lightweight vertical voting pass."""

    try:
        arr = np.asarray(binary, dtype=np.uint8)
    except Exception:
        return []
    if arr.ndim != 2 or arr.size == 0:
        return []
    ink = arr.sum(axis=0).astype(np.float32)
    if ink.size == 0:
        return []
    norm = ink / float(max(1.0, ink.max()))
    blank_score = 1.0 - norm
    window = max(3, min_gap // 2 * 2 + 1)
    kernel = np.ones(window, dtype=np.float32)
    kernel /= float(kernel.sum() or 1.0)
    smoothed = np.convolve(blank_score, kernel, mode="same")
    if smoothed.size == 0:
        return []
    threshold = float(max(0.2, min(0.85, np.percentile(smoothed, 75))))
    picks: List[int] = []
    for idx in range(1, len(smoothed) - 1):
        if smoothed[idx] < threshold:
            continue
        if smoothed[idx] < smoothed[idx - 1] or smoothed[idx] < smoothed[idx + 1]:
            continue
        if picks and idx - picks[-1] < min_gap:
            if smoothed[idx] > smoothed[picks[-1]]:
                picks[-1] = idx
            continue
        picks.append(idx)
        if len(picks) >= max_candidates:
            break
    return [int(val) for val in picks]


def _apply_column_continuity_prior(
    candidates_by_row: List[List[int]], tolerance: int = 5
) -> List[List[int]]:
    if not candidates_by_row:
        return candidates_by_row
    refined: List[List[int]] = []
    prev: Optional[List[int]] = None
    tol = max(1, int(tolerance))
    for row in candidates_by_row:
        base = sorted({int(v) for v in row})
        if prev:
            augmented = list(base)
            for anchor in prev:
                if all(abs(anchor - cur) > tol for cur in base):
                    augmented.append(int(anchor))
            base = sorted({int(v) for v in augmented})
        refined.append(base)
        prev = base
    return refined


def _refine_column_bounds_alignment(
    col_bounds: List[int], smooth_strength: float = 0.25
) -> List[int]:
    if len(col_bounds) <= 2:
        return col_bounds
    widths = [col_bounds[i + 1] - col_bounds[i] for i in range(len(col_bounds) - 1)]
    target = float(max(1.0, median([abs(w) for w in widths]) if widths else 1.0))
    new_bounds = [int(col_bounds[0])]
    for width in widths:
        deviation = float(width) - target
        adjusted = float(width) - deviation * smooth_strength
        adjusted = max(2.0, adjusted)
        new_bounds.append(int(round(new_bounds[-1] + adjusted)))
    span_orig = float(col_bounds[-1] - col_bounds[0]) or 1.0
    span_new = float(new_bounds[-1] - new_bounds[0]) or 1.0
    scale = span_orig / span_new
    scaled = [int(round(col_bounds[0] + (val - new_bounds[0]) * scale)) for val in new_bounds]
    scaled[0] = col_bounds[0]
    scaled[-1] = col_bounds[-1]
    return scaled


def _column_blank_ratios(binary: "np.ndarray", col_bounds: Sequence[int]) -> List[float]:
    count = max(0, len(col_bounds) - 1)
    if count <= 0:
        return []
    try:
        arr = (np.asarray(binary, dtype=np.uint8) > 0).astype(np.uint8)
    except Exception:
        return [0.0] * count
    H, W = arr.shape
    ratios: List[float] = []
    for idx in range(count):
        x0 = int(max(0, min(W, col_bounds[idx])))
        x1 = int(max(0, min(W, col_bounds[idx + 1])))
        if x1 <= x0:
            ratios.append(1.0)
            continue
        slice_arr = arr[:, x0:x1]
        total = float(max(1, slice_arr.size))
        ink = float(slice_arr.sum())
        ratios.append(max(0.0, min(1.0, 1.0 - ink / total)))
    return ratios


def _find_projection_valleys(values: "np.ndarray", threshold: float, min_gap: int) -> List[int]:
    if values.size <= 2:
        return []
    valleys: List[int] = []
    for idx in range(1, values.size - 1):
        if values[idx] > threshold:
            continue
        if values[idx] > values[idx - 1] or values[idx] >= values[idx + 1]:
            continue
        if valleys and idx - valleys[-1] < min_gap:
            if values[idx] < values[valleys[-1]]:
                valleys[-1] = idx
            continue
        valleys.append(idx)
    return valleys


def _align_row_band_centers(
    row_bands: List[Tuple[int, int]], height: int, med_h: float
) -> List[Tuple[int, int]]:
    if len(row_bands) <= 1:
        return row_bands
    centers = [0.5 * (y0 + y1) for (y0, y1) in row_bands]
    diffs = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    target_gap = float(max(2.0, median([abs(d) for d in diffs]) if diffs else med_h))
    aligned = [centers[0]]
    clamp_delta = med_h * 0.6
    for idx in range(1, len(centers)):
        expected = aligned[-1] + target_gap
        delta = clamp(centers[idx] - expected, -clamp_delta, clamp_delta)
        aligned.append(expected + delta * 0.35)
    refined: List[Tuple[int, int]] = []
    for idx, center in enumerate(aligned):
        span = max(2.0, row_bands[idx][1] - row_bands[idx][0])
        start = int(round(center - span / 2.0))
        end = int(round(center + span / 2.0))
        start = max(0, start)
        end = min(height, max(start + 2, end))
        refined.append((start, end))
    return refined


def _refine_row_bands_by_projection(
    row_bands: List[Tuple[int, int]],
    binary: "np.ndarray",
    med_h: float,
    height: int,
) -> Tuple[List[Tuple[int, int]], Dict[str, int]]:
    stats = {
        "segments": 0,
        "valley_splits": 0,
        "merged": 0,
    }
    if not row_bands:
        return row_bands, stats
    try:
        proj = (np.asarray(binary, dtype=np.uint8) > 0).sum(axis=1).astype(np.float64)
    except Exception:
        return row_bands, stats
    if proj.size == 0:
        return row_bands, stats
    kernel = np.ones(5, dtype=np.float64)
    kernel /= float(kernel.sum() or 1.0)
    smooth = np.convolve(proj, kernel, mode="same")
    refined: List[Tuple[int, int]] = []
    min_gap = int(max(2, med_h * 0.35))
    for y0, y1 in row_bands:
        stats["segments"] += 1
        if y1 - y0 <= 0:
            continue
        segment = smooth[y0:y1]
        if (y1 - y0) > med_h * 1.6 and segment.size > 4:
            thr = float(min(np.percentile(segment, 40), np.mean(segment) * 0.7))
            valleys = _find_projection_valleys(segment, thr, min_gap)
            if valleys:
                stats["valley_splits"] += len(valleys)
                points = [y0] + [y0 + v for v in valleys] + [y1]
                for a, b in zip(points, points[1:]):
                    if b - a >= max(2, int(med_h * 0.45)):
                        refined.append((int(a), int(b)))
                continue
        refined.append((int(y0), int(y1)))
    merged: List[Tuple[int, int]] = []
    for band in refined:
        if merged and (band[1] - band[0]) < med_h * 0.45:
            prev = merged[-1]
            if band[0] - prev[1] <= med_h * 0.25:
                merged[-1] = (prev[0], max(prev[1], band[1]))
                stats["merged"] += 1
                continue
        merged.append(band)
    aligned = _align_row_band_centers(merged, height, med_h)
    stats["refined"] = len(aligned)
    return aligned, stats

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
    bin_img=_apply_italic_guard(bin_img)
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
    row_bands_raw = list(row_bands)
    row_bands, row_proj_stats = _refine_row_bands_by_projection(row_bands, bin_img, med_h, H)
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
    candidates_initial = [list(row) for row in candidates_by_row]
    global_mid_seeds: List[int] = []
    if len(btree_seed) >= 2:
        global_mid_seeds.extend(int((btree_seed[i]+btree_seed[i+1])/2.0) for i in range(len(btree_seed)-1))
    if len(centers)>=2:
        global_mid_seeds.extend(int((centers[i]+centers[i+1])/2.0) for i in range(len(centers)-1))
    vertical_votes = _vertical_vote_boundaries(
        bin_img,
        max_candidates=32,
        min_gap=int(max(4, med_w * 0.5)),
    )
    if vertical_votes:
        global_mid_seeds.extend(vertical_votes)
    if global_mid_seeds:
        mids_global = sorted({int(val) for val in global_mid_seeds})
        merged_rows: List[List[int]] = []
        for row in candidates_by_row:
            merged_rows.append(sorted({*row, *mids_global}))
        candidates_by_row = merged_rows
    continuity_tol = int(max(3, med_w * 0.35))
    candidates_by_row = _apply_column_continuity_prior(candidates_by_row, tolerance=continuity_tol)
    candidates_after = [list(row) for row in candidates_by_row]
    shape_lambda = float(params.get("shape_lambda", 4.0))
    col_bounds=_smooth_per_column(candidates_by_row, W, lam=shape_lambda, H_sched=max(1,H))
    col_bounds = _refine_column_bounds_alignment(col_bounds)
    column_blank_ratios = _column_blank_ratios(bin_img, col_bounds)
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
    blank_thr = float(params.get("colspan_blank_threshold", 0.9))
    blank_sigma = float(params.get("colspan_blank_sigma", 0.08))

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

    def try_expand_colspan(r: int, c: int, rs: int, cs: int, ch: List[int]) -> int:
        step = 0
        while (c + cs) < C:
            blank_ratio = column_blank_ratios[c + cs] if (c + cs) < len(column_blank_ratios) else 0.0
            if blank_ratio < blank_thr:
                break
            block = cell_block_bbox(r, rs, c, cs + 1)
            coverage_rows: List[float] = []
            for rr in range(r, min(R, r + rs)):
                coverage_rows.append(_coverage_ratio(chunks_by_row[rr], *block))
            cov = float(np.median(coverage_rows)) if coverage_rows else 0.0
            p_iou = _sigmoid((cov - tau) / max(1e-6, sigma))
            blank_boost = _sigmoid((blank_ratio - blank_thr) / max(1e-3, blank_sigma))
            gap_block = (
                col_bounds[c + cs],
                row_bands[r][0],
                col_bounds[c + cs + 1],
                row_bands[min(R - 1, r + rs - 1)][1],
            )
            gap_hits = 0
            for rr in range(r, min(R, r + rs)):
                cov_gap = _coverage_ratio(
                    chunks_by_row[rr],
                    gap_block[0],
                    row_bands[rr][0],
                    gap_block[2],
                    row_bands[rr][1],
                )
                if cov_gap < 0.05:
                    gap_hits += 1
            p_gap = _sigmoid(((gap_hits / float(max(1, rs))) - 0.5) / 0.2)
            p_comb = p_iou * blank_boost * p_gap
            if p_comb < p_cons_thr:
                break
            cs += 1
            step += 1
        return cs
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
                cs = try_expand_colspan(r,c,rs,cs,ch)
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
    row_diag = {
        "initial_bands": int(len(row_bands_raw)),
        "refined_bands": int(len(row_bands)),
        "projection_splits": int(row_proj_stats.get("valley_splits", 0)) if isinstance(row_proj_stats, dict) else 0,
        "segments": int(row_proj_stats.get("segments", 0)) if isinstance(row_proj_stats, dict) else 0,
        "merged": int(row_proj_stats.get("merged", 0)) if isinstance(row_proj_stats, dict) else 0,
    }
    row_bands_rel = [
        (int(max(0, min(H, yt))), int(max(0, min(H, yb))))
        for (yt, yb) in row_bands
    ]
    column_diag = {
        "btree_seed": len(btree_seed),
        "dp_centers": len(centers),
        "vertical_votes": len(vertical_votes),
        "global_mid_seeds": len({int(v) for v in global_mid_seeds}) if global_mid_seeds else 0,
        "continuity_tol": continuity_tol,
        "candidates_initial": sum(len(r) for r in candidates_initial),
        "candidates_final": sum(len(r) for r in candidates_after),
        "rows_with_candidates": sum(1 for r in candidates_after if r),
        "blank_columns": sum(1 for ratio in column_blank_ratios if ratio >= blank_thr),
        "blank_ratio_mean": round(float(sum(column_blank_ratios)) / float(len(column_blank_ratios) or 1), 4)
        }
    segmentation_stats = {
        "row": row_diag,
        "column": column_diag,
    }
    dbg = {
        "rows":R,"cols":C,"row_counts": row_counts,
        "col_bounds":col_bounds,"smear_wx": wx, "smear_wy": wy,
        "med_h": med_h, "col_jitter": col_jitter,
        "baselines_segs": baselines,
        "row_bands_rel": row_bands_rel,
        "segmentation_stats": segmentation_stats,
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
_AUTOCALIB_DEFAULT_SAMPLES = 3
_AUTOTUNE_DEFAULT_TRIALS = 6


def _positive_cli_value(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return value if value > 0 else None


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
    p.add_argument(
        "--autocalib",
        nargs="?",
        type=int,
        const=_AUTOCALIB_DEFAULT_SAMPLES,
        default=None,
        metavar="N",
        help=(
            "Auto-calibrate CC params using N sample pages (default %(const)s when "
            "the flag is provided without a value). Pass 0 or omit the flag to disable."
        ),
    )
    p.add_argument(
        "--autotune",
        nargs="?",
        type=int,
        const=_AUTOTUNE_DEFAULT_TRIALS,
        default=None,
        metavar="N",
        help=(
            "Run the unsupervised autotuner for N trials (default %(const)s when the "
            "flag has no explicit value). Pass 0 or omit to skip autotuning."
        ),
    )
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
    autocalib_samples = _positive_cli_value(args.autocalib)
    autotune_trials = _positive_cli_value(args.autotune)
    if autocalib_samples:
        tab_cfg.update(auto_calibrate_params(pages, autocalib_samples))
    if autotune_trials:
        tab_cfg.update(autotune_params(pages, tab_cfg, trials=autotune_trials))
    cfg={"table":tab_cfg,"bench_iterations":args.bench_iterations,"eval":True}
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

def _initial_toy_sweep_limit(default: int = 2) -> int:
    candidates = ("ZOCR_TOY_SWEEPS", "ZOCR_TOY_SWEEP_LIMIT")
    for key in candidates:
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            return max(1, int(raw.strip()))
        except Exception:
            continue
    return max(1, int(default))


_INITIAL_TOY_SWEEP_LIMIT = _initial_toy_sweep_limit()
_GLYPH_VARIANT_LIMIT = int(_INITIAL_TOY_SWEEP_LIMIT)


@dataclass
class MotionPriorCfg:
    enabled: bool = False
    sigma_px: float = 8.0
    cutoff_sigma: float = 2.5
    accept_ratio: float = 0.5
    k_sigma_window: float = 2.5
    auto_sigma: bool = False
    sigma_min_ratio: float = 0.15
    sigma_max_ratio: float = 1.5
    table_signature: Optional[str] = None
    cache_dir: Optional[str] = None
    bandit_action: Optional[str] = None


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


try:  # pragma: no cover - optional orchestrator helper
    from ..orchestrator.prior import estimate_sigma_px as _estimate_sigma_px  # type: ignore
except Exception:  # pragma: no cover - fallback when orchestrator is unavailable

    def _estimate_sigma_px(
        delta_y: Sequence[float],
        median_row_h: float,
        s_min_ratio: float = 0.15,
        s_max_ratio: float = 1.5,
    ) -> float:
        if not delta_y:
            return max(1.0, 0.5 * float(median_row_h or 1.0))
        med = median(delta_y)
        mad = median([abs(d - med) for d in delta_y])
        sigma = 1.4826 * mad
        sigma_min = s_min_ratio * max(1.0, median_row_h)
        sigma_max = s_max_ratio * max(1.0, median_row_h)
        return float(min(max(sigma, sigma_min), sigma_max))


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
    enabled = _env_flag("ZOCR_EXPORT_MOTION_PRIOR", True) or _env_flag("ZOCR_USE_PRIOR", False)
    cfg = MotionPriorCfg(enabled=enabled)
    cfg.table_signature = os.environ.get("ZOCR_TABLE_SIGNATURE")
    cfg.cache_dir = os.environ.get("ZOCR_PRIOR_CACHE")
    cfg.bandit_action = os.environ.get("ZOCR_PRIOR_ACTION")
    cfg.k_sigma_window = float(max(0.1, _env_float("ZOCR_K_SIGMA_WINDOW", cfg.k_sigma_window)))
    cfg.sigma_min_ratio = float(max(0.01, _env_float("ZOCR_PRIOR_SIGMA_MIN_RATIO", cfg.sigma_min_ratio)))
    cfg.sigma_max_ratio = float(max(cfg.sigma_min_ratio, _env_float("ZOCR_PRIOR_SIGMA_MAX_RATIO", cfg.sigma_max_ratio)))
    sigma_raw = os.environ.get("ZOCR_PRIOR_SIGMA")
    sigma_override: Optional[float] = None
    if sigma_raw:
        if sigma_raw.strip().lower() == "auto":
            cfg.auto_sigma = True
        else:
            try:
                sigma_override = float(sigma_raw)
            except Exception:
                sigma_override = None
    elif cfg.enabled:
        cfg.auto_sigma = True
    if sigma_override is None:
        sigma_env = os.environ.get("ZOCR_EXPORT_MOTION_SIGMA")
        if sigma_env:
            try:
                sigma_override = float(sigma_env)
            except Exception:
                sigma_override = None
    if sigma_override is not None:
        cfg.sigma_px = max(0.5, float(sigma_override))
        cfg.auto_sigma = False
    cutoff_raw = os.environ.get("ZOCR_EXPORT_MOTION_CUTOFF")
    if cutoff_raw:
        try:
            cfg.cutoff_sigma = max(0.1, float(cutoff_raw))
        except Exception:
            pass
    accept_raw = os.environ.get("ZOCR_EXPORT_MOTION_ACCEPT")
    if accept_raw:
        try:
            cfg.accept_ratio = float(max(0.0, min(1.0, float(accept_raw))))
        except Exception:
            pass
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


def _prior_cache_path(cache_dir: str, signature: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", signature.strip()) or "unknown"
    return os.path.join(cache_dir, f"{safe}.ykeys.json")


def _load_prior_cache_ykeys(signature: Optional[str], cache_dir: Optional[str]) -> Optional[List[float]]:
    if not signature or not cache_dir:
        return None
    path = _prior_cache_path(cache_dir, signature)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    values = payload.get("y_keys") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        return None
    result: List[float] = []
    for val in values:
        try:
            result.append(float(val))
        except Exception:
            continue
    return result or None


def _store_prior_cache_ykeys(
    signature: Optional[str], cache_dir: Optional[str], y_keys: Sequence[float]
) -> bool:
    if not signature or not cache_dir or not y_keys:
        return False
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        return False
    path = _prior_cache_path(cache_dir, signature)
    payload = {"y_keys": [float(y) for y in y_keys]}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


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
    row_heights: Optional[Sequence[int]] = None,
) -> Tuple[List[Tuple[int, int]], bool, Dict[str, Any]]:
    stats: Dict[str, Any] = {}
    if not cfg.enabled or not prev_keys or not row_bands:
        return row_bands, False, stats
    reseeded, matches = _reseed_row_bands_with_prior(prev_keys, row_bands)
    if not reseeded:
        return row_bands, False, stats
    heights = row_heights or [int(max(1, band[1] - band[0])) for band in row_bands]
    med_height = float(median(heights)) if heights else max(1.0, abs(row_bands[0][1] - row_bands[0][0]))
    deltas = [cand_mid - prev_val for cand_mid, prev_val in matches]
    sigma_px = float(cfg.sigma_px)
    auto_used = False
    if cfg.auto_sigma:
        sigma_px = _estimate_sigma_px(deltas, med_height, cfg.sigma_min_ratio, cfg.sigma_max_ratio)
        auto_used = True
    sigma_px = max(1e-3, float(sigma_px))
    k_sigma = max(0.1, float(cfg.k_sigma_window))
    window = k_sigma * sigma_px
    cutoff = float(cfg.cutoff_sigma) * sigma_px
    stats.update(
        {
            "sigma_px": float(sigma_px),
            "window_px": float(window),
            "auto_sigma": bool(auto_used),
            "deltas": len(deltas),
            "median_row_h": float(med_height),
            "k_sigma_window": float(k_sigma),
        }
    )
    limited: List[Tuple[int, int]] = []
    trimmed = 0
    if window > 0:
        for idx, band in enumerate(reseeded):
            top, bottom = band
            mid = (float(top) + float(bottom)) * 0.5
            target = float(prev_keys[idx]) if idx < len(prev_keys) else mid
            if abs(mid - target) > window:
                trimmed += 1
                height = max(1.0, float(bottom - top))
                center = target + max(-window, min(window, mid - target))
                half = height * 0.5
                new_top = int(round(center - half))
                new_bottom = int(round(center + half))
                limited.append((new_top, new_bottom))
            else:
                limited.append(band)
    else:
        limited = list(reseeded)
    if trimmed:
        stats["trimmed"] = int(trimmed)
    reseeded = limited
    inside = 0
    total = len(matches)
    for cand_mid, prev_val in matches:
        if abs(cand_mid - prev_val) <= cutoff:
            inside += 1
    stats["inliers"] = int(inside)
    stats["matches"] = int(total)
    actual_ratio = inside / float(total or 1)
    stats["accept_ratio"] = float(actual_ratio)
    if actual_ratio >= cfg.accept_ratio:
        return reseeded, True, stats
    return row_bands, False, stats


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


def _radial_signature(arr_f: "np.ndarray") -> Tuple[float, float]:
    if arr_f.ndim != 2 or arr_f.size == 0:
        return 0.0, 0.0
    arr = np.asarray(arr_f, dtype=np.float32)
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = float(max(1.0, max(cy, cx)))
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / radius
    inner_mask = dist <= 0.4
    outer_mask = dist >= 0.8
    inner_val = float(arr[inner_mask].mean()) if inner_mask.any() else 0.0
    outer_val = float(arr[outer_mask].mean()) if outer_mask.any() else 0.0
    return inner_val, outer_val


def _compute_glyph_features_from_array(arr: "np.ndarray") -> Dict[str, float]:
    arr_f = np.asarray(arr, dtype=np.float32)
    if arr_f.size == 0:
        return {
            "aspect": 1.0,
            "density": 0.0,
            "symmetry": 0.0,
            "style_var": 0.0,
            "radial_inner": 0.0,
            "radial_outer": 0.0,
            "count": 0,
        }
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
    inner_ring = 0.0
    outer_ring = 0.0
    if arr_f.ndim == 2 and arr_f.size:
        inner_ring, outer_ring = _radial_signature(arr_f)
    return {
        "aspect": aspect,
        "density": density,
        "symmetry": symmetry,
        "style_var": style_var,
        "radial_inner": inner_ring,
        "radial_outer": outer_ring,
        "count": 1,
    }

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
_TESSLITE_LAST_SOURCE: str = "none"

_TESSLITE_BUILTIN_SIGNATURE = "tesslite_builtin_v1"
if _tesslite_defaults is not None:
    _TESSLITE_BUILTIN_SIGNATURE = str(
        getattr(_tesslite_defaults, "DEFAULT_SIGNATURE", _TESSLITE_BUILTIN_SIGNATURE)
    )


def _tesslite_builtin_available() -> bool:
    disable = os.environ.get("ZOCR_TESSLITE_DISABLE_BUILTIN", "").strip().lower()
    if disable in {"1", "true", "yes", "on"}:
        return False
    return bool(
        _tesslite_defaults
        and getattr(_tesslite_defaults, "DEFAULT_GLYPHS", None)
        and len(getattr(_tesslite_defaults, "DEFAULT_GLYPHS", [])) > 0
    )


def _tesslite_builtin_payload() -> Tuple[
    Set[str],
    Dict[str, Set[str]],
    Dict[str, str],
    Set[str],
    Dict[str, Dict[str, float]],
]:
    glyphs: Set[str] = set()
    ambiguous: Dict[str, Set[str]] = defaultdict(set)
    categories: Dict[str, str] = {}
    dictionary: Set[str] = set()
    bigrams: Dict[str, Dict[str, float]] = {}
    if not _tesslite_builtin_available():
        return glyphs, ambiguous, categories, dictionary, bigrams
    glyphs.update(getattr(_tesslite_defaults, "DEFAULT_GLYPHS", []))
    for src, targets in getattr(_tesslite_defaults, "DEFAULT_AMBIGUOUS", {}).items():
        if not targets:
            continue
        ambiguous[src].update(targets)
    categories.update(getattr(_tesslite_defaults, "DEFAULT_CHAR_CATEGORIES", {}))
    dictionary.update(getattr(_tesslite_defaults, "DEFAULT_DICTIONARY", []))
    raw_bigrams = getattr(_tesslite_defaults, "DEFAULT_BIGRAMS", {})
    if isinstance(raw_bigrams, dict):
        for prev, mapping in raw_bigrams.items():
            if not isinstance(mapping, dict):
                continue
            bigrams[prev] = {str(ch): float(val) for ch, val in mapping.items() if isinstance(val, (int, float))}
    return glyphs, ambiguous, categories, dictionary, bigrams


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


def _tesslite_effective_source() -> Tuple[bool, str, bool]:
    unichar = os.environ.get("ZOCR_TESS_UNICHARSET") or None
    wordlist = os.environ.get("ZOCR_TESS_WORDLIST") or None
    bigram = os.environ.get("ZOCR_TESS_BIGRAM_JSON") or None
    env_supplied = any([unichar, wordlist, bigram])
    if env_supplied:
        return True, _tesslite_env_signature(), False
    if _tesslite_builtin_available():
        return True, f"builtin:{_TESSLITE_BUILTIN_SIGNATURE}", True
    return False, "", False


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
    global _TESSLITE_MODEL, _TESSLITE_MODEL_SIG, _TESSLITE_LAST_SOURCE
    enabled, signature, use_builtin = _tesslite_effective_source()
    if not enabled:
        _TESSLITE_MODEL = None
        _TESSLITE_MODEL_SIG = None
        _TESSLITE_LAST_SOURCE = "none"
        return None
    if _TESSLITE_MODEL is not None and signature == _TESSLITE_MODEL_SIG:
        return _TESSLITE_MODEL
    if use_builtin:
        glyphs, ambiguous, categories, dictionary, bigrams = _tesslite_builtin_payload()
    else:
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
    _TESSLITE_LAST_SOURCE = "builtin" if use_builtin else "env"
    return model


def get_tesslite_status() -> Dict[str, Any]:
    enabled, signature, use_builtin = _tesslite_effective_source()
    unichar = os.environ.get("ZOCR_TESS_UNICHARSET") or None
    wordlist = os.environ.get("ZOCR_TESS_WORDLIST") or None
    bigram = os.environ.get("ZOCR_TESS_BIGRAM_JSON") or None
    source = "builtin" if use_builtin else ("env" if enabled else "none")
    return {
        "enabled": bool(enabled),
        "signature": signature or None,
        "source": source,
        "unicharset": unichar if unichar else ("builtin" if use_builtin else None),
        "wordlist": wordlist if wordlist else ("builtin" if use_builtin else None),
        "bigram_json": bigram if bigram else ("builtin" if use_builtin else None),
    }


def _feature_source(*names: str) -> str:
    for name in names:
        if name and os.environ.get(name) is not None:
            return "env"
    return "default"


def get_toy_feature_status() -> Dict[str, Any]:
    """Return a snapshot of the active toy OCR feature knobs."""

    status: Dict[str, Any] = {}

    conf_cfg = _confidence_boost_cfg_from_env()
    status["confidence_boost"] = {
        "enabled": bool(conf_cfg.enabled),
        "target": float(conf_cfg.target),
        "min_input": float(conf_cfg.min_input),
        "source": _feature_source("ZOCR_CONF_BOOST_NUMERIC"),
    }

    lex_cfg = _lexical_boost_cfg_from_env()
    status["lexical_boost"] = {
        "enabled": bool(lex_cfg.enabled),
        "target": float(lex_cfg.target),
        "min_input": float(lex_cfg.min_input),
        "min_quality": float(lex_cfg.min_quality),
        "source": _feature_source("ZOCR_CONF_BOOST_LEXICAL"),
    }

    alpha_raw = os.environ.get("ZOCR_NGRAM_EMA_ALPHA")
    try:
        alpha_val = float(alpha_raw) if alpha_raw is not None else 0.05
    except Exception:
        alpha_val = 0.05
    status["ngram_ema"] = {
        "alpha": float(max(0.0, alpha_val)),
        "source": _feature_source("ZOCR_NGRAM_EMA_ALPHA"),
    }

    motion_cfg = _motion_prior_cfg_from_env()
    status["motion_prior"] = {
        "enabled": bool(motion_cfg.enabled),
        "sigma_px": float(motion_cfg.sigma_px),
        "auto_sigma": bool(getattr(motion_cfg, "auto_sigma", False)),
        "k_sigma_window": float(motion_cfg.k_sigma_window),
        "cutoff_sigma": float(motion_cfg.cutoff_sigma),
        "accept_ratio": float(motion_cfg.accept_ratio),
        "source": _feature_source("ZOCR_EXPORT_MOTION_PRIOR", "ZOCR_USE_PRIOR"),
    }

    blank_cfg = _blank_skip_cfg_from_env()
    status["blank_skip"] = {
        "enabled": bool(blank_cfg.enabled),
        "dark_threshold": int(blank_cfg.dark_threshold),
        "min_dark_pixels": int(blank_cfg.min_dark_pixels),
        "min_dark_ratio": float(blank_cfg.min_dark_ratio),
        "min_area": int(blank_cfg.min_area),
        "source": _feature_source(
            "ZOCR_EXPORT_SKIP_BLANK",
            "ZOCR_EXPORT_BLANK_THRESHOLD",
            "ZOCR_EXPORT_BLANK_MIN_PIXELS",
            "ZOCR_EXPORT_BLANK_MIN_RATIO",
            "ZOCR_EXPORT_BLANK_MIN_AREA",
        ),
    }

    status["tesslite"] = get_tesslite_status()

    status["pytesseract"] = {
        "allowed": _pytesseract_allowed(),
        "source": _feature_source("ZOCR_ALLOW_PYTESSERACT"),
    }

    flush_raw = os.environ.get("ZOCR_EXPORT_FLUSH_EVERY")
    if flush_raw:
        try:
            flush_val = int(flush_raw)
        except Exception:
            flush_val = None
        if flush_val is not None:
            status["export_flush"] = {
                "interval": max(1, flush_val),
                "source": "env",
            }

    return status


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
_NUMERIC_RX = re.compile(r"[+\-]?\d[\d,]*(?:\.\d+)?%?")
_NUMERIC_HEADER_INFERRED_LAST = 0


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
        return True
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
_TOY_SWEEPS = int(_INITIAL_TOY_SWEEP_LIMIT)
_FORCE_NUMERIC = _env_flag(
    "ZOCR_COERCE_NUMERIC",
    _env_flag("ZOCR_FORCE_NUMERIC", True),
)
_LAST_EXPORT_STATS: Dict[str, Any] = {}


def toy_runtime_config() -> Dict[str, Any]:
    """Return the currently active toy OCR runtime knobs."""

    return {
        "threshold_sweeps": int(_TOY_SWEEPS),
        "glyph_variant_limit": int(_GLYPH_VARIANT_LIMIT),
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
    global _TOY_SWEEPS, _FORCE_NUMERIC, _GLYPH_VARIANT_LIMIT
    if sweeps is not None:
        try:
            new_sweeps = max(1, int(sweeps))
        except Exception:
            new_sweeps = _TOY_SWEEPS
        if new_sweeps != _TOY_SWEEPS:
            _TOY_SWEEPS = new_sweeps
            _GLYPH_VARIANT_LIMIT = int(max(1, new_sweeps))
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
    ("subtotal", re.compile(r"(小計|subtotal)", re.I)),
    (
        "amount",
        re.compile(
            r"(金額|見積金額|御見積金額|御見積合計|合計金額|合計|総計|計|税込|税別|総額|amount|total|grand\s*total|balance|amount\s*due|total\s*due)",
            re.I,
        ),
    ),
    ("tax_amount", re.compile(r"(税額|消費税額|tax\s*amount|vat\s*amount|gst\s*amount)", re.I)),
    ("tax_rate", re.compile(r"(税率|消費税率|税%|tax%|tax(\s*rate)?|vat|gst)", re.I)),
]

_FULLWIDTH_NUMBERS = str.maketrans(
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
        "．": ".",
        "，": ",",
        "－": "-",
        "＋": "+",
        "％": "%",
    }
)
_NUMERIC_SANITIZE_RX = re.compile(r"[^0-9.+-]")


def _normalize_numeric_bits(text: Any, kind: str) -> Tuple[str, Optional[float], Optional[bool]]:
    raw = "" if text is None else str(text)
    raw = raw.strip()
    if not raw:
        return "", None, None
    work = raw.translate(_FULLWIDTH_NUMBERS)
    work = work.replace("円", "").replace("￥", "").replace("¥", "").replace("$", "")
    work = work.replace(",", "")
    work = work.replace("，", "").replace("．", ".")
    work = work.replace("％", "%")
    neg = False
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
        if c < len(kinds) and kinds[c]:
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
                for candidate, rx in _NUMERIC_HEADER_KIND:
                    if variant and rx.search(variant):
                        kind = candidate
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


_DATE_ROLE_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    (
        "due",
        (
            "due date",
            "payment due",
            "due on",
            "due",
            "支払期限",
            "支払期日",
            "支払日",
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
            "交付日",
            "売上日",
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
            "搭乗日",
            "乗車日",
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
    norm = norm.translate(_FULLWIDTH_NUMBERS)
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
        "pattern": re.compile(r"(item|品名|品目|摘要|description|desc|details)", re.I),
        "title": "Item",
        "normalizer": None,
    },
    {
        "key": "qty",
        "pattern": re.compile(r"(qty|数量|quantity|q'?ty|pcs?|units?|個数|台数)", re.I),
        "title": "Qty",
        "normalizer": "qty",
    },
    {
        "key": "unit_price",
        "pattern": re.compile(r"(unit\s*(price|cost)|単価|単価金額)", re.I),
        "title": "Unit Price",
        "normalizer": "currency",
    },
    {
        "key": "amount",
        "pattern": re.compile(r"(amount|line\s*total|line\s*amount|金額|total|合計|総計|見積金額)", re.I),
        "title": "Amount",
        "normalizer": "currency",
    },
]

_SCHEMA_SEMANTIC_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "item": (
        "item",
        "品名",
        "品目",
        "摘要",
        "内容",
        "名称",
        "品番",
        "項目",
        "description",
        "desc",
        "details",
        "備考",
        "仕様",
    ),
    "qty": (
        "qty",
        "quantity",
        "数量",
        "個数",
        "台数",
        "数",
        "pcs",
        "unit",
    ),
    "unit_price": (
        "unitprice",
        "unitcost",
        "単価",
        "単価金額",
        "単価(税込)",
        "単価(税抜)",
    ),
    "amount": (
        "amount",
        "total",
        "lineamount",
        "lineTotal",
        "金額",
        "合計",
        "総計",
        "見積金額",
        "御見積金額",
        "計",
        "税込金額",
    ),
}


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


def _build_schema_synonym_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for key, tokens in _SCHEMA_SEMANTIC_SYNONYMS.items():
        for token in tokens:
            norm = _schema_normalize_header_token(token)
            if not norm:
                continue
            mapping.setdefault(norm, key)
    return mapping


_SCHEMA_SYNONYM_MAP = _build_schema_synonym_map()


def _schema_candidate_headers(
    grid_text: Sequence[Sequence[str]], max_rows: int = 3
) -> List[Tuple[str, List[str]]]:
    if not grid_text:
        return []
    sample_rows = [row for row in grid_text[:max_rows] if row]
    if not sample_rows:
        sample_rows = [grid_text[0]]
    width = max((len(row) for row in sample_rows), default=len(grid_text[0] or []))
    width = max(width, len(grid_text[0])) if grid_text and grid_text[0] else width

    def _pad(row: Sequence[str]) -> List[str]:
        return [
            str(row[idx]) if idx < len(row) and row[idx] is not None else ""
            for idx in range(width)
        ]

    candidates: List[Tuple[str, List[str]]] = []
    seen: Set[Tuple[str, ...]] = set()

    def _push(label: str, row: List[str]) -> None:
        if not row:
            return
        if not any(str(cell or "").strip() for cell in row):
            return
        key = tuple(row)
        if key in seen:
            return
        seen.add(key)
        candidates.append((label, row))

    for ridx, row in enumerate(sample_rows):
        _push(f"row{ridx}", _pad(row))

    max_span = min(len(sample_rows), max_rows)
    for span in range(2, max_span + 1):
        merged = ["" for _ in range(width)]
        for ridx in range(span):
            row = _pad(sample_rows[ridx])
            for idx, token in enumerate(row):
                token_clean = str(token or "").strip()
                if not token_clean:
                    continue
                current = str(merged[idx] or "").strip()
                if not current or len(current) <= 2:
                    merged[idx] = token
                elif len(token_clean) > len(current):
                    merged[idx] = token
        _push("+".join(f"row{ridx}" for ridx in range(span)), merged)

    return candidates


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
                mapped = _SCHEMA_SYNONYM_MAP.get(token)
                if mapped == column.get("key"):
                    best_idx = idx
                    break
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


def _schema_token_role(text: Optional[str]) -> Optional[str]:
    token = _schema_normalize_header_token(text)
    if not token:
        return None
    mapped = _SCHEMA_SYNONYM_MAP.get(token)
    if mapped:
        return mapped
    for column in _ITEM_QTY_SCHEMA_COLUMNS:
        pattern = column.get("pattern")
        if pattern and pattern.search(token):
            return column.get("key")
    return None


def _collect_schema_token_bonuses(
    grid_text: Sequence[Sequence[str]], num_cols: int
) -> List[defaultdict]:  # type: ignore[type-arg]
    bonuses: List[defaultdict] = [defaultdict(float) for _ in range(max(0, num_cols))]
    if not grid_text or num_cols <= 0:
        return bonuses
    sample_rows: List[Sequence[str]] = []
    sample_rows.extend(grid_text[:2])
    sample_rows.extend(grid_text[-2:])
    for ridx, row in enumerate(sample_rows):
        weight = 1.2 if ridx == 0 else (1.0 if ridx <= 2 else 0.9)
        for idx in range(num_cols):
            token = row[idx] if idx < len(row) else None
            role = _schema_token_role(token)
            if not role:
                continue
            bonuses[idx][role] += weight
    return bonuses


def _semantic_item_qty_schema(grid_text: Sequence[Sequence[str]]) -> Optional[List[int]]:
    if not grid_text:
        return None
    num_cols = max(len(row) for row in grid_text)
    if num_cols < len(_ITEM_QTY_SCHEMA_COLUMNS):
        return None
    profiles = _column_numeric_profiles(grid_text)
    if not profiles or len(profiles) < num_cols:
        return None
    bonuses = _collect_schema_token_bonuses(grid_text, num_cols)

    def _score(idx: int, key: str) -> float:
        profile = profiles[idx] if idx < len(profiles) else {
            "numeric_ratio": 0.0,
            "currency_ratio": 0.0,
            "decimal_ratio": 0.0,
            "text_ratio": 0.0,
        }
        score = 0.0
        if key == "item":
            score = profile.get("text_ratio", 0.0) * 1.2 + (1.0 - profile.get("numeric_ratio", 0.0)) * 0.4
        elif key == "qty":
            score = profile.get("numeric_ratio", 0.0) * 1.1 + (1.0 - profile.get("currency_ratio", 0.0)) * 0.3
            score -= profile.get("text_ratio", 0.0) * 0.2
        elif key == "unit_price":
            score = profile.get("decimal_ratio", 0.0) * 1.0 + profile.get("currency_ratio", 0.0) * 0.6
            score += profile.get("numeric_ratio", 0.0) * 0.25
        elif key == "amount":
            score = profile.get("currency_ratio", 0.0) * 1.2 + profile.get("numeric_ratio", 0.0) * 0.6
        score += bonuses[idx].get(key, 0.0)
        # prefer columns closer to the right for amount but left for item
        if key == "item":
            score -= idx * 0.02
        elif key == "amount":
            score += idx * 0.01
        return score

    def _select_best(indices: Sequence[int], key: str) -> Optional[int]:
        best_idx = None
        best_score = -1e9
        for idx in indices:
            sc = _score(idx, key)
            if sc > best_score:
                best_score = sc
                best_idx = idx
        if best_idx is None or best_score < 0.1:
            return None
        return best_idx

    all_indices = list(range(num_cols))
    amount_idx = _select_best(all_indices, "amount")
    if amount_idx is None:
        return None
    unit_price_candidates = [idx for idx in all_indices if idx < amount_idx]
    unit_price_idx = _select_best(unit_price_candidates, "unit_price")
    if unit_price_idx is None:
        return None
    qty_candidates = [idx for idx in all_indices if idx < unit_price_idx]
    qty_idx = _select_best(qty_candidates, "qty")
    if qty_idx is None:
        return None
    item_candidates = [idx for idx in all_indices if idx < qty_idx]
    item_idx = _select_best(item_candidates, "item")
    if item_idx is None:
        return None
    indices = [item_idx, qty_idx, unit_price_idx, amount_idx]
    if any(indices[i] >= indices[i + 1] for i in range(len(indices) - 1)):
        return None
    return indices


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


def _looks_like_numeric_token(text: Optional[str]) -> bool:
    if text is None:
        return False
    body = str(text).strip()
    if not body:
        return False
    compact = body.replace(",", "").replace("，", "").replace("．", ".")
    if compact.endswith("%"):
        compact = compact[:-1]
    return bool(re.fullmatch(r"[+\-]?\d+(?:\.\d+)?", compact))


def _looks_like_item_context(text: Optional[str]) -> bool:
    if text is None:
        return False
    body = str(text).strip()
    if not body:
        return False
    if _looks_like_numeric_token(body):
        return False
    if any(ch.isalpha() for ch in body):
        return True
    for ch in body:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            return True
    return len(body) >= 4


def _looks_like_item_noise(text: Optional[str]) -> bool:
    if text is None:
        return True
    body = str(text).strip()
    if not body:
        return True
    if len(body) <= 2:
        return True
    if body.isupper() and len(body) <= 4 and body.isalpha():
        return True
    return False


def _rectify_item_qty_amount_schema(
    grid_text: List[List[str]],
    grid_conf: List[List[float]],
    col_bounds: List[int],
) -> Optional[Tuple[List[List[str]], List[List[float]], List[int], Dict[str, Any]]]:
    if not grid_text or not grid_text[0]:
        return None
    strategy = "header"
    header_candidates = _schema_candidate_headers(grid_text)
    header_source = None
    match: Optional[List[int]] = None
    for label, header_row in header_candidates:
        match = _match_item_qty_schema(header_row)
        if match:
            header_source = label
            break
    if not match:
        match = _semantic_item_qty_schema(grid_text)
        if match:
            strategy = "semantic"
    if not match:
        match = _approximate_item_qty_schema(grid_text)
        if not match:
            return None
        strategy = "heuristic"
    if any(idx + 1 >= len(col_bounds) for idx in match):
        return None
    noise_cols = [idx for idx in range(len(grid_text[0])) if idx not in match]
    left_item_cols = [idx for idx in noise_cols if idx < match[0]] if match else []
    right_amount_cols = [idx for idx in noise_cols if idx > match[-1]] if match else []
    width = len(match)
    new_text: List[List[str]] = []
    new_conf: List[List[float]] = []
    rows_adjusted = 0
    cells_salvaged = 0
    cells_cleared = 0
    item_aux_rows = 0
    item_aux_cells = 0
    item_aux_examples: List[str] = []
    trailing_notes = 0
    trailing_note_rows = 0
    trailing_note_examples: List[str] = []

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
            left_candidates: List[Tuple[str, float]] = []
            trailing_candidates: List[str] = []
            for idx in sorted(noise_cols, reverse=True):
                txt, cf = _extract(row, conf_row, idx)
                if not txt:
                    continue
                noise_pool.append((txt, cf))
                if idx in left_item_cols and _looks_like_item_context(txt):
                    cleaned = str(txt).strip()
                    if cleaned:
                        left_candidates.append((cleaned, cf))
                elif idx in right_amount_cols:
                    cleaned = str(txt).strip()
                    if not cleaned or _looks_like_numeric_token(cleaned):
                        continue
                    trailing_candidates.append(cleaned)
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
            if left_candidates:
                left_candidates.reverse()
                base_val = new_row[0] or ""
                base_clean = base_val.strip()
                allow_merge = (not base_clean) or _looks_like_item_noise(base_clean) or len(base_clean) <= 3
                if allow_merge:
                    merged_tokens: List[str] = []
                    for token, cf in left_candidates:
                        token_clean = token.strip()
                        if not token_clean:
                            continue
                        if token_clean in merged_tokens:
                            continue
                        merged_tokens.append(token_clean)
                        item_aux_cells += 1
                        if len(item_aux_examples) < 5:
                            item_aux_examples.append(token_clean)
                        new_conf_row[0] = max(new_conf_row[0], cf)
                    keep_existing = bool(base_clean and not _looks_like_item_noise(base_clean))
                    if keep_existing and base_clean:
                        merged_tokens.append(base_clean)
                    if merged_tokens:
                        merged_value = "\n".join(merged_tokens)
                        if merged_value != new_row[0]:
                            new_row[0] = merged_value
                            row_changed = True
                            item_aux_rows += 1
            if trailing_candidates:
                trailing_notes += len(trailing_candidates)
                trailing_note_rows += 1
                for token in trailing_candidates:
                    if len(trailing_note_examples) < 5 and token not in trailing_note_examples:
                        trailing_note_examples.append(token)
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
        "item_aux_rows": item_aux_rows,
        "item_aux_cells": item_aux_cells,
        "item_aux_examples": item_aux_examples[:5],
        "trailing_notes": trailing_notes,
        "trailing_note_rows": trailing_note_rows,
        "trailing_note_examples": trailing_note_examples[:5],
        "header_candidates": len(header_candidates),
        "header_source": header_source,
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
    cell_inner = 0.0
    cell_outer = 0.0
    if cell_scaled.ndim == 2 and cell_scaled.size:
        row_profile = cell_scaled.mean(axis=1)
        col_profile = cell_scaled.mean(axis=0)
        cell_style = float(_np.var(row_profile) + _np.var(col_profile))
        cell_inner, cell_outer = _radial_signature(cell_scaled)
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
        glyph_inner = feats.get("radial_inner", cell_inner)
        glyph_outer = feats.get("radial_outer", cell_outer)
        aspect_penalty = math.exp(-abs(math.log((cell_aspect + 1e-3)/(glyph_aspect + 1e-3))) * 0.75)
        density_penalty = 1.0 - min(0.4, abs(cell_density - glyph_density) * 1.6)
        if glyph_sym > 0.5:
            sym_cell = float(1.0 - _np.mean(_np.abs(cell_arr - _np.flip(cell_arr, axis=1))) / 255.0)
            symmetry_penalty = 0.8 + 0.2 * max(0.0, sym_cell)
        else:
            symmetry_penalty = 1.0
        style_penalty = 1.0 - min(0.35, abs(cell_style - glyph_style) * 0.8)
        radial_penalty = 1.0 - min(0.3, abs(cell_inner - glyph_inner) * 1.2 + abs(cell_outer - glyph_outer) * 0.9)
        variant_best *= (
            aspect_penalty
            * max(0.4, density_penalty)
            * symmetry_penalty
            * max(0.45, style_penalty)
            * max(0.5, radial_penalty)
        )
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
    avg_width: float = 0.0
    avg_height: float = 0.0
    stroke_density: float = 0.0
    aspect_median: float = 1.0


def _estimate_baseline_stats(boxes: Sequence[Tuple[int, int, int, int, float]]) -> Optional[_BaselineStats]:
    if not boxes:
        return None
    bottoms: List[float] = []
    heights: List[float] = []
    widths: List[float] = []
    densities: List[float] = []
    ascenders: List[float] = []
    descenders: List[float] = []
    for x1, y1, x2, y2, area in boxes:
        h = float(max(1, y2 - y1))
        w = float(max(1, x2 - x1))
        heights.append(h)
        widths.append(w)
        bottoms.append(float(y2))
        box_area = float(max(1.0, (x2 - x1) * (y2 - y1)))
        densities.append(float(max(0.0, area)) / box_area)
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
    avg_w = float(median(widths)) if widths else xheight
    density = float(median(densities)) if densities else 0.5
    aspect = float(max(0.25, min(4.0, avg_w / max(1.0, xheight))))
    return _BaselineStats(
        baseline=baseline,
        xheight=xheight,
        ascender=asc,
        descender=desc,
        avg_width=avg_w,
        avg_height=float(median(heights) if heights else xheight),
        stroke_density=density,
        aspect_median=aspect,
    )


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
    density_hint = 0.0
    if baseline:
        density_hint = baseline.stroke_density or 0.0
        aspect_hint = baseline.aspect_median or 1.0
        wide_ratio = max(1.1, min(1.8, 1.0 + aspect_hint * 0.35))
        tall_ratio = max(1.2, min(2.2, (baseline.avg_height or baseline.xheight) / max(1.0, baseline.avg_width or baseline.xheight)))
        gap_ratio = 0.06 * (0.85 if density_hint > 0.45 else 1.1)
    if width > height * wide_ratio:
        segments = _component_projection_splits(arr, axis=1, gap_ratio=gap_ratio, min_span_ratio=0.22)
    if not segments and baseline and width > max(1.0, baseline.avg_width) * 1.75:
        segments = _component_projection_splits(arr, axis=1, gap_ratio=gap_ratio * 0.8, min_span_ratio=0.2)
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

    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return 0

    log_every = max(1, _parse_env_int("ZOCR_EXPORT_LOG_EVERY", 200, 1))
    flush_every = max(0, _parse_env_int("ZOCR_EXPORT_FLUSH_EVERY", 200, 0))
    max_cells = _parse_env_int("ZOCR_EXPORT_MAX_CELLS", 0, 0)
    cells_done = 0
    t0 = time.time()
    stop_due_to_limit = False
    motion_cfg = _motion_prior_cfg_from_env()
    sweep_tracker = _EXPORT_SWEEP_TRACKER
    motion_applied = 0
    motion_rejected = 0
    motion_sigma_samples: List[float] = []
    motion_window_samples: List[float] = []
    motion_auto_estimated = 0
    motion_prior_attempts: List[Dict[str, Any]] = []
    guard_ms = _parse_env_int("ZOCR_EXPORT_GUARD_MS", 0, 0)
    guard_timeouts = 0
    doc_identifier = str(
        doc.get("doc_id")
        or doc.get("document_id")
        or doc.get("id")
        or os.path.splitext(os.path.basename(doc_json_path))[0]
    )
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
    prior_cache_seed: Optional[List[float]] = None
    prior_cache_hits = 0
    prior_cache_writes = 0
    schema_tables = 0
    schema_noise_columns = 0
    schema_rows_adjusted = 0
    schema_cells_salvaged = 0
    schema_cells_cleared = 0
    schema_item_aux_rows = 0
    schema_item_aux_cells = 0
    schema_trailing_notes = 0
    schema_trailing_note_rows = 0
    schema_item_aux_examples: List[str] = []
    schema_trailing_examples: List[str] = []
    schema_header_candidates = 0
    schema_header_sources: Dict[str, int] = defaultdict(int)
    schema_strategy_breakdown: Dict[str, int] = defaultdict(int)
    total_rows_seen = 0
    total_rows_reflowed = 0
    total_rows_ocr_attempts = 0
    total_rows_ocr_success = 0
    seg_tables = 0
    seg_row_initial = 0
    seg_row_final = 0
    seg_row_splits = 0
    seg_row_merged = 0
    seg_col_btree = 0
    seg_col_dp = 0
    seg_col_votes = 0
    seg_col_mid = 0
    seg_col_candidates_initial = 0
    seg_col_candidates_final = 0
    seg_rows_with_candidates = 0
    if motion_cfg.enabled:
        cached = _load_prior_cache_ykeys(motion_cfg.table_signature, motion_cfg.cache_dir)
        if cached:
            prior_cache_seed = list(cached)
            prior_cache_hits += 1

    pages = doc.get("pages") if isinstance(doc, dict) else None
    if not isinstance(pages, list):
        pages = []

    for page in pages:
        if not isinstance(page, dict):
            continue
        for table in page.get("tables", []) or []:
            if not isinstance(table, dict):
                continue
            dbg = table.get("dbg")
            if not isinstance(dbg, dict):
                continue
            seg_dbg = dbg.get("segmentation_stats")
            if not isinstance(seg_dbg, dict):
                continue
            seg_tables += 1
            row_diag = seg_dbg.get("row") or {}
            col_diag = seg_dbg.get("column") or {}
            seg_row_initial += _safe_int(row_diag.get("initial_bands"))
            seg_row_final += _safe_int(row_diag.get("refined_bands"))
            seg_row_splits += _safe_int(row_diag.get("projection_splits"))
            seg_row_merged += _safe_int(row_diag.get("merged"))
            seg_col_btree += _safe_int(col_diag.get("btree_seed"))
            seg_col_dp += _safe_int(col_diag.get("dp_centers"))
            seg_col_votes += _safe_int(col_diag.get("vertical_votes"))
            seg_col_mid += _safe_int(col_diag.get("global_mid_seeds"))
            seg_col_candidates_initial += _safe_int(col_diag.get("candidates_initial"))
            seg_col_candidates_final += _safe_int(col_diag.get("candidates_final"))
            seg_rows_with_candidates += _safe_int(col_diag.get("rows_with_candidates"))

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
            page_image = _load_page_image(page_image_path, lookup_idx)
            if page_image is None:
                continue
            page_w, page_h = page_image.size
            page_index_int = int(lookup_idx) if lookup_idx is not None else int(enum_idx)
            tables = p.get("tables", []) if isinstance(p, dict) else []
            for ti, t in enumerate(tables):
                if stop_due_to_limit:
                    break
                if not isinstance(t, dict):
                    continue
                tables_processed += 1
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
                row_heights = [int(max(1, band[1] - band[0])) for band in row_bands]
                prev_keys = sweep_tracker.get(doc_identifier, page_index_int, ti)
                if not prev_keys and prior_cache_seed:
                    prev_keys = list(prior_cache_seed)
                motion_entry_stats: Dict[str, Any] = {}
                if motion_cfg.enabled:
                    row_bands_prior, applied, motion_entry_stats = _apply_motion_prior_to_bands(
                        prev_keys,
                        row_bands,
                        motion_cfg,
                        row_heights=row_heights,
                    )
                    if motion_entry_stats:
                        motion_prior_attempts.append(motion_entry_stats)
                        sigma_val = motion_entry_stats.get("sigma_px")
                        window_val = motion_entry_stats.get("window_px")
                        if isinstance(sigma_val, (int, float)):
                            motion_sigma_samples.append(float(sigma_val))
                        if isinstance(window_val, (int, float)):
                            motion_window_samples.append(float(window_val))
                        if motion_entry_stats.get("auto_sigma"):
                            motion_auto_estimated += 1
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
                guard_deadline = (time.time() + guard_ms / 1000.0) if guard_ms > 0 else None
                guard_triggered = False
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
                    if stop_due_to_limit:
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
                    mids = _row_band_midpoints(row_bands)
                    sweep_tracker.put(
                        doc_identifier,
                        page_index_int,
                        ti,
                        mids,
                    )
                    if _store_prior_cache_ykeys(
                        motion_cfg.table_signature,
                        motion_cfg.cache_dir,
                        mids,
                    ):
                        prior_cache_writes += 1
                        prior_cache_seed = list(mids)
                schema_adjust = _rectify_item_qty_amount_schema(grid_text, grid_conf, col_bounds)
                if schema_adjust:
                    grid_text, grid_conf, col_bounds, schema_meta = schema_adjust
                    C = max(1, len(col_bounds) - 1)
                    schema_tables += 1
                    schema_noise_columns += int(schema_meta.get("noise_columns", 0))
                    schema_rows_adjusted += int(schema_meta.get("rows_adjusted", 0))
                    schema_cells_salvaged += int(schema_meta.get("cells_salvaged", 0))
                    schema_cells_cleared += int(schema_meta.get("cells_cleared", 0))
                    schema_item_aux_rows += int(schema_meta.get("item_aux_rows", 0))
                    schema_item_aux_cells += int(schema_meta.get("item_aux_cells", 0))
                    schema_trailing_notes += int(schema_meta.get("trailing_notes", 0))
                    schema_trailing_note_rows += int(schema_meta.get("trailing_note_rows", 0))
                    schema_header_candidates += int(schema_meta.get("header_candidates", 0))
                    strategy_name = schema_meta.get("strategy")
                    if strategy_name:
                        schema_strategy_breakdown[str(strategy_name)] += 1
                    header_source_meta = schema_meta.get("header_source")
                    if header_source_meta:
                        schema_header_sources[str(header_source_meta)] += 1
                    aux_samples = schema_meta.get("item_aux_examples") or []
                    if isinstance(aux_samples, list):
                        for sample in aux_samples:
                            if len(schema_item_aux_examples) >= 5:
                                break
                            schema_item_aux_examples.append(str(sample))
                    trailing_samples = schema_meta.get("trailing_note_examples") or []
                    if isinstance(trailing_samples, list):
                        for sample in trailing_samples:
                            if len(schema_trailing_examples) >= 5:
                                break
                            schema_trailing_examples.append(str(sample))
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
                        row_texts = grid_text[r]
                        # build search/synthesis
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
                        if col_kind:
                            filters["numeric_header_kind"] = col_kind
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
            "item_aux_rows": int(schema_item_aux_rows),
            "item_aux_cells": int(schema_item_aux_cells),
            "trailing_notes": int(schema_trailing_notes),
            "trailing_note_rows": int(schema_trailing_note_rows),
        }
        if schema_item_aux_examples:
            export_stats["schema_alignment"]["item_aux_examples"] = schema_item_aux_examples[:5]
        if schema_trailing_examples:
            export_stats["schema_alignment"]["trailing_note_examples"] = schema_trailing_examples[:5]
        if schema_header_candidates:
            export_stats["schema_alignment"]["header_candidates"] = int(schema_header_candidates)
        if schema_header_sources:
            export_stats["schema_alignment"]["header_sources"] = dict(schema_header_sources)
        if schema_strategy_breakdown:
            export_stats["schema_alignment"]["strategy_breakdown"] = dict(schema_strategy_breakdown)
    if total_rows_seen:
        export_stats["total_rows"] = {
            "rows": int(total_rows_seen),
            "reflowed": int(total_rows_reflowed),
            "ocr_attempts": int(total_rows_ocr_attempts),
            "ocr_success": int(total_rows_ocr_success),
        }
    if seg_tables:
        inv_tables = 1.0 / float(max(1, seg_tables))
        export_stats["segmentation"] = {
            "tables": int(seg_tables),
            "row_initial_mean": round(seg_row_initial * inv_tables, 3),
            "row_final_mean": round(seg_row_final * inv_tables, 3),
            "row_projection_splits_mean": round(seg_row_splits * inv_tables, 3),
            "row_merged_mean": round(seg_row_merged * inv_tables, 3),
            "column_btree_mean": round(seg_col_btree * inv_tables, 3),
            "column_dp_mean": round(seg_col_dp * inv_tables, 3),
            "column_vertical_votes_mean": round(seg_col_votes * inv_tables, 3),
            "column_mid_seed_mean": round(seg_col_mid * inv_tables, 3),
            "column_candidates_initial_mean": round(seg_col_candidates_initial * inv_tables, 3),
            "column_candidates_final_mean": round(seg_col_candidates_final * inv_tables, 3),
            "rows_with_candidates_mean": round(seg_rows_with_candidates * inv_tables, 3),
        }
    if motion_cfg.enabled:
        sigma_summary = None
        if motion_sigma_samples:
            sigma_summary = {
                "median": float(median(motion_sigma_samples)),
                "min": float(min(motion_sigma_samples)),
                "max": float(max(motion_sigma_samples)),
                "count": len(motion_sigma_samples),
            }
        window_summary = None
        if motion_window_samples:
            window_summary = {
                "median": float(median(motion_window_samples)),
                "min": float(min(motion_window_samples)),
                "max": float(max(motion_window_samples)),
                "count": len(motion_window_samples),
            }
        export_stats["motion_prior"] = {
            "sigma_px": float(motion_cfg.sigma_px),
            "cutoff_sigma": float(motion_cfg.cutoff_sigma),
            "accept_ratio": float(motion_cfg.accept_ratio),
            "k_sigma_window": float(motion_cfg.k_sigma_window),
            "applied": int(motion_applied),
            "rejected": int(motion_rejected),
            "auto_estimated": int(motion_auto_estimated),
            "bandit_action": motion_cfg.bandit_action,
            "table_signature": motion_cfg.table_signature,
            "sigma_summary": sigma_summary,
            "window_summary": window_summary,
            "cache": {
                "hits": int(prior_cache_hits),
                "writes": int(prior_cache_writes),
            },
            "attempts": len(motion_prior_attempts),
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
    use_ext_variants = (
        os.environ.get("ZOCR_EXPORT_EXT_VARIANTS", "0") == "1"
        and ocr_engine in ("tess", "easyocr")
    )

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
                if use_ext_variants:
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
    runtime_overrides: Dict[str, Any] = {}
    sweeps = getattr(args, "toy_sweeps", None)
    if sweeps is not None and sweeps > 0:
        runtime_overrides["sweeps"] = sweeps
    force_numeric = getattr(args, "force_numeric", None)
    if force_numeric is not None:
        runtime_overrides["force_numeric"] = force_numeric
    if runtime_overrides:
        configure_toy_runtime(**runtime_overrides)
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
        sp.add_argument(
            "--toy-sweeps",
            type=int,
            default=None,
            help="Clamp toy OCR threshold sweeps (overrides ZOCR_TOY_SWEEPS)",
        )
        group = sp.add_mutually_exclusive_group()
        group.add_argument(
            "--force-numeric",
            dest="force_numeric",
            action="store_true",
            help="Force numeric coercion based on headers",
        )
        group.add_argument(
            "--no-force-numeric",
            dest="force_numeric",
            action="store_false",
            help="Disable numeric coercion",
        )
        sp.set_defaults(force_numeric=None)
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
