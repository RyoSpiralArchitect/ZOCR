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

import concurrent.futures
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - optional built-in tesslite tables
    from ..resources import tesslite_defaults as _tesslite_defaults  # type: ignore
except Exception:  # pragma: no cover - fallback when package data is unavailable
    _tesslite_defaults = None  # type: ignore

try:  # pragma: no cover - optional domain dictionary access
    from ..resources.domain_dictionary import get_domain_keywords as _get_domain_keywords  # type: ignore
except Exception:  # pragma: no cover - fallback when package data is unavailable
    _get_domain_keywords = None  # type: ignore

try:
    import numpy as np
except Exception:
    np = None

try:  # shared utility helpers when running as a package
    from ..utils.json_utils import json_ready as _json_ready  # type: ignore
except Exception:  # pragma: no cover - fallback when relative import fails
    try:
        from zocr.utils.json_utils import json_ready as _json_ready  # type: ignore
    except Exception:  # pragma: no cover - standalone fallback
        _json_ready = None  # type: ignore

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageEnhance
try:
    import pytesseract  # type: ignore
    from pytesseract import Output as _PYTESS_OUTPUT  # type: ignore
    from pytesseract import TesseractError as _PYTESS_ERR  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore
    _PYTESS_OUTPUT = None  # type: ignore
    _PYTESS_ERR = None  # type: ignore
from html.parser import HTMLParser

from .runtime_primitives import (
    _apply_italic_guard,
    _binarize_pure,
    _btree_column_centers,
    _cc_label_rle,
    _dilate_binary_rect,
    _dp_means_1d,
    _find_projection_valleys,
    _smooth_per_column,
    _vertical_vote_boundaries,
    clamp,
    _align_row_band_centers,
)


def _env_truthy_local(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int_local(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except Exception:
        return default


_DEBUG_CELLS_ENABLED = _env_truthy_local("ZOCR_EXPORT_DEBUG_CELLS", False)
_DEBUG_CELLS_PER_PAGE = max(0, _env_int_local("ZOCR_EXPORT_DEBUG_CELLS_PER_PAGE", 0))
_DEBUG_CELL_MIN_AREA = max(0, _env_int_local("ZOCR_EXPORT_DEBUG_CELL_MIN_AREA", 4096))
_DEBUG_CELL_PAGE_COUNT: Dict[str, int] = defaultdict(int)


def _allow_debug_cell_dump(page_key: str, area: int) -> bool:
    if not _DEBUG_CELLS_ENABLED or _DEBUG_CELLS_PER_PAGE <= 0:
        return False
    if area < _DEBUG_CELL_MIN_AREA:
        return False
    if _DEBUG_CELL_PAGE_COUNT[page_key] >= _DEBUG_CELLS_PER_PAGE:
        return False
    _DEBUG_CELL_PAGE_COUNT[page_key] += 1
    return True


def _toy_runtime_module():
    """Import the lazily-loaded toy runtime module."""

    from . import toy_runtime as _toy_runtime

    return _toy_runtime


def push_toy_self_correction(config: Optional[Dict[str, Any]]) -> None:
    """Push a toy OCR self-correction configuration onto the stack."""

    _toy_runtime_module().push_toy_self_correction(config)


def pop_toy_self_correction() -> None:
    """Pop the most recently applied toy OCR self-correction configuration."""

    _toy_runtime_module().pop_toy_self_correction()


def current_toy_self_correction() -> Dict[str, Any]:
    """Return the merged toy OCR self-correction configuration."""

    return _toy_runtime_module().current_toy_self_correction()


def toy_self_correction_scope(config: Optional[Dict[str, Any]]):
    """Context manager delegating to the toy runtime implementation."""

    return _toy_runtime_module().toy_self_correction_scope(config)


def toy_runtime_config() -> Dict[str, Any]:
    """Expose the active toy runtime configuration."""

    return _toy_runtime_module().toy_runtime_config()


def configure_toy_runtime(*, sweeps: Optional[int] = None, force_numeric: Optional[bool] = None) -> Dict[str, Any]:
    """Update toy OCR runtime knobs at runtime."""

    return _toy_runtime_module().configure_toy_runtime(sweeps=sweeps, force_numeric=force_numeric)


def last_export_stats() -> Dict[str, Any]:
    """Return metrics captured during the most recent contextual export."""

    return _toy_runtime_module().last_export_stats()


def reset_toy_recognition_stats() -> None:
    """Clear per-run toy OCR recognition diagnostics."""

    _toy_runtime_module().reset_toy_recognition_stats()


def toy_recognition_stats(reset: bool = False) -> Dict[str, Any]:
    """Return aggregate diagnostics gathered during toy OCR recognition."""

    return _toy_runtime_module().toy_recognition_stats(reset=reset)


def toy_memory_snapshot() -> Dict[str, Any]:
    """Return aggregate statistics describing the current toy OCR memory."""

    return _toy_runtime_module().toy_memory_snapshot()


def toy_memory_delta(before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Compute deltas between two toy OCR memory snapshots."""

    return _toy_runtime_module().toy_memory_delta(before, after)


def save_toy_memory(path: str) -> Dict[str, Any]:
    """Persist the toy OCR memory state to disk."""

    return _toy_runtime_module().save_toy_memory(path)


def load_toy_memory(path: str) -> Dict[str, Any]:
    """Load toy OCR memory state from disk."""

    return _toy_runtime_module().load_toy_memory(path)


def get_toy_feature_status() -> Dict[str, Any]:
    """Return a snapshot of the active toy OCR feature knobs."""

    return _toy_runtime_module().get_toy_feature_status()


def export_jsonl_with_ocr(
    doc_json_path: str,
    source_images: Union[str, Sequence[str], Mapping[int, str]],
    out_jsonl_path: str,
    ocr_engine: str = "toy",
    contextual: bool = True,
    ocr_min_conf: float = 0.58,
) -> int:
    """Delegate contextual export with OCR to the toy runtime implementation."""

    return _toy_runtime_module().export_jsonl_with_ocr(
        doc_json_path,
        source_images,
        out_jsonl_path,
        ocr_engine=ocr_engine,
        contextual=contextual,
        ocr_min_conf=ocr_min_conf,
    )


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


if _json_ready is None:  # pragma: no cover - standalone fallback

    def _json_ready(obj: Any):
        """Best-effort conversion of numpy/complex objects into JSON-safe values."""

        if isinstance(obj, dict):
            return {k: _json_ready(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_ready(v) for v in obj]
        if isinstance(obj, set):
            return [_json_ready(v) for v in obj]
        if np is not None:
            if isinstance(obj, np.generic):  # type: ignore[attr-defined]
                return obj.item()
            if isinstance(obj, np.ndarray):  # type: ignore[attr-defined]
                return obj.tolist()
        return obj

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
    debug_page_key = os.path.abspath(image_path) if image_path else ""
    if amb_crops and _DEBUG_CELLS_ENABLED and _DEBUG_CELLS_PER_PAGE > 0:
        vdir: Optional[str] = None
        for (bl, r0, c0, st) in amb_crops[:64]:
            xl, yt, xr, yb = bl
            area = max(0, int(xr - xl) * int(yb - yt))
            if not _allow_debug_cell_dump(debug_page_key, area):
                continue
            if vdir is None:
                vdir = os.path.join(os.path.dirname(image_path), "views_cells")
                ensure_dir(vdir)
            crop = imc.crop((xl, yt, xr, yb))
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
@dataclass
class _PdfInspection:
    page_count: int
    max_width_pt: float
    max_height_pt: float


def _pdf_parallel_workers(page_count: int) -> int:
    if page_count <= 1:
        return 1
    min_pages = max(2, _env_int_local("ZOCR_PDF_PARALLEL_MIN_PAGES", 6))
    if page_count < min_pages:
        return 1
    forced = _env_int_local("ZOCR_PDF_WORKERS", 0)
    if forced > 0:
        return max(1, min(forced, page_count))
    cpu = os.cpu_count() or 1
    if cpu <= 2:
        return 1
    default = min(4, cpu - 1)
    return max(1, min(default, page_count))


def _pdf_chunk_ranges(page_count: int, workers: int) -> List[Tuple[int, int]]:
    if workers <= 1 or page_count <= 0:
        return [(0, page_count)]
    chunk = max(1, math.ceil(page_count / workers))
    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < page_count:
        end = min(page_count, start + chunk)
        ranges.append((start, end))
        start = end
    return ranges


def _pdf_inspect_with_doc(doc: Any, sample_pages: Optional[int] = None) -> _PdfInspection:
    page_count = len(doc)
    if page_count <= 0:
        return _PdfInspection(0, 0.0, 0.0)
    limit = sample_pages
    if limit is None:
        limit = max(1, _env_int_local("ZOCR_PDF_INSPECT_PAGES", 12))
    limit = min(limit, page_count)
    max_w = 0.0
    max_h = 0.0
    for idx in range(limit):
        page = doc[idx]
        try:
            try:
                w, h = page.get_size()  # type: ignore[attr-defined]
            except Exception:
                w = h = 0.0
        finally:
            page.close()
        max_w = max(max_w, float(w or 0.0))
        max_h = max(max_h, float(h or 0.0))
    return _PdfInspection(page_count, max_w, max_h)


def _pdf_inspect_via_pdfinfo(pdf_path: str) -> Optional[_PdfInspection]:
    exe = shutil.which("pdfinfo")
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, pdf_path],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    pages = 0
    max_w = 0.0
    max_h = 0.0
    for line in proc.stdout.splitlines():
        norm = line.strip().lower()
        if norm.startswith("pages:") and pages <= 0:
            try:
                pages = int(line.split(":", 1)[1].strip().split()[0])
            except Exception:
                pages = 0
        elif "page size" in norm:
            nums = re.findall(r"([0-9]+(?:\\.[0-9]+)?)", line)
            if len(nums) >= 2:
                try:
                    w = float(nums[0])
                    h = float(nums[1])
                except Exception:
                    continue
                max_w = max(max_w, w)
                max_h = max(max_h, h)
    if pages <= 0:
        return None
    if max_w <= 0.0:
        max_w = 612.0
    if max_h <= 0.0:
        max_h = 792.0
    return _PdfInspection(pages, max_w, max_h)


def _pdf_lazy_inspection(pdf_path: str) -> Optional[_PdfInspection]:
    try:
        import pypdfium2  # type: ignore

        doc = pypdfium2.PdfDocument(pdf_path)
    except Exception:
        return _pdf_inspect_via_pdfinfo(pdf_path)
    try:
        return _pdf_inspect_with_doc(doc)
    finally:
        doc.close()


def _pdf_resolve_raster_plan(
    base_dpi: int, inspection: Optional[_PdfInspection]
) -> Tuple[int, Optional[int]]:
    dpi = max(1, base_dpi or 200)
    snapshot_mode = _env_truthy_local("ZOCR_PIPELINE_SNAPSHOT", False)
    if snapshot_mode:
        snapshot_pct = _env_int_local("ZOCR_PDF_SNAPSHOT_DPI_PCT", 80)
        if 0 < snapshot_pct < 100:
            dpi = max(1, int(dpi * (snapshot_pct / 100.0)))
    hard_floor = max(48, _env_int_local("ZOCR_PDF_MIN_DPI_FLOOR", 72))
    soft_min = max(hard_floor, _env_int_local("ZOCR_PDF_MIN_DPI", 120))
    budget = max(0, _env_int_local("ZOCR_PDF_PIXEL_BUDGET", 320_000_000))
    if snapshot_mode:
        snap_budget = max(0, _env_int_local("ZOCR_PDF_SNAPSHOT_PIXEL_BUDGET", 220_000_000))
        if snap_budget > 0:
            budget = snap_budget if budget <= 0 else min(budget, snap_budget)
    ideal_dpi = dpi
    if inspection and budget > 0 and dpi > 0:
        width_pt = inspection.max_width_pt or 612.0
        height_pt = inspection.max_height_pt or 792.0
        width_px = (width_pt / 72.0) * dpi
        height_px = (height_pt / 72.0) * dpi
        per_page = max(width_px * height_px, 1.0)
        total_pixels = per_page * max(1, inspection.page_count)
        if total_pixels > budget:
            scale = math.sqrt(budget / total_pixels)
            ideal_dpi = max(1, int(max(1.0, dpi) * scale))
    if ideal_dpi < soft_min:
        dpi = max(hard_floor, ideal_dpi)
    else:
        dpi = max(soft_min, min(dpi, ideal_dpi))
    limit = max(0, _env_int_local("ZOCR_PDF_MAX_PAGES", 0))
    if snapshot_mode and limit <= 0:
        limit = max(0, _env_int_local("ZOCR_PDF_SNAPSHOT_MAX_PAGES", 0))
    if limit <= 0:
        page_limit: Optional[int] = None
    else:
        page_limit = limit
        if inspection:
            page_limit = min(page_limit, inspection.page_count)
    return dpi, page_limit


def _pdfium_render_linear(
    doc: Any, tmpdir: str, scale: float, limit: Optional[int]
) -> List[str]:
    total = len(doc) if limit is None else min(limit, len(doc))
    out_paths: List[str] = []
    try:
        for i in range(total):
            page = doc[i]
            try:
                bitmap = page.render(scale=scale)
                im = bitmap.to_pil()
            finally:
                page.close()
            page_path = os.path.join(tmpdir, f"page-{i+1:04d}.png")
            im.convert("RGB").save(page_path, format="PNG")
            out_paths.append(page_path)
    finally:
        doc.close()
    return out_paths


def _render_pdf_chunk_task(args: Tuple[str, Tuple[int, int], float, str]) -> List[Tuple[int, str]]:
    pdf_path, bounds, scale, tmpdir = args
    start, end = bounds
    import pypdfium2

    doc = pypdfium2.PdfDocument(pdf_path)
    out: List[Tuple[int, str]] = []
    try:
        for idx in range(start, min(end, len(doc))):
            page = doc[idx]
            try:
                bitmap = page.render(scale=scale)
                im = bitmap.to_pil()
            finally:
                page.close()
            path = os.path.join(tmpdir, f"page-{idx+1:04d}.png")
            im.convert("RGB").save(path, format="PNG")
            out.append((idx, path))
    finally:
        doc.close()
    return out


def _pdfium_render_parallel(pdf_path: str, page_count: int, tmpdir: str, scale: float, workers: int) -> List[str]:
    ranges = _pdf_chunk_ranges(page_count, workers)
    tasks = []
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for bounds in ranges:
                if bounds[0] >= bounds[1]:
                    continue
                tasks.append(pool.submit(_render_pdf_chunk_task, (pdf_path, bounds, scale, tmpdir)))
            ordered: List[Tuple[int, str]] = []
            for fut in concurrent.futures.as_completed(tasks):
                ordered.extend(fut.result())
    except Exception:
        # Fallback to linear rendering if parallel execution fails for any reason.
        import pypdfium2

        doc = pypdfium2.PdfDocument(pdf_path)
        return _pdfium_render_linear(doc, tmpdir, scale, page_count)
    ordered.sort(key=lambda item: item[0])
    return [path for _, path in ordered]


def _pdf_to_images_via_pdfium(pdf_path: str, dpi: int = 200) -> List[str]:
    try:
        import pypdfium2
    except ImportError as exc:
        raise RuntimeError(
            "pdftoppm not found and pypdfium2 is unavailable; install poppler-utils "
            "or `pip install pypdfium2` to rasterize PDFs"
        ) from exc

    tmpdir = tempfile.mkdtemp(prefix="zocr_pdfium_")
    doc = pypdfium2.PdfDocument(pdf_path)
    inspection = _pdf_inspect_with_doc(doc)
    effective_dpi, page_limit = _pdf_resolve_raster_plan(dpi, inspection)
    scale = float(effective_dpi) / 72.0 if effective_dpi else 1.0
    target_pages = inspection.page_count
    if page_limit is not None:
        target_pages = min(target_pages, page_limit)
    if target_pages <= 0:
        doc.close()
        return []
    workers = _pdf_parallel_workers(target_pages)
    if workers <= 1:
        return _pdfium_render_linear(doc, tmpdir, scale, target_pages)
    doc.close()
    return _pdfium_render_parallel(pdf_path, target_pages, tmpdir, scale, workers)


def pdf_to_images_via_poppler(pdf_path: str, dpi: int=200) -> List[str]:
    exe=shutil.which("pdftoppm")
    if exe:
        inspection = _pdf_lazy_inspection(pdf_path)
        effective_dpi, page_limit = _pdf_resolve_raster_plan(dpi, inspection)
        tmpdir=tempfile.mkdtemp(prefix="zocr_pdf_"); out_prefix=os.path.join(tmpdir,"page")
        cmd=[exe,"-r",str(effective_dpi),"-png"]
        if page_limit is not None:
            cmd += ["-f","1","-l",str(page_limit)]
        cmd += [pdf_path,out_prefix]
        subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return [os.path.join(tmpdir,fn) for fn in sorted(os.listdir(tmpdir)) if fn.lower().endswith(".png")]
    return _pdf_to_images_via_pdfium(pdf_path, dpi=dpi)


def detect_pdf_raster_backends() -> Dict[str, Any]:
    """Report the availability of Poppler/pdfium raster backends."""

    status: Dict[str, Any] = {"status": "missing", "active": None, "hint": None}
    poppler_path = shutil.which("pdftoppm")
    poppler_hint = "Install poppler-utils (pdftoppm) for the fastest multi-page rasterization"
    status["poppler_pdftoppm"] = {
        "status": "available" if poppler_path else "missing",
        "path": poppler_path,
        "hint": None if poppler_path else poppler_hint,
    }
    try:
        import importlib.util as _importlib_util

        pdfium_available = _importlib_util.find_spec("pypdfium2") is not None
    except Exception:
        pdfium_available = False
    pdfium_hint = "`pip install pypdfium2` to enable the builtin PDF raster fallback"
    status["pypdfium2"] = {
        "status": "available" if pdfium_available else "missing",
        "hint": None if pdfium_available else pdfium_hint,
    }
    if poppler_path:
        status["status"] = "ready"
        status["active"] = "poppler_pdftoppm"
    elif pdfium_available:
        status["status"] = "ready"
        status["active"] = "pypdfium2"
        status["hint"] = "Poppler not found; falling back to pypdfium2"
    else:
        status["hint"] = "Install poppler-utils (pdftoppm) or `pip install pypdfium2`"
    return status

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
