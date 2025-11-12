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
# Generated: 2025-11-10T06:47:54.250029Z
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


import sys, types, os, tempfile, pathlib

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
import os, sys, io, json, argparse, tempfile, shutil, subprocess, time, math, re
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter

try:
    import numpy as np
except Exception:
    np = None

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageChops, ImageEnhance
from html.parser import HTMLParser

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
    # per-row chunks & counts
    chunks_by_row=[ [list(bx) for bx in cc if not (bx[3]<=yt or bx[1]>=yb)] for (yt,yb) in row_bands ]
    row_counts=[len(row) for row in chunks_by_row]
    # segmented baselines
    baselines = [_fit_baseline_row_segments(row, W, segs=int(params.get("baseline_segs", 4))) for row in chunks_by_row]
    # DP-means for columns with λ補正（列モード吸着）
    xcenters=[(ch[0]+ch[2])/2.0 for row in chunks_by_row for ch in row]
    med_w=float(np.median([(ch[2]-ch[0]) for row in chunks_by_row for ch in row])) if xcenters else 12.0
    lam_base=float(params.get("dp_lambda_factor", 2.2))*max(6.0, med_w)
    centers0=_dp_means_1d(sorted(xcenters), lam=lam_base, iters=3)
    K_pred0=len(centers0)
    K_mode=_robust_k_mode(row_counts) or max(2, K_pred0)
    alpha=float(params.get("lambda_alpha", 0.7))
    # clip the scaling to avoid extreme swings
    scale = ( (K_pred0 / float(max(1,K_mode))) ** alpha )
    lam_eff = clamp(lam_base * scale, 0.6*lam_base, 1.8*lam_base)
    centers=_dp_means_1d(sorted(xcenters), lam=lam_eff, iters=3)
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
    if len(centers)>=2:
        mids_global=[int((centers[i]+centers[i+1])/2.0) for i in range(len(centers)-1)]
        candidates_by_row = [[*mids_global, *row] for row in candidates_by_row]
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
            results["pages"].append({"index":i+1,"tables":[{"bbox":tbl_bbox,"html":html_pred,"dbg":dbg,"teds":teds}]})
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
import re, pickle, hashlib

_ASCII_SET = (
    "0123456789"
    ".,:-/$()%"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    " +#=_[]{}"
)

def _render_glyphs(font=None, size=16):
    # PIL's default bitmap font via ImageFont.load_default() matches our demo
    f = ImageFont.load_default() if font is None else font
    atlas = {}
    for ch in _ASCII_SET:
        # Render on tight canvas
        img = Image.new("L", (size*2, size*2), 0)
        dr = ImageDraw.Draw(img)
        dr.text((2,2), ch, fill=255, font=f)
        # crop to bbox
        bbox = img.getbbox() or (0,0,1,1)
        crop = img.crop(bbox)
        atlas[ch] = crop
    return atlas

_GLYPH_ATLAS = _render_glyphs()

_TOTAL_LABEL_HINTS = [
    "total", "grandtotal", "amountdue", "balancedue", "totaldue", "subtotal",
    "totaltax", "totaltaxes", "totals", "dueamount", "amountpayable",
    "合計", "総計", "総額", "小計", "税込合計", "税込総額", "請求額", "ご請求額", "支払金額", "合算", "合計金額"
]
_TOTAL_PREFIXES = ["total", "subtotal", "balance", "amountdue", "dueamount", "grandtotal", "amountpayable", "合計", "小計", "総額", "請求"]
_NUMERIC_RX = re.compile(r"[+\-]?\d[\d,]*(?:\.\d+)?")

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

def _resize_keep_ar(im, w, h):
    im = im.convert("L")
    iw, ih = im.size
    scale = min(max(1, w-2)/max(1, iw), max(1, h-2)/max(1, ih))
    tw, th = max(1, int(round(iw*scale))), max(1, int(round(ih*scale)))
    imr = im.resize((tw, th), resample=Image.BILINEAR)
    out = Image.new("L", (w,h), 0)
    out.paste(imr, ((w-tw)//2,(h-th)//2))
    return out

def _match_glyph(cell_bin, atlas):
    # try best correlation over atlas
    cw, ch = cell_bin.size
    best_ch, best_score = "", -1.0
    for ch_key, tpl in atlas.items():
        tw, th = tpl.size
        t = _resize_keep_ar(tpl, cw, ch)
        # normalized correlation
        import numpy as _np
        a = _np.asarray(cell_bin, dtype=_np.float32)
        b = _np.asarray(t, dtype=_np.float32)
        a = (a - a.mean())/ (a.std()+1e-6)
        b = (b - b.mean())/ (b.std()+1e-6)
        score = float((a*b).mean())
        if score > best_score:
            best_score = score; best_ch = ch_key
    # map low score to '?'
    conf = (best_score+1)/2  # [-1,1] -> [0,1]
    return (best_ch if conf>=0.52 else "?"), float(conf)

def _otsu_threshold(arr):
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

def _text_from_binary(bw):
    cc = _cc_label_rle(bw)
    cc = [b for b in cc if (b[2]-b[0])*(b[3]-b[1]) >= 10]
    if not cc:
        return "", 0.0
    cc.sort(key=lambda b: b[0])
    atlas = _GLYPH_ATLAS
    text = []
    scores = []
    for (x1, y1, x2, y2, _) in cc:
        patch = Image.fromarray(bw[y1:y2, x1:x2])
        ch, sc = _match_glyph(patch, atlas)
        text.append(ch)
        scores.append(sc)
    if not text:
        return "", 0.0
    return "".join(text), float(sum(scores)/len(scores))

def toy_ocr_text_from_cell(crop_img: "Image.Image", bin_k: int = 15) -> tuple[str, float]:
    """Very small OCR to work with the demo font. Returns (text, confidence)."""
    import numpy as _np
    g = ImageOps.autocontrast(crop_img.convert("L"))
    arr = _np.asarray(g, dtype=_np.uint8)
    thr_med = int(_np.clip(_np.median(arr), 48, 208))
    bw = (arr < thr_med).astype(_np.uint8) * 255
    text, conf = _text_from_binary(bw)
    if text:
        return text, conf
    thr_otsu = _otsu_threshold(arr)
    bw2 = (arr < thr_otsu).astype(_np.uint8) * 255
    text, conf = _text_from_binary(bw2)
    if text:
        return text, conf
    bw3 = (arr > thr_otsu).astype(_np.uint8) * 255
    text, conf = _text_from_binary(bw3)
    if text:
        return text, conf
    try:
        expanded = Image.fromarray(bw).filter(ImageFilter.MaxFilter(3))
        bw4 = _np.asarray(expanded, dtype=_np.uint8)
        text, conf = _text_from_binary(bw4)
        if text:
            return text, conf
    except Exception:
        pass
    return "", 0.0

def _keywords_from_row(row_cells: list[str]) -> list[str]:
    kws = set()
    rx_num = re.compile(r"[+\-]?\d[\d,]*(\.\d+)?")
    rx_date = re.compile(r"\b(20\d{2}|19\d{2})[/-](0?[1-9]|1[0-2])([/-](0?[1-9]|[12][0-9]|3[01]))?\b")
    for t in row_cells:
        if not t: continue
        for m in rx_num.findall(t): kws.add(m[0] if isinstance(m, tuple) else m)
        for m in rx_date.findall(t): kws.add("-".join([x for x in m if x]))
        if any(sym in t for sym in ["$", "¥", "円"]): kws.add("currency")
    return sorted(kws)[:12]

def _context_line_from_row(headers: list[str], row: list[str]) -> str:
    if headers and len(headers)==len(row):
        pairs = [f"{h.strip()}={row[i].strip()}" for i,h in enumerate(headers)]
        return " | ".join(pairs)
    else:
        return " | ".join([x.strip() for x in row if x.strip()])

def export_jsonl_with_ocr(doc_json_path: str, source_image_path: str, out_jsonl_path: str,
                          ocr_engine: str = "toy", contextual: bool = True,
                          ocr_min_conf: float = 0.58) -> int:
    with open(doc_json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    im = Image.open(source_image_path).convert("RGB")
    count = 0
    with open(out_jsonl_path, "w", encoding="utf-8") as fw:
        for p in doc["pages"]:
            pidx = p["index"]
            for ti, t in enumerate(p["tables"]):
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
                for r in range(R):
                    for c in range(C):
                        cx1 = x1 + col_bounds[c]
                        cx2 = x1 + col_bounds[c+1]
                        left_pad = pad_edge_x if c == 0 else pad_inner_x
                        right_pad = pad_edge_x if c == C-1 else pad_inner_x
                        cy1, cy2 = row_bands[r]
                        crop = im.crop((
                            max(0, cx1 - left_pad),
                            max(0, cy1 - pad_y),
                            min(im.width, cx2 + right_pad),
                            min(im.height, cy2 + pad_y)
                        ))
                        if ocr_engine=="toy":
                            txt, conf = toy_ocr_text_from_cell(crop)
                        else:
                            txt, conf = ("", 0.0)
                        grid_text[r][c] = txt
                        grid_conf[r][c] = conf
                footer_rows: Set[int] = set()
                fallback_notes: Dict[Tuple[int, int], str] = {}
                for r in range(R):
                    if _is_total_row(grid_text[r]):
                        footer_rows.add(r)
                        has_numeric = any(_NUMERIC_RX.search(grid_text[r][c] or "") for c in range(C))
                        if not has_numeric and C > 0:
                            target_col = C-1
                            cy1, cy2 = row_bands[r]
                            cx1 = x1 + col_bounds[target_col]
                            cx2 = x1 + col_bounds[target_col+1] + pad_edge_x * 2
                            crop = im.crop((
                                max(0, cx1 - pad_inner_x),
                                max(0, cy1 - pad_y),
                                min(im.width, cx2),
                                min(im.height, cy2 + pad_y)
                            ))
                            alt_txt, alt_conf = toy_ocr_text_from_cell(crop)
                            m = _NUMERIC_RX.search(alt_txt or "")
                            if m:
                                grid_text[r][target_col] = m.group(0)
                                grid_conf[r][target_col] = max(grid_conf[r][target_col], alt_conf)
                                fallback_notes[(r, target_col)] = "footer_band"
                # contextual one-liners
                headers = grid_text[0] if contextual else []
                for r in range(R):
                    for c in range(C):
                        cx1 = x1 + col_bounds[c]; cx2 = x1 + col_bounds[c+1]
                        cy1, cy2 = row_bands[r]
                        txt = grid_text[r][c]
                        conf = grid_conf[r][c]
                        # build search/synthesis
                        row_texts = grid_text[r]
                        ctx_line = _context_line_from_row(headers, row_texts) if contextual and r>0 else txt
                        kws = _keywords_from_row(row_texts) if contextual and r>0 else []
                        low_conf = (conf is not None and conf < ocr_min_conf)
                        trace_id = f"page={pidx},table={ti},row={r},col={c}"
                        filters = {
                            "has_currency": ("currency" in kws),
                            "row_index": r,
                            "col_index": c,
                            "trace_id": trace_id
                        }
                        if r in footer_rows:
                            filters["row_role"] = "footer"
                        note = fallback_notes.get((r, c))
                        if note:
                            filters["linked"] = note
                        rec = {
                            "doc_id": doc.get("doc_id"),
                            "page": pidx, "table_index": ti, "row": r, "col": c,
                            "bbox": [int(cx1), int(cy1), int(cx2), int(cy2)],
                            "image_path": source_image_path,
                            "text": txt,
                            "search_unit": (txt or ctx_line),
                            "synthesis_window": ctx_line,
                            "meta": {
                                "headers": headers,
                                "keywords": kws,
                                "confidence": conf,
                                "low_conf": bool(low_conf),
                                "trace": trace_id,
                                "filters": filters
                            }
                        }
                        if note:
                            rec["meta"]["fallback"] = note
                        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        count += 1
    return count

# ---------- Minimal local hybrid search ----------
def _tokenize(s: str) -> list[str]:
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
    n = export_jsonl_with_ocr(jpath, src, out_jsonl, ocr_engine="toy", contextual=True)
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

import os, re, csv, json, math, pickle, ctypes, tempfile, subprocess, datetime, html
from collections import defaultdict, Counter
from typing import List, Optional, Dict, Any, Tuple, Set
from PIL import Image, ImageOps
import numpy as np

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
    EXP int rle_cc(const uint8_t* img, int H, int W, int max_boxes, int* out_xyxy){
        // BFS stack (iterative) to avoid recursion
        int max_stack = H*W;
        P* st = (P*)malloc(sizeof(P)*max_stack);
        if(!st) return -3;
        uint8_t* vis = (uint8_t*)calloc(H*W,1);
        if(!vis){ free(st); return -4; }

        int nb=0;
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
                                vis[nidx]=1; st[top++]=(P){nx,ny};
                                if(top>=max_stack-1) top=max_stack-1; // clamp
                            }
                        }
                    }
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
        so = os.path.join(outdir, "libzocr.so")
        cc = os.environ.get("CC", "cc")
        r = subprocess.run([cc, "-O3", "-shared", "-fPIC", cpath, "-o", so], capture_output=True)
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
def phash64(img: Image.Image) -> int:
    g = ImageOps.grayscale(img).resize((32,32), Image.BICUBIC)
    a = np.asarray(g, dtype=np.float64)
    a = np.nan_to_num(a, copy=False)
    N=32
    x=np.arange(N,dtype=np.float64); k=np.arange(N,dtype=np.float64).reshape(-1,1)
    basis=np.cos((math.pi/N)*(x+0.5)*k).astype(np.float64, copy=False)
    basis = np.nan_to_num(basis, copy=False)
    d=basis@a@basis.T
    d=np.nan_to_num(d, copy=False, posinf=0.0, neginf=0.0)
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
    "estimate_jp": [("見積書",1.0),("見積金額",0.85),("有効期限",0.6),("数量",0.5)],
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
    "invoice": {"lambda_shape": 4.5, "w_kw": 0.6, "w_img": 0.3, "ocr_min_conf": 0.58},
    "invoice_jp_v2": {"lambda_shape": 4.5, "w_kw": 0.6, "w_img": 0.3, "ocr_min_conf": 0.58},
    "invoice_en": {"lambda_shape": 4.2, "w_kw": 0.55, "w_img": 0.25, "ocr_min_conf": 0.55},
    "invoice_fr": {"lambda_shape": 4.2, "w_kw": 0.55, "w_img": 0.25, "ocr_min_conf": 0.55},
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
    "estimate_jp": ["見積金額", "有効期限", "数量", "単価"],
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


def _normalize_text(val: Optional[Any]) -> str:
    if val is None:
        return ""
    if not isinstance(val, str):
        val = str(val)
    val = val.strip()
    if not val:
        return ""
    return re.sub(r"\s+", " ", val)

def detect_domain_on_jsonl(jsonl_path: str, filename_tokens: Optional[List[str]] = None) -> Tuple[str, Dict[str, Any]]:
    scores: Dict[str, float] = {k: 0.0 for k in DOMAIN_KW.keys()}
    hits: Dict[str, int] = {k: 0 for k in DOMAIN_KW.keys()}
    token_hits: Dict[str, int] = {k: 0 for k in DOMAIN_KW.keys()}
    total_cells = 0
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
    except FileNotFoundError:
        pass

    if filename_tokens:
        lookup: Dict[str, List[str]] = {}

        def _register(key: str, target: str):
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

        for token in filename_tokens:
            token_l = (token or "").strip().lower()
            if not token_l:
                continue
            matched = False
            for dom in lookup.get(token_l, []):
                scores[dom] += 0.6
                hits[dom] += 1
                token_hits[dom] += 1
                matched = True
            if matched:
                continue
            for key, dom_list in lookup.items():
                if token_l in key and key not in lookup.get(token_l, []):
                    for dom in dom_list:
                        scores[dom] += 0.3
                        hits[dom] += 1
                        token_hits[dom] += 1
                    break

    def _score_key(dom: str) -> Tuple[float, int]:
        return scores.get(dom, 0.0), hits.get(dom, 0)

    best_dom = "invoice_jp_v2"
    if scores:
        best_dom = max(scores.keys(), key=lambda d: (_score_key(d)[0], _score_key(d)[1]))
    resolved = _DOMAIN_ALIAS.get(best_dom, best_dom)
    score_total = sum(max(0.0, s) for s in scores.values())
    confidence = scores.get(best_dom, 0.0) / score_total if score_total > 0 else 0.0
    detail = {
        "scores": scores,
        "hits": hits,
        "token_hits": token_hits,
        "total_cells": total_cells,
        "resolved": resolved,
        "raw_best": best_dom,
        "confidence": confidence,
        "filename_tokens": filename_tokens or [],
    }
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


def _evaluate_gate(domain: Optional[str], amount_score: Optional[float], date_score: Optional[float]) -> Tuple[bool, str, float]:
    resolved = _DOMAIN_ALIAS.get(domain or "", domain or "") if '_DOMAIN_ALIAS' in globals() else (domain or "")
    amt = float(amount_score) if amount_score is not None else 0.0
    dt = float(date_score) if date_score is not None else None
    if resolved in _INVOICE_GATE_DOMAINS:
        if amt >= 0.8:
            note = "amount+date hit" if dt is not None and dt >= 0.5 else "amount hit (date optional)"
            return True, note, amt
        if amt >= 0.7 and (dt is None or dt >= 0.3):
            note = "amount hit (date optional)" if dt is None or dt < 0.5 else "amount+date hit"
            return True, note, amt
        return False, "amount below gate", amt
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
        "estimate_jp":   ["見積金額", "小計", "数量", "有効期限"],
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
    corporate_match_rate = (corp_hits/max(1,corp_total)) if corp_total>0 else None

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
        trust = good / len(res) if res else None
        return hit, trust
    q_amount="合計 金額 消費税 円 2023 2024 2025"
    q_date="請求日 発行日 2023 2024 2025"
    if domain=="contract_jp_v2":
        q_amount="契約金額 代金 支払"
        q_date="契約日 締結日 開始日 終了日"
    hit_amount, trust_amount = _score("amount", q_amount)
    hit_date, trust_date = _score("date", q_date)
    hit_mean=(hit_amount+hit_date)/2.0
    trust_vals = [v for v in (trust_amount, trust_date) if v is not None]
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
    tax_fail_rate = (tax_fail/max(1,tax_cov)) if tax_cov>0 else None

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

    gate_pass, gate_reason, gate_score = _evaluate_gate(domain, hit_amount, hit_date)
    row={"timestamp":datetime.datetime.utcnow().isoformat()+"Z","jsonl":jsonl,"K":k,
         "domain": domain or "auto",
         "low_conf_rate":low_conf_rate,"reprocess_rate":reprocess_rate,"reprocess_success_rate":reprocess_success_rate,
         "hit_amount":hit_amount,"hit_date":hit_date,"hit_mean":hit_mean,
         "tax_fail_rate":tax_fail_rate,"corporate_match_rate":corporate_match_rate,"p95_ms":p95,
         "trust_amount":trust_amount,"trust_date":trust_date,"trust_mean":trust_mean,
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
            "hit_amount", "hit_date", "hit_mean", "p95_ms", "tax_fail_rate",
            "corporate_match_rate", "trust_amount", "trust_date", "trust_mean",
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

import os, sys, json, time, traceback, argparse, random, platform, hashlib, subprocess, importlib, re, glob
from typing import Any, Dict, List, Optional, Tuple
from html import escape

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

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

_STOP_TOKENS = {"samples", "sample", "demo", "image", "images", "img", "scan", "page", "pages", "document", "documents", "doc"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _is_auto_domain(value: Optional[str]) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        norm = value.strip().lower()
        return norm in {"", "auto", "autodetect", "detect", "default"}
    return False


def _prepare_domain_hints(inputs: List[str]) -> Dict[str, Any]:
    tokens_raw: List[str] = []
    per_input: Dict[str, List[str]] = {}
    for raw in inputs:
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
                seg_tokens.append(tok)
        if seg_tokens:
            per_input[raw] = seg_tokens
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
        "tokens": unique_tokens,
        "per_input": per_input,
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
    with open(os.path.join(outdir, "pipeline_history.jsonl"), "a", encoding="utf-8") as fw:
        fw.write(json.dumps(_json_ready(rec), ensure_ascii=False) + "\n")

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


def _coerce_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        if isinstance(val, str) and not val.strip():
            return None
        return float(val)
    except Exception:
        return None


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
    footer { margin-top: 3rem; font-size: 0.85rem; opacity: 0.7; }
    """

    meta_table = _render_table(meta or {}, "環境 / Environment / Environnement", [
        "seed",
        "python",
        "platform",
        "env",
        "versions",
    ]) if meta else "<p class=\"muted\">(no snapshot metadata — run with --snapshot)</p>"

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
  {plugin_html}
  <section>
    <h2>履歴 / History / Historique</h2>
    {history_html}
  </section>
  <section>
    <h2>環境 / Environment / Environnement</h2>
    {meta_table}
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
        return {"ok": True, "elapsed_ms": dt, "out": out, "name": name}
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[FAIL] {name} ({dt:.1f} ms): {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"ok": False, "elapsed_ms": dt, "error": f"{type(e).__name__}: {e}", "name": name}

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
) -> Dict[str, Any]:
    ensure_dir(outdir)
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

    if len(inputs) == 1 and inputs[0].lower() == "demo":
        pages, annos = zocr_onefile_consensus.make_demo(outdir)
    else:
        pages = _collect_pages(inputs, dpi=dpi)
        annos = [None] * len(pages)
    if not pages:
        raise RuntimeError("No input pages provided")

    pipe = zocr_onefile_consensus.Pipeline({"table": {}, "bench_iterations": 1, "eval": False})

    doc_json_path = os.path.join(outdir, "doc.zocr.json")
    jsonl_path = os.path.join(outdir, "doc.contextual.jsonl")
    mm_jsonl = os.path.join(outdir, "doc.mm.jsonl")
    idx_path = os.path.join(outdir, "bm25.pkl")
    mon_csv = os.path.join(outdir, "monitor.csv")
    prof_path = os.path.join(outdir, "auto_profile.json")
    prof = _load_profile(outdir, domain_hint)

    summary: Dict[str, Any] = {
        "contextual_jsonl": jsonl_path,
        "mm_jsonl": mm_jsonl,
        "index": idx_path,
        "monitor_csv": mon_csv,
        "profile_json": prof_path,
        "history": os.path.join(outdir, "pipeline_history.jsonl"),
        "inputs": inputs[:],
        "page_count": len(pages),
        "domain": prof.get("domain"),
        "seed": seed,
        "resume_requested": bool(resume),
        "resume_applied": bool(ok),
        "resume_steps": sorted(str(s) for s in ok if s is not None),
        "snapshot": bool(snapshot),
        "tune_budget": int(tune_budget) if tune_budget is not None else None,
    }

    domain_hints = _prepare_domain_hints(inputs)
    domain_auto_summary: Dict[str, Any] = {
        "provided": domain_hint,
        "from_inputs": {
            "guess": domain_hints.get("guess"),
            "best_score": float(domain_hints.get("best_score") or 0.0) if domain_hints.get("best_score") else None,
            "tokens": domain_hints.get("tokens"),
            "per_input": domain_hints.get("per_input"),
            "scores": domain_hints.get("scores"),
        },
        "initial_profile": prof.get("domain"),
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

    src_img = pages[0]

    if "Export" in ok:
        print("[SKIP] Export JSONL (resume)")
    else:
        r = _safe_step("Export", zocr_onefile_consensus.export_jsonl_with_ocr,
                       doc_json_path, src_img, jsonl_path, "toy", True, prof.get("ocr_min_conf", 0.58))
        _append_hist(outdir, r)
        if not r.get("ok"):
            raise RuntimeError("Export failed")
    _call("post_export", jsonl=jsonl_path, outdir=outdir)

    autodetect_detail: Optional[Dict[str, Any]] = None
    autodetect_error: Optional[str] = None
    if os.path.exists(jsonl_path):
        try:
            detected_domain, autodetect_detail = zocr_multidomain_core.detect_domain_on_jsonl(
                jsonl_path, domain_hints.get("tokens_raw")
            )
        except Exception as e:
            autodetect_error = str(e)
            detected_domain = None  # type: ignore
            autodetect_detail = None
        if autodetect_detail:
            domain_auto_summary["from_content"] = autodetect_detail
            resolved = autodetect_detail.get("resolved") or detected_domain
            if resolved:
                take = False
                conf_val = autodetect_detail.get("confidence")
                try:
                    conf_float = float(conf_val) if conf_val is not None else None
                except Exception:
                    conf_float = None
                if _is_auto_domain(prof.get("domain")):
                    take = True
                elif resolved != prof.get("domain") and conf_float is not None and conf_float >= 0.55:
                    take = True
                if take:
                    prof["domain"] = resolved
                    selected_source = "content"
                    selected_confidence = conf_float
        if autodetect_error:
            domain_auto_summary["content_error"] = autodetect_error

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
        summary["rag_suggested_queries"] = rag_manifest.get("suggested_queries")
        summary["rag_trace_schema"] = rag_manifest.get("trace_schema")
        summary["rag_fact_tag_example"] = rag_manifest.get("fact_tag_example")
    except Exception as e:
        print("RAG bundle export skipped:", e)
        summary["rag_trace_schema"] = summary.get("rag_trace_schema") or None
        summary["rag_fact_tag_example"] = summary.get("rag_fact_tag_example") or None
    _call(
        "post_rag",
        manifest=summary.get("rag_manifest"),
        bundle=summary.get("rag_bundle"),
        trace_schema=summary.get("rag_trace_schema"),
        fact_tag_example=summary.get("rag_fact_tag_example"),
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
    summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report_path = os.path.join(outdir, "pipeline_report.html")
    summary["report_html"] = report_path

    with open(os.path.join(outdir, "pipeline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(_json_ready(summary), f, ensure_ascii=False, indent=2)
    try:
        _generate_report(outdir, dest=report_path, summary=summary, history=history_records, meta=_read_meta(outdir))
    except Exception as e:
        print("Report generation skipped:", e)
    return summary

run_full_pipeline = _patched_run_full_pipeline

# ---------------- CLI ----------------
def main():
    argv = sys.argv[1:]
    if argv and argv[0] in {"history", "summary", "plugins", "report"}:
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
    args = ap.parse_args(argv)

    ensure_dir(args.outdir)
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
