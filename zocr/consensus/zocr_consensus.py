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
import re, pickle, hashlib, math

_ASCII_SET = (
    "0123456789"
    ".,:-/$()%"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    " +#=_[]{}"
)

_GLYPH_VARIANT_LIMIT = 6
_THRESHOLD_MEMORY: Dict[Tuple[int, int, int], int] = {}
_NGRAM_COUNTS: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
_NGRAM_TOTALS: Dict[str, int] = defaultdict(int)
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
        return {"aspect": 1.0, "density": 0.0, "symmetry": 0.0, "count": 0}
    if arr_f.max() > 1.5:
        arr_f = arr_f / 255.0
    h, w = arr_f.shape
    aspect = float(w) / float(h or 1)
    density = float(arr_f.mean())
    flipped = np.flip(arr_f, axis=1) if arr_f.ndim == 2 else arr_f
    symmetry = 1.0 - float(np.mean(np.abs(arr_f - flipped))) if arr_f.size else 0.0
    return {"aspect": aspect, "density": density, "symmetry": symmetry, "count": 1}

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

def _blend_glyph_features(ch: str, feats: Dict[str, float]) -> None:
    if not feats:
        return
    cur = _GLYPH_FEATS.setdefault(ch, {"aspect": feats.get("aspect", 1.0),
                                        "density": feats.get("density", 0.0),
                                        "symmetry": feats.get("symmetry", 0.0),
                                        "count": feats.get("count", 1) or 1})
    count = max(1, int(cur.get("count", 1)))
    new_count = min(_GLYPH_VARIANT_LIMIT, count + 1)
    alpha = 1.0 / float(min(count + 1, _GLYPH_VARIANT_LIMIT))
    for key in ("aspect", "density", "symmetry"):
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

def _ngram_coherence(text: str) -> float:
    if len(text) < 2 or not _NGRAM_TOTALS:
        return 0.0
    prev = "\0"
    total_pairs = 0
    log_sum = 0.0
    vocab = len(_ASCII_SET) + 4
    for ch in text:
        totals = _NGRAM_TOTALS.get(prev, 0)
        counts = _NGRAM_COUNTS.get(prev)
        prob = None
        if counts and totals:
            prob = (counts.get(ch, 0) + 1.0) / (totals + vocab)
        elif counts:
            prob = (counts.get(ch, 0) + 1.0) / (len(counts) + vocab)
        if prob is not None:
            log_sum += math.log(prob)
            total_pairs += 1
        prev = ch
    if total_pairs == 0:
        return 0.0
    return float(math.exp(log_sum / total_pairs))

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
    for ch in text:
        _NGRAM_COUNTS[prev][ch] += 1
        _NGRAM_TOTALS[prev] += 1
        prev = ch

def _self_augment_views(arr: "np.ndarray", best_bw: Optional["np.ndarray"]) -> List[Tuple["np.ndarray", Dict[str, Any]]]:
    variants: List[Tuple["np.ndarray", Dict[str, Any]]] = []
    try:
        gray_img = Image.fromarray(arr.astype(np.uint8)) if arr is not None else None
    except Exception:
        gray_img = None
    if best_bw is not None:
        try:
            bw_img = Image.fromarray(best_bw.astype(np.uint8))
            for size in (3, 5):
                try:
                    variants.append((np.asarray(bw_img.filter(ImageFilter.MaxFilter(size)), dtype=np.uint8), {"type": "augment_max", "size": size}))
                    variants.append((np.asarray(bw_img.filter(ImageFilter.MinFilter(size)), dtype=np.uint8), {"type": "augment_min", "size": size}))
                except Exception:
                    continue
        except Exception:
            pass
    if gray_img is not None:
        fill = int(float(np.median(arr))) if arr.size else 0
        for angle in (-3, -1, 1, 3):
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

def _match_glyph(cell_bin, atlas):
    # try best correlation over atlas with light shift tolerance and feature penalties
    cw, ch = cell_bin.size
    import numpy as _np
    cell_arr = _np.asarray(cell_bin, dtype=_np.float32)
    if cell_arr.size == 0:
        return "", 0.0
    cell_norm = (cell_arr - cell_arr.mean()) / (cell_arr.std() + 1e-6)
    cell_density = float((cell_arr > 0).mean())
    cell_aspect = float(cw) / float(ch or 1)
    best_ch, best_score = "", -1.0
    for ch_key, tpl in atlas.items():
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
        aspect_penalty = math.exp(-abs(math.log((cell_aspect + 1e-3)/(glyph_aspect + 1e-3))) * 0.75)
        density_penalty = 1.0 - min(0.4, abs(cell_density - glyph_density) * 1.6)
        if glyph_sym > 0.5:
            sym_cell = float(1.0 - _np.mean(_np.abs(cell_arr - _np.flip(cell_arr, axis=1))) / 255.0)
            symmetry_penalty = 0.8 + 0.2 * max(0.0, sym_cell)
        else:
            symmetry_penalty = 1.0
        variant_best *= aspect_penalty * max(0.4, density_penalty) * symmetry_penalty
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

def _text_from_binary(bw):
    cc = _cc_label_rle(bw)
    cc = [b for b in cc if (b[2]-b[0])*(b[3]-b[1]) >= 10]
    if not cc:
        return "", 0.0
    refined: List[Tuple[int,int,int,int,float]] = []
    for (x1, y1, x2, y2, area) in cc:
        sub = bw[y1:y2, x1:x2]
        if sub.size == 0:
            continue
        if (x2 - x1) > max(2, int(1.8 * (y2 - y1))):
            col_density = (sub > 0).sum(axis=0)
            gap_thr = max(1, int(0.12 * sub.shape[0]))
            mask = col_density <= gap_thr
            segments = []
            idx = 0
            L = mask.shape[0]
            while idx < L:
                while idx < L and mask[idx]:
                    idx += 1
                if idx >= L:
                    break
                j = idx
                while j < L and not mask[j]:
                    j += 1
                if j - idx >= max(2, int(0.4 * (y2 - y1))):
                    segments.append((x1 + idx, y1, x1 + j, y2))
                idx = j
            if segments:
                refined.extend([(sx1, sy1, sx2, sy2, (sx2 - sx1) * (sy2 - sy1)) for (sx1, sy1, sx2, sy2) in segments])
                continue
        refined.append((x1, y1, x2, y2, area))
    if not refined:
        refined = cc
    refined.sort(key=lambda b: b[0])
    atlas = _GLYPH_ATLAS
    text = []
    scores = []
    for (x1, y1, x2, y2, _) in refined:
        patch = Image.fromarray(bw[y1:y2, x1:x2])
        ch, sc = _match_glyph(patch, atlas)
        if ch and ch != "?" and sc > 0.6:
            _adapt_glyph(ch, patch)
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

def toy_ocr_text_from_cell(crop_img: "Image.Image", bin_k: int = 15) -> tuple[str, float]:
    """Very small OCR to work with the demo font. Returns (text, confidence)."""
    import numpy as _np
    g = ImageOps.autocontrast(crop_img.convert("L"))
    g = g.filter(ImageFilter.MedianFilter(3))
    arr = _np.asarray(g, dtype=_np.uint8)
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
    if key in _THRESHOLD_MEMORY:
        thr_mem = int(_THRESHOLD_MEMORY[key])
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
    thr_min = max(16, thr_med - 60)
    thr_max = min(240, thr_med + 70)
    for thr_candidate in range(thr_min, thr_max + 1, 10):
        _add_candidate((arr < thr_candidate).astype(_np.uint8) * 255, {"type": "sweep", "thr": thr_candidate})
    spread = int(max(6, arr.std()))
    for delta in (-spread, spread):
        thr_val = int(_np.clip(thr_med + delta, 16, 240))
        _add_candidate((arr < thr_val).astype(_np.uint8) * 255, {"type": "sweep_local", "thr": thr_val})

    candidate_scores: Dict[str, float] = {}
    best_text, best_conf = "", 0.0
    best_meta: Optional[Dict[str, Any]] = None
    best_bw: Optional[np.ndarray] = None

    def _evaluate_from(idx: int) -> int:
        nonlocal best_text, best_conf, best_meta, best_bw
        total = len(candidates)
        for i in range(idx, total):
            bw, meta = candidates[i]
            text, conf = _text_from_binary(bw)
            if text:
                prev = candidate_scores.get(text)
                if prev is None or conf > prev:
                    candidate_scores[text] = conf
            if text and conf > best_conf:
                best_text, best_conf, best_meta, best_bw = text, conf, meta, bw
        return total

    _evaluate_from(0)
    if best_conf < 0.5:
        start_len = len(candidates)
        for bw_aug, meta_aug in _self_augment_views(arr, best_bw):
            _add_candidate(bw_aug, meta_aug)
        _evaluate_from(start_len)

    if best_meta and best_meta.get("thr") is not None:
        _THRESHOLD_MEMORY[key] = int(best_meta["thr"])

    final_text, final_conf = best_text, best_conf
    if candidate_scores:
        reranked_text, reranked_conf = _contextual_rerank_candidates(candidate_scores)
        if reranked_text:
            if reranked_conf >= final_conf - 1e-6:
                final_text = reranked_text
                final_conf = max(final_conf, reranked_conf)
            elif not final_text:
                final_text, final_conf = reranked_text, reranked_conf

    if final_text:
        _update_ngram_model(final_text)
    if final_text:
        return final_text, float(max(0.0, min(1.0, final_conf)))
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
