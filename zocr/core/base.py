"""Shared primitives for the multi-domain core."""

from __future__ import annotations

import datetime
import json
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .._compat import optional_numpy

np = optional_numpy(__name__)

try:  # pragma: no cover - optional dependency
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover - fallback stubs
    Image = None  # type: ignore
    ImageOps = None  # type: ignore

__all__ = [
    "PREFS",
    "CO_KW",
    "UNIT_KW",
    "RX_AMT",
    "RX_DATE",
    "RX_TAXID",
    "RX_POST",
    "RX_PHONE",
    "RX_PERCENT",
    "RX_CORP13",
    "_NUMERIC_HEADER_HINTS",
    "hamm64",
    "thomas_tridiag",
    "_second_diff_tridiag",
    "cc_label",
    "lambda_schedule",
    "phash64",
    "tiny_vec",
    "norm_amount",
    "norm_date",
    "norm_company",
    "norm_address",
    "parse_kv_window",
    "infer_row_fields",
    "_normalize_text",
    "detect_domain_on_jsonl",
]

_NUMERIC_HEADER_HINTS = {
    "qty": "qty",
    "q'ty": "qty",
    "quantity": "qty",
    "unit price": "unit_price",
    "price": "unit_price",
    "unit cost": "unit_price",
    "amount": "amount",
    "total": "total",
    "total amount": "total",
    "subtotal": "subtotal",
    "tax": "tax",
    "tax amount": "tax_amount",
    "tax %": "tax_rate",
    "tax%": "tax_rate",
    "tax rate": "tax_rate",
    "vat": "tax",
}
_NUMERIC_HEADER_HINTS.update({alias: "qty" for alias in (
    "数量",
    "個数",
    "台数",
    "件数",
    "口数",
    "本数",
    "点数",
    "数量/qty",
)})
_NUMERIC_HEADER_HINTS.update({alias: "unit_price" for alias in (
    "単価",
    "単価(円)",
    "単価[円]",
    "単価(税込)",
    "単価(税抜)",
    "単価(税別)",
)})
_NUMERIC_HEADER_HINTS.update({alias: "amount" for alias in (
    "金額",
    "金額(税込)",
    "金額(税抜)",
    "金額(税別)",
    "金額[円]",
    "金額(円)",
    "税込",
    "税抜",
    "税別",
)})
_NUMERIC_HEADER_HINTS.update({alias: "total" for alias in (
    "見積金額",
    "御見積金額",
    "御見積合計",
    "合計金額",
    "合計",
    "総計",
    "総額",
    "計",
)})
_NUMERIC_HEADER_HINTS.update({alias: "subtotal" for alias in ("小計", "小計(税込)", "小計(税抜)")})
_NUMERIC_HEADER_HINTS.update({alias: "tax" for alias in (
    "消費税",
    "tax",
    "vat",
    "gst",
)})
_NUMERIC_HEADER_HINTS.update({alias: "tax_amount" for alias in (
    "消費税額",
    "税額",
    "税金",
)})
_NUMERIC_HEADER_HINTS.update({alias: "tax_rate" for alias in (
    "税率",
    "消費税率",
    "税%",
    "tax率",
)})

# -------------------- Image helpers --------------------

def hamm64(a: int, b: int) -> int:
    """Return the Hamming distance between two 64-bit integers."""
    return int((int(a) ^ int(b)).bit_count())


def thomas_tridiag(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Solve a tri-diagonal system via the Thomas algorithm."""
    n = int(b.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    cp = np.zeros(max(1, n - 1), dtype=np.float64)
    dp = np.zeros(n, dtype=np.float64)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
    x = np.zeros(n, dtype=np.float64)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _second_diff_tridiag(n: int, lam: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(n)
    lam = float(max(0.0, lam))
    if n <= 0:
        return np.array([]), np.array([]), np.array([])
    a = -lam * np.ones(max(0, n - 1), dtype=np.float64)
    c = -lam * np.ones(max(0, n - 1), dtype=np.float64)
    b = np.ones(n, dtype=np.float64) + 2.0 * lam
    if n >= 1:
        b[0] = 1.0 + lam
        b[-1] = 1.0 + lam
    return a, b, c


def cc_label(bw: np.ndarray, max_boxes: int = 65536) -> List[Tuple[int, int, int, int]]:
    """Simple BFS based connected component labeling for binary images."""
    H, W = bw.shape
    lab = np.zeros((H, W), dtype=np.int32)
    cur = 0
    boxes: List[Tuple[int, int, int, int]] = []
    for y in range(H):
        for x in range(W):
            if bw[y, x] == 1 and lab[y, x] == 0:
                cur += 1
                q = [(x, y)]
                lab[y, x] = cur
                x1 = x2 = x
                y1 = y2 = y
                while q:
                    xx, yy = q.pop()
                    x1 = min(x1, xx)
                    x2 = max(x2, xx)
                    y1 = min(y1, yy)
                    y2 = max(y2, yy)
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = xx + dx, yy + dy
                        if 0 <= nx < W and 0 <= ny < H and bw[ny, nx] == 1 and lab[ny, nx] == 0:
                            lab[ny, nx] = cur
                            q.append((nx, ny))
                boxes.append((x1, y1, x2 + 1, y2 + 1))
                if len(boxes) >= max_boxes:
                    return boxes
    return boxes


def lambda_schedule(page_height: int, base_lambda: float, ref_height: int = 1000, alpha: float = 0.7) -> float:
    if page_height <= 0 or ref_height <= 0:
        return base_lambda
    return float(base_lambda) * ((float(page_height) / float(ref_height)) ** float(alpha))


@lru_cache(maxsize=1)
def _dct_basis_32() -> np.ndarray:
    N = 32
    x = np.arange(N, dtype=np.float64)
    k = np.arange(N, dtype=np.float64).reshape(-1, 1)
    basis = np.cos((math.pi / N) * (x + 0.5) * k).astype(np.float64, copy=False)
    return np.nan_to_num(basis, copy=False)


def phash64(img: Image.Image) -> int:
    if ImageOps is None:
        raise RuntimeError("Pillow is required for phash64")
    g = ImageOps.grayscale(img).resize((32, 32), Image.BICUBIC)
    a = np.asarray(g, dtype=np.float64)
    a = np.nan_to_num(a, copy=False)
    basis = _dct_basis_32()
    d = basis @ a @ basis.T
    d = np.nan_to_num(d, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    d += 1e-9
    blk = d[:8, :8].copy()
    blk[0, 0] = 0.0
    m = float(np.median(blk))
    bits = (blk > m).astype(np.uint8).reshape(-1)
    v = 0
    for i, bit in enumerate(bits):
        if bit:
            v |= (1 << i)
    return int(v)


def tiny_vec(img: Image.Image, n: int = 16) -> np.ndarray:
    if ImageOps is None:
        raise RuntimeError("Pillow is required for tiny_vec")
    g = ImageOps.grayscale(img).resize((n, n), Image.BICUBIC)
    v = np.asarray(g, dtype=np.float32).reshape(-1)
    v = (v - v.mean()) / (v.std() + 1e-6)
    return v

# -------------------- Normalization / Filters --------------
PREFS = [
    "北海道","青森県","岩手県","宮城県","秋田県","山形県","福島県","茨城県","栃木県","群馬県","埼玉県","千葉県","東京都","神奈川県",
    "新潟県","富山県","石川県","福井県","山梨県","長野県","岐阜県","静岡県","愛知県","三重県","滋賀県","京都府","大阪府","兵庫県",
    "奈良県","和歌山県","鳥取県","島根県","岡山県","広島県","山口県","徳島県","香川県","愛媛県","高知県","福岡県","佐賀県",
    "長崎県","熊本県","大分県","宮崎県","鹿児島県","沖縄県",
]
CO_KW = ["株式会社","有限会社","合名会社","合資会社","合同会社","Inc.","Co.","Co.,","LLC","G.K.","K.K."]
RX_AMT = re.compile(r"[¥￥$]?\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
RX_DATE = re.compile(r"(20\d{2}|19\d{2})[./\-年](0?[1-9]|1[0-2])([./\-月](0?[1-9]|[12]\d|3[01])日?)?")
RX_TAXID = re.compile(r"\bT\d{10,13}\b", re.IGNORECASE)
RX_POST = re.compile(r"\b\d{3}-\d{4}\b")
RX_PHONE = re.compile(r"\b0\d{1,3}-\d{2,4}-\d{3,4}\b")
RX_PERCENT = re.compile(r"(\d{1,2}(?:\.\d+)?)\s*%")
RX_CORP13 = re.compile(r"\b\d{13}\b")
UNIT_KW = ["個","式","セット","本","枚","箱","袋","台","pcs","set","kg","g","cm","m","h","時間","回"]

def norm_amount(s: str):
    m = RX_AMT.search(s or "")
    if not m:
        return None
    try:
        return int(float(m.group(1).replace(",", "")))
    except Exception:
        return None

def norm_date(s: str):
    m = RX_DATE.search(s or "")
    if not m:
        return None
    y = int(m.group(1))
    mo = int(m.group(2))
    d = m.group(3)
    dd = 1
    if d:
        d = re.sub(r"[^0-9]", "", d)
        if d:
            dd = int(d)
    try:
        datetime.date(y, mo, dd)
    except Exception:
        return None
    return f"{y:04d}-{mo:02d}-{dd:02d}"

def norm_company(s: str, org_dict: Optional[Dict[str, str]] = None):
    s = s or ""
    m = RX_CORP13.search(s)
    corp_id = m.group(0) if m else None
    normalized = unicodedata.normalize("NFKC", s).strip()
    if corp_id and org_dict and corp_id in org_dict:
        return org_dict[corp_id].strip(), corp_id
    for kw in CO_KW:
        if kw in normalized:
            return normalized, corp_id
    return None, corp_id

def norm_address(s: str):
    s = (s or "").strip()
    if not s:
        return None
    s = unicodedata.normalize("NFKC", s)
    for pref in PREFS:
        if s.startswith(pref):
            return s
    return None

def parse_kv_window(swin: str) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    if not swin:
        return kv
    for seg in swin.split("|"):
        seg = seg.strip()
        if "=" in seg:
            k, v = seg.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k:
                kv[k] = v
    return kv

def infer_row_fields(swin: str) -> Dict[str, Any]:
    kv = parse_kv_window(swin)
    out: Dict[str, Any] = {}
    cand = kv.get("税率") or kv.get("tax") or kv.get("tax_rate") or ""
    m = RX_PERCENT.search(cand or swin or "")
    if m:
        out["tax_rate"] = float(m.group(1)) / 100.0
    qstr = kv.get("数量") or kv.get("qty") or ""
    qm = re.search(r"\d+", qstr)
    if qm:
        out["qty"] = int(qm.group(0))
    else:
        qm = re.search(r"数量[^0-9]*?(\d+)", swin or "")
        if qm:
            out["qty"] = int(qm.group(1))
    u = kv.get("単位") or kv.get("unit") or ""
    if not u:
        for w in UNIT_KW:
            if w in (qstr or "") or w in (swin or ""):
                u = w
                break
    if u:
        out["unit"] = u
    sub = kv.get("金額") or kv.get("小計") or kv.get("subtotal") or ""
    a = norm_amount(sub)
    if a is not None:
        out["subtotal"] = a
    tax_line = kv.get("消費税") or kv.get("税額") or kv.get("tax_amount") or ""
    ta = norm_amount(tax_line)
    if ta is not None:
        out["tax_amount"] = ta
    return out

def _normalize_text(val: Optional[Any]) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip().lower()
    return str(val).strip().lower()

def detect_domain_on_jsonl(
    jsonl_path: str,
    filename_tokens: Optional[Sequence[Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    from .domains import DOMAIN_KW  # local import to avoid cycles

    kw = DOMAIN_KW
    counts = Counter({dom: 0.0 for dom in kw})
    def _accumulate(text: str, weight: float = 1.0) -> None:
        text = (text or "").lower()
        for dom, entries in kw.items():
            for phrase, w in entries:
                if phrase in text:
                    counts[dom] += weight * w
    if filename_tokens:
        for tok in filename_tokens:
            _accumulate(str(tok), weight=2.0)
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fr:
            for line in fr:
                if not line.strip():
                    continue
                try:
                    ob = json.loads(line)
                except Exception:
                    continue
                meta = ob.get("meta") or {}
                _accumulate(meta.get("title") or "")
                _accumulate(meta.get("domain_hint") or "")
                for field in (ob.get("text"), meta.get("filters")):
                    if isinstance(field, dict):
                        for val in field.values():
                            _accumulate(str(val))
                    elif isinstance(field, str):
                        _accumulate(field)
    except FileNotFoundError:
        pass
    domain = max(counts.items(), key=lambda kv: kv[1])[0]
    return domain, {"scores": counts}
