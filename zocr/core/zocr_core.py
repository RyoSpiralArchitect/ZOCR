
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

if __name__.startswith("zocr."):
    sys.modules.setdefault("zocr_multidomain_core", sys.modules[__name__])

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
            # try fuzzy match when no direct key
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
    seen_keys = set()
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
            seen_keys.add(key)
            score += key_hits
    if seen_keys:
        score += min(len(seen_keys), 4) * 0.5
    return float(score)


def query(index_pkl: str, jsonl: str, q_text: str="", q_image: Optional[str]=None, topk:int=10,
          w_bm25:float=1.0, w_kw:float=0.6, w_img:float=0.3, w_sym:float=0.45,
          domain:str="invoice"):
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
    """Return (reprocess_set, success_set) from a Views JSONL log."""
    if not views_log:
        return set(), set()
    try:
        logs = _read_views_log(views_log)
        return set(logs.get("reprocess", set())), set(logs.get("success", set()))
    except Exception:
        return set(), set()

def _read_gt(gt_jsonl: Optional[str]) -> Dict[str, Set]:
    """Load ground-truth cell identifiers grouped by semantic label."""
    labels: Dict[str, Set] = {"amount": set(), "date": set()}
    if not gt_jsonl or not os.path.exists(gt_jsonl):
        return labels
    with open(gt_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ob = json.loads(line)
            except Exception:
                continue
            lab = str(ob.get("label", "")).lower()
            if lab not in labels:
                continue
            key = (
                ob.get("doc_id"),
                int(ob.get("page", 0)),
                int(ob.get("table_index", 0)),
                int(ob.get("row", 0)),
                int(ob.get("col", 0)),
            )
            labels[lab].add(key)
    return labels


_INVOICE_GATE_DOMAINS = {"invoice", "invoice_jp_v2", "invoice_en", "invoice_fr"}


def _evaluate_gate(domain: Optional[str], amount_score: Optional[float], date_score: Optional[float]) -> Tuple[bool, str, float]:
    resolved = _DOMAIN_ALIAS.get(domain or "", domain or "")
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
    ap=argparse.ArgumentParser("ZOCR Multi‑Domain Core")
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
    """Estimate how often table rows deviate from the dominant column count."""
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
    """Measure how consistently cells stay within the same logical chunks/sections."""
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
    """Blend the latest monitor metrics back into the auto profile."""
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
