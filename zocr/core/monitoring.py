"""Monitoring and autotune helpers."""
from __future__ import annotations

import csv
import datetime
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .._compat import optional_numpy

np = optional_numpy(__name__)

from .base import (
    _normalize_text,
    detect_domain_on_jsonl,
    lambda_schedule,
    thomas_tridiag,
    _second_diff_tridiag,
)
from .domains import DOMAIN_ALIAS, DOMAIN_DEFAULTS, DOMAIN_MONITOR_QUERIES
from .indexer import build_index, _bm25_numba_score, _bm25_py_score
from .numba_support import HAS_NUMBA
from .query_engine import query, _kw_meta_boost, _symbolic_match_score
from .tokenization import tokenize_jp

__all__ = [
    "monitor",
    "learn_from_monitor",
    "metric_col_over_under_rate",
    "metric_chunk_consistency",
    "metric_col_alignment_energy_cached",
    "autotune_unlabeled",
]


def _read_views_log(views_log: Optional[str]) -> Dict[str, Set]:
    R: Set = set()
    S: Set = set()
    if not views_log or not os.path.exists(views_log):
        return {"reprocess": R, "success": S}
    with open(views_log, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ob = json.loads(line)
                key = (
                    ob.get("doc_id"),
                    int(ob.get("page", 0)),
                    int(ob.get("table_index", 0)),
                    int(ob.get("row", 0)),
                    int(ob.get("col", 0)),
                )
                ev = ob.get("event")
                if ev in ("reprocess", "view_reprocess", "llm_completion", "reocr"):
                    R.add(key)
                if ev in ("reocr_success", "llm_completion_success"):
                    S.add(key)
            except Exception:
                continue
    return {"reprocess": R, "success": S}


def _read_views_sets(views_log: Optional[str]) -> Tuple[Set, Set]:
    if not views_log:
        return set(), set()
    try:
        logs = _read_views_log(views_log)
        return set(logs.get("reprocess", set())), set(logs.get("success", set()))
    except Exception:
        return set(), set()


def _read_gt(gt_jsonl: Optional[str]) -> Dict[str, Set]:
    labels: Dict[str, Set] = {"amount": set(), "date": set(), "due": set()}
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


def _evaluate_gate(
    domain: Optional[str],
    amount_score: Optional[float],
    date_score: Optional[float],
    due_score: Optional[float],
    corporate_rate: Optional[float],
    tax_fail_rate: Optional[float],
) -> Tuple[bool, str, float]:
    resolved = DOMAIN_ALIAS.get(domain or "", domain or "")
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


def _preload_index_and_raws(index_pkl: str, jsonl: str):
    with open(index_pkl, "rb") as f:
        ix = pickle.load(f)
    raws: List[Dict[str, Any]] = []
    with open(jsonl, "r", encoding="utf-8") as fr:
        for line in fr:
            raws.append(json.loads(line))
    return ix, raws


def _query_scores_preloaded(
    ix: Dict[str, Any],
    raws: List[Dict[str, Any]],
    q_text: str,
    domain: Optional[str],
    w_kw: float,
    w_img: float,
    w_sym: float,
) -> float:
    vocab = ix["vocab"]
    df = np.array(ix["df"], dtype=np.int32)
    N = int(ix["N"])
    avgdl = float(ix["avgdl"])
    tokens = tokenize_jp(q_text or "")
    q_ids = [vocab[t] for t in tokens if t in vocab]
    if not q_ids:
        q_ids = [-1]
    q_ids = np.array(q_ids, dtype=np.int32)
    best = -1e9
    for i, doc_ids in enumerate(ix["docs_tokens"]):
        di = np.array(doc_ids + [-1], dtype=np.int32)
        dl = len(doc_ids)
        sb = (
            _bm25_numba_score(N, avgdl, df, dl, q_ids, di)
            if HAS_NUMBA
            else _bm25_py_score(N, avgdl, df, dl, q_ids, di)
        )
        ob = raws[i]
        sk = _kw_meta_boost(ob, tokens, domain or "invoice")
        filters = ((ob.get("meta") or {}).get("filters") or {})
        sym = _symbolic_match_score(filters, q_text or "", tokens)
        s = sb + w_kw * sk + w_img * 0.0 + w_sym * sym
        if s > best:
            best = s
    return float(best)


def _time_queries_preloaded(
    ix: Dict[str, Any],
    raws: List[Dict[str, Any]],
    domain: Optional[str],
    w_kw: float,
    w_img: float,
    w_sym: float,
    trials: int = 60,
    warmup: int = 8,
) -> Dict[str, float]:
    import random
    import time

    queries = {
        "invoice": ["合計", "金額", "消費税", "小計", "請求", "振込"],
        "invoice_jp_v2": ["合計", "金額", "消費税", "小計", "請求日", "発行日"],
        "invoice_en": ["invoice total", "amount due", "tax", "balance", "payment"],
        "invoice_fr": ["facture", "montant", "tva", "total", "date"],
        "purchase_order": ["purchase order", "po", "vendor", "ship", "qty"],
        "expense": ["expense", "category", "total", "tax", "reimburse"],
        "timesheet": ["timesheet", "hours", "project", "rate", "total"],
        "shipping_notice": ["shipment", "tracking", "carrier", "delivery", "ship"],
        "medical_receipt": ["診療", "点数", "保険", "負担金", "薬剤"],
        "delivery": ["納品", "数量", "受領", "出荷", "品名"],
        "delivery_jp": ["納品", "数量", "品番", "伝票", "受領"],
        "delivery_en": ["delivery", "tracking", "carrier", "qty", "item"],
        "estimate": ["見積", "単価", "小計", "有効期限"],
        "estimate_jp": ["御見積金額", "見積金額", "有効期限", "納期"],
        "estimate_en": ["estimate", "quote", "valid", "subtotal", "project"],
        "receipt": ["領収", "合計", "発行日", "住所", "税込"],
        "receipt_jp": ["領収書", "税込", "受領", "発行日", "現金"],
        "receipt_en": ["receipt", "paid", "total", "tax", "cash"],
        "contract": ["契約", "締結", "署名", "条", "甲"],
        "contract_jp_v2": ["契約", "甲", "乙", "条", "締結日", "署名"],
        "contract_en": ["contract", "signature", "party", "term", "agreement"],
        "rental_agreement_en": ["monthly rent", "lease", "tenant", "landlord", "deposit"],
        "rental_agreement_jp": ["賃貸借", "賃料", "借主", "貸主", "敷金"],
        "loan_statement_en": ["loan", "interest", "principal", "installment", "balance"],
        "loan_statement_jp": ["返済", "利息", "元金", "残高", "返済日"],
        "travel_itinerary_en": ["itinerary", "flight", "departure", "arrival", "hotel"],
        "travel_itinerary_jp": ["旅程", "出発", "到着", "航空券", "宿泊"],
    }
    fallback = queries["invoice_jp_v2"]
    dom_q = queries.get(domain or "invoice_jp_v2", fallback)
    rnd = random.Random(0x5A17)
    lat: List[float] = []
    total = warmup + trials
    for t in range(total):
        q = " ".join(rnd.sample(dom_q, min(3, len(dom_q))))
        t0 = time.perf_counter()
        _ = _query_scores_preloaded(ix, raws, q_text=q, domain=domain, w_kw=w_kw, w_img=w_img, w_sym=w_sym)
        dt = (time.perf_counter() - t0) * 1000.0
        if t >= warmup:
            lat.append(dt)
    if not lat:
        return {"p50": None, "p95": None}
    lat = sorted(lat)
    p50 = lat[int(0.50 * (len(lat) - 1))]
    p95 = lat[int(0.95 * (len(lat) - 1))]
    return {"p50": float(p50), "p95": float(p95)}


def _prepare_alignment_cache(jsonl_mm: str):
    by_tbl = defaultdict(list)
    with open(jsonl_mm, "r", encoding="utf-8") as f:
        for line in f:
            ob = json.loads(line)
            key = (ob.get("doc_id"), int(ob.get("page", 0)), int(ob.get("table_index", 0)))
            by_tbl[key].append(ob)
    tbls = []
    for cells in by_tbl.values():
        max_r = max(int(c.get("row", 0)) for c in cells) + 1
        max_c = max(int(c.get("col", 0)) for c in cells) + 1
        left = [[None] * max_c for _ in range(max_r)]
        right = [[None] * max_c for _ in range(max_r)]
        H = None
        ymax = 0
        for c in cells:
            meta = c.get("meta") or {}
            if H is None and meta.get("page_height"):
                H = int(meta["page_height"])
            x1, y1, x2, y2 = c.get("bbox", [0, 0, 0, 0])
            ymax = max(ymax, int(y2))
            r = int(c.get("row", 0))
            co = int(c.get("col", 0))
            left[r][co] = int(x1)
            right[r][co] = int(x2)
        if H is None:
            H = int(max(1000, ymax))
        tbls.append({"H": H, "left": left, "right": right})
    return tbls


def _second_diff_energy(vec: np.ndarray) -> float:
    if vec.shape[0] < 3:
        return 0.0
    v = 0.0
    for i in range(1, vec.shape[0] - 1):
        d = vec[i + 1] - 2.0 * vec[i] + vec[i - 1]
        v += float(d * d)
    return v / float(vec.shape[0] - 2)


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


def metric_col_alignment_energy_cached(
    tbl_cache: List[Dict[str, Any]],
    lambda_shape: float,
    height_ref: float = 1000.0,
    exp: float = 0.7,
) -> float:
    num = 0.0
    den = 0.0
    for tb in tbl_cache:
        H = tb["H"]
        lam_eff = lambda_schedule(H, lambda_shape, height_ref, exp)
        for mat in [tb["left"], tb["right"]]:
            max_r = len(mat)
            max_c = len(mat[0]) if max_r > 0 else 0
            for co in range(max_c):
                arr = [mat[r][co] for r in range(max_r)]
                idx = [i for i, v in enumerate(arr) if v is not None]
                if len(idx) < 3:
                    continue
                y = np.array([arr[i] for i in idx], dtype=np.float64)
                e_before = _second_diff_energy(y)
                if e_before <= 1e-9:
                    num += 1.0
                    den += 1.0
                    continue
                a, b, c = _second_diff_tridiag(len(y), lam_eff)
                x = thomas_tridiag(a, b, c, y)
                e_after = _second_diff_energy(x)
                num += float(e_after)
                den += float(e_before)
    if den <= 0.0:
        return 1.0
    return float(num / den)


def autotune_unlabeled(
    jsonl_mm: str,
    index_pkl: str,
    outdir: str,
    method: str = "random",
    budget: int = 30,
    domain_hint: Optional[str] = None,
    seed: int = 0,
    p95_target_ms: float = 300.0,
    use_smoothing_metric: bool = True,
) -> Dict[str, Any]:
    import random as pyrand

    np.random.seed(seed)
    pyrand.seed(seed)
    os.makedirs(outdir, exist_ok=True)

    domain, _ = detect_domain_on_jsonl(jsonl_mm)
    if domain_hint:
        domain = domain_hint
    base = DOMAIN_DEFAULTS.get(domain, DOMAIN_DEFAULTS["invoice_jp_v2"])

    col_rate = metric_col_over_under_rate(jsonl_mm)
    chunk_c = metric_chunk_consistency(jsonl_mm)

    ix, raws = _preload_index_and_raws(index_pkl, jsonl_mm)
    tbl_cache = _prepare_alignment_cache(jsonl_mm) if use_smoothing_metric else None

    log_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    def _score(p95, lam_shape):
        p95n = (p95 or p95_target_ms) / max(1.0, p95_target_ms)
        if use_smoothing_metric and tbl_cache is not None:
            align_ratio = metric_col_alignment_energy_cached(tbl_cache, lam_shape, 1000.0, 0.7)
            f_align = 0.3 + 0.7 * float(align_ratio)
        else:
            f_align = 1.0
        return col_rate * p95n * (1.0 - chunk_c + 0.05) * f_align, f_align

    def sample(center=None, scale=1.0):
        if center is None:
            lam = float(np.random.uniform(1.0, 6.0))
            wkw = float(np.random.uniform(0.3, 0.8))
            wimg = float(np.random.uniform(0.0, 0.5))
            wsym = float(np.random.uniform(0.3, 0.7))
            ocr = float(np.random.uniform(0.4, 0.8))
        else:
            lam = float(np.clip(np.random.normal(center["lambda_shape"], 0.5 * scale), 1.0, 6.0))
            wkw = float(np.clip(np.random.normal(center["w_kw"], 0.1 * scale), 0.2, 0.9))
            wimg = float(np.clip(np.random.normal(center["w_img"], 0.1 * scale), 0.0, 0.6))
            wsym = float(
                np.clip(
                    np.random.normal(center.get("w_sym", base.get("w_sym", 0.45)), 0.08 * scale),
                    0.2,
                    0.85,
                )
            )
            ocr = float(
                np.clip(
                    np.random.normal(center["ocr_min_conf"], 0.05 * scale),
                    0.3,
                    0.9,
                )
            )
        return {
            "lambda_shape": lam,
            "w_kw": wkw,
            "w_img": wimg,
            "w_sym": wsym,
            "ocr_min_conf": ocr,
        }

    n_init = max(8, min(15, budget // 2))
    for i in range(n_init):
        params = sample()
        lat = _time_queries_preloaded(ix, raws, domain, params["w_kw"], params["w_img"], params["w_sym"], trials=48, warmup=8)
        score, f_align = _score(lat["p95"], params["lambda_shape"])
        row = {
            "iter": i,
            "phase": "init",
            "domain": domain,
            "col_rate": col_rate,
            "chunk_c": chunk_c,
            "p95": lat["p95"],
            "score": score,
            "align_factor": f_align,
            **params,
        }
        log_rows.append(row)
        if best is None or score < best["score"]:
            best = row

    remain = max(0, budget - n_init)
    for j in range(remain):
        params = sample(center=best, scale=max(0.5, 1.5 * (remain - j) / max(1, remain)))
        lat = _time_queries_preloaded(ix, raws, domain, params["w_kw"], params["w_img"], params["w_sym"], trials=48, warmup=8)
        score, f_align = _score(lat["p95"], params["lambda_shape"])
        row = {
            "iter": n_init + j,
            "phase": "refine",
            "domain": domain,
            "col_rate": col_rate,
            "chunk_c": chunk_c,
            "p95": lat["p95"],
            "score": score,
            "align_factor": f_align,
            **params,
        }
        log_rows.append(row)
        if score < best["score"]:
            best = row

    csv_path = os.path.join(outdir, "autotune_log.csv")
    hdr = [
        "iter",
        "phase",
        "domain",
        "lambda_shape",
        "w_kw",
        "w_img",
        "w_sym",
        "ocr_min_conf",
        "col_rate",
        "chunk_c",
        "p95",
        "align_factor",
        "score",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fw:
        wr = csv.DictWriter(fw, fieldnames=hdr)
        wr.writeheader()
        for r in log_rows:
            wr.writerow({k: r.get(k) for k in hdr})

    prof_path = os.path.join(outdir, "auto_profile.json")
    try:
        prof = json.load(open(prof_path, "r", encoding="utf-8"))
    except Exception:
        prof = {"domain": domain}
    prof.update(
        {
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
            "tune_score": float(best["score"]),
        }
    )
    with open(prof_path, "w", encoding="utf-8") as fw:
        json.dump(prof, fw, ensure_ascii=False, indent=2)
    return {"best": best, "log_csv": csv_path, "profile_json": prof_path}


def _compute_p95_if_needed(jsonl: str, index_pkl: str, domain: Optional[str]) -> Optional[float]:
    try:
        ix, raws = _preload_index_and_raws(index_pkl, jsonl)
        d = domain or detect_domain_on_jsonl(jsonl)[0]
        base = DOMAIN_DEFAULTS.get(d, DOMAIN_DEFAULTS["invoice_jp_v2"])
        lat = _time_queries_preloaded(ix, raws, d, base["w_kw"], base["w_img"], base.get("w_sym", 0.45), trials=60, warmup=8)
        return float(lat["p95"]) if lat["p95"] is not None else None
    except Exception:
        return None


def monitor(
    jsonl: str,
    index_pkl: str,
    k: int,
    out_csv: str,
    views_log: Optional[str] = None,
    gt_jsonl: Optional[str] = None,
    domain: Optional[str] = None,
):
    total = 0
    low = 0
    corp_hits = 0
    corp_total = 0
    lc_keys = set()
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            ob = json.loads(line)
            total += 1
            meta = ob.get("meta") or {}
            filt = meta.get("filters", {})
            if meta.get("low_conf"):
                lc_keys.add((ob.get("doc_id"), ob.get("page"), ob.get("table_index"), ob.get("row"), ob.get("col")))
                low += 1
            if filt.get("corporate_id") is not None:
                corp_total += 1
                if filt.get("company_canonical"):
                    corp_hits += 1
    low_conf_rate = low / max(1, total)
    corporate_match_rate = (corp_hits / max(1, corp_total)) if corp_total > 0 else 0.0

    S_reproc, S_success = _read_views_sets(views_log)
    reprocess_rate = len(S_reproc & lc_keys) / max(1, len(lc_keys)) if lc_keys else 0.0
    reprocess_success_rate = len(S_success & S_reproc) / max(1, len(S_reproc)) if S_reproc else 0.0

    if not os.path.exists(index_pkl):
        build_index(jsonl, index_pkl)
    G = _read_gt(gt_jsonl)

    def _score(label: str, queries: Sequence[str]) -> Tuple[int, Optional[float]]:
        res = query(index_pkl, jsonl, " ".join(queries), None, topk=k, domain=domain or "invoice")
        if not res:
            return 0, None
        rel = G.get(label) or set()
        good = 0
        hit = 0
        for _, ob in res:
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
                elif label == "due" and (
                    filt.get("due_date")
                    or filt.get("due")
                    or filt.get("payment_due")
                    or filt.get("deadline")
                ):
                    hit = 1
                    good += 1
        trust = good / len(res) if res else None
        return hit, trust

    domain_key = domain or "default"
    resolved_monitor_key = DOMAIN_ALIAS.get(domain_key, domain_key)
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
    hit_mean = sum(metrics) / len(metrics) if metrics else 0.0
    trust_vals = [v for v in (trust_amount, trust_date, trust_due) if v is not None]
    trust_mean = sum(trust_vals) / len(trust_vals) if trust_vals else None

    tax_fail = 0
    tax_cov = 0
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            ob = json.loads(line)
            filt = (ob.get("meta") or {}).get("filters", {})
            if filt.get("tax_amount") is not None and filt.get("tax_amount_expected") is not None:
                tax_cov += 1
                if abs(int(filt["tax_amount"]) - int(filt["tax_amount_expected"])) > 1:
                    tax_fail += 1
    tax_fail_rate = (tax_fail / max(1, tax_cov)) if tax_cov > 0 else 0.0

    p95 = None
    agg = os.path.join(os.path.dirname(jsonl), "metrics_aggregate.csv")
    if os.path.exists(agg):
        try:
            import pandas as pd

            df = pd.read_csv(agg)
            if "latency_p95_ms" in df.columns:
                p95 = float(df["latency_p95_ms"].iloc[0])
        except Exception:
            p95 = None
    if p95 is None:
        p95 = _compute_p95_if_needed(jsonl, index_pkl, domain)

    gate_pass, gate_reason, gate_score = _evaluate_gate(
        domain,
        hit_amount,
        hit_date,
        hit_due,
        corporate_match_rate,
        tax_fail_rate,
    )
    row = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "jsonl": jsonl,
        "K": k,
        "domain": domain or "auto",
        "low_conf_rate": low_conf_rate,
        "reprocess_rate": reprocess_rate,
        "reprocess_success_rate": reprocess_success_rate,
        "hit_amount": hit_amount,
        "hit_date": hit_date,
        "hit_due": hit_due,
        "hit_mean": hit_mean,
        "tax_fail_rate": tax_fail_rate,
        "tax_coverage": tax_cov,
        "corporate_match_rate": corporate_match_rate,
        "corporate_coverage": corp_total,
        "p95_ms": p95,
        "trust_amount": trust_amount,
        "trust_date": trust_date,
        "trust_due": trust_due,
        "trust_mean": trust_mean,
        "gate_pass": gate_pass,
        "gate_reason": gate_reason,
        "gate_score": gate_score,
    }
    hdr = not os.path.exists(out_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "a", encoding="utf-8-sig", newline="") as fw:
        wr = csv.DictWriter(fw, fieldnames=list(row.keys()))
        if hdr:
            wr.writeheader()
        wr.writerow(row)
    return row


def learn_from_monitor(
    monitor_csv: str,
    profile_json_in: Optional[str],
    profile_json_out: Optional[str] = None,
    domain_hint: Optional[str] = None,
    ema: float = 0.5,
) -> Dict[str, Any]:
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
            with open(monitor_csv, "r", encoding="utf-8-sig", newline="") as fr:
                rows = list(csv.DictReader(fr))
            if rows:
                metrics = rows[-1]
        except Exception:
            metrics = {}

    if metrics:
        numeric_keys = [
            "low_conf_rate",
            "reprocess_rate",
            "reprocess_success_rate",
            "hit_amount",
            "hit_date",
            "hit_due",
            "hit_mean",
            "p95_ms",
            "tax_fail_rate",
            "tax_coverage",
            "corporate_match_rate",
            "corporate_coverage",
            "trust_amount",
            "trust_date",
            "trust_due",
            "trust_mean",
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
                target = max(0.1, min(0.99, target - 0.25 * (metrics["low_conf_rate"] - 0.1)))
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
