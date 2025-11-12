
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

import os, sys, json, time, traceback, argparse, random, platform, hashlib, subprocess, importlib, re, glob, shutil
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


def _resolve_toy_memory_path(outdir: str) -> str:
    env_path = os.environ.get("ZOCR_TOY_MEMORY")
    if env_path:
        return env_path
    return os.path.join(outdir, "toy_memory.json")


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


def _reanalyze_output_paths(learning_jsonl: str, outdir: str) -> Tuple[str, str]:
    base = os.path.basename(learning_jsonl)
    if base.endswith(".jsonl"):
        base = base[:-6]
    output = os.path.join(outdir, f"{base}.reanalyzed.jsonl")
    return output, output + ".summary.json"


def _profile_snapshot(prof: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(prof))


def _profile_diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    diff: Dict[str, Tuple[Any, Any]] = {}
    keys = set(before.keys()) | set(after.keys())
    for key in sorted(keys):
        if before.get(key) != after.get(key):
            diff[key] = (before.get(key), after.get(key))
    return diff


def _derive_intent(monitor_row: Optional[Dict[str, Any]], export_signals: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    intent: Dict[str, Any] = {"action": "steady", "reason": "metrics within guardrails"}
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
    if hit_mean is None:
        intent = {"action": "recover", "reason": "monitor missing", "priority": "high"}
    elif hit_mean < 0.8:
        intent = {"action": "focus_headers", "reason": f"hit_mean={hit_mean:.3f} below 0.8", "priority": "high"}
    elif p95 is not None and p95 > 400.0:
        intent = {"action": "optimize_speed", "reason": f"p95_ms={p95:.1f} > 400", "priority": "medium"}
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
    else:
        intent = {"action": "explore_footer", "reason": "metrics nominal", "priority": "low"}
    intent["signals"] = {
        "hit_mean": hit_mean,
        "p95_ms": p95,
        "low_conf_ratio": low_conf_ratio,
        "high_surprisal_ratio": high_surprisal_ratio,
        "surprisal_threshold": export_signals.get("surprisal_threshold") if export_signals else None,
        "learning_samples": export_signals.get("learning_samples") if export_signals else None,
    }
    intent["profile_domain"] = profile.get("domain")
    return intent


def _apply_intent_to_profile(intent: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    updates: Dict[str, Tuple[Any, Any]] = {}
    action = intent.get("action")
    if action == "focus_headers":
        old = profile.get("header_boost", 1.0)
        profile["header_boost"] = float(old) * 1.15 if isinstance(old, (int, float)) else 1.2
        updates["header_boost"] = (old, profile["header_boost"])
        targets = list(profile.get("reanalyze_target") or [])
        if "headers" not in targets:
            targets.append("headers")
        profile["reanalyze_target"] = targets
    elif action == "optimize_speed":
        old = profile.get("lambda_shape", 4.5)
        try:
            new_val = max(2.5, float(old) * 0.9)
        except Exception:
            new_val = 3.8
        profile["lambda_shape"] = new_val
        updates["lambda_shape"] = (old, new_val)
        profile.setdefault("speed_priority", True)
    elif action == "reanalyze_cells":
        prev = list(profile.get("reanalyze_target") or [])
        if "learning_cells" not in prev:
            prev.append("learning_cells")
        profile["reanalyze_target"] = prev
        updates["reanalyze_target"] = (None, prev)
    elif action == "recover":
        profile.setdefault("force_monitor_refresh", True)
        updates["force_monitor_refresh"] = (None, True)
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


def _apply_rag_feedback(manifest_path: Optional[str], profile: Dict[str, Any], profile_path: str) -> Dict[str, Any]:
    if not manifest_path or not os.path.exists(manifest_path):
        return {"applied": []}
    try:
        with open(manifest_path, "r", encoding="utf-8") as fr:
            payload = json.load(fr)
    except Exception:
        return {"applied": [], "error": "manifest_unreadable"}
    feedback = payload.get("feedback") if isinstance(payload, dict) else None
    applied: List[str] = []
    if isinstance(feedback, dict):
        profile_overrides = feedback.get("profile_overrides")
        if isinstance(profile_overrides, dict):
            for key, value in profile_overrides.items():
                before = profile.get(key)
                profile[key] = value
                applied.append(key)
    if applied:
        try:
            with open(profile_path, "w", encoding="utf-8") as fw:
                json.dump(_json_ready(profile), fw, ensure_ascii=False, indent=2)
        except Exception as e:
            return {"applied": applied, "error": str(e)}
    return {"applied": applied}

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

    toy_memory_path = _resolve_toy_memory_path(outdir)
    toy_memory_info_load = zocr_onefile_consensus.load_toy_memory(toy_memory_path)
    toy_memory_after_load = toy_memory_info_load.get("snapshot_after") or toy_memory_info_load.get("snapshot_before")
    if hasattr(zocr_onefile_consensus, "reset_toy_recognition_stats"):
        zocr_onefile_consensus.reset_toy_recognition_stats()

    if len(inputs) == 1 and inputs[0].lower() == "demo" and not os.path.exists(inputs[0]):
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
        "toy_memory": {
            "path": toy_memory_path,
            "load": _json_ready(toy_memory_info_load),
        },
    }

    toy_memory_run_baseline = toy_memory_after_load or zocr_onefile_consensus.toy_memory_snapshot()

    domain_hints = _prepare_domain_hints(inputs)
    content_conf_threshold = float(os.environ.get("ZOCR_DOMAIN_CONF_THRESHOLD", "0.25"))
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
        r = _safe_step("Export", zocr_onefile_consensus.export_jsonl_with_ocr,
                       doc_json_path, pages, jsonl_path, "toy", True, prof.get("ocr_min_conf", 0.58))
        _append_hist(outdir, r)
        if not r.get("ok"):
            raise RuntimeError("Export failed")
    _call("post_export", jsonl=jsonl_path, outdir=outdir)
    export_signals = _load_export_signals(jsonl_path)
    if export_signals:
        summary["export_signals"] = export_signals
        if export_signals.get("learning_jsonl"):
            summary["learning_jsonl"] = export_signals.get("learning_jsonl")

    reanalysis_summary: Optional[Dict[str, Any]] = None
    learning_jsonl_path = export_signals.get("learning_jsonl") if export_signals else None
    re_targets = {str(t) for t in (prof.get("reanalyze_target") or []) if t}
    if learning_jsonl_path and "learning_cells" in re_targets:
        re_dir = os.path.join(outdir, "reanalyze")
        ensure_dir(re_dir)
        if "ReanalyzeLearning" in ok:
            print("[SKIP] Reanalyze learning cells (resume)")
            _, summary_path = _reanalyze_output_paths(learning_jsonl_path, re_dir)
            try:
                with open(summary_path, "r", encoding="utf-8") as fr:
                    loaded = json.load(fr)
                    if isinstance(loaded, dict):
                        reanalysis_summary = loaded
            except Exception:
                reanalysis_summary = None
        else:
            try:
                re_limit = int(prof.get("reanalyze_limit") or 64)
            except Exception:
                re_limit = 64
            r = _safe_step("ReanalyzeLearning", zocr_onefile_consensus.reanalyze_learning_jsonl,
                           learning_jsonl_path, re_dir, re_limit)
            _append_hist(outdir, r)
            if r.get("ok"):
                reanalysis_summary = r.get("out")
        if reanalysis_summary:
            summary["reanalyze_learning"] = _json_ready(reanalysis_summary)
            if isinstance(reanalysis_summary, dict) and reanalysis_summary.get("output_jsonl"):
                summary["learning_reanalyzed_jsonl"] = reanalysis_summary.get("output_jsonl")

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

    prof_after = _load_profile(outdir, prof.get("domain"))
    profile_diff = _profile_diff(profile_before_feedback, prof_after)
    intent = _derive_intent(summary.get("monitor_row"), export_signals, prof_after)
    summary["intent"] = intent
    intent_updates = _apply_intent_to_profile(intent, prof_after)
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
    rerun_flags = _needs_rerun_for_keys(list(combined_updates.keys())) if combined_updates else {"augment": False, "monitor": False}
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

    intent_runs: List[str] = []
    if intent.get("action") == "reanalyze_cells" and learning_jsonl_path and not summary.get("reanalyze_learning"):
        re_dir = os.path.join(outdir, "reanalyze")
        ensure_dir(re_dir)
        try:
            re_limit = int(prof_after.get("reanalyze_limit") or 64)
        except Exception:
            re_limit = 64
        r = _safe_step("ReanalyzeLearningIntent", zocr_onefile_consensus.reanalyze_learning_jsonl,
                       learning_jsonl_path, re_dir, re_limit)
        _append_hist(outdir, r)
        if r.get("ok"):
            intent_runs.append("reanalyze_learning")
            reanalysis_summary = r.get("out")
            if isinstance(reanalysis_summary, dict):
                summary["reanalyze_learning"] = _json_ready(reanalysis_summary)
                if reanalysis_summary.get("output_jsonl"):
                    summary["learning_reanalyzed_jsonl"] = reanalysis_summary.get("output_jsonl")
    if intent_runs:
        summary["intent_runs"] = intent_runs

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
    if summary.get("rag_manifest"):
        summary["rag_feedback"] = _apply_rag_feedback(summary.get("rag_manifest"), prof, prof_path)

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

    toy_memory_after_run = zocr_onefile_consensus.toy_memory_snapshot()
    toy_memory_delta_run = zocr_onefile_consensus.toy_memory_delta(toy_memory_run_baseline, toy_memory_after_run)
    summary.setdefault("toy_memory", {})
    summary["toy_memory"]["before_run"] = _json_ready(toy_memory_run_baseline)
    summary["toy_memory"]["after_run"] = _json_ready(toy_memory_after_run)
    summary["toy_memory"]["delta_run"] = _json_ready(toy_memory_delta_run)
    if hasattr(zocr_onefile_consensus, "toy_recognition_stats"):
        summary["toy_memory"]["recognition"] = _json_ready(
            zocr_onefile_consensus.toy_recognition_stats(reset=True)
        )

    toy_memory_saved = zocr_onefile_consensus.save_toy_memory(toy_memory_path)
    summary["toy_memory"]["save"] = _json_ready(toy_memory_saved)

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
