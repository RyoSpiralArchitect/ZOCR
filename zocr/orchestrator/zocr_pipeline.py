
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

import os, sys, json, time, traceback, argparse, random, platform, hashlib, subprocess, importlib
from typing import Any, Dict, List, Optional
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
    pages = []
    for it in inputs:
        ext = os.path.splitext(it)[1].lower()
        if ext == ".pdf":
            try:
                pages += zocr_onefile_consensus.pdf_to_images_via_poppler(it, dpi=dpi)
            except Exception as e:
                raise RuntimeError(f"PDF rasterization failed for {it}: {e}")
        else:
            pages.append(it)
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

    if "OCR" in ok:
        print("[SKIP] OCR (resume)")
    else:
        r = _safe_step("OCR", pipe.run, "doc", pages, outdir, annos)
        _append_hist(outdir, r)
        if not r.get("ok"):
            raise RuntimeError("OCR failed")
        try:
            _, doc_json_path = r.get("out", (None, doc_json_path))
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
            with open(mon_csv, "r", encoding="utf-8", newline="") as fr:
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
                with open(mon_csv, "r", encoding="utf-8", newline="") as fr:
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
    except Exception as e:
        print("RAG bundle export skipped:", e)
    _call("post_rag", manifest=summary.get("rag_manifest"), bundle=summary.get("rag_bundle"))

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
