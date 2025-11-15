
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
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set, TypedDict
try:
    from typing import Literal  # py39+
except ImportError:  # pragma: no cover - fallback for very old Python
    from typing_extensions import Literal  # type: ignore
from html import escape

from .prior import PriorBandit, normalize_headers_to_signature, decide_success

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

_AUTOCALIB_DEFAULT_SAMPLES = 3
_AUTOTUNE_DEFAULT_TRIALS = 6


def _positive_cli_value(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        intval = int(value)
    except Exception:
        return None
    return intval if intval > 0 else None


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


def _current_episode_id() -> Optional[str]:
    eid = _EPISODE_CONTEXT.get("id")
    return str(eid) if eid else None


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


def _episode_artifact_path(outdir: str, episode_id: str, filename: str) -> str:
    return os.path.join(_episodes_root(outdir), episode_id, filename)


def _finalize_episode(outdir: str, summary: Dict[str, Any]) -> None:
    info = summary.get("episode") or {}
    if not info.get("id"):
        info = _EPISODE_CONTEXT
    episode_id = str(info.get("id") or "").strip()
    if not episode_id:
        return
    ep_dir = info.get("path") or _episode_artifact_path(outdir, episode_id, "")
    if not ep_dir:
        ep_dir = _episode_artifact_path(outdir, episode_id, "")
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
        stage_path = os.path.join(ep_dir, "stage_trace.json")
        try:
            with open(stage_path, "w", encoding="utf-8") as fw:
                json.dump(_json_ready(stage_trace), fw, ensure_ascii=False, indent=2)
            artifacts["stage_trace.json"] = _rel(stage_path)
        except Exception as exc:
            print(f"[WARN] episode stage trace write failed: {exc}")

    toy_delta = ((summary.get("toy_memory") or {}).get("delta_run"))
    if toy_delta:
        toy_path = os.path.join(ep_dir, "toy_memory_delta.json")
        try:
            with open(toy_path, "w", encoding="utf-8") as fw:
                json.dump(_json_ready(toy_delta), fw, ensure_ascii=False, indent=2)
            artifacts["toy_memory_delta.json"] = _rel(toy_path)
        except Exception as exc:
            print(f"[WARN] episode toy delta write failed: {exc}")

    for key in ("learning_hotspots", "selective_reanalysis_plan", "hotspot_gallery"):
        if not summary.get(key):
            continue
        path = os.path.join(ep_dir, f"{key}.json")
        try:
            with open(path, "w", encoding="utf-8") as fw:
                json.dump(_json_ready(summary.get(key)), fw, ensure_ascii=False, indent=2)
            artifacts[f"{key}.json"] = _rel(path)
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
            parts.append(
                f"<p class=\"muted\"><a href=\"{escape(str(gallery['story']))}\">gallery notes</a></p>"
            )
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
    """Remove RAG suggested queries that duplicate existing insights."""
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


def _signature_state_path(outdir: str) -> str:
    return os.path.join(outdir, "table_signature.json")


def _load_saved_signature(outdir: str) -> Tuple[Optional[str], Optional[List[str]]]:
    path = _signature_state_path(outdir)
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None, None
    signature = payload.get("signature") if isinstance(payload, dict) else None
    headers = payload.get("headers") if isinstance(payload, dict) else None
    if isinstance(signature, str):
        sig_val = signature
    else:
        sig_val = None
    header_list = headers if isinstance(headers, list) else None
    return sig_val, header_list


def _save_signature(outdir: str, signature: str, headers: Optional[List[str]]) -> None:
    payload = {
        "signature": signature,
        "headers": headers,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    path = _signature_state_path(outdir)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _extract_headers_from_jsonl(jsonl_path: str) -> Optional[List[str]]:
    if not os.path.exists(jsonl_path):
        return None
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                headers = rec.get("headers") if isinstance(rec, dict) else None
                if isinstance(headers, list) and headers:
                    return [str(h) for h in headers]
    except Exception:
        return None
    return None

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
                    entry.setdefault("avg_conf" , 0.0)
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
    trace_rank = sorted(trace_scores.items(), key=lambda item: (-float(item[1].get("score") or 0.0), item[1].get("count", 0)),)[:24]
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
    trace_order = sorted(
        trace_scores.items(), key=lambda item: (-float(item[1].get("score") or 0.0), item[1].get("count", 0))
    )
    trace_ids = [trace for trace, _ in trace_order[:max_traces] if trace]
    row_order = sorted(
        row_stats.items(), key=lambda item: (-float(item[1].get("score") or 0.0), -float(item[1].get("count") or 0))
    )
    row_keys = [row for row, _ in row_order[:16]]
    table_order = sorted(
        table_stats.items(), key=lambda item: (-float(item[1].get("score") or 0.0), -float(item[1].get("count") or 0))
    )
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

def _normalize_optional_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    trimmed = str(path).strip()
    if not trimmed:
        return None
    return os.path.abspath(os.path.expanduser(trimmed))


def _validate_file_if_supplied(path: Optional[str], label: str) -> Optional[str]:
    resolved = _normalize_optional_path(path)
    if not resolved:
        return None
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{label} path does not exist: {resolved}")
    return resolved


def _enforce_default_toy_feature_flags(
    motion_prior_enabled: Optional[bool] = None,
    tesslite_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ensure core toy OCR helpers are enabled without manual env wiring."""

    def _ensure_truthy_env(name: str, default_value: str = "1") -> Tuple[bool, str]:
        source = "env"
        if not os.environ.get(name):
            os.environ[name] = default_value
            source = "default"
        return _env_truthy(name, True), source

    feature_status: Dict[str, Any] = {}

    conf_enabled, conf_source = _ensure_truthy_env("ZOCR_CONF_BOOST_NUMERIC")
    lex_enabled, lex_source = _ensure_truthy_env("ZOCR_CONF_BOOST_LEXICAL")

    alpha_source = "env"
    if not os.environ.get("ZOCR_NGRAM_EMA_ALPHA"):
        os.environ["ZOCR_NGRAM_EMA_ALPHA"] = "0.05"
        alpha_source = "default"
    try:
        alpha_value = float(os.environ.get("ZOCR_NGRAM_EMA_ALPHA", "0.05") or 0.05)
    except Exception:
        alpha_value = 0.05
    snapshot_fn = getattr(zocr_onefile_consensus, "get_toy_feature_status", None)
    if callable(snapshot_fn):
        try:
            snapshot = snapshot_fn()
            if isinstance(snapshot, dict):
                feature_status.update(snapshot)
        except Exception:
            feature_status = {}

    confidence_entry = feature_status.setdefault("confidence_boost", {})
    confidence_entry.setdefault("enabled", conf_enabled)
    confidence_entry.setdefault("source", conf_source)

    lexical_entry = feature_status.setdefault("lexical_boost", {})
    lexical_entry.setdefault("enabled", lex_enabled)
    lexical_entry.setdefault("source", lex_source)

    ngram_entry = feature_status.setdefault("ngram_ema", {})
    ngram_entry.setdefault("alpha", alpha_value)
    ngram_entry.setdefault("source", alpha_source)

    motion_entry = dict(feature_status.get("motion_prior", {}))
    if motion_prior_enabled is not None:
        motion_entry["enabled"] = bool(motion_prior_enabled)
        motion_entry["source"] = "cli"
    else:
        motion_entry.setdefault("enabled", _env_truthy("ZOCR_EXPORT_MOTION_PRIOR", True))
        motion_entry.setdefault(
            "source",
            "env" if os.environ.get("ZOCR_EXPORT_MOTION_PRIOR") is not None else "default",
        )
    feature_status["motion_prior"] = motion_entry

    blank_entry = feature_status.setdefault("blank_skip", {})
    if "source" not in blank_entry:
        blank_entry["source"] = (
            "env"
            if any(
                os.environ.get(key) is not None
                for key in (
                    "ZOCR_EXPORT_SKIP_BLANK",
                    "ZOCR_EXPORT_BLANK_THRESHOLD",
                    "ZOCR_EXPORT_BLANK_MIN_PIXELS",
                    "ZOCR_EXPORT_BLANK_MIN_RATIO",
                    "ZOCR_EXPORT_BLANK_MIN_AREA",
                )
            )
            else "default"
        )

    if tesslite_status:
        feature_status["tesslite"] = dict(tesslite_status)
    else:
        feature_status.setdefault("tesslite", {"enabled": False, "source": "none"})

    if "pytesseract" not in feature_status:
        feature_status["pytesseract"] = {
            "allowed": _env_truthy("ZOCR_ALLOW_PYTESSERACT", True),
            "source": "env" if os.environ.get("ZOCR_ALLOW_PYTESSERACT") else "default",
        }

    feature_status["hotspot_detection"] = {
        "enabled": True,
        "mode": "learning_hotspots",
    }
    feature_status["view_generation"] = {
        "enabled": True,
        "mode": "microscope_xray",
    }
    feature_status["intent_simulations"] = {
        "enabled": True,
        "mode": "auto_profile",
    }

    return feature_status


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
    motion_prior: Optional[bool] = None,
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
    autocalib_samples: Optional[int] = None,
    autotune_trials: Optional[int] = None,
) -> Dict[str, Any]:
    if sweeps_fixed is not None and sweeps_fixed > 0:
        toy_sweeps = int(sweeps_fixed)
        os.environ["ZOCR_TOY_SWEEPS"] = str(toy_sweeps)
        os.environ["ZOCR_TOY_SWEEP_LIMIT"] = str(toy_sweeps)
    if motion_prior is None:
        motion_prior = True
    os.environ["ZOCR_EXPORT_MOTION_PRIOR"] = "1" if motion_prior else "0"
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
    tess_unicharset = _validate_file_if_supplied(tess_unicharset, "--tess-unicharset")
    tess_wordlist = _validate_file_if_supplied(tess_wordlist, "--tess-wordlist")
    tess_bigram_json = _validate_file_if_supplied(tess_bigram_json, "--tess-bigram-json")
    if tess_unicharset is not None:
        os.environ["ZOCR_TESS_UNICHARSET"] = tess_unicharset
    else:
        os.environ.pop("ZOCR_TESS_UNICHARSET", None)
    if tess_wordlist is not None:
        os.environ["ZOCR_TESS_WORDLIST"] = tess_wordlist
    else:
        os.environ.pop("ZOCR_TESS_WORDLIST", None)
    if tess_bigram_json is not None:
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

    table_params: Dict[str, Any] = {}
    table_autocalib_status: Optional[Dict[str, Any]] = None
    table_autotune_status: Optional[Dict[str, Any]] = None

    autocalib_count = _positive_cli_value(autocalib_samples)
    autotune_count = _positive_cli_value(autotune_trials)

    if autocalib_count:
        def _run_autocalib_stage() -> Dict[str, Any]:
            status: Dict[str, Any] = {"samples": int(autocalib_count)}
            updates: Dict[str, Any] = {}
            auto_calib_fn = getattr(zocr_onefile_consensus, "auto_calibrate_params", None)
            if not callable(auto_calib_fn):
                status["status"] = "unavailable"
                return {"status": status, "updates": updates}
            try:
                calib_cfg = auto_calib_fn(pages, autocalib_count) or {}
            except Exception as exc:  # pragma: no cover - defensive log path
                print(f"[WARN] Auto-calibration failed: {exc}")
                status["status"] = "error"
                status["error"] = str(exc)
                return {"status": status, "updates": updates}
            if isinstance(calib_cfg, dict) and calib_cfg:
                updates = calib_cfg
                status["status"] = "applied"
                status["keys"] = sorted(calib_cfg.keys())
            else:
                status["status"] = "no_change"
            return {"status": status, "updates": updates}

        calib_stage = _safe_step(f"AutoCalib ({autocalib_count})", _run_autocalib_stage)
        payload = calib_stage.get("out") if isinstance(calib_stage, dict) else None
        if isinstance(payload, dict):
            status = payload.get("status")
            updates = payload.get("updates")
            if isinstance(status, dict):
                table_autocalib_status = status
            if isinstance(updates, dict) and updates:
                table_params.update(updates)

    if autotune_count:
        def _run_autotune_stage() -> Dict[str, Any]:
            status: Dict[str, Any] = {"trials": int(autotune_count)}
            updates: Dict[str, Any] = {}
            autotune_fn = getattr(zocr_onefile_consensus, "autotune_params", None)
            if not callable(autotune_fn):
                status["status"] = "unavailable"
                return {"status": status, "updates": updates}
            try:
                base_for_tune = table_params.copy()
                tuned_cfg = autotune_fn(pages, base_for_tune, trials=autotune_count) or {}
            except Exception as exc:  # pragma: no cover - defensive log path
                print(f"[WARN] Autotune failed: {exc}")
                status["status"] = "error"
                status["error"] = str(exc)
                return {"status": status, "updates": updates}
            if isinstance(tuned_cfg, dict) and tuned_cfg:
                updates = tuned_cfg
                status["status"] = "applied"
                status["keys"] = sorted(tuned_cfg.keys())
            else:
                status["status"] = "no_change"
            return {"status": status, "updates": updates}

        tune_stage = _safe_step(f"AutoTune ({autotune_count})", _run_autotune_stage)
        payload = tune_stage.get("out") if isinstance(tune_stage, dict) else None
        if isinstance(payload, dict):
            status = payload.get("status")
            updates = payload.get("updates")
            if isinstance(status, dict):
                table_autotune_status = status
            if isinstance(updates, dict) and updates:
                table_params.update(updates)

    pipe_cfg = {"table": table_params, "bench_iterations": 1, "eval": False}
    pipe = zocr_onefile_consensus.Pipeline(pipe_cfg)

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
    bandit: Optional[PriorBandit] = None
    bandit_action: Optional[str] = None
    bandit_signature: Optional[str] = None
    bandit_headers: Optional[List[str]] = None

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
    bandit: Optional[PriorBandit] = None
    bandit_action: Optional[str] = None
    bandit_signature: Optional[str] = None
    bandit_headers: Optional[List[str]] = None

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
    bandit: Optional[PriorBandit] = None
    bandit_action: Optional[str] = None
    bandit_signature: Optional[str] = None
    bandit_headers: Optional[List[str]] = None

    tesslite_cfg = {
        "unicharset": os.environ.get("ZOCR_TESS_UNICHARSET") or None,
        "wordlist": os.environ.get("ZOCR_TESS_WORDLIST") or None,
        "bigram_json": os.environ.get("ZOCR_TESS_BIGRAM_JSON") or None,
    }
    tesslite_status_fn = getattr(zocr_onefile_consensus, "get_tesslite_status", None)
    if callable(tesslite_status_fn):
        summary["tesslite"] = tesslite_status_fn()
    else:
        summary["tesslite"] = {
            "enabled": any(tesslite_cfg.values()),
            **{k: v for k, v in tesslite_cfg.items() if v},
        }
    bandit: Optional[PriorBandit] = None
    bandit_action: Optional[str] = None
    bandit_signature: Optional[str] = None
    bandit_headers: Optional[List[str]] = None

    tesslite_cfg = {
        "unicharset": os.environ.get("ZOCR_TESS_UNICHARSET") or None,
        "wordlist": os.environ.get("ZOCR_TESS_WORDLIST") or None,
        "bigram_json": os.environ.get("ZOCR_TESS_BIGRAM_JSON") or None,
    }
    tesslite_status_fn = getattr(zocr_onefile_consensus, "get_tesslite_status", None)
    if callable(tesslite_status_fn):
        summary["tesslite"] = tesslite_status_fn()
    else:
        summary["tesslite"] = {
            "enabled": any(tesslite_cfg.values()),
            **{k: v for k, v in tesslite_cfg.items() if v},
        }

    toy_feature_defaults = _enforce_default_toy_feature_flags(
        motion_prior_enabled=motion_prior,
        tesslite_status=summary.get("tesslite"),
    )
    if toy_feature_defaults:
        summary["toy_feature_defaults"] = _json_ready(toy_feature_defaults)
    bandit: Optional[PriorBandit] = None
    bandit_action: Optional[str] = None
    bandit_signature: Optional[str] = None
    bandit_headers: Optional[List[str]] = None

    tesslite_cfg = {
        "unicharset": os.environ.get("ZOCR_TESS_UNICHARSET") or None,
        "wordlist": os.environ.get("ZOCR_TESS_WORDLIST") or None,
        "bigram_json": os.environ.get("ZOCR_TESS_BIGRAM_JSON") or None,
    }
    tesslite_status_fn = getattr(zocr_onefile_consensus, "get_tesslite_status", None)
    if callable(tesslite_status_fn):
        summary["tesslite"] = tesslite_status_fn()
    else:
        summary["tesslite"] = {
            "enabled": any(tesslite_cfg.values()),
            **{k: v for k, v in tesslite_cfg.items() if v},
        }

    toy_feature_defaults = _enforce_default_toy_feature_flags(
        motion_prior_enabled=motion_prior,
        tesslite_status=summary.get("tesslite"),
    )
    if toy_feature_defaults:
        summary["toy_feature_defaults"] = _json_ready(toy_feature_defaults)
    bandit: Optional[PriorBandit] = None
    bandit_action: Optional[str] = None
    bandit_signature: Optional[str] = None
    bandit_headers: Optional[List[str]] = None

    tesslite_cfg = {
        "unicharset": os.environ.get("ZOCR_TESS_UNICHARSET") or None,
        "wordlist": os.environ.get("ZOCR_TESS_WORDLIST") or None,
        "bigram_json": os.environ.get("ZOCR_TESS_BIGRAM_JSON") or None,
    }
    tesslite_status_fn = getattr(zocr_onefile_consensus, "get_tesslite_status", None)
    if callable(tesslite_status_fn):
        summary["tesslite"] = tesslite_status_fn()
    else:
        summary["tesslite"] = {
            "enabled": any(tesslite_cfg.values()),
            **{k: v for k, v in tesslite_cfg.items() if v},
        }

    toy_feature_defaults = _enforce_default_toy_feature_flags(
        motion_prior_enabled=motion_prior,
        tesslite_status=summary.get("tesslite"),
    )
    if toy_feature_defaults:
        summary["toy_feature_defaults"] = _json_ready(toy_feature_defaults)
    bandit: Optional[PriorBandit] = None
    bandit_action: Optional[str] = None
    bandit_signature: Optional[str] = None
    bandit_headers: Optional[List[str]] = None

    tesslite_cfg = {
        "unicharset": os.environ.get("ZOCR_TESS_UNICHARSET") or None,
        "wordlist": os.environ.get("ZOCR_TESS_WORDLIST") or None,
        "bigram_json": os.environ.get("ZOCR_TESS_BIGRAM_JSON") or None,
    }
    tesslite_status_fn = getattr(zocr_onefile_consensus, "get_tesslite_status", None)
    if callable(tesslite_status_fn):
        summary["tesslite"] = tesslite_status_fn()
    else:
        summary["tesslite"] = {
            "enabled": any(tesslite_cfg.values()),
            **{k: v for k, v in tesslite_cfg.items() if v},
        }

    toy_feature_defaults = _enforce_default_toy_feature_flags(
        motion_prior_enabled=motion_prior,
        tesslite_status=summary.get("tesslite"),
    )
    if toy_feature_defaults:
        summary["toy_feature_defaults"] = _json_ready(toy_feature_defaults)
    bandit: Optional[PriorBandit] = None
    bandit_action: Optional[str] = None
    bandit_signature: Optional[str] = None
    bandit_headers: Optional[List[str]] = None

    tesslite_cfg = {
        "unicharset": os.environ.get("ZOCR_TESS_UNICHARSET") or None,
        "wordlist": os.environ.get("ZOCR_TESS_WORDLIST") or None,
        "bigram_json": os.environ.get("ZOCR_TESS_BIGRAM_JSON") or None,
    }
    tesslite_status_fn = getattr(zocr_onefile_consensus, "get_tesslite_status", None)
    if callable(tesslite_status_fn):
        summary["tesslite"] = tesslite_status_fn()
    else:
        summary["tesslite"] = {
            "enabled": any(tesslite_cfg.values()),
            **{k: v for k, v in tesslite_cfg.items() if v},
        }

    toy_feature_defaults = _enforce_default_toy_feature_flags(
        motion_prior_enabled=motion_prior,
        tesslite_status=summary.get("tesslite"),
    )
    if toy_feature_defaults:
        summary["toy_feature_defaults"] = _json_ready(toy_feature_defaults)
    bandit: Optional[PriorBandit] = None
    bandit_action: Optional[str] = None
    bandit_signature: Optional[str] = None
    bandit_headers: Optional[List[str]] = None

    tesslite_cfg = {
        "unicharset": os.environ.get("ZOCR_TESS_UNICHARSET") or None,
        "wordlist": os.environ.get("ZOCR_TESS_WORDLIST") or None,
        "bigram_json": os.environ.get("ZOCR_TESS_BIGRAM_JSON") or None,
    }
    tesslite_status_fn = getattr(zocr_onefile_consensus, "get_tesslite_status", None)
    if callable(tesslite_status_fn):
        summary["tesslite"] = tesslite_status_fn()
    else:
        summary["tesslite"] = {
            "enabled": any(tesslite_cfg.values()),
            **{k: v for k, v in tesslite_cfg.items() if v},
        }

    toy_feature_defaults = _enforce_default_toy_feature_flags(
        motion_prior_enabled=motion_prior,
        tesslite_status=summary.get("tesslite"),
    )
    if toy_feature_defaults:
        summary["toy_feature_defaults"] = _json_ready(toy_feature_defaults)
    bandit: Optional[PriorBandit] = None
    bandit_action: Optional[str] = None
    bandit_signature: Optional[str] = None
    bandit_headers: Optional[List[str]] = None

    tesslite_cfg = {
        "unicharset": os.environ.get("ZOCR_TESS_UNICHARSET") or None,
        "wordlist": os.environ.get("ZOCR_TESS_WORDLIST") or None,
        "bigram_json": os.environ.get("ZOCR_TESS_BIGRAM_JSON") or None,
    }
    tesslite_status_fn = getattr(zocr_onefile_consensus, "get_tesslite_status", None)
    if callable(tesslite_status_fn):
        summary["tesslite"] = tesslite_status_fn()
    else:
        summary["tesslite"] = {
            "enabled": any(tesslite_cfg.values()),
            **{k: v for k, v in tesslite_cfg.items() if v},
        }

    toy_feature_defaults = _enforce_default_toy_feature_flags(
        motion_prior_enabled=motion_prior,
        tesslite_status=summary.get("tesslite"),
    )
    if toy_feature_defaults:
        summary["toy_feature_defaults"] = _json_ready(toy_feature_defaults)
    bandit: Optional[PriorBandit] = None
    bandit_action: Optional[str] = None
    bandit_signature: Optional[str] = None
    bandit_headers: Optional[List[str]] = None

    tesslite_cfg = {
        "unicharset": os.environ.get("ZOCR_TESS_UNICHARSET") or None,
        "wordlist": os.environ.get("ZOCR_TESS_WORDLIST") or None,
        "bigram_json": os.environ.get("ZOCR_TESS_BIGRAM_JSON") or None,
    }
    tesslite_status_fn = getattr(zocr_onefile_consensus, "get_tesslite_status", None)
    if callable(tesslite_status_fn):
        summary["tesslite"] = tesslite_status_fn()
    else:
        summary["tesslite"] = {
            "enabled": any(tesslite_cfg.values()),
            **{k: v for k, v in tesslite_cfg.items() if v},
        }

    toy_feature_defaults = _enforce_default_toy_feature_flags(
        motion_prior_enabled=motion_prior,
        tesslite_status=summary.get("tesslite"),
    )
    if toy_feature_defaults:
        summary["toy_feature_defaults"] = _json_ready(toy_feature_defaults)
    bandit: Optional[PriorBandit] = None
    bandit_action: Optional[str] = None
    bandit_signature: Optional[str] = None
    bandit_headers: Optional[List[str]] = None

    if table_params:
        summary["table_params"] = _json_ready(table_params)
    if table_autocalib_status:
        summary["table_autocalib"] = _json_ready(table_autocalib_status)
    if table_autotune_status:
        summary["table_autotune"] = _json_ready(table_autotune_status)

    tesslite_cfg = {
        "unicharset": os.environ.get("ZOCR_TESS_UNICHARSET") or None,
        "wordlist": os.environ.get("ZOCR_TESS_WORDLIST") or None,
        "bigram_json": os.environ.get("ZOCR_TESS_BIGRAM_JSON") or None,
    }
    tesslite_status_fn = getattr(zocr_onefile_consensus, "get_tesslite_status", None)
    if callable(tesslite_status_fn):
        summary["tesslite"] = tesslite_status_fn()
    else:
        summary["tesslite"] = {
            "enabled": any(tesslite_cfg.values()),
            **{k: v for k, v in tesslite_cfg.items() if v},
        }

    toy_feature_defaults = _enforce_default_toy_feature_flags(
        motion_prior_enabled=motion_prior,
        tesslite_status=summary.get("tesslite"),
    )
    if toy_feature_defaults:
        summary["toy_feature_defaults"] = _json_ready(toy_feature_defaults)
    bandit: Optional[PriorBandit] = None
    bandit_action: Optional[str] = None
    bandit_signature: Optional[str] = None
    bandit_headers: Optional[List[str]] = None

    if table_params:
        summary["table_params"] = _json_ready(table_params)
    if table_autocalib_status:
        summary["table_autocalib"] = _json_ready(table_autocalib_status)
    if table_autotune_status:
        summary["table_autotune"] = _json_ready(table_autotune_status)

    tesslite_cfg = {
        "unicharset": os.environ.get("ZOCR_TESS_UNICHARSET") or None,
        "wordlist": os.environ.get("ZOCR_TESS_WORDLIST") or None,
        "bigram_json": os.environ.get("ZOCR_TESS_BIGRAM_JSON") or None,
    }
    tesslite_status_fn = getattr(zocr_onefile_consensus, "get_tesslite_status", None)
    if callable(tesslite_status_fn):
        summary["tesslite"] = tesslite_status_fn()
    else:
        summary["tesslite"] = {
            "enabled": any(tesslite_cfg.values()),
            **{k: v for k, v in tesslite_cfg.items() if v},
        }

    toy_feature_defaults = _enforce_default_toy_feature_flags(
        motion_prior_enabled=motion_prior,
        tesslite_status=summary.get("tesslite"),
    )
    if toy_feature_defaults:
        summary["toy_feature_defaults"] = _json_ready(toy_feature_defaults)
    bandit: Optional[PriorBandit] = None
    bandit_action: Optional[str] = None
    bandit_signature: Optional[str] = None
    bandit_headers: Optional[List[str]] = None

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
        if isinstance(export_ocr_engine, str) and export_ocr_engine.lower().startswith("toy"):
            saved_sig, saved_headers = _load_saved_signature(outdir)
            cached_headers = _extract_headers_from_jsonl(jsonl_path)
            headers_source = None
            candidate_headers = None
            if cached_headers:
                candidate_headers = cached_headers
                headers_source = "contextual_jsonl"
            elif saved_headers:
                candidate_headers = saved_headers
                headers_source = "signature_cache"
            if candidate_headers is None:
                fallback_tokens: List[str] = []
                domain_token = prof.get("domain") or domain_hint
                if domain_token:
                    fallback_tokens.append(str(domain_token))
                if not fallback_tokens and inputs:
                    fallback_tokens.append(os.path.splitext(os.path.basename(inputs[0]))[0])
                if not fallback_tokens:
                    fallback_tokens.append("unknown")
                candidate_headers = fallback_tokens
                headers_source = "fallback"
            bandit_headers = candidate_headers
            bandit_signature = normalize_headers_to_signature(bandit_headers)
            bandit_state_path = os.path.join(outdir, "bandit_state.json")
            bandit = PriorBandit(bandit_state_path)
            bandit_action = bandit.decide(bandit_signature)
            os.environ["ZOCR_USE_PRIOR"] = "1" if bandit_action == "WITH_PRIOR" else "0"
            os.environ["ZOCR_PRIOR_ACTION"] = bandit_action
            os.environ.setdefault("ZOCR_PRIOR_SIGMA", "auto")
            os.environ.setdefault("ZOCR_K_SIGMA_WINDOW", "2.5")
            prior_cache_dir = os.path.join(outdir, ".prior_cache")
            ensure_dir(prior_cache_dir)
            os.environ["ZOCR_PRIOR_CACHE"] = prior_cache_dir
            os.environ["ZOCR_TABLE_SIGNATURE"] = bandit_signature
            if not os.environ.get("ZOCR_TEMPLATE_CACHE"):
                template_cache_path = os.path.join(outdir, "toy_template_cache.json")
                os.environ["ZOCR_TEMPLATE_CACHE"] = template_cache_path
                summary.setdefault("toy_templates", {})["cache"] = template_cache_path
            summary["prior_bandit"] = {
                "signature": bandit_signature,
                "action": bandit_action,
                "headers_preview": bandit_headers[:8] if bandit_headers else None,
                "headers_source": headers_source,
                "state": bandit_state_path,
            }
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
        new_headers = _extract_headers_from_jsonl(jsonl_path)
        if new_headers:
            final_sig = normalize_headers_to_signature(new_headers)
            bandit_headers = new_headers
            bandit_signature = final_sig
            _save_signature(outdir, final_sig, new_headers)
            prior_meta = summary.setdefault("prior_bandit", {})
            prior_meta["final_signature"] = final_sig
            prior_meta["headers_preview"] = new_headers[:8]
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
                # ensure we persist the effective config used
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
    if profile_guard:
        summary["profile_guard"] = profile_guard.report()

    toy_memory_saved = zocr_onefile_consensus.save_toy_memory(toy_memory_path)
    summary["toy_memory"]["save"] = _json_ready(toy_memory_saved)

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

    if bandit and bandit_signature and bandit_action:
        try:
            success_flag = decide_success(summary)
            bandit.update(bandit_signature, bandit_action, bool(success_flag))
            bandit.save()
            prior_meta = summary.setdefault("prior_bandit", {})
            prior_meta["signature"] = bandit_signature
            prior_meta["action"] = bandit_action
            prior_meta["success"] = bool(success_flag)
        except Exception as exc:
            print(f"[WARN] bandit update skipped: {exc}")

    _finalize_episode(outdir, summary)

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
        "--autocalib",
        nargs="?",
        type=int,
        const=_AUTOCALIB_DEFAULT_SAMPLES,
        default=None,
        metavar="N",
        help=(
            "Auto-calibrate table detection using N sample pages (default %(const)s when "
            "no explicit value is supplied). Pass 0 or omit to skip."
        ),
    )
    ap.add_argument(
        "--autotune",
        nargs="?",
        type=int,
        const=_AUTOTUNE_DEFAULT_TRIALS,
        default=None,
        metavar="N",
        help=(
            "Run the unsupervised table autotuner for N trials (default %(const)s when the "
            "flag is value-less). Pass 0 or omit to disable."
        ),
    )
    ap.add_argument(
        "--force-numeric-by-header",
        action="store_true",
        help="Normalize numeric columns according to header heuristics",
    )
    ap.add_argument(
        "--motion-prior",
        dest="motion_prior",
        action="store_true",
        default=None,
        help="Force-enable motion prior seeding between export sweeps",
    )
    ap.add_argument(
        "--no-motion-prior",
        dest="motion_prior",
        action="store_false",
        help="Disable motion prior reseeding",
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
        help="Explicitly allow pytesseract variants (enabled by default unless --no-allow-pytesseract or ZOCR_ALLOW_PYTESSERACT=0 is set)",
    )
    ap.add_argument(
        "--no-allow-pytesseract",
        dest="allow_pytesseract",
        action="store_false",
        default=None,
        help="Force-disable pytesseract even when the environment would otherwise allow it",
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
        help="JSON mapping of bigram probabilities used to penalize unlikely glyph transitions",
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
            autocalib_samples=args.autocalib,
            autotune_trials=args.autotune,
        )
        print("\n[SUCCESS] Summary written:", os.path.join(args.outdir, "pipeline_summary.json"))
        print(json.dumps(res, ensure_ascii=False, indent=2))
    except Exception as e:
        print("\n💀 Pipeline crashed:", e)
        sys.exit(1)

if __name__=="__main__":
    main()
