from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .._version import __version__

MANIFEST_FILENAME = "zocr.manifest.json"
MANIFEST_SCHEMA = "zocr.manifest"
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_HASH_MAX_BYTES = 16 * 1024 * 1024

_STANDARD_ARTIFACTS: Dict[str, str] = {
    "doc_zocr": "doc.zocr.json",
    "pipeline_summary": "pipeline_summary.json",
    "pipeline_meta": "pipeline_meta.json",
    "auto_profile": "auto_profile.json",
    "contextual_jsonl": "doc.contextual.jsonl",
    "contextual_signals": "doc.contextual.jsonl.signals.json",
    "contextual_learning": "doc.contextual.jsonl.learning.jsonl",
    "contextual_reanalyzed": "doc.contextual.reanalyzed.jsonl",
    "mm_jsonl": "doc.mm.jsonl",
    "index": "bm25.pkl",
    "monitor_csv": "monitor.csv",
    "report_html": "report.html",
    "simple_pages": "pages.json",
    "simple_summary": "summary.json",
    "simple_regions": "regions.jsonl",
    "simple_manifest": "manifest.json",
    "rag_manifest": "rag/manifest.json",
    "rag_cells": "rag/cells.jsonl",
    "rag_sections": "rag/sections.jsonl",
    "rag_tables_json": "rag/tables.json",
    "rag_markdown": "rag/digest.md",
    "rag_feedback_request": "rag/feedback_request.json",
    "rag_feedback_markdown": "rag/feedback_request.md",
    "rag_conversation": "rag/conversation.jsonl",
}

_SUMMARY_ARTIFACT_KEYS = {
    "contextual_jsonl",
    "mm_jsonl",
    "index",
    "monitor_csv",
    "profile_json",
    "rag_manifest",
    "rag_cells",
    "rag_sections",
    "rag_tables_json",
    "rag_markdown",
    "rag_bundle",
    "report_html",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _datetime_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _relpath(path: str, base_dir: Path) -> str:
    raw = Path(path)
    try:
        resolved = raw if raw.is_absolute() else (Path.cwd() / raw)
        resolved = resolved.resolve()
    except Exception:
        resolved = raw
    try:
        return resolved.relative_to(base_dir.resolve()).as_posix()
    except Exception:
        pass
    # Heuristic: summary paths may include the outdir name prefix (e.g. "out/foo.json").
    try:
        parts = raw.parts
        if parts and parts[0] == base_dir.name:
            return Path(*parts[1:]).as_posix()
    except Exception:
        pass
    return raw.as_posix()


def _guess_kind(path: Path) -> str:
    if path.is_dir():
        return "dir"
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".zip":
        return "zip"
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return "image"
    if suffix in {".csv"}:
        return "csv"
    if suffix in {".pkl"}:
        return "pickle"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix in {".md"}:
        return "markdown"
    return "file"


def _env_hash_limit(default: int = DEFAULT_ARTIFACT_HASH_MAX_BYTES) -> int:
    raw = os.environ.get("ZOCR_MANIFEST_HASH_MAX_BYTES")
    if raw is None:
        return int(default)
    try:
        return max(0, int(raw))
    except Exception:
        return int(default)


def _string_path(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Path):
        return value.as_posix()
    return None


def _resolve_artifact_path(value: str, outdir: Path) -> tuple[Optional[str], Optional[Path]]:
    rel = _relpath(value, outdir)
    artifact_path = outdir / rel
    if artifact_path.exists():
        return rel, artifact_path
    raw = Path(value)
    if raw.exists():
        return _relpath(str(raw), outdir), raw
    return rel, None


def _artifact_entry(
    rel: str,
    path: Path,
    *,
    hash_artifacts: bool,
    hash_artifact_max_bytes: int,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "path": rel,
        "kind": _guess_kind(path),
    }
    try:
        stat = path.stat()
    except Exception:
        return entry
    if path.is_file():
        entry["bytes"] = int(stat.st_size)
        entry["modified_at"] = _datetime_from_timestamp(stat.st_mtime)
        if hash_artifacts:
            if hash_artifact_max_bytes <= 0 or stat.st_size <= hash_artifact_max_bytes:
                try:
                    entry["sha256"] = _sha256_file(path)
                except Exception as exc:
                    entry["sha256_error"] = f"{type(exc).__name__}: {exc}"
            else:
                entry["sha256_skipped"] = "size_limit"
                entry["sha256_max_bytes"] = int(hash_artifact_max_bytes)
    elif path.is_dir():
        entry["modified_at"] = _datetime_from_timestamp(stat.st_mtime)
    return entry


def _artifact_kind_counts(artifacts: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in artifacts.values():
        kind = str(entry.get("kind") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def build_manifest(
    outdir: str | Path,
    *,
    summary: Optional[Mapping[str, Any]] = None,
    inputs: Optional[Sequence[str]] = None,
    run_id: Optional[str] = None,
    hash_inputs: bool = False,
    hash_artifacts: bool = True,
    hash_artifact_max_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    outdir_path = Path(outdir)
    created_at = None
    if summary is not None and isinstance(summary.get("generated_at"), str):
        created_at = summary.get("generated_at")

    artifacts: Dict[str, Dict[str, Any]] = {}
    missing_references: list[dict[str, str]] = []
    candidates: Dict[str, str] = dict(_STANDARD_ARTIFACTS)
    if summary is not None:
        for key in sorted(_SUMMARY_ARTIFACT_KEYS):
            value = _string_path(summary.get(key))
            if value:
                candidates[key] = value

    seen_paths: set[str] = set()
    max_hash_bytes = _env_hash_limit() if hash_artifact_max_bytes is None else max(0, int(hash_artifact_max_bytes))
    for name, value in candidates.items():
        rel, artifact_path = _resolve_artifact_path(value, outdir_path)
        if artifact_path is None or rel is None:
            if summary is not None and name in _SUMMARY_ARTIFACT_KEYS:
                missing_references.append({"artifact": name, "path": value, "source": "pipeline_summary"})
            continue
        rel_key = rel
        if rel_key in seen_paths and name not in {"manifest_json"}:
            continue
        seen_paths.add(rel_key)
        artifacts[name] = _artifact_entry(
            rel,
            artifact_path,
            hash_artifacts=hash_artifacts,
            hash_artifact_max_bytes=max_hash_bytes,
        )

    input_entries: list[dict[str, Any]] = []
    if inputs:
        for p in inputs:
            entry: dict[str, Any] = {"path": str(p)}
            path_obj = Path(p)
            if path_obj.exists() and path_obj.is_file():
                try:
                    entry["bytes"] = path_obj.stat().st_size
                except Exception:
                    pass
                if hash_inputs:
                    try:
                        entry["sha256"] = _sha256_file(path_obj)
                    except Exception:
                        pass
            input_entries.append(entry)

    total_bytes = sum(int(entry.get("bytes") or 0) for entry in artifacts.values())
    audit_warnings: list[str] = []
    if missing_references:
        audit_warnings.append(f"{len(missing_references)} summary artifact reference(s) were missing")
    audit = {
        "artifact_count": len(artifacts),
        "artifact_kinds": _artifact_kind_counts(artifacts),
        "total_bytes": int(total_bytes),
        "hash_algorithm": "sha256",
        "hash_max_bytes": int(max_hash_bytes),
        "missing_references": missing_references,
        "warnings": audit_warnings,
    }

    manifest: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "zocr_version": __version__,
        "created_at": created_at or _utc_now_iso(),
        "run_id": run_id,
        "outdir": outdir_path.as_posix(),
        "inputs": input_entries,
        "artifacts": artifacts,
        "audit": audit,
    }
    return manifest


def write_manifest(
    outdir: str | Path,
    *,
    summary: Optional[Mapping[str, Any]] = None,
    inputs: Optional[Sequence[str]] = None,
    run_id: Optional[str] = None,
    hash_inputs: bool = False,
    hash_artifacts: bool = True,
    hash_artifact_max_bytes: Optional[int] = None,
    filename: str = MANIFEST_FILENAME,
) -> Path:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        outdir_path,
        summary=summary,
        inputs=inputs,
        run_id=run_id,
        hash_inputs=hash_inputs,
        hash_artifacts=hash_artifacts,
        hash_artifact_max_bytes=hash_artifact_max_bytes,
    )
    dest = outdir_path / filename
    tmp = outdir_path / f".{filename}.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(dest)
    return dest
