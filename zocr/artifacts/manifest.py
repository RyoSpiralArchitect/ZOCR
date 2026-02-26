from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .._version import __version__

MANIFEST_FILENAME = "zocr.manifest.json"
MANIFEST_SCHEMA = "zocr.manifest"
MANIFEST_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def build_manifest(
    outdir: str | Path,
    *,
    summary: Optional[Mapping[str, Any]] = None,
    inputs: Optional[Sequence[str]] = None,
    run_id: Optional[str] = None,
    hash_inputs: bool = False,
) -> Dict[str, Any]:
    outdir_path = Path(outdir)
    created_at = None
    if summary is not None and isinstance(summary.get("generated_at"), str):
        created_at = summary.get("generated_at")

    artifacts: Dict[str, Dict[str, Any]] = {}
    candidates: Dict[str, Optional[str]] = {
        "doc_zocr": "doc.zocr.json",
        "pipeline_summary": "pipeline_summary.json",
        "pipeline_meta": "pipeline_meta.json",
    }
    if summary is not None:
        candidates.update(
            {
                "contextual_jsonl": str(summary.get("contextual_jsonl")) if summary.get("contextual_jsonl") else None,
                "mm_jsonl": str(summary.get("mm_jsonl")) if summary.get("mm_jsonl") else None,
                "index": str(summary.get("index")) if summary.get("index") else None,
                "monitor_csv": str(summary.get("monitor_csv")) if summary.get("monitor_csv") else None,
                "profile_json": str(summary.get("profile_json")) if summary.get("profile_json") else None,
                "rag_manifest": str(summary.get("rag_manifest")) if summary.get("rag_manifest") else None,
                "rag_cells": str(summary.get("rag_cells")) if summary.get("rag_cells") else None,
                "rag_sections": str(summary.get("rag_sections")) if summary.get("rag_sections") else None,
                "rag_tables_json": str(summary.get("rag_tables_json")) if summary.get("rag_tables_json") else None,
                "rag_markdown": str(summary.get("rag_markdown")) if summary.get("rag_markdown") else None,
                "rag_bundle": str(summary.get("rag_bundle")) if summary.get("rag_bundle") else None,
                "report_html": str(summary.get("report_html")) if summary.get("report_html") else None,
            }
        )

    for name, value in candidates.items():
        if not value:
            continue
        rel = _relpath(value, outdir_path)
        artifact_path = outdir_path / rel
        if not artifact_path.exists():
            # Try the raw value relative to cwd (legacy summary paths).
            alt = Path(value)
            if alt.exists():
                artifact_path = alt
                rel = _relpath(str(alt), outdir_path)
            else:
                continue
        artifacts[name] = {
            "path": rel,
            "kind": _guess_kind(artifact_path),
        }

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

    manifest: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "zocr_version": __version__,
        "created_at": created_at or _utc_now_iso(),
        "run_id": run_id,
        "outdir": outdir_path.as_posix(),
        "inputs": input_entries,
        "artifacts": artifacts,
    }
    return manifest


def write_manifest(
    outdir: str | Path,
    *,
    summary: Optional[Mapping[str, Any]] = None,
    inputs: Optional[Sequence[str]] = None,
    run_id: Optional[str] = None,
    hash_inputs: bool = False,
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
    )
    dest = outdir_path / filename
    tmp = outdir_path / f".{filename}.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(dest)
    return dest

