"""Thin ingestion/query interface to expose Z-OCR artifacts to downstream layers.

This module intentionally keeps the surface area small while wiring through to the
existing orchestrator. It provides two helpers:

- ``ingest_job``: run (or schedule) an ingestion job that executes the full OCR
  pipeline and returns canonical artifact locations keyed by ``job_id``.
- ``query_job``: read the previously generated artifacts for a ``job_id`` and
  return a lightweight response envelope ready to send to a downstream RAG/LLM
  stack.

Both functions are synchronous wrappers so that they can be called directly from
API or CLI glue code before a richer async/job manager is introduced.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from jsonschema import Draft202012Validator
import httpx

from zocr.orchestrator import zocr_pipeline
from zocr.api_spec import (
    INGEST_REQUEST_SCHEMA_V0,
    QUERY_REQUEST_SCHEMA_V0,
    render_user_prompt_analysis_v0,
)
from zocr.core.indexer import build_index
from zocr.core.query_engine import hybrid_query

__all__ = [
    "FileSpec",
    "IngestRequest",
    "IngestResult",
    "QueryRequest",
    "QueryResult",
    "ingest_job",
    "query_job",
    "ingest_response_payload_v0",
    "query_response_payload_v0",
    "validate_ingest_request_payload",
    "validate_query_request_payload",
    "render_user_prompt_analysis_v0",
]


@dataclass
class FileSpec:
    """Descriptor for an input file provided to ingestion."""

    id: str
    uri: str
    kind: str = "auto"
    tags: List[str] = field(default_factory=list)


@dataclass
class IngestRequest:
    """Parameters controlling an ingestion job."""

    tenant_id: str
    files: Iterable[Union[str, Dict[str, Any], FileSpec]]
    domain_hint: Optional[str] = None
    job_id: Optional[str] = None
    out_root: str = "episodes"
    resume: bool = False
    dry_run: bool = False
    pipeline_kwargs: Dict[str, Any] = field(default_factory=dict)
    async_mode: bool = False

    def resolved_job_id(self) -> str:
        return self.job_id or f"episode_{uuid.uuid4().hex[:12]}"


@dataclass
class QueryRequest:
    """Parameters for querying an existing job."""

    tenant_id: str
    job_id: str
    conversation_id: Optional[str] = None
    query: str = ""
    language: str = "ja"
    mode: str = "analysis"
    base_dir: str = "episodes"


@dataclass
class IngestResult:
    """Structured result of an ingestion attempt."""

    job_id: str
    tenant_id: str
    status: str
    outdir: str
    artifacts: Dict[str, Optional[str]]
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_path: Optional[str] = None


@dataclass
class QueryResult:
    """Response envelope for a query over an existing job's artifacts."""

    job_id: str
    tenant_id: str
    conversation_id: Optional[str]
    mode: str
    answer: Dict[str, str]
    artifacts: Dict[str, Any]
    provenance: List[Dict[str, str]]
    flags: Dict[str, Any]
    status: str


_JOB_LOCK = threading.Lock()
_STATUS_FILENAME = "job_status.json"


def _safe_segment(raw: str, label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in raw.strip())
    cleaned = cleaned.strip("._")
    if not cleaned:
        raise ValueError(f"Invalid {label}")
    return cleaned


def _tenant_root(out_root: str, tenant_id: str) -> Path:
    return Path(out_root) / _safe_segment(tenant_id, "tenant_id")


def _job_root(out_root: str, tenant_id: str, job_id: str) -> Path:
    return _tenant_root(out_root, tenant_id) / _safe_segment(job_id, "job_id")


def _status_path(job_dir: Path) -> Path:
    return job_dir / _STATUS_FILENAME


def _write_job_status(job_dir: Path, payload: Dict[str, Any]) -> None:
    os.makedirs(job_dir, exist_ok=True)
    with open(_status_path(job_dir), "w", encoding="utf-8") as fw:
        json.dump(payload, fw, ensure_ascii=False, indent=2)


def _read_job_status(job_dir: Path) -> Dict[str, Any]:
    path = _status_path(job_dir)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fr:
        return json.load(fr)


def _download_remote(uri: str, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", uri, timeout=60.0) as response:
        response.raise_for_status()
        with open(target, "wb") as fw:
            for chunk in response.iter_bytes():
                fw.write(chunk)
    return str(target)


def _resolve_uri(uri: str, target_dir: Path, label: str) -> str:
    if uri.startswith("http://") or uri.startswith("https://"):
        filename = f"{label}{Path(uri).suffix or '.bin'}"
        return _download_remote(uri, target_dir / filename)
    if uri.startswith("s3://"):
        try:
            import boto3  # type: ignore
        except Exception as exc:
            raise RuntimeError("boto3 is required to fetch s3:// URIs") from exc
        bucket_key = uri[5:].split("/", 1)
        if len(bucket_key) != 2:
            raise ValueError(f"Invalid s3 URI: {uri}")
        bucket, key = bucket_key
        filename = f"{label}{Path(key).suffix or '.bin'}"
        target = target_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3").download_file(bucket, key, str(target))
        return str(target)
    return uri


def _resolve_inputs(files: Iterable[FileSpec], job_dir: Path) -> List[FileSpec]:
    resolved: List[FileSpec] = []
    cache_dir = job_dir / "inputs"
    for idx, item in enumerate(files):
        label = _safe_segment(item.id or f"file_{idx}", "file_id")
        uri = _resolve_uri(item.uri, cache_dir, label)
        resolved.append(
            FileSpec(id=item.id, uri=uri, kind=item.kind, tags=list(item.tags))
        )
    return resolved


def _build_trace_label(cell: Dict[str, Any]) -> str:
    doc_id = cell.get("doc_id") or ""
    page = cell.get("page")
    table_index = cell.get("table_index")
    row = cell.get("row")
    col = cell.get("col")
    parts = [
        f"doc={doc_id}",
        f"page={page}" if page is not None else "page=",
        f"table={table_index}" if table_index is not None else "table=",
        f"row={row}" if row is not None else "row=",
        f"col={col}" if col is not None else "col=",
    ]
    return ";".join(parts)


def _existing(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def _normalize_files(files: Iterable[Union[str, Dict[str, Any], FileSpec]]) -> List[FileSpec]:
    normalized: List[FileSpec] = []
    for idx, item in enumerate(files):
        if isinstance(item, FileSpec):
            normalized.append(item)
            continue
        if isinstance(item, str):
            normalized.append(FileSpec(id=os.path.basename(item) or f"file_{idx}", uri=item))
            continue
        if isinstance(item, dict):
            normalized.append(
                FileSpec(
                    id=str(item.get("id") or f"file_{idx}"),
                    uri=str(item.get("uri")),
                    kind=str(item.get("kind", "auto")),
                    tags=list(item.get("tags", [])),
                )
            )
            continue
        raise TypeError(f"Unsupported file descriptor: {item!r}")
    return normalized


def ingest_job(request: IngestRequest) -> IngestResult:
    """Run the orchestrator and expose canonical artifact paths.

    The call is synchronous by default. Set ``request.dry_run`` to avoid running
    the pipeline while still returning the computed job identifiers and paths.
    Additional keyword arguments for the pipeline can be supplied via
    ``request.pipeline_kwargs``.
    """

    job_id = _safe_segment(request.resolved_job_id(), "job_id")
    outdir = _job_root(request.out_root, request.tenant_id, job_id)
    os.makedirs(outdir, exist_ok=True)

    artifact_paths = {
        "pipeline_summary": str(outdir / "pipeline_summary.json"),
        "doc_jsonl": str(outdir / "doc.mm.jsonl"),
        "rag_manifest": str(outdir / "rag" / "manifest.json"),
        "sql_dir": str(outdir / "sql"),
    }

    if request.dry_run:
        queued = IngestResult(
            job_id=job_id,
            tenant_id=request.tenant_id,
            status="queued",
            outdir=str(outdir),
            artifacts=artifact_paths,
            summary=None,
            error=None,
            status_path=str(_status_path(outdir)),
        )
        _write_job_status(
            outdir,
            {
                "job_id": job_id,
                "tenant_id": request.tenant_id,
                "status": "queued",
                "updated_at": _iso_now(),
                "artifacts": artifact_paths,
            },
        )
        return queued

    files = _normalize_files(request.files)
    resolved_files = _resolve_inputs(files, outdir)
    pipeline_inputs = [fs.uri for fs in resolved_files]

    def _run_pipeline() -> None:
        with _JOB_LOCK:
            _write_job_status(
                outdir,
                {
                    "job_id": job_id,
                    "tenant_id": request.tenant_id,
                    "status": "running",
                    "started_at": _iso_now(),
                    "artifacts": artifact_paths,
                },
            )
        try:
            summary = zocr_pipeline.run_full_pipeline(
                pipeline_inputs,
                str(outdir),
                domain_hint=request.domain_hint,
                resume=request.resume,
                **request.pipeline_kwargs,
            )
            status = "completed"
            error = None
        except Exception as exc:  # pragma: no cover - surfaced to caller
            summary = None
            status = "failed"
            error = str(exc)
        with _JOB_LOCK:
            payload = _read_job_status(outdir)
            payload.update(
                {
                    "status": status,
                    "updated_at": _iso_now(),
                    "summary": summary,
                    "error": error,
                }
            )
            _write_job_status(outdir, payload)

    if request.async_mode:
        _write_job_status(
            outdir,
            {
                "job_id": job_id,
                "tenant_id": request.tenant_id,
                "status": "queued",
                "updated_at": _iso_now(),
                "artifacts": artifact_paths,
            },
        )
        thread = threading.Thread(target=_run_pipeline, daemon=True)
        thread.start()
        return IngestResult(
            job_id=job_id,
            tenant_id=request.tenant_id,
            status="queued",
            outdir=str(outdir),
            artifacts=artifact_paths,
            summary=None,
            error=None,
            status_path=str(_status_path(outdir)),
        )

    _run_pipeline()
    status_payload = _read_job_status(outdir)
    return IngestResult(
        job_id=job_id,
        tenant_id=request.tenant_id,
        status=status_payload.get("status", "failed"),
        outdir=str(outdir),
        artifacts={k: _existing(Path(v)) or v for k, v in artifact_paths.items()},
        summary=status_payload.get("summary"),
        error=status_payload.get("error"),
        status_path=str(_status_path(outdir)),
    )


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fr:
        return json.load(fr)


def _summarize_manifest(manifest: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    cell_count = manifest.get("cell_count") or manifest.get("cells", {}).get("count")
    page_sections = manifest.get("page_sections")
    table_sections = manifest.get("table_sections")
    parts: List[str] = []
    if cell_count is not None:
        parts.append(f"cells={cell_count}")
    if page_sections is not None:
        parts.append(f"page_sections={page_sections}")
    if table_sections is not None:
        parts.append(f"table_sections={table_sections}")
    summary = f"Artifacts for {job_id}" if parts else f"No manifest stats for {job_id}"
    if parts:
        summary = f"Artifacts for {job_id}: " + ", ".join(parts)
    return {
        "data_summary": summary,
        "table_meta": {"count": table_sections},
        "page_meta": {"count": page_sections},
    }


def _tabulate_cells(cells: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rows: Dict[int, Dict[int, str]] = {}
    max_col = -1
    for cell in cells:
        row_idx = cell.get("row")
        col_idx = cell.get("col")
        if row_idx is None or col_idx is None:
            continue
        text = cell.get("text") or ""
        rows.setdefault(int(row_idx), {})[int(col_idx)] = text
        max_col = max(max_col, int(col_idx))
    if max_col < 0 or not rows:
        return None
    ordered_rows = []
    for row_idx in sorted(rows):
        row = [rows[row_idx].get(col, "") for col in range(max_col + 1)]
        ordered_rows.append(row)
    columns = [f"col_{idx}" for idx in range(max_col + 1)]
    return {"columns": columns, "rows": ordered_rows}


def _load_tables_from_manifest(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    tables_path = manifest.get("tables")
    if not tables_path or not os.path.exists(tables_path):
        return []
    with open(tables_path, "r", encoding="utf-8") as fr:
        raw_tables = json.load(fr)
    output: List[Dict[str, Any]] = []
    for key, payload in raw_tables.items():
        cells = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(cells, list):
            continue
        table = _tabulate_cells(cells)
        if not table:
            continue
        output.append(
            {
                "id": str(key),
                "title": str(key),
                "columns": table["columns"],
                "rows": table["rows"],
            }
        )
    return output


def _provenance_from_manifest(manifest: Dict[str, Any]) -> List[Dict[str, str]]:
    provenance: List[Dict[str, str]] = []
    trace_schema = manifest.get("trace_schema") or {}
    label = trace_schema.get("label") or ""
    fact_example = manifest.get("fact_tag_example")
    if label or fact_example:
        provenance.append(
            {
                "trace": label or "",
                "fact_text": fact_example or "",
            }
        )
    prov_info = manifest.get("provenance") or {}
    prov_path = prov_info.get("prov_bundle")
    bundle_id = prov_info.get("bundle_id")
    if prov_path:
        provenance.append(
            {
                "trace": bundle_id or "prov_bundle",
                "fact_text": f"PROV trace at {prov_path}",
            }
        )
    return provenance


def query_job(
    *,
    job_id: str,
    tenant_id: str,
    base_dir: str = "episodes",
    conversation_id: Optional[str] = None,
    query: str = "",
    mode: str = "analysis",
) -> QueryResult:
    """Build a minimal query response envelope grounded on existing artifacts."""

    job_dir = _job_root(base_dir, tenant_id, job_id)
    manifest_path = job_dir / "rag" / "manifest.json"
    summary_path = job_dir / "pipeline_summary.json"

    if not manifest_path.exists():
        legacy_job_dir = Path(base_dir) / _safe_segment(job_id, "job_id")
        legacy_manifest = legacy_job_dir / "rag" / "manifest.json"
        if legacy_manifest.exists():
            job_dir = legacy_job_dir
            manifest_path = legacy_manifest
            summary_path = job_dir / "pipeline_summary.json"

    manifest = _load_json(manifest_path)
    pipeline_summary = _load_json(summary_path)
    status_payload = _read_job_status(job_dir)
    job_status = status_payload.get("status") or ("ready" if manifest else "pending")

    manifest_summary = _summarize_manifest(manifest, job_id)
    status = "ready" if manifest else job_status
    facts: List[Dict[str, Any]] = []

    cells_path = manifest.get("cells")
    if isinstance(cells_path, str) and os.path.exists(cells_path):
        index_path = job_dir / "rag" / "cells_index.pkl"
        if not index_path.exists():
            build_index(cells_path, str(index_path))
        try:
            hits = hybrid_query(
                str(index_path),
                cells_path,
                q_text=query,
                topk=8,
            )
        except Exception:
            hits = []
        for score, cell in hits:
            text = cell.get("text") or ""
            if not text:
                continue
            facts.append(
                {
                    "trace": cell.get("trace") or _build_trace_label(cell),
                    "text": text,
                    "score": float(score),
                }
            )

    if status in {"queued", "running"} and not manifest:
        query_label = query.strip() or "<empty>"
        prefix = f"Received query '{query_label}' for job {job_id}."
        answer = {
            "data_summary": f"ジョブ {job_id} は現在 {status} です。",
            "business_commentary": f"{prefix} Job is {status}; retry after completion.",
        }
    else:
        query_label = query.strip() or "<empty>"
        prefix = f"Received query '{query_label}' for job {job_id}."
        answer = {
            "data_summary": (
                "\n".join(f"- {fact['text']}" for fact in facts)
                if facts
                else manifest_summary["data_summary"]
            ),
            "business_commentary": (
                f"{prefix} No indexed facts found; refine the query or rebuild the index."
                if not facts
                else f"{prefix} Retrieved {len(facts)} facts; review them for trends and deltas."
            ),
        }

    artifacts: Dict[str, Any] = {
        "tables": _load_tables_from_manifest(manifest),
        "charts": [],
        "sources": {
            "manifest": _existing(manifest_path),
            "pipeline_summary": _existing(summary_path),
            "doc_jsonl": _existing(job_dir / "doc.mm.jsonl"),
            "sql_dir": _existing(job_dir / "sql"),
        },
        "metadata": {"pipeline_summary": pipeline_summary, "manifest": manifest},
    }

    provenance = _provenance_from_manifest(manifest)
    if facts:
        provenance.extend(
            {
                "trace": fact.get("trace", ""),
                "fact_text": fact.get("text", ""),
            }
            for fact in facts
        )

    flags = {"facts_insufficient": (not bool(manifest)) and (not bool(facts))}

    return QueryResult(
        job_id=job_id,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        mode=mode,
        answer=answer,
        artifacts=artifacts,
        provenance=provenance,
        flags=flags,
        status=status,
    )


def validate_ingest_request_payload(payload: Dict[str, Any]) -> None:
    """Validate a payload against the canonical ingest request schema."""

    Draft202012Validator(INGEST_REQUEST_SCHEMA_V0).validate(payload)


def validate_query_request_payload(payload: Dict[str, Any]) -> None:
    """Validate a payload against the canonical query request schema."""

    Draft202012Validator(QUERY_REQUEST_SCHEMA_V0).validate(payload)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ingest_response_payload_v0(result: IngestResult) -> Dict[str, Any]:
    """Map an ``IngestResult`` to the canonical v0 response schema."""

    payload: Dict[str, Any] = {
        "job_id": result.job_id,
        "tenant_id": result.tenant_id,
        "status": result.status,
        "created_at": _iso_now(),
    }

    if result.summary is not None:
        payload["summary"] = result.summary
        estimated = None
        if isinstance(result.summary, dict):
            estimated = result.summary.get("page_count") or result.summary.get("pages")
        if estimated is not None:
            payload["estimated_pages"] = estimated
    if result.error:
        payload["error"] = result.error

    return payload


def query_response_payload_v0(result: QueryResult) -> Dict[str, Any]:
    """Map a ``QueryResult`` to the canonical v0 response schema."""

    tables = result.artifacts.get("tables") or []
    charts = result.artifacts.get("charts") or []
    provenance = result.provenance or []
    flags = result.flags or {}

    return {
        "answer": result.answer,
        "artifacts": {"tables": tables, "charts": charts},
        "provenance": provenance,
        "flags": flags,
    }
