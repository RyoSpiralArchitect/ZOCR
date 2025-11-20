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
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from zocr.orchestrator import zocr_pipeline

__all__ = ["IngestRequest", "IngestResult", "QueryResult", "ingest_job", "query_job"]


@dataclass
class IngestRequest:
    """Parameters controlling an ingestion job."""

    tenant_id: str
    files: Iterable[str]
    domain_hint: Optional[str] = None
    job_id: Optional[str] = None
    out_root: str = "episodes"
    resume: bool = False
    dry_run: bool = False
    pipeline_kwargs: Dict[str, Any] = field(default_factory=dict)

    def resolved_job_id(self) -> str:
        return self.job_id or f"episode_{uuid.uuid4().hex[:12]}"


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
    status: str


def _existing(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def ingest_job(request: IngestRequest) -> IngestResult:
    """Run the orchestrator and expose canonical artifact paths.

    The call is synchronous by default. Set ``request.dry_run`` to avoid running
    the pipeline while still returning the computed job identifiers and paths.
    Additional keyword arguments for the pipeline can be supplied via
    ``request.pipeline_kwargs``.
    """

    job_id = request.resolved_job_id()
    outdir = Path(request.out_root) / job_id
    os.makedirs(outdir, exist_ok=True)

    artifact_paths = {
        "pipeline_summary": str(outdir / "pipeline_summary.json"),
        "doc_jsonl": str(outdir / "doc.mm.jsonl"),
        "rag_manifest": str(outdir / "rag" / "manifest.json"),
        "sql_dir": str(outdir / "sql"),
    }

    if request.dry_run:
        return IngestResult(
            job_id=job_id,
            tenant_id=request.tenant_id,
            status="queued",
            outdir=str(outdir),
            artifacts=artifact_paths,
            summary=None,
            error=None,
        )

    try:
        summary = zocr_pipeline.run_full_pipeline(
            list(request.files),
            str(outdir),
            domain_hint=request.domain_hint,
            resume=request.resume,
            **request.pipeline_kwargs,
        )
        status = "succeeded"
        error = None
    except Exception as exc:  # pragma: no cover - surfaced to caller
        summary = None
        status = "failed"
        error = str(exc)

    return IngestResult(
        job_id=job_id,
        tenant_id=request.tenant_id,
        status=status,
        outdir=str(outdir),
        artifacts={k: _existing(Path(v)) or v for k, v in artifact_paths.items()},
        summary=summary,
        error=error,
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

    job_dir = Path(base_dir) / job_id
    manifest_path = job_dir / "rag" / "manifest.json"
    summary_path = job_dir / "pipeline_summary.json"

    manifest = _load_json(manifest_path)
    pipeline_summary = _load_json(summary_path)

    manifest_summary = _summarize_manifest(manifest, job_id)
    status = "ready" if manifest else "pending"
    answer = {
        "data_summary": manifest_summary["data_summary"],
        "business_commentary": f"Received query='{query}' for job {job_id} (mode={mode}).",
    }

    artifacts: Dict[str, Any] = {
        "tables": [manifest_summary.get("table_meta")],
        "charts": [],
        "sources": {
            "manifest": _existing(manifest_path),
            "pipeline_summary": _existing(summary_path),
            "doc_jsonl": _existing(job_dir / "doc.mm.jsonl"),
            "sql_dir": _existing(job_dir / "sql"),
        },
        "metadata": {"pipeline_summary": pipeline_summary},
    }

    provenance = _provenance_from_manifest(manifest)

    return QueryResult(
        job_id=job_id,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        mode=mode,
        answer=answer,
        artifacts=artifacts,
        provenance=provenance,
        status=status,
    )
