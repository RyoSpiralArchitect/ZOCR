"""HTTP-friendly helpers that wrap the ingest/query primitives.

These utilities keep a minimal dependency footprint while providing a
straightforward way for web handlers (FastAPI, Flask, Lambda, etc.) to:

- Validate incoming JSON bodies against the v0 schemas,
- Convert them into ``IngestRequest`` / ``QueryRequest`` instances, and
- Emit public response payloads that conform to the canonical schemas.

The goal is to avoid duplicating validation/mapping glue across services while
still leaving concurrency/job-management choices to the caller.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from jsonschema import Draft202012Validator

from zocr.api import (
    IngestRequest,
    QueryRequest,
    ingest_job,
    ingest_response_payload_v0,
    query_job,
    query_response_payload_v0,
)
from zocr.api_spec import (
    INGEST_REQUEST_SCHEMA_V0,
    INGEST_RESPONSE_SCHEMA_V0,
    QUERY_REQUEST_SCHEMA_V0,
    QUERY_RESPONSE_SCHEMA_V0,
)

__all__ = [
    "ingest_request_from_payload",
    "query_request_from_payload",
    "ingest_from_payload",
    "query_from_payload",
]


def ingest_request_from_payload(
    payload: Dict[str, Any],
    *,
    out_root: str = "episodes",
    resume: bool = False,
    dry_run: bool = False,
    pipeline_kwargs: Dict[str, Any] | None = None,
) -> IngestRequest:
    """Validate and normalize an ``IngestRequest`` from a JSON body.

    ``options`` fields present in the payload are forwarded to the pipeline
    kwargs to keep the public contract and orchestrator behaviour aligned.
    """

    Draft202012Validator(INGEST_REQUEST_SCHEMA_V0).validate(payload)
    options = payload.get("options", {}) or {}
    pipeline_kw = dict(pipeline_kwargs or {})
    for key in ("snapshot", "seed", "priority"):
        if key in options and options[key] is not None:
            pipeline_kw.setdefault(key, options[key])

    return IngestRequest(
        tenant_id=payload["tenant_id"],
        files=payload["files"],
        domain_hint=payload.get("domain_hint"),
        out_root=out_root,
        resume=resume,
        dry_run=dry_run,
        pipeline_kwargs=pipeline_kw,
    )


def query_request_from_payload(
    payload: Dict[str, Any],
    *,
    base_dir: str = "episodes",
) -> QueryRequest:
    """Validate and normalize a ``QueryRequest`` from a JSON body."""

    Draft202012Validator(QUERY_REQUEST_SCHEMA_V0).validate(payload)
    return QueryRequest(
        tenant_id=payload["tenant_id"],
        job_id=payload["job_id"],
        conversation_id=payload.get("conversation_id"),
        query=payload.get("query", ""),
        language=payload.get("language", "ja"),
        mode=payload.get("mode", "analysis"),
        base_dir=base_dir,
    )


def ingest_from_payload(
    payload: Dict[str, Any],
    *,
    out_root: str = "episodes",
    resume: bool = False,
    dry_run: bool = False,
    pipeline_kwargs: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run ingestion from an HTTP payload and return (result, response payload)."""

    request = ingest_request_from_payload(
        payload,
        out_root=out_root,
        resume=resume,
        dry_run=dry_run,
        pipeline_kwargs=pipeline_kwargs,
    )
    result = ingest_job(request)
    response = ingest_response_payload_v0(result)
    Draft202012Validator(INGEST_RESPONSE_SCHEMA_V0).validate(response)
    return result.__dict__, response


def query_from_payload(
    payload: Dict[str, Any],
    *,
    base_dir: str = "episodes",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Answer a query from an HTTP payload and return (result, response payload)."""

    request = query_request_from_payload(payload, base_dir=base_dir)
    result = query_job(
        job_id=request.job_id,
        tenant_id=request.tenant_id,
        base_dir=request.base_dir,
        conversation_id=request.conversation_id,
        query=request.query,
        mode=request.mode,
    )
    response = query_response_payload_v0(result)
    Draft202012Validator(QUERY_RESPONSE_SCHEMA_V0).validate(response)
    return result.__dict__, response
