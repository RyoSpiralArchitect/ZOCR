"""FastAPI application exposing the ingest/query surfaces.

This module keeps the HTTP wiring lightweight and reuses the validation and
response mappers defined in :mod:`zocr.api_http`. Callers can provide custom
``ingest_runner`` / ``query_runner`` callbacks for dependency injection in
tests or to wrap job orchestration in their own queues.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from fastapi import FastAPI, HTTPException
from jsonschema import Draft202012Validator, ValidationError

from zocr.api_http import ingest_from_payload, query_from_payload
from zocr.api_spec import INGEST_REQUEST_SCHEMA_V0, QUERY_REQUEST_SCHEMA_V0

__all__ = ["create_app"]

Runner = Callable[[Dict[str, Any]], Tuple[Dict[str, Any], Dict[str, Any]]]


def _wrap_validation_errors(fn: Runner) -> Runner:
    def _wrapped(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            return fn(payload)
        except ValidationError as exc:  # pragma: no cover - exercised via API
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _wrapped


def create_app(
    *,
    out_root: str = "episodes",
    resume: bool = False,
    dry_run: bool = False,
    base_dir: str = "episodes",
    pipeline_kwargs: Dict[str, Any] | None = None,
    ingest_runner: Runner | None = None,
    query_runner: Runner | None = None,
) -> FastAPI:
    """Return a FastAPI instance exposing ``/ingest`` and ``/query``.

    Parameters mirror those used by :func:`zocr.api_http.ingest_from_payload` and
    :func:`zocr.api_http.query_from_payload` so services can tune output roots or
    pipeline kwargs while keeping the HTTP contract stable.
    """

    ingest_cb: Runner = ingest_runner or (
        lambda payload: ingest_from_payload(
            payload,
            out_root=out_root,
            resume=resume,
            dry_run=dry_run,
            pipeline_kwargs=pipeline_kwargs,
        )
    )
    query_cb: Runner = query_runner or (
        lambda payload: query_from_payload(payload, base_dir=base_dir)
    )

    app = FastAPI(title="ZOCR API", version="0.1.0")

    safe_ingest = _wrap_validation_errors(ingest_cb)
    safe_query = _wrap_validation_errors(query_cb)

    @app.get("/healthz")
    def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest")
    def ingest(payload: Dict[str, Any]):
        try:
            Draft202012Validator(INGEST_REQUEST_SCHEMA_V0).validate(payload)
            _, response = safe_ingest(payload)
            return response
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/query")
    def query(payload: Dict[str, Any]):
        try:
            Draft202012Validator(QUERY_REQUEST_SCHEMA_V0).validate(payload)
            _, response = safe_query(payload)
            return response
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app

