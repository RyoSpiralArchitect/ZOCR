"""FastAPI application exposing the ingest/query surfaces.

This module keeps the HTTP wiring lightweight and reuses the validation and
response mappers defined in :mod:`zocr.api_http`. Callers can provide custom
``ingest_runner`` / ``query_runner`` callbacks for dependency injection in
tests or to wrap job orchestration in their own queues.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Tuple

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from jsonschema import Draft202012Validator, ValidationError

from zocr.api_http import ingest_from_payload, query_from_payload
from zocr.api_spec import INGEST_REQUEST_SCHEMA_V0, QUERY_REQUEST_SCHEMA_V0

__all__ = ["create_app"]

Runner = Callable[[Dict[str, Any]], Tuple[Dict[str, Any], Dict[str, Any]]]

_DEFAULT_RATE_LIMIT_PER_MIN = 60
_DEFAULT_MAX_BODY_BYTES = 2 * 1024 * 1024
_DEFAULT_REQUEST_TIMEOUT_S = 30.0


class _RateLimiter:
    def __init__(self, *, limit_per_minute: int) -> None:
        self._limit = max(1, int(limit_per_minute))
        self._hits: Dict[str, list[float]] = {}

    def allow(self, key: str, now: float) -> bool:
        window_start = now - 60.0
        hits = self._hits.setdefault(key, [])
        while hits and hits[0] < window_start:
            hits.pop(0)
        if len(hits) >= self._limit:
            return False
        hits.append(now)
        return True


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
    rate_limit_per_minute: int = _DEFAULT_RATE_LIMIT_PER_MIN,
    max_body_bytes: int = _DEFAULT_MAX_BODY_BYTES,
    request_timeout_s: float = _DEFAULT_REQUEST_TIMEOUT_S,
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
    limiter = _RateLimiter(limit_per_minute=rate_limit_per_minute)

    @app.middleware("http")
    async def _limits_middleware(request: Request, call_next):
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client, anyio.current_time()):
            return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})
        if request.headers.get("content-length"):
            try:
                length = int(request.headers["content-length"])
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "invalid content-length"})
            if length > max_body_bytes:
                return JSONResponse(status_code=413, content={"detail": "payload too large"})
        return await call_next(request)

    safe_ingest = _wrap_validation_errors(ingest_cb)
    safe_query = _wrap_validation_errors(query_cb)

    @app.get("/healthz")
    def healthcheck() -> Dict[str, Any]:
        errors = []
        for path in (out_root, base_dir):
            try:
                os.makedirs(path, exist_ok=True)
            except OSError as exc:
                errors.append(f"{path}: {exc}")
        status = "ok" if not errors else "degraded"
        return {"status": status, "errors": errors}

    @app.post("/ingest")
    async def ingest(payload: Dict[str, Any]):
        try:
            Draft202012Validator(INGEST_REQUEST_SCHEMA_V0).validate(payload)
            with anyio.fail_after(request_timeout_s):
                _, response = await anyio.to_thread.run_sync(safe_ingest, payload)
            return response
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="request timeout") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/query")
    async def query(payload: Dict[str, Any]):
        try:
            Draft202012Validator(QUERY_REQUEST_SCHEMA_V0).validate(payload)
            with anyio.fail_after(request_timeout_s):
                _, response = await anyio.to_thread.run_sync(safe_query, payload)
            return response
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="request timeout") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app
