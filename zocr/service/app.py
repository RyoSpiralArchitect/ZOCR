import asyncio
import functools
import logging
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from .._version import __version__
from ..artifacts.manifest import MANIFEST_FILENAME, write_manifest
from ..utils.json_utils import json_ready

_UPLOAD_CHUNK_BYTES = 1024 * 1024
_JOB_SCHEMA = "zocr.api_job"
_JOB_SCHEMA_VERSION = 1
_JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _zip_dir(src_dir: str, dest_zip: str) -> None:
    compression_raw = (os.environ.get("ZOCR_API_ZIP_COMPRESSION") or "deflated").strip().lower()
    if compression_raw in {"store", "stored", "none", "0"}:
        compression = zipfile.ZIP_STORED
    else:
        compression = zipfile.ZIP_DEFLATED

    compresslevel = None
    raw_level = os.environ.get("ZOCR_API_ZIP_COMPRESSLEVEL")
    if raw_level is not None and raw_level.strip():
        try:
            compresslevel = max(0, min(9, int(raw_level)))
        except ValueError:
            compresslevel = None

    src_path = Path(src_dir)
    zip_kwargs = {}
    if compression == zipfile.ZIP_DEFLATED and compresslevel is not None:
        zip_kwargs["compresslevel"] = compresslevel
    with zipfile.ZipFile(dest_zip, "w", compression=compression, **zip_kwargs) as zf:
        for path in src_path.rglob("*"):
            if path.is_dir():
                continue
            rel = path.relative_to(src_path)
            zf.write(path, rel.as_posix())


def _cleanup_tree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _allowed_suffix(filename: str) -> bool:
    suffix = Path(filename or "").suffix.lower()
    return suffix in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def create_app():
    try:
        from fastapi import BackgroundTasks, Body, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
        from fastapi.responses import FileResponse, RedirectResponse, Response
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "FastAPI dependencies are missing. Install with `pip install -e '.[api]'` "
            "(or `pip install 'zocr-suite[api]'`)."
        ) from exc

    try:
        import python_multipart  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "python-multipart is required for file uploads. Install with `pip install -e '.[api]'` "
            "(or `pip install 'zocr-suite[api]'`)."
        ) from exc

    try:
        import anyio
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("anyio is required (it should be installed with FastAPI).") from exc

    from zocr.orchestrator.zocr_pipeline import run_full_pipeline
    from .auth import AuthError, Principal, RequestAuthContext, build_auth_backend
    from .metadata import build_job_repository, default_tenant_from_env, normalize_tenant_id
    from .policy import (
        build_audit_logger,
        build_rate_limiter,
        build_tenant_approval_notifier,
        redis_rate_limit_settings_from_env,
    )
    from .queue import build_queue_backend, build_redis_queue, redis_settings_from_env
    from .storage import build_job_store

    storage_backend, job_store = build_job_store()
    metadata_backend, job_repository = build_job_repository(job_store)
    queue_backend = build_queue_backend()
    redis_queue = None
    redis_queue_settings = None
    if queue_backend == "redis":
        redis_queue_settings = redis_settings_from_env()
        redis_queue = build_redis_queue()
        redis_queue.ensure_consumer_group(redis_queue_settings.group)
    elif queue_backend != "inline":
        raise RuntimeError(
            f"Unsupported ZOCR_API_QUEUE_BACKEND={queue_backend!r}. Supported backends: inline, redis."
        )

    max_upload_mb = max(1, _env_int("ZOCR_API_MAX_UPLOAD_MB", 50))
    max_upload_bytes = int(max_upload_mb) * 1024 * 1024
    run_timeout_sec = _env_float("ZOCR_API_RUN_TIMEOUT_SEC", 15 * 60.0)
    if run_timeout_sec <= 0:
        run_timeout_sec = 0.0
    concurrency = max(1, _env_int("ZOCR_API_CONCURRENCY", 1))
    run_semaphore = asyncio.BoundedSemaphore(concurrency)
    jobs_max_count = max(0, _env_int("ZOCR_API_JOBS_MAX_COUNT", 200))
    jobs_ttl_hours = max(0.0, _env_float("ZOCR_API_JOBS_TTL_HOURS", 7.0 * 24.0))
    jobs_resume_on_startup = _env_truthy("ZOCR_API_JOBS_RESUME_ON_STARTUP", True)
    jobs_cleanup_on_startup = _env_truthy("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", True)
    jobs_cleanup_on_create = _env_truthy("ZOCR_API_JOBS_CLEANUP_ON_CREATE", True)
    log_format = (os.environ.get("ZOCR_API_LOG_FORMAT") or "json").strip().lower()
    log_level = (os.environ.get("ZOCR_API_LOG_LEVEL") or "INFO").strip().upper()
    metrics_enabled = _env_truthy("ZOCR_API_METRICS_ENABLED", True)
    multi_tenant_enabled = _env_truthy("ZOCR_API_MULTI_TENANT_ENABLED", False)
    default_tenant = default_tenant_from_env(multi_tenant_enabled=multi_tenant_enabled)
    auth_backend = build_auth_backend(
        multi_tenant_enabled=multi_tenant_enabled,
        default_tenant=default_tenant,
    )
    rate_limit_per_min = max(0, _env_int("ZOCR_API_RATE_LIMIT_PER_MIN", 0))
    tenant_max_active_jobs = max(0, _env_int("ZOCR_API_TENANT_MAX_ACTIVE_JOBS", 0))
    tenant_max_stored_jobs = max(0, _env_int("ZOCR_API_TENANT_MAX_STORED_JOBS", 0))
    tenant_approval_required = _env_truthy("ZOCR_API_TENANT_APPROVAL_REQUIRED", False)
    tenant_approval_allow_self = _env_truthy("ZOCR_API_TENANT_APPROVAL_ALLOW_SELF", False)
    tenant_approval_min_approvers = max(1, _env_int("ZOCR_API_TENANT_APPROVAL_MIN_APPROVERS", 1))
    rate_limit_settings = redis_rate_limit_settings_from_env(
        default_url=redis_queue_settings.url if redis_queue_settings else None
    )
    rate_limit_backend, rate_limiter = build_rate_limiter(
        limit_per_min=rate_limit_per_min,
        default_redis_url=rate_limit_settings.url,
    )
    audit_settings, audit_logger = build_audit_logger(
        default_dsn=os.environ.get("ZOCR_API_DATABASE_URL") or ""
    )
    tenant_approval_notification_settings, tenant_approval_notifier = build_tenant_approval_notifier()

    logger = logging.getLogger("zocr.api")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.propagate = False

    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    storage_dir = job_store.storage_dir
    jobs_dir = job_store.jobs_dir
    active_jobs: set[str] = set()
    metrics_lock = Lock()
    http_requests_total: Counter[tuple[str, str, str]] = Counter()
    http_duration_sum: dict[tuple[str, str], float] = defaultdict(float)
    http_duration_count: Counter[tuple[str, str]] = Counter()
    job_creates_total = 0
    job_completions_total: Counter[str] = Counter()
    job_duration_sum: dict[str, float] = defaultdict(float)
    job_duration_count: Counter[str] = Counter()

    def _route_template(request: Request) -> str:
        route = request.scope.get("route")
        path = getattr(route, "path", None)
        if isinstance(path, str) and path:
            return path
        return request.url.path

    def _log(event: str, payload: Dict[str, Any], *, level: str = "info") -> None:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        if not logger.isEnabledFor(numeric_level):
            return
        record = {"ts": _utc_now_iso(), "event": event, **payload}
        if log_format == "json":
            msg = json.dumps(record, ensure_ascii=False)
        else:
            msg = f"{record.get('ts')} {event} {payload}"
        fn = getattr(logger, level, logger.info)
        fn(msg)

    def _audit(
        event: str,
        payload: Dict[str, Any],
        *,
        request: Optional[Request] = None,
        principal: Optional[Principal] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        if not audit_logger.enabled:
            return
        request_id = None
        client = None
        if request is not None:
            request_id = getattr(request.state, "request_id", None)
            client = request.client.host if request.client else None
            if principal is None:
                principal = getattr(request.state, "principal", None)
            if tenant_id is None:
                tenant_id = getattr(request.state, "tenant_id", None)
        audit_payload: Dict[str, Any] = dict(payload)
        if request_id:
            audit_payload.setdefault("request_id", request_id)
        if request is not None:
            audit_payload.setdefault("method", request.method)
            audit_payload.setdefault("path", _route_template(request))
        if client:
            audit_payload.setdefault("client", client)
        if principal is not None:
            audit_payload.setdefault("auth_mode", principal.auth_mode)
            audit_payload.setdefault("subject", principal.subject)
            audit_payload.setdefault("roles", list(principal.roles))
            audit_payload.setdefault("scopes", list(principal.scopes))
            audit_payload.setdefault("is_admin", bool(principal.is_admin))
            if tenant_id is None:
                tenant_id = principal.tenant_id
        if tenant_id:
            audit_payload.setdefault("tenant_id", tenant_id)
        audit_logger.write(event, audit_payload)

    def _prom_escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')

    def _render_metrics() -> str:
        lines: list[str] = []
        lines.append("# HELP zocr_api_http_requests_total Total HTTP requests.")
        lines.append("# TYPE zocr_api_http_requests_total counter")
        lines.append("# HELP zocr_api_http_request_duration_seconds Request duration (sum/count).")
        lines.append("# TYPE zocr_api_http_request_duration_seconds summary")
        lines.append("# HELP zocr_api_jobs_active Active job workers.")
        lines.append("# TYPE zocr_api_jobs_active gauge")
        lines.append("# HELP zocr_api_job_creates_total Total created jobs.")
        lines.append("# TYPE zocr_api_job_creates_total counter")
        lines.append("# HELP zocr_api_job_completions_total Job completions by status.")
        lines.append("# TYPE zocr_api_job_completions_total counter")
        lines.append("# HELP zocr_api_job_duration_seconds Job runtime duration by status (sum/count).")
        lines.append("# TYPE zocr_api_job_duration_seconds summary")

        with metrics_lock:
            for (method, path, status), count in sorted(http_requests_total.items()):
                lines.append(
                    'zocr_api_http_requests_total{method="%s",path="%s",status="%s"} %d'
                    % (_prom_escape(method), _prom_escape(path), _prom_escape(status), int(count))
                )
            for (method, path), total in sorted(http_duration_sum.items()):
                count = int(http_duration_count.get((method, path), 0))
                lines.append(
                    'zocr_api_http_request_duration_seconds_sum{method="%s",path="%s"} %.6f'
                    % (_prom_escape(method), _prom_escape(path), float(total))
                )
                lines.append(
                    'zocr_api_http_request_duration_seconds_count{method="%s",path="%s"} %d'
                    % (_prom_escape(method), _prom_escape(path), count)
                )
            lines.append("zocr_api_jobs_active %d" % (len(active_jobs),))
            lines.append("zocr_api_job_creates_total %d" % (int(job_creates_total),))
            for status, count in sorted(job_completions_total.items()):
                lines.append(
                    'zocr_api_job_completions_total{status="%s"} %d'
                    % (_prom_escape(status), int(count))
                )
            for status, total in sorted(job_duration_sum.items()):
                count = int(job_duration_count.get(status, 0))
                lines.append(
                    'zocr_api_job_duration_seconds_sum{status="%s"} %.6f'
                    % (_prom_escape(status), float(total))
                )
                lines.append(
                    'zocr_api_job_duration_seconds_count{status="%s"} %d'
                    % (_prom_escape(status), count)
                )

        return "\n".join(lines) + "\n"

    def _validate_job_id(job_id: str) -> None:
        if not _JOB_ID_RE.match(job_id or ""):
            raise HTTPException(status_code=404, detail="Unknown job id")

    def _tenant_job_key(job_id: str, tenant_id: str) -> str:
        return f"{tenant_id}:{job_id}"

    def _rate_limit_scope_label(required_scopes: tuple[str, ...]) -> str:
        if not required_scopes:
            return "unscoped"
        if len(required_scopes) == 1:
            return required_scopes[0]
        return ",".join(sorted(required_scopes))

    def _effective_tenant_policy(tenant_id: Optional[str], *, request: Optional[Request] = None) -> Dict[str, Any]:
        cached = getattr(request.state, "tenant_policy", None) if request is not None else None
        cached_tenant = cached.get("tenant_id") if isinstance(cached, dict) else None
        if cached and cached_tenant == tenant_id:
            return dict(cached)

        policy: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "source": "env",
            "plan_name": None,
            "rate_limit_per_min": rate_limit_per_min,
            "max_active_jobs": tenant_max_active_jobs,
            "max_stored_jobs": tenant_max_stored_jobs,
        }
        if tenant_id:
            try:
                resolved = job_repository.resolve_tenant_policy(tenant_id)
            except Exception:
                resolved = None
            if resolved is not None:
                policy["source"] = resolved.source
                policy["plan_name"] = resolved.plan_name
                if resolved.rate_limit_per_min is not None:
                    policy["rate_limit_per_min"] = resolved.rate_limit_per_min
                if resolved.max_active_jobs is not None:
                    policy["max_active_jobs"] = resolved.max_active_jobs
                if resolved.max_stored_jobs is not None:
                    policy["max_stored_jobs"] = resolved.max_stored_jobs
        if request is not None:
            request.state.tenant_policy = dict(policy)
        return policy

    def _parse_query_datetime(name: str, value: Optional[str]) -> Optional[datetime]:
        if value is None:
            return None
        parsed = _parse_iso_datetime(value)
        if parsed is None:
            raise HTTPException(status_code=400, detail=f"Invalid {name} timestamp")
        return parsed.astimezone(timezone.utc)

    def _request_change(
        *,
        target_type: str,
        target_id: str,
        action: str,
        payload: Optional[Dict[str, Any]],
        principal: Principal,
    ):
        return job_repository.create_tenant_change_request(
            {
                "target_type": target_type,
                "target_id": target_id,
                "action": action,
                "payload": dict(payload or {}),
                "requested_by": principal.subject,
            }
        )

    def _resolve_audit_tenant_filter(principal: Principal, tenant_id: Optional[str]) -> Optional[str]:
        requested_tenant = None
        if tenant_id is not None and str(tenant_id).strip():
            try:
                requested_tenant = normalize_tenant_id(str(tenant_id))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not auth_backend.enabled or auth_backend.can_override_tenant(principal):
            return requested_tenant
        principal_tenant = principal.tenant_id
        if requested_tenant and requested_tenant != principal_tenant:
            raise HTTPException(status_code=403, detail="Tenant override not allowed")
        return principal_tenant

    def _approve_change_request(
        request_id: str,
        *,
        principal: Principal,
        review_reason: Optional[str] = None,
    ):
        try:
            approved = job_repository.approve_tenant_change_request(
                request_id,
                reviewed_by=principal.subject,
                review_reason=review_reason,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to approve change request: {exc}") from exc
        if approved is None:
            raise HTTPException(status_code=404, detail="Unknown change request")
        return approved

    def _reject_change_request(
        request_id: str,
        *,
        principal: Principal,
        review_reason: Optional[str] = None,
    ):
        try:
            rejected = job_repository.reject_tenant_change_request(
                request_id,
                reviewed_by=principal.subject,
                review_reason=review_reason,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to reject change request: {exc}") from exc
        if rejected is None:
            raise HTTPException(status_code=404, detail="Unknown change request")
        return rejected

    def _change_request_tenant_id(change_request) -> Optional[str]:
        if getattr(change_request, "target_type", None) == "policy":
            return str(getattr(change_request, "target_id", "")).strip() or None
        return None

    def _change_request_response(change_request) -> Dict[str, Any]:
        return {"change_request": json_ready(change_request.__dict__)}

    def _change_request_audit_payload(change_request) -> Dict[str, Any]:
        return {
            "change_request_id": change_request.request_id,
            "target_type": change_request.target_type,
            "target_id": change_request.target_id,
            "action": change_request.action,
            "status": change_request.status,
            "approvals_required": getattr(change_request, "approvals_required", 1),
            "approvals_received": getattr(change_request, "approvals_received", 0),
            "requested_by": change_request.requested_by,
            "reviewed_by": change_request.reviewed_by,
            "review_reason": change_request.review_reason,
        }

    def _change_request_approval_event(change_request) -> str:
        if str(getattr(change_request, "status", "")) == "approved":
            return "tenant_change_request.approved"
        return "tenant_change_request.reviewed"

    def _notify_change_request(
        event: str,
        change_request,
        *,
        principal: Principal,
        request: Optional[Request] = None,
    ) -> None:
        if not tenant_approval_notifier.enabled:
            return
        tenant_approval_notifier.notify(
            event,
            {
                "source": "api",
                "tenant_id": _change_request_tenant_id(change_request),
                "http_request_id": getattr(request.state, "request_id", None) if request is not None else None,
                "actor": {
                    "subject": principal.subject,
                    "auth_mode": principal.auth_mode,
                    "tenant_id": principal.tenant_id,
                    "is_admin": bool(principal.is_admin),
                    "roles": list(principal.roles),
                    "scopes": list(principal.scopes),
                },
                "change_request": json_ready(change_request.__dict__),
            },
        )

    def _review_reason_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[str]:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Review payload must be an object")
        return str(payload.get("reason") or "").strip() or None

    def _find_tenant_plan(plan_name: str):
        for plan in job_repository.list_tenant_plans():
            if plan.plan_name == str(plan_name or "").strip():
                return plan
        return None

    def _apply_rate_limit(
        request: Request,
        *,
        principal: Principal,
        tenant_id: Optional[str],
        required_scopes: tuple[str, ...],
    ) -> None:
        policy = _effective_tenant_policy(tenant_id or principal.tenant_id, request=request)
        effective_limit = max(0, int(policy.get("rate_limit_per_min") or 0))
        request.state.rate_limit_limit = effective_limit
        request.state.rate_limit_remaining = effective_limit
        if effective_limit <= 0:
            return
        client = request.client.host if request.client else None
        key = rate_limiter.build_key(
            tenant_id=tenant_id or principal.tenant_id,
            subject=principal.subject,
            client=client,
        )
        decision = rate_limiter.check(key, limit_per_min=effective_limit)
        request.state.rate_limit_limit = decision.limit
        request.state.rate_limit_remaining = decision.remaining
        if decision.allowed:
            return
        scope_label = _rate_limit_scope_label(required_scopes)
        _audit(
            "policy.rate_limited",
            {
                "scope": scope_label,
                "plan_name": policy.get("plan_name"),
                "limit_per_min": decision.limit,
                "retry_after_sec": decision.retry_after_sec,
            },
            request=request,
            principal=principal,
            tenant_id=tenant_id or principal.tenant_id,
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {scope_label}",
            headers={"Retry-After": str(decision.retry_after_sec)},
        )

    def _tenant_quota_snapshot(tenant_id: str) -> Dict[str, int]:
        return {
            "active": int(job_repository.count_jobs(tenant_id, statuses=("queued", "running"))),
            "stored": int(job_repository.count_jobs(tenant_id)),
        }

    def _enforce_tenant_quotas(request: Request, *, principal: Principal, tenant_id: str) -> None:
        policy = _effective_tenant_policy(tenant_id, request=request)
        max_active_jobs = max(0, int(policy.get("max_active_jobs") or 0))
        max_stored_jobs = max(0, int(policy.get("max_stored_jobs") or 0))
        if max_active_jobs <= 0 and max_stored_jobs <= 0:
            return
        snapshot = _tenant_quota_snapshot(tenant_id)
        request.state.tenant_job_counts = dict(snapshot)
        if max_active_jobs > 0 and snapshot["active"] >= max_active_jobs:
            _audit(
                "policy.tenant_quota_denied",
                {
                    "quota": "active_jobs",
                    "plan_name": policy.get("plan_name"),
                    "limit": max_active_jobs,
                    "current": snapshot["active"],
                },
                request=request,
                principal=principal,
                tenant_id=tenant_id,
            )
            raise HTTPException(
                status_code=429,
                detail=f"Tenant active job quota exceeded ({snapshot['active']}/{max_active_jobs})",
            )
        if max_stored_jobs > 0 and snapshot["stored"] >= max_stored_jobs:
            _audit(
                "policy.tenant_quota_denied",
                {
                    "quota": "stored_jobs",
                    "plan_name": policy.get("plan_name"),
                    "limit": max_stored_jobs,
                    "current": snapshot["stored"],
                },
                request=request,
                principal=principal,
                tenant_id=tenant_id,
            )
            raise HTTPException(
                status_code=429,
                detail=f"Tenant stored job quota exceeded ({snapshot['stored']}/{max_stored_jobs})",
            )

    def _resolve_request_context(
        request: Request,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        authorization: Optional[str] = Header(default=None, alias="Authorization"),
        x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    ) -> RequestAuthContext:
        try:
            auth_ctx = auth_backend.authenticate(
                x_api_key=x_api_key,
                authorization=authorization,
                x_tenant_id=x_tenant_id,
            )
            request.state.principal = auth_ctx.principal
            request.state.auth_context = auth_ctx
            request.state.tenant_id = auth_ctx.tenant_id
            return auth_ctx
        except AuthError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    def _require_principal(*required_scopes: str):
        def _dependency(
            request: Request,
            x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
            authorization: Optional[str] = Header(default=None, alias="Authorization"),
        ) -> Principal:
            try:
                principal = auth_backend.authenticate_principal(
                    x_api_key=x_api_key,
                    authorization=authorization,
                )
                auth_backend.require_scopes(principal, *required_scopes)
                request.state.principal = principal
                request.state.tenant_id = principal.tenant_id
                _apply_rate_limit(
                    request,
                    principal=principal,
                    tenant_id=principal.tenant_id,
                    required_scopes=tuple(required_scopes),
                )
                return principal
            except AuthError as exc:
                raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

        return _dependency

    def _require_request_context(*required_scopes: str):
        def _dependency(
            request: Request,
            auth_ctx: RequestAuthContext = Depends(_resolve_request_context),
        ) -> RequestAuthContext:
            try:
                auth_backend.require_scopes(auth_ctx.principal, *required_scopes)
                _apply_rate_limit(
                    request,
                    principal=auth_ctx.principal,
                    tenant_id=auth_ctx.tenant_id,
                    required_scopes=tuple(required_scopes),
                )
            except AuthError as exc:
                raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
            return auth_ctx

        return _dependency

    def _job_paths(job_id: str) -> Dict[str, Path]:
        paths = job_store.job_paths(job_id)
        return {
            "root": paths.root,
            "job_json": paths.job_json,
            "input_dir": paths.input_dir,
            "out_dir": paths.out_dir,
            "artifacts_zip": paths.artifacts_zip,
        }

    def _read_job(job_id: str, tenant_id: str) -> Dict[str, Any]:
        _validate_job_id(job_id)
        try:
            data = job_repository.read_job(job_id, tenant_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Job metadata unreadable: {exc}") from exc
        if data is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        if not isinstance(data, dict):
            raise HTTPException(status_code=500, detail="Job metadata invalid")
        return data

    def _write_job(job_id: str, tenant_id: str, job: Dict[str, Any]) -> None:
        job_repository.write_job(job_id, tenant_id, job)

    def _parse_iso_datetime(value: Any) -> Optional[datetime]:
        if not isinstance(value, str) or not value.strip():
            return None
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except Exception:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _cleanup_jobs() -> Dict[str, int]:
        """Best-effort cleanup for persisted jobs."""

        deleted = 0
        candidates = job_repository.list_job_refs()
        if not candidates:
            return {"deleted": 0, "kept": 0}

        job_infos: list[tuple[float, str, str, Optional[Dict[str, Any]]]] = []
        now = datetime.now(timezone.utc)
        for meta in candidates:
            job_obj = job_repository.read_job(meta.job_id, meta.tenant_id)
            mtime = float(meta.last_modified)
            created_at = _parse_iso_datetime(job_obj.get("created_at") if job_obj else None)
            if created_at is not None:
                age_hours = (now - created_at).total_seconds() / 3600.0
            else:
                age_hours = 0.0
            status = (job_obj or {}).get("status") if job_obj else None
            is_terminal = status in {"succeeded", "failed"}
            ttl_expired = bool(jobs_ttl_hours > 0 and is_terminal and age_hours > jobs_ttl_hours)
            if ttl_expired:
                job_repository.delete_job(meta.job_id, meta.tenant_id)
                job_store.delete_job(meta.job_id)
                deleted += 1
                continue
            job_infos.append((mtime, meta.tenant_id, meta.job_id, job_obj))

        if jobs_max_count <= 0:
            return {"deleted": deleted, "kept": len(job_infos)}

        # Enforce max-count on terminal jobs per tenant (avoid deleting queued/running).
        job_infos_by_tenant: dict[str, list[tuple[float, str, str, Optional[Dict[str, Any]]]]] = defaultdict(list)
        for info in job_infos:
            job_infos_by_tenant[info[1]].append(info)

        kept_count = 0
        for tenant_infos in job_infos_by_tenant.values():
            tenant_infos.sort(key=lambda item: item[0], reverse=True)
            kept: list[tuple[float, str, str, Optional[Dict[str, Any]]]] = []
            terminal: list[tuple[float, str, str, Optional[Dict[str, Any]]]] = []
            for info in tenant_infos:
                status = (info[3] or {}).get("status") if info[3] else None
                if status in {"succeeded", "failed"}:
                    terminal.append(info)
                else:
                    kept.append(info)
            remaining_terminal_budget = max(0, jobs_max_count - len(kept))
            kept.extend(terminal[:remaining_terminal_budget])
            kept_count += len(kept)
            for _, tenant_id, job_id, _ in terminal[remaining_terminal_budget:]:
                job_repository.delete_job(job_id, tenant_id)
                job_store.delete_job(job_id)
                deleted += 1
        return {"deleted": deleted, "kept": kept_count}

    def _schedule_job(job_id: str, tenant_id: str) -> Dict[str, Any]:
        job_key = _tenant_job_key(job_id, tenant_id)
        if queue_backend == "redis":
            if redis_queue is None:
                raise RuntimeError("Redis queue backend is not initialized.")
            message_id = redis_queue.enqueue(job_id, tenant_id)
            _log(
                "job_enqueued",
                {
                    "job_id": job_id,
                    "tenant_id": tenant_id,
                    "queue_backend": queue_backend,
                    "message_id": message_id,
                    "stream": redis_queue_settings.stream if redis_queue_settings else None,
                },
            )
            return {"backend": queue_backend, "message_id": message_id}

        if job_key in active_jobs:
            return {"backend": queue_backend, "deduped": True}
        active_jobs.add(job_key)
        try:
            asyncio.create_task(_process_job(job_id, tenant_id))
        except Exception:
            active_jobs.discard(job_key)
            raise
        return {"backend": queue_backend}

    async def _save_upload(upload: UploadFile, dest: Path) -> int:
        written = 0
        try:
            with dest.open("wb") as f:
                while True:
                    chunk = await upload.read(_UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File too large (>{max_upload_mb} MB).",
                        )
                    f.write(chunk)
        finally:
            try:
                await upload.close()
            except Exception:
                pass
        return written

    async def _run_pipeline(*, ignore_cancel: bool = False, **kwargs):
        acquired = False
        try:
            while True:
                try:
                    await run_semaphore.acquire()
                    acquired = True
                    break
                except asyncio.CancelledError:
                    if not ignore_cancel:
                        raise
                    current_task = asyncio.current_task()
                    if current_task is not None:
                        try:
                            while current_task.cancelling():
                                current_task.uncancel()
                        except AttributeError:
                            pass
                    continue

            runner = functools.partial(run_full_pipeline, **kwargs)
            thread_task = asyncio.create_task(anyio.to_thread.run_sync(runner, abandon_on_cancel=True))
            waitable = asyncio.shield(thread_task) if ignore_cancel else thread_task
            try:
                if run_timeout_sec > 0:
                    return await asyncio.wait_for(waitable, timeout=run_timeout_sec)
                return await waitable
            except asyncio.CancelledError:
                if not ignore_cancel:
                    raise
                current_task = asyncio.current_task()
                if current_task is not None:
                    try:
                        while current_task.cancelling():
                            current_task.uncancel()
                    except AttributeError:
                        pass
                if run_timeout_sec > 0:
                    return await asyncio.wait_for(asyncio.shield(thread_task), timeout=run_timeout_sec)
                return await asyncio.shield(thread_task)
        finally:
            if acquired:
                run_semaphore.release()

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        if jobs_cleanup_on_startup:
            _cleanup_jobs()
        if jobs_resume_on_startup:
            candidates = job_repository.list_job_refs()
            for meta in candidates:
                job_id = meta.job_id
                if not _JOB_ID_RE.match(job_id):
                    continue
                job = job_repository.read_job(job_id, meta.tenant_id)
                if not job:
                    continue
                if job.get("status") not in {"queued", "running"}:
                    continue
                job["status"] = "queued"
                job["updated_at"] = _utc_now_iso()
                job["error"] = None
                _write_job(job_id, meta.tenant_id, job)
                _schedule_job(job_id, meta.tenant_id)
        yield

    app = FastAPI(
        title="Z-OCR Suite API (Reference)",
        version=__version__,
        description="Reference FastAPI wrapper around the Z-OCR orchestrator.",
        lifespan=_lifespan,
    )

    @app.middleware("http")
    async def _request_observability(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        t0 = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            dt = time.perf_counter() - t0
            status_code = int(getattr(response, "status_code", 500)) if response is not None else 500
            try:
                route_path = _route_template(request)
            except Exception:
                route_path = request.url.path
            if response is not None:
                try:
                    response.headers["X-Request-ID"] = request_id
                except Exception:
                    pass
                if hasattr(request.state, "rate_limit_limit"):
                    try:
                        response.headers["X-RateLimit-Limit"] = str(getattr(request.state, "rate_limit_limit", rate_limit_per_min))
                        response.headers["X-RateLimit-Remaining"] = str(
                            max(0, int(getattr(request.state, "rate_limit_remaining", rate_limit_per_min)))
                        )
                    except Exception:
                        pass
            if metrics_enabled:
                with metrics_lock:
                    http_requests_total[(request.method, route_path, str(status_code))] += 1
                    http_duration_sum[(request.method, route_path)] += float(dt)
                    http_duration_count[(request.method, route_path)] += 1
            client = request.client.host if request.client else None
            _log(
                "http_request",
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": route_path,
                    "status": status_code,
                    "duration_ms": int(round(dt * 1000.0)),
                    "client": client,
                },
                level="info" if status_code < 500 else "error",
            )
            _audit(
                "http.request",
                {
                    "status": status_code,
                    "duration_ms": int(round(dt * 1000.0)),
                    "route_path": route_path,
                    "job_id": getattr(request.state, "job_id", None),
                    "rate_limit_remaining": getattr(request.state, "rate_limit_remaining", None),
                    "tenant_job_counts": getattr(request.state, "tenant_job_counts", None),
                },
                request=request,
            )

    if metrics_enabled:
        @app.get("/metrics", dependencies=[Depends(_require_principal("zocr.metrics.read"))] if auth_backend.enabled else [])
        async def metrics() -> Response:
            return Response(_render_metrics(), media_type="text/plain; version=0.0.4")

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {
            "ok": True,
            "version": __version__,
            "auth": {
                "enabled": auth_backend.enabled,
                "mode": auth_backend.mode,
                "api_key_enabled": auth_backend.api_key_enabled,
                "jwt_enabled": auth_backend.jwt_enabled,
                "authz_strict": auth_backend.authz_strict,
                "role_presets": sorted(auth_backend.role_scopes),
                "implicit_allow_when_unscoped": not auth_backend.authz_strict,
                "jwt_shared_secret": bool(auth_backend.jwt and auth_backend.jwt.secret),
                "jwks_enabled": bool(auth_backend.jwks.enabled),
                "oidc_discovery_enabled": bool(auth_backend.jwks.oidc_discovery_url),
            },
            "limits": {"max_upload_mb": max_upload_mb, "concurrency": concurrency, "timeout_sec": run_timeout_sec},
            "storage": {"backend": storage_backend, "dir": str(storage_dir)},
            "metadata": {"backend": metadata_backend},
            "queue": {
                "backend": queue_backend,
                "stream": redis_queue_settings.stream if redis_queue_settings else None,
                "group": redis_queue_settings.group if redis_queue_settings else None,
            },
            "tenancy": {
                "multi_tenant_enabled": multi_tenant_enabled,
                "default_tenant": default_tenant or None,
                "header": "X-Tenant-ID",
            },
            "policies": {
                "rate_limit_per_min": rate_limit_per_min,
                "rate_limit_backend": rate_limit_backend,
                "rate_limit_window_sec": rate_limiter.window_sec,
                "rate_limit_redis_prefix": rate_limit_settings.prefix if rate_limit_backend == "redis" else None,
                "tenant_max_active_jobs": tenant_max_active_jobs,
                "tenant_max_stored_jobs": tenant_max_stored_jobs,
                "tenant_approval_required": tenant_approval_required,
                "tenant_approval_allow_self": tenant_approval_allow_self,
                "tenant_approval_min_approvers": tenant_approval_min_approvers,
                "tenant_approval_notify_enabled": tenant_approval_notification_settings.enabled,
                "tenant_approval_notify_url": tenant_approval_notification_settings.url,
                "tenant_policy_source": metadata_backend if metadata_backend == "postgres" else "env",
                "tenant_plans_configured": bool(os.environ.get("ZOCR_API_TENANT_PLANS_JSON")),
                "tenant_policies_configured": bool(os.environ.get("ZOCR_API_TENANT_POLICIES_JSON")),
                "audit_log_enabled": audit_logger.enabled,
                "audit_backends": list(audit_settings.backends),
                "audit_read_backend": audit_settings.read_backend,
                "audit_readable": audit_logger.readable,
                "audit_log_path": str(audit_logger.path) if audit_logger.path else None,
                "audit_http_url": audit_settings.http_url,
            },
            "jobs": {
                "dir": str(jobs_dir),
                "max_count": jobs_max_count,
                "ttl_hours": jobs_ttl_hours,
                "resume_on_startup": jobs_resume_on_startup,
            },
            "metrics": {"enabled": bool(metrics_enabled), "path": "/metrics"},
        }

    @app.get("/v1/admin/tenant-plans")
    async def list_tenant_plans(
        request: Request,
        principal: Principal = Depends(_require_principal("zocr.tenants.read")),
    ) -> Dict[str, Any]:
        items = [json_ready(plan.__dict__) for plan in job_repository.list_tenant_plans()]
        _audit(
            "tenant_plan.listed",
            {"count": len(items)},
            request=request,
            principal=principal,
        )
        return {"plans": items, "count": len(items)}

    @app.get("/v1/admin/tenant-change-requests")
    async def list_tenant_change_requests(
        request: Request,
        limit: int = 100,
        status: Optional[str] = None,
        target_type: Optional[str] = None,
        principal: Principal = Depends(_require_principal("zocr.tenants.read")),
    ) -> Dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        try:
            items = [
                json_ready(item.__dict__)
                for item in job_repository.list_tenant_change_requests(
                    status=status,
                    target_type=target_type,
                    limit=limit,
                )
            ]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _audit(
            "tenant_change_request.listed",
            {
                "count": len(items),
                "filters": {
                    "status": str(status).strip() or None if status is not None else None,
                    "target_type": str(target_type).strip() or None if target_type is not None else None,
                    "limit": limit,
                },
            },
            request=request,
            principal=principal,
        )
        return {"change_requests": items, "count": len(items)}

    @app.get("/v1/admin/audit-events")
    async def list_audit_events(
        request: Request,
        limit: int = 100,
        tenant_id: Optional[str] = None,
        event: Optional[str] = None,
        subject: Optional[str] = None,
        request_id: Optional[str] = None,
        contains: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        principal: Principal = Depends(_require_principal("zocr.audit.read")),
    ) -> Dict[str, Any]:
        if not audit_logger.readable:
            raise HTTPException(status_code=503, detail="Audit read backend is not configured")
        limit = max(1, min(int(limit), 200))
        tenant_filter = _resolve_audit_tenant_filter(principal, tenant_id)
        since_dt = _parse_query_datetime("since", since)
        until_dt = _parse_query_datetime("until", until)
        if since_dt and until_dt and since_dt > until_dt:
            raise HTTPException(status_code=400, detail="`since` must be earlier than `until`")
        items = audit_logger.search(
            limit=limit,
            tenant_id=tenant_filter,
            event=str(event).strip() or None if event is not None else None,
            subject=str(subject).strip() or None if subject is not None else None,
            request_id=str(request_id).strip() or None if request_id is not None else None,
            contains=str(contains).strip() or None if contains is not None else None,
            since=since_dt,
            until=until_dt,
        )
        filters = {
            "tenant_id": tenant_filter,
            "event": str(event).strip() or None if event is not None else None,
            "subject": str(subject).strip() or None if subject is not None else None,
            "request_id": str(request_id).strip() or None if request_id is not None else None,
            "contains": str(contains).strip() or None if contains is not None else None,
            "since": since_dt.isoformat() if since_dt else None,
            "until": until_dt.isoformat() if until_dt else None,
            "limit": limit,
        }
        _audit(
            "audit_event.listed",
            {
                "count": len(items),
                "backend": audit_logger.read_backend,
                "filters": filters,
            },
            request=request,
            principal=principal,
            tenant_id=tenant_filter or principal.tenant_id,
        )
        return {
            "events": items,
            "count": len(items),
            "backend": audit_logger.read_backend,
            "filters": filters,
        }

    @app.post("/v1/admin/tenant-change-requests/{request_id}/approve")
    async def approve_tenant_change_request(
        request: Request,
        request_id: str,
        payload: Optional[Dict[str, Any]] = Body(default=None),
        principal: Principal = Depends(_require_principal("zocr.tenants.approve")),
    ) -> Dict[str, Any]:
        approved = _approve_change_request(
            request_id,
            principal=principal,
            review_reason=_review_reason_from_payload(payload),
        )
        event = _change_request_approval_event(approved)
        _audit(
            event,
            _change_request_audit_payload(approved),
            request=request,
            principal=principal,
            tenant_id=_change_request_tenant_id(approved),
        )
        _notify_change_request(
            event,
            approved,
            principal=principal,
            request=request,
        )
        return _change_request_response(approved)

    @app.post("/v1/admin/tenant-change-requests/{request_id}/reject")
    async def reject_tenant_change_request(
        request: Request,
        request_id: str,
        payload: Optional[Dict[str, Any]] = Body(default=None),
        principal: Principal = Depends(_require_principal("zocr.tenants.approve")),
    ) -> Dict[str, Any]:
        rejected = _reject_change_request(
            request_id,
            principal=principal,
            review_reason=_review_reason_from_payload(payload),
        )
        _audit(
            "tenant_change_request.rejected",
            _change_request_audit_payload(rejected),
            request=request,
            principal=principal,
            tenant_id=_change_request_tenant_id(rejected),
        )
        _notify_change_request(
            "tenant_change_request.rejected",
            rejected,
            principal=principal,
            request=request,
        )
        return _change_request_response(rejected)

    @app.put("/v1/admin/tenant-plans/{plan_name}")
    async def upsert_tenant_plan(
        request: Request,
        response: Response,
        plan_name: str,
        payload: Dict[str, Any] = Body(...),
        principal: Principal = Depends(_require_principal("zocr.tenants.write")),
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Tenant plan payload must be an object")
        if tenant_approval_required:
            try:
                change_request = _request_change(
                    target_type="plan",
                    target_id=plan_name,
                    action="upsert",
                    payload=payload,
                    principal=principal,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to create change request: {exc}") from exc
            response.status_code = 202
            _audit(
                "tenant_change_request.created",
                _change_request_audit_payload(change_request),
                request=request,
                principal=principal,
            )
            _notify_change_request(
                "tenant_change_request.created",
                change_request,
                principal=principal,
                request=request,
            )
            return _change_request_response(change_request)
        try:
            plan = job_repository.upsert_tenant_plan(plan_name, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to store tenant plan: {exc}") from exc
        _audit(
            "tenant_plan.upserted",
            {"plan_name": plan.plan_name, "plan": json_ready(plan.__dict__)},
            request=request,
            principal=principal,
        )
        return {"plan": json_ready(plan.__dict__)}

    @app.delete("/v1/admin/tenant-plans/{plan_name}")
    async def delete_tenant_plan(
        request: Request,
        response: Response,
        plan_name: str,
        principal: Principal = Depends(_require_principal("zocr.tenants.write")),
    ) -> Dict[str, Any]:
        if tenant_approval_required:
            try:
                existing = _find_tenant_plan(plan_name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if existing is None or existing.source == "env":
                raise HTTPException(status_code=404, detail="Unknown tenant plan")
            try:
                change_request = _request_change(
                    target_type="plan",
                    target_id=plan_name,
                    action="delete",
                    payload=None,
                    principal=principal,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to create change request: {exc}") from exc
            response.status_code = 202
            _audit(
                "tenant_change_request.created",
                _change_request_audit_payload(change_request),
                request=request,
                principal=principal,
            )
            _notify_change_request(
                "tenant_change_request.created",
                change_request,
                principal=principal,
                request=request,
            )
            return _change_request_response(change_request)
        try:
            deleted = job_repository.delete_tenant_plan(plan_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to delete tenant plan: {exc}") from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Unknown tenant plan")
        _audit(
            "tenant_plan.deleted",
            {"plan_name": plan_name},
            request=request,
            principal=principal,
        )
        return {"ok": True, "plan_name": plan_name}

    @app.get("/v1/admin/tenant-policies")
    async def list_tenant_policies(
        request: Request,
        principal: Principal = Depends(_require_principal("zocr.tenants.read")),
    ) -> Dict[str, Any]:
        items = [json_ready(policy.__dict__) for policy in job_repository.list_tenant_policies()]
        _audit(
            "tenant_policy.listed",
            {"count": len(items)},
            request=request,
            principal=principal,
        )
        return {"policies": items, "count": len(items)}

    @app.put("/v1/admin/tenant-policies/{tenant_id}")
    async def upsert_tenant_policy(
        request: Request,
        response: Response,
        tenant_id: str,
        payload: Dict[str, Any] = Body(...),
        principal: Principal = Depends(_require_principal("zocr.tenants.write")),
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Tenant policy payload must be an object")
        if tenant_approval_required:
            try:
                change_request = _request_change(
                    target_type="policy",
                    target_id=tenant_id,
                    action="upsert",
                    payload=payload,
                    principal=principal,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to create change request: {exc}") from exc
            response.status_code = 202
            _audit(
                "tenant_change_request.created",
                _change_request_audit_payload(change_request),
                request=request,
                principal=principal,
                tenant_id=_change_request_tenant_id(change_request),
            )
            _notify_change_request(
                "tenant_change_request.created",
                change_request,
                principal=principal,
                request=request,
            )
            return _change_request_response(change_request)
        try:
            policy = job_repository.upsert_tenant_policy(tenant_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to store tenant policy: {exc}") from exc
        _audit(
            "tenant_policy.upserted",
            {"tenant_id": policy.tenant_id, "policy": json_ready(policy.__dict__)},
            request=request,
            principal=principal,
            tenant_id=policy.tenant_id,
        )
        return {"policy": json_ready(policy.__dict__)}

    @app.delete("/v1/admin/tenant-policies/{tenant_id}")
    async def delete_tenant_policy(
        request: Request,
        response: Response,
        tenant_id: str,
        principal: Principal = Depends(_require_principal("zocr.tenants.write")),
    ) -> Dict[str, Any]:
        if tenant_approval_required:
            try:
                existing = job_repository.resolve_tenant_policy(tenant_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to read tenant policy: {exc}") from exc
            if existing is None or existing.source == "env":
                raise HTTPException(status_code=404, detail="Unknown tenant policy")
            try:
                change_request = _request_change(
                    target_type="policy",
                    target_id=tenant_id,
                    action="delete",
                    payload=None,
                    principal=principal,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to create change request: {exc}") from exc
            response.status_code = 202
            _audit(
                "tenant_change_request.created",
                _change_request_audit_payload(change_request),
                request=request,
                principal=principal,
                tenant_id=_change_request_tenant_id(change_request),
            )
            _notify_change_request(
                "tenant_change_request.created",
                change_request,
                principal=principal,
                request=request,
            )
            return _change_request_response(change_request)
        try:
            deleted = job_repository.delete_tenant_policy(tenant_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to delete tenant policy: {exc}") from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Unknown tenant policy")
        _audit(
            "tenant_policy.deleted",
            {"tenant_id": tenant_id},
            request=request,
            principal=principal,
            tenant_id=tenant_id,
        )
        return {"ok": True, "tenant_id": tenant_id}

    @app.post("/v1/run", dependencies=[Depends(_require_principal("zocr.jobs.run"))])
    async def run_pipeline(
        file: UploadFile = File(...),
        *,
        domain: Optional[str] = None,
        dpi: int = 200,
        k: int = 10,
        seed: int = 24601,
        snapshot: bool = False,
        toy_lite: bool = False,
    ) -> Dict[str, Any]:
        filename = file.filename or "upload"
        if not _allowed_suffix(filename):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename!r}")

        with tempfile.TemporaryDirectory(prefix="zocr_api_") as tmp_root:
            tmp_root_path = Path(tmp_root)
            input_path = tmp_root_path / filename
            outdir = tmp_root_path / "out"
            outdir.mkdir(parents=True, exist_ok=True)

            await _save_upload(file, input_path)
            try:
                summary = await _run_pipeline(
                    inputs=[str(input_path)],
                    outdir=str(outdir),
                    dpi=int(dpi),
                    domain_hint=domain,
                    k=int(k),
                    seed=int(seed),
                    snapshot=bool(snapshot),
                    toy_lite=bool(toy_lite),
                )
            except TimeoutError as exc:
                raise HTTPException(status_code=504, detail="Pipeline timeout") from exc

            return {"summary": json_ready(summary)}

    @app.post("/v1/run.zip", dependencies=[Depends(_require_principal("zocr.jobs.run"))])
    async def run_pipeline_zip(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        *,
        domain: Optional[str] = None,
        dpi: int = 200,
        k: int = 10,
        seed: int = 24601,
        snapshot: bool = False,
        toy_lite: bool = False,
    ) -> FileResponse:
        filename = file.filename or "upload"
        if not _allowed_suffix(filename):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename!r}")

        tmp_root = tempfile.mkdtemp(prefix="zocr_api_zip_")
        background_tasks.add_task(_cleanup_tree, tmp_root)
        tmp_root_path = Path(tmp_root)
        input_path = tmp_root_path / filename
        outdir = tmp_root_path / "out"
        outdir.mkdir(parents=True, exist_ok=True)

        await _save_upload(file, input_path)
        try:
            await _run_pipeline(
                inputs=[str(input_path)],
                outdir=str(outdir),
                dpi=int(dpi),
                domain_hint=domain,
                k=int(k),
                seed=int(seed),
                snapshot=bool(snapshot),
                toy_lite=bool(toy_lite),
            )
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Pipeline timeout") from exc

        zip_path = str(tmp_root_path / "zocr_artifacts.zip")
        _zip_dir(str(outdir), zip_path)
        return FileResponse(zip_path, media_type="application/zip", filename="zocr_artifacts.zip")

    async def _process_job(job_id: str, tenant_id: str) -> None:
        job_t0 = time.perf_counter()
        job_key = _tenant_job_key(job_id, tenant_id)
        status = "unknown"
        error: Optional[str] = None
        try:
            with anyio.CancelScope(shield=True):
                paths = _job_paths(job_id)
                try:
                    job = _read_job(job_id, tenant_id)
                except Exception:
                    return

                now = _utc_now_iso()
                job["status"] = "running"
                job["updated_at"] = now
                _write_job(job_id, tenant_id, job)
                status = "running"
                _log("job_started", {"job_id": job_id, "tenant_id": tenant_id})
                _audit("job.started", {"job_id": job_id}, tenant_id=tenant_id)

                input_path = paths["input_dir"] / (job.get("input", {}) or {}).get("filename", "input")
                try:
                    job_store.ensure_local_file(input_path, force=True)
                except FileNotFoundError:
                    job["status"] = "failed"
                    job["updated_at"] = _utc_now_iso()
                    job["error"] = "Input file missing"
                    status = "failed"
                    error = str(job["error"])
                    _write_job(job_id, tenant_id, job)
                    return
                if not input_path.exists():
                    job["status"] = "failed"
                    job["updated_at"] = _utc_now_iso()
                    job["error"] = "Input file missing"
                    status = "failed"
                    error = str(job["error"])
                    _write_job(job_id, tenant_id, job)
                    return

                out_dir = paths["out_dir"]
                out_dir.mkdir(parents=True, exist_ok=True)
                params = job.get("params") if isinstance(job.get("params"), dict) else {}
                summary: Optional[Dict[str, Any]] = None
                pipeline_ok = False
                try:
                    summary = await _run_pipeline(
                        ignore_cancel=True,
                        inputs=[str(input_path)],
                        outdir=str(out_dir),
                        dpi=int(params.get("dpi") or 200),
                        domain_hint=params.get("domain"),
                        k=int(params.get("k") or 10),
                        seed=int(params.get("seed") or 24601),
                        snapshot=bool(params.get("snapshot") or False),
                        toy_lite=bool(params.get("toy_lite") or False),
                    )
                except TimeoutError:
                    job["status"] = "failed"
                    job["error"] = "Pipeline timeout"
                    status = "failed"
                    error = str(job["error"])
                except asyncio.CancelledError:
                    summary_path = out_dir / "pipeline_summary.json"
                    for _ in range(50):
                        if not summary_path.exists():
                            time.sleep(0.01)
                            continue
                        try:
                            summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        except Exception:
                            summary = None
                        if summary:
                            break
                        time.sleep(0.01)
                    if summary:
                        pipeline_ok = True
                    else:
                        job["status"] = "failed"
                        job["error"] = "Pipeline cancelled"
                        status = "failed"
                        error = str(job["error"])
                except Exception as exc:
                    job["status"] = "failed"
                    job["error"] = str(exc)
                    status = "failed"
                    error = str(job["error"])
                else:
                    pipeline_ok = True
                if pipeline_ok and summary is not None:
                    try:
                        manifest_path = write_manifest(out_dir, summary=summary, inputs=[str(input_path)], run_id=job_id)
                        job.setdefault("artifacts", {})["manifest_json"] = str(manifest_path.relative_to(paths["root"]))
                    except Exception:
                        pass
                    try:
                        _zip_dir(str(out_dir), str(paths["artifacts_zip"]))
                    except Exception:
                        pass
                    try:
                        job_store.persist_tree(out_dir)
                        job_store.persist_file(paths["artifacts_zip"])
                    except Exception as exc:
                        job["status"] = "failed"
                        job["error"] = f"Persist failed: {exc}"
                        status = "failed"
                        error = str(job["error"])
                    else:
                        job["status"] = "succeeded"
                        job["error"] = None
                        status = "succeeded"
                job["updated_at"] = _utc_now_iso()
                _write_job(job_id, tenant_id, job)
        finally:
            dt = time.perf_counter() - job_t0
            with metrics_lock:
                job_completions_total[status] += 1
                job_duration_sum[status] += float(dt)
                job_duration_count[status] += 1
            active_jobs.discard(job_key)
            _log(
                "job_finished",
                {
                    "job_id": job_id,
                    "tenant_id": tenant_id,
                    "status": status,
                    "duration_ms": int(round(dt * 1000.0)),
                    "error": error,
                },
                level="info" if status == "succeeded" else "error",
            )
            _audit(
                "job.finished",
                {
                    "job_id": job_id,
                    "status": status,
                    "duration_ms": int(round(dt * 1000.0)),
                    "error": error,
                },
                tenant_id=tenant_id,
            )

    @app.post("/v1/jobs")
    async def create_job(
        request: Request,
        file: UploadFile = File(...),
        *,
        domain: Optional[str] = None,
        dpi: int = 200,
        k: int = 10,
        seed: int = 24601,
        snapshot: bool = False,
        toy_lite: bool = False,
        auth_ctx: RequestAuthContext = Depends(_require_request_context("zocr.jobs.write")),
    ) -> Dict[str, Any]:
        nonlocal job_creates_total
        tenant_id = auth_ctx.tenant_id
        _enforce_tenant_quotas(request, principal=auth_ctx.principal, tenant_id=tenant_id)
        filename = Path(file.filename or "upload").name
        if not _allowed_suffix(filename):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename!r}")

        job_id = uuid.uuid4().hex
        paths = _job_paths(job_id)
        paths["input_dir"].mkdir(parents=True, exist_ok=True)
        paths["out_dir"].mkdir(parents=True, exist_ok=True)

        input_path = paths["input_dir"] / filename
        written = await _save_upload(file, input_path)
        try:
            job_store.persist_file(input_path)
        except Exception as exc:
            job_store.delete_job(job_id)
            raise HTTPException(status_code=500, detail=f"Failed to persist upload: {exc}") from exc

        now = _utc_now_iso()
        job: Dict[str, Any] = {
            "schema": _JOB_SCHEMA,
            "schema_version": _JOB_SCHEMA_VERSION,
            "id": job_id,
            "tenant_id": tenant_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "params": {
                "domain": domain,
                "dpi": int(dpi),
                "k": int(k),
                "seed": int(seed),
                "snapshot": bool(snapshot),
                "toy_lite": bool(toy_lite),
            },
            "input": {
                "filename": filename,
                "bytes": int(written),
            },
            "artifacts": {
                "out_dir": "out",
                "artifacts_zip": "artifacts.zip",
            },
            "error": None,
        }
        _write_job(job_id, tenant_id, job)
        with metrics_lock:
            job_creates_total += 1
        _log(
            "job_created",
            {
                "job_id": job_id,
                "tenant_id": tenant_id,
                "filename": filename,
                "bytes": int(written),
                "params": job.get("params"),
            },
        )
        request.state.job_id = job_id
        _audit(
            "job.created",
            {
                "job_id": job_id,
                "filename": filename,
                "bytes": int(written),
                "dispatch_backend": queue_backend,
            },
            request=request,
            principal=auth_ctx.principal,
            tenant_id=tenant_id,
        )

        if jobs_cleanup_on_create:
            _cleanup_jobs()

        try:
            dispatch = _schedule_job(job_id, tenant_id)
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = f"Failed to dispatch job: {exc}"
            job["updated_at"] = _utc_now_iso()
            _write_job(job_id, tenant_id, job)
            raise HTTPException(status_code=500, detail=f"Failed to dispatch job: {exc}") from exc

        return {
            "job": job,
            "dispatch": dispatch,
            "urls": {"self": f"/v1/jobs/{job_id}", "zip": f"/v1/jobs/{job_id}/artifacts.zip"},
        }

    @app.get("/v1/jobs")
    async def list_jobs(
        limit: int = 20,
        auth_ctx: RequestAuthContext = Depends(_require_request_context("zocr.jobs.read")),
    ) -> Dict[str, Any]:
        tenant_id = auth_ctx.tenant_id
        limit = max(1, min(int(limit), 200))
        items: list[Dict[str, Any]] = []
        candidates = job_repository.list_job_refs(tenant_id, limit=limit)
        for meta in candidates:
            job_id = meta.job_id
            try:
                job = _read_job(job_id, tenant_id)
            except Exception:
                continue
            items.append(job)
        return {"jobs": items}

    @app.get("/v1/jobs/{job_id}")
    async def get_job(
        job_id: str,
        include_summary: bool = False,
        auth_ctx: RequestAuthContext = Depends(_require_request_context("zocr.jobs.read")),
    ) -> Dict[str, Any]:
        tenant_id = auth_ctx.tenant_id
        job = _read_job(job_id, tenant_id)
        if include_summary:
            paths = _job_paths(job_id)
            summary_path = paths["out_dir"] / "pipeline_summary.json"
            try:
                job_store.ensure_local_file(summary_path, force=True)
            except FileNotFoundError:
                pass
            if summary_path.exists():
                try:
                    job = dict(job)
                    job["pipeline_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return {"job": job}

    @app.get("/v1/jobs/{job_id}/artifacts.zip", response_model=None)
    async def get_job_zip(
        request: Request,
        job_id: str,
        auth_ctx: RequestAuthContext = Depends(_require_request_context("zocr.jobs.read")),
    ) -> Response:
        tenant_id = auth_ctx.tenant_id
        job = _read_job(job_id, tenant_id)
        request.state.job_id = job_id
        paths = _job_paths(job_id)
        zip_path = paths["artifacts_zip"]
        redirect_url = job_store.presigned_get_url(zip_path) if job_store.has_file(zip_path) else None
        if redirect_url:
            _audit(
                "job.artifacts_zip.requested",
                {"job_id": job_id, "redirect": True},
                request=request,
                principal=auth_ctx.principal,
                tenant_id=tenant_id,
            )
            return RedirectResponse(redirect_url, status_code=307)
        try:
            job_store.ensure_local_file(zip_path, force=True)
        except FileNotFoundError:
            if job.get("status") != "succeeded":
                raise HTTPException(status_code=409, detail="Job has no artifacts yet")
            if not paths["out_dir"].exists():
                raise HTTPException(status_code=404, detail="Artifacts zip missing")
            try:
                _zip_dir(str(paths["out_dir"]), str(zip_path))
                job_store.persist_file(zip_path)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to build zip: {exc}") from exc
        _audit(
            "job.artifacts_zip.requested",
            {"job_id": job_id, "redirect": False},
            request=request,
            principal=auth_ctx.principal,
            tenant_id=tenant_id,
        )
        return FileResponse(str(zip_path), media_type="application/zip", filename=f"zocr_job_{job_id}.zip")

    @app.get(
        "/v1/jobs/{job_id}/artifacts/{artifact}",
        response_model=None,
    )
    async def get_job_artifact(
        request: Request,
        job_id: str,
        artifact: str,
        auth_ctx: RequestAuthContext = Depends(_require_request_context("zocr.jobs.read")),
    ) -> Response:
        tenant_id = auth_ctx.tenant_id
        _read_job(job_id, tenant_id)
        request.state.job_id = job_id
        artifact = str(artifact or "").strip()
        if not artifact:
            raise HTTPException(status_code=404, detail="Unknown artifact")
        paths = _job_paths(job_id)
        out_dir = paths["out_dir"]
        manifest_path = out_dir / MANIFEST_FILENAME
        try:
            job_store.ensure_local_file(manifest_path, force=True)
        except FileNotFoundError:
            raise HTTPException(status_code=409, detail="Manifest not available yet")
        if not manifest_path.exists():
            raise HTTPException(status_code=409, detail="Manifest not available yet")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Manifest unreadable: {exc}") from exc
        artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
        if not isinstance(artifacts, dict) or artifact not in artifacts:
            raise HTTPException(status_code=404, detail="Unknown artifact")
        entry = artifacts.get(artifact)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise HTTPException(status_code=404, detail="Unknown artifact")
        rel = entry["path"]
        target = (out_dir / rel).resolve()
        try:
            out_resolved = out_dir.resolve()
        except Exception:
            out_resolved = out_dir
        if out_resolved not in target.parents and target != out_resolved:
            raise HTTPException(status_code=400, detail="Invalid artifact path")
        redirect_url = job_store.presigned_get_url(target) if job_store.has_file(target) else None
        if redirect_url:
            _audit(
                "job.artifact.requested",
                {"job_id": job_id, "artifact": artifact, "redirect": True},
                request=request,
                principal=auth_ctx.principal,
                tenant_id=tenant_id,
            )
            return RedirectResponse(redirect_url, status_code=307)
        try:
            job_store.ensure_local_file(target, force=True)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Artifact missing")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact missing")
        _audit(
            "job.artifact.requested",
            {"job_id": job_id, "artifact": artifact, "redirect": False},
            request=request,
            principal=auth_ctx.principal,
            tenant_id=tenant_id,
        )
        return FileResponse(str(target))

    @app.delete("/v1/jobs/{job_id}")
    async def delete_job(
        request: Request,
        job_id: str,
        auth_ctx: RequestAuthContext = Depends(_require_request_context("zocr.jobs.delete")),
    ) -> Dict[str, Any]:
        tenant_id = auth_ctx.tenant_id
        _read_job(job_id, tenant_id)
        request.state.job_id = job_id
        job_repository.delete_job(job_id, tenant_id)
        job_store.delete_job(job_id)
        _audit(
            "job.deleted",
            {"job_id": job_id},
            request=request,
            principal=auth_ctx.principal,
            tenant_id=tenant_id,
        )
        return {"ok": True, "job_id": job_id}

    return app
