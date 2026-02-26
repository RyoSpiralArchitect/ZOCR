import asyncio
import functools
import logging
import hmac
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
        from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
        from fastapi.responses import FileResponse
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
    from .storage import build_job_store

    storage_backend, job_store = build_job_store()

    api_key = os.environ.get("ZOCR_API_KEY") or None
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
        record = {"ts": _utc_now_iso(), "event": event, **payload}
        if log_format == "json":
            msg = json.dumps(record, ensure_ascii=False)
        else:
            msg = f"{record.get('ts')} {event} {payload}"
        fn = getattr(logger, level, logger.info)
        fn(msg)

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

    def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
        job_store.atomic_write_json(path, payload)

    def _job_paths(job_id: str) -> Dict[str, Path]:
        paths = job_store.job_paths(job_id)
        return {
            "root": paths.root,
            "job_json": paths.job_json,
            "input_dir": paths.input_dir,
            "out_dir": paths.out_dir,
            "artifacts_zip": paths.artifacts_zip,
        }

    def _read_job(job_id: str) -> Dict[str, Any]:
        _validate_job_id(job_id)
        paths = _job_paths(job_id)
        job_path = paths["job_json"]
        if not job_path.exists():
            raise HTTPException(status_code=404, detail="Unknown job id")
        try:
            data = json.loads(job_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Job metadata unreadable: {exc}") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=500, detail="Job metadata invalid")
        return data

    def _write_job(job_id: str, job: Dict[str, Any]) -> None:
        paths = _job_paths(job_id)
        _atomic_write_json(paths["job_json"], job)

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
        candidates = job_store.list_job_roots()
        if not candidates:
            return {"deleted": 0, "kept": 0}

        job_infos: list[tuple[float, Path, Optional[Dict[str, Any]]]] = []
        now = datetime.now(timezone.utc)
        for root in candidates:
            job_path = root / "job.json"
            job_obj = job_store.read_json(job_path)
            try:
                mtime = root.stat().st_mtime
            except Exception:
                mtime = 0.0
            created_at = _parse_iso_datetime(job_obj.get("created_at") if job_obj else None)
            if created_at is not None:
                age_hours = (now - created_at).total_seconds() / 3600.0
            else:
                age_hours = 0.0
            status = (job_obj or {}).get("status") if job_obj else None
            is_terminal = status in {"succeeded", "failed"}
            ttl_expired = bool(jobs_ttl_hours > 0 and is_terminal and age_hours > jobs_ttl_hours)
            if ttl_expired:
                job_store.delete_tree(root)
                deleted += 1
                continue
            job_infos.append((mtime, root, job_obj))

        if jobs_max_count <= 0:
            return {"deleted": deleted, "kept": len(job_infos)}

        # Enforce max-count on terminal jobs only (avoid deleting queued/running).
        job_infos.sort(key=lambda item: item[0], reverse=True)
        kept: list[tuple[float, Path, Optional[Dict[str, Any]]]] = []
        terminal: list[tuple[float, Path, Optional[Dict[str, Any]]]] = []
        for info in job_infos:
            status = (info[2] or {}).get("status") if info[2] else None
            if status in {"succeeded", "failed"}:
                terminal.append(info)
            else:
                kept.append(info)
        remaining_terminal_budget = max(0, jobs_max_count - len(kept))
        kept.extend(terminal[:remaining_terminal_budget])
        for _, root, _ in terminal[remaining_terminal_budget:]:
            job_store.delete_tree(root)
            deleted += 1
        return {"deleted": deleted, "kept": len(kept)}

    def _schedule_job(job_id: str) -> bool:
        if job_id in active_jobs:
            return False
        active_jobs.add(job_id)
        try:
            asyncio.create_task(_process_job(job_id))
        except Exception:
            active_jobs.discard(job_id)
            raise
        return True

    def _extract_api_key(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        token = value.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        return token or None

    def _require_api_key(
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        authorization: Optional[str] = Header(default=None, alias="Authorization"),
    ) -> None:
        if not api_key:
            return
        candidate = _extract_api_key(x_api_key) or _extract_api_key(authorization)
        if not candidate:
            raise HTTPException(status_code=401, detail="Missing API key")
        if not hmac.compare_digest(candidate, api_key):
            raise HTTPException(status_code=403, detail="Invalid API key")

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
            candidates = job_store.list_job_roots()
            for root in candidates:
                job_id = root.name
                if not _JOB_ID_RE.match(job_id):
                    continue
                job_path = root / "job.json"
                job = job_store.read_json(job_path)
                if not job:
                    continue
                if job.get("status") not in {"queued", "running"}:
                    continue
                job["status"] = "queued"
                job["updated_at"] = _utc_now_iso()
                job["error"] = None
                _atomic_write_json(job_path, job)
                _schedule_job(job_id)
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

    if metrics_enabled:
        from fastapi.responses import Response

        @app.get("/metrics", dependencies=[Depends(_require_api_key)] if api_key else [])
        async def metrics() -> Response:
            return Response(_render_metrics(), media_type="text/plain; version=0.0.4")

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {
            "ok": True,
            "version": __version__,
            "auth": {"enabled": bool(api_key)},
            "limits": {"max_upload_mb": max_upload_mb, "concurrency": concurrency, "timeout_sec": run_timeout_sec},
            "storage": {"backend": storage_backend, "dir": str(storage_dir)},
            "jobs": {
                "dir": str(jobs_dir),
                "max_count": jobs_max_count,
                "ttl_hours": jobs_ttl_hours,
                "resume_on_startup": jobs_resume_on_startup,
            },
            "metrics": {"enabled": bool(metrics_enabled), "path": "/metrics"},
        }

    @app.post("/v1/run", dependencies=[Depends(_require_api_key)])
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

    @app.post("/v1/run.zip", dependencies=[Depends(_require_api_key)])
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

    async def _process_job(job_id: str) -> None:
        job_t0 = time.perf_counter()
        status = "unknown"
        error: Optional[str] = None
        try:
            with anyio.CancelScope(shield=True):
                paths = _job_paths(job_id)
                try:
                    job = _read_job(job_id)
                except Exception:
                    return

                now = _utc_now_iso()
                job["status"] = "running"
                job["updated_at"] = now
                _write_job(job_id, job)
                status = "running"
                _log("job_started", {"job_id": job_id})

                input_path = paths["input_dir"] / (job.get("input", {}) or {}).get("filename", "input")
                if not input_path.exists():
                    job["status"] = "failed"
                    job["updated_at"] = _utc_now_iso()
                    job["error"] = "Input file missing"
                    status = "failed"
                    error = str(job["error"])
                    _write_job(job_id, job)
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
                    job["status"] = "succeeded"
                    status = "succeeded"
                job["updated_at"] = _utc_now_iso()
                _write_job(job_id, job)
        finally:
            dt = time.perf_counter() - job_t0
            with metrics_lock:
                job_completions_total[status] += 1
                job_duration_sum[status] += float(dt)
                job_duration_count[status] += 1
            active_jobs.discard(job_id)
            _log(
                "job_finished",
                {
                    "job_id": job_id,
                    "status": status,
                    "duration_ms": int(round(dt * 1000.0)),
                    "error": error,
                },
                level="info" if status == "succeeded" else "error",
            )

    @app.post("/v1/jobs", dependencies=[Depends(_require_api_key)])
    async def create_job(
        file: UploadFile = File(...),
        *,
        domain: Optional[str] = None,
        dpi: int = 200,
        k: int = 10,
        seed: int = 24601,
        snapshot: bool = False,
        toy_lite: bool = False,
    ) -> Dict[str, Any]:
        nonlocal job_creates_total
        filename = Path(file.filename or "upload").name
        if not _allowed_suffix(filename):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename!r}")

        job_id = uuid.uuid4().hex
        paths = _job_paths(job_id)
        paths["input_dir"].mkdir(parents=True, exist_ok=True)
        paths["out_dir"].mkdir(parents=True, exist_ok=True)

        input_path = paths["input_dir"] / filename
        written = await _save_upload(file, input_path)

        now = _utc_now_iso()
        job: Dict[str, Any] = {
            "schema": _JOB_SCHEMA,
            "schema_version": _JOB_SCHEMA_VERSION,
            "id": job_id,
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
        _write_job(job_id, job)
        with metrics_lock:
            job_creates_total += 1
        _log(
            "job_created",
            {
                "job_id": job_id,
                "filename": filename,
                "bytes": int(written),
                "params": job.get("params"),
            },
        )

        if jobs_cleanup_on_create:
            _cleanup_jobs()

        _schedule_job(job_id)

        return {"job": job, "urls": {"self": f"/v1/jobs/{job_id}", "zip": f"/v1/jobs/{job_id}/artifacts.zip"}}

    @app.get("/v1/jobs", dependencies=[Depends(_require_api_key)])
    async def list_jobs(limit: int = 20) -> Dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        items: list[Dict[str, Any]] = []
        try:
            candidates = sorted(jobs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            candidates = []
        for path in candidates[:limit]:
            if not path.is_dir():
                continue
            job_id = path.name
            try:
                job = _read_job(job_id)
            except Exception:
                continue
            items.append(job)
        return {"jobs": items}

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(_require_api_key)])
    async def get_job(job_id: str, include_summary: bool = False) -> Dict[str, Any]:
        job = _read_job(job_id)
        if include_summary:
            paths = _job_paths(job_id)
            summary_path = paths["out_dir"] / "pipeline_summary.json"
            if summary_path.exists():
                try:
                    job = dict(job)
                    job["pipeline_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return {"job": job}

    @app.get("/v1/jobs/{job_id}/artifacts.zip", dependencies=[Depends(_require_api_key)])
    async def get_job_zip(job_id: str) -> FileResponse:
        _validate_job_id(job_id)
        paths = _job_paths(job_id)
        if not paths["job_json"].exists():
            raise HTTPException(status_code=404, detail="Unknown job id")
        zip_path = paths["artifacts_zip"]
        if not zip_path.exists():
            if not paths["out_dir"].exists():
                raise HTTPException(status_code=409, detail="Job has no artifacts yet")
            try:
                _zip_dir(str(paths["out_dir"]), str(zip_path))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to build zip: {exc}") from exc
        return FileResponse(str(zip_path), media_type="application/zip", filename=f"zocr_job_{job_id}.zip")

    @app.get("/v1/jobs/{job_id}/artifacts/{artifact}", dependencies=[Depends(_require_api_key)])
    async def get_job_artifact(job_id: str, artifact: str) -> FileResponse:
        _validate_job_id(job_id)
        artifact = str(artifact or "").strip()
        if not artifact:
            raise HTTPException(status_code=404, detail="Unknown artifact")
        paths = _job_paths(job_id)
        out_dir = paths["out_dir"]
        manifest_path = out_dir / MANIFEST_FILENAME
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
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact missing")
        return FileResponse(str(target))

    @app.delete("/v1/jobs/{job_id}", dependencies=[Depends(_require_api_key)])
    async def delete_job(job_id: str) -> Dict[str, Any]:
        _validate_job_id(job_id)
        paths = _job_paths(job_id)
        if not paths["root"].exists():
            raise HTTPException(status_code=404, detail="Unknown job id")
        job_store.delete_tree(paths["root"])
        return {"ok": True, "job_id": job_id}

    return app
