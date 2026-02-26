from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
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


def _zip_dir(src_dir: str, dest_zip: str) -> None:
    src_path = Path(src_dir)
    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
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
        from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, UploadFile
        from fastapi.responses import FileResponse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "FastAPI dependencies are missing. Install with `pip install -e '.[api]'` "
            "(or `pip install 'zocr-suite[api]'`)."
        ) from exc

    try:
        import anyio
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("anyio is required (it should be installed with FastAPI).") from exc

    from zocr.orchestrator.zocr_pipeline import run_full_pipeline

    api_key = os.environ.get("ZOCR_API_KEY") or None
    max_upload_mb = max(1, _env_int("ZOCR_API_MAX_UPLOAD_MB", 50))
    max_upload_bytes = int(max_upload_mb) * 1024 * 1024
    run_timeout_sec = _env_float("ZOCR_API_RUN_TIMEOUT_SEC", 15 * 60.0)
    if run_timeout_sec <= 0:
        run_timeout_sec = 0.0
    concurrency = max(1, _env_int("ZOCR_API_CONCURRENCY", 1))
    run_semaphore = asyncio.BoundedSemaphore(concurrency)

    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _pick_storage_dir() -> Path:
        env = os.environ.get("ZOCR_API_STORAGE_DIR")
        if env:
            return Path(env)
        candidates = [
            Path("/data"),
            Path(tempfile.gettempdir()) / "zocr_api_store",
        ]
        for candidate in candidates:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
            except Exception:
                continue
        return Path(tempfile.gettempdir()) / "zocr_api_store"

    storage_dir = _pick_storage_dir()
    jobs_dir = storage_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    def _validate_job_id(job_id: str) -> None:
        if not _JOB_ID_RE.match(job_id or ""):
            raise HTTPException(status_code=404, detail="Unknown job id")

    def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    def _job_paths(job_id: str) -> Dict[str, Path]:
        job_root = jobs_dir / job_id
        return {
            "root": job_root,
            "job_json": job_root / "job.json",
            "input_dir": job_root / "input",
            "out_dir": job_root / "out",
            "artifacts_zip": job_root / "artifacts.zip",
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

    async def _run_pipeline(**kwargs):
        await run_semaphore.acquire()
        try:
            if run_timeout_sec > 0:
                with anyio.fail_after(run_timeout_sec):
                    return await anyio.to_thread.run_sync(run_full_pipeline, **kwargs)
            return await anyio.to_thread.run_sync(run_full_pipeline, **kwargs)
        finally:
            run_semaphore.release()

    app = FastAPI(
        title="Z-OCR Suite API (Reference)",
        version=__version__,
        description="Reference FastAPI wrapper around the Z-OCR orchestrator.",
    )

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {
            "ok": True,
            "version": __version__,
            "auth": {"enabled": bool(api_key)},
            "limits": {"max_upload_mb": max_upload_mb, "concurrency": concurrency, "timeout_sec": run_timeout_sec},
            "storage": {"dir": str(storage_dir)},
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
        paths = _job_paths(job_id)
        try:
            job = _read_job(job_id)
        except Exception:
            return

        now = _utc_now_iso()
        job["status"] = "running"
        job["updated_at"] = now
        _write_job(job_id, job)

        input_path = paths["input_dir"] / (job.get("input", {}) or {}).get("filename", "input")
        if not input_path.exists():
            job["status"] = "failed"
            job["updated_at"] = _utc_now_iso()
            job["error"] = "Input file missing"
            _write_job(job_id, job)
            return

        out_dir = paths["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        params = job.get("params") if isinstance(job.get("params"), dict) else {}
        try:
            summary = await _run_pipeline(
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
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
        else:
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
        job["updated_at"] = _utc_now_iso()
        _write_job(job_id, job)

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

        asyncio.create_task(_process_job(job_id))

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
        _cleanup_tree(str(paths["root"]))
        return {"ok": True, "job_id": job_id}

    return app
