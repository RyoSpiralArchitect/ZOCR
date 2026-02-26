from __future__ import annotations

import asyncio
import hmac
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from .._version import __version__
from ..utils.json_utils import json_ready

_UPLOAD_CHUNK_BYTES = 1024 * 1024


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

    async def _run_guarded(**kwargs):
        await run_semaphore.acquire()
        try:
            if run_timeout_sec > 0:
                with anyio.fail_after(run_timeout_sec):
                    return await anyio.to_thread.run_sync(run_full_pipeline, **kwargs)
            return await anyio.to_thread.run_sync(run_full_pipeline, **kwargs)
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Pipeline timeout") from exc
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
            summary = await _run_guarded(
                inputs=[str(input_path)],
                outdir=str(outdir),
                dpi=int(dpi),
                domain_hint=domain,
                k=int(k),
                seed=int(seed),
                snapshot=bool(snapshot),
                toy_lite=bool(toy_lite),
            )

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
        await _run_guarded(
            inputs=[str(input_path)],
            outdir=str(outdir),
            dpi=int(dpi),
            domain_hint=domain,
            k=int(k),
            seed=int(seed),
            snapshot=bool(snapshot),
            toy_lite=bool(toy_lite),
        )

        zip_path = str(tmp_root_path / "zocr_artifacts.zip")
        _zip_dir(str(outdir), zip_path)
        return FileResponse(zip_path, media_type="application/zip", filename="zocr_artifacts.zip")

    return app
