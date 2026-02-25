from __future__ import annotations

import asyncio
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from .._version import __version__


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
        from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
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

    app = FastAPI(
        title="Z-OCR Suite API (Reference)",
        version=__version__,
        description="Reference FastAPI wrapper around the Z-OCR orchestrator.",
    )

    run_lock = asyncio.Lock()

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {"ok": True, "version": __version__}

    @app.post("/v1/run")
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

            with input_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)

            async with run_lock:
                summary = await anyio.to_thread.run_sync(
                    run_full_pipeline,
                    inputs=[str(input_path)],
                    outdir=str(outdir),
                    dpi=int(dpi),
                    domain_hint=domain,
                    k=int(k),
                    seed=int(seed),
                    snapshot=bool(snapshot),
                    toy_lite=bool(toy_lite),
                )

            return {"summary": summary}

    @app.post("/v1/run.zip")
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

        with input_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        async with run_lock:
            await anyio.to_thread.run_sync(
                run_full_pipeline,
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

