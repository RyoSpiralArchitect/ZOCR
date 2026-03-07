from __future__ import annotations

import json
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from ..artifacts.manifest import write_manifest


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def process_job(
    *,
    job_id: str,
    tenant_id: str,
    job_store: Any,
    job_repository: Any,
    run_full_pipeline: Callable[..., Any],
) -> tuple[str, Optional[str], float]:
    t0 = time.perf_counter()
    job = job_repository.read_job(job_id, tenant_id)
    if not isinstance(job, dict):
        return "missing", "Job metadata missing", time.perf_counter() - t0

    job["status"] = "running"
    job["updated_at"] = _utc_now_iso()
    job["error"] = None
    job_repository.write_job(job_id, tenant_id, job)

    paths = job_store.job_paths(job_id)
    input_meta = job.get("input") if isinstance(job.get("input"), dict) else {}
    input_filename = str((input_meta or {}).get("filename") or "input")
    input_path = paths.input_dir / input_filename

    try:
        job_store.ensure_local_file(input_path, force=True)
    except FileNotFoundError:
        job["status"] = "failed"
        job["error"] = "Input file missing"
        job["updated_at"] = _utc_now_iso()
        job_repository.write_job(job_id, tenant_id, job)
        return "failed", str(job["error"]), time.perf_counter() - t0

    out_dir = paths.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    params = job.get("params") if isinstance(job.get("params"), dict) else {}

    try:
        summary = run_full_pipeline(
            inputs=[str(input_path)],
            outdir=str(out_dir),
            dpi=int(params.get("dpi") or 200),
            domain_hint=params.get("domain"),
            k=int(params.get("k") or 10),
            seed=int(params.get("seed") or 24601),
            snapshot=bool(params.get("snapshot") or False),
            toy_lite=bool(params.get("toy_lite") or False),
        )
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["updated_at"] = _utc_now_iso()
        job_repository.write_job(job_id, tenant_id, job)
        return "failed", str(job["error"]), time.perf_counter() - t0

    summary_path = out_dir / "pipeline_summary.json"
    if not summary_path.exists():
        try:
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    try:
        manifest_path = write_manifest(out_dir, summary=summary, inputs=[str(input_path)], run_id=job_id)
        job.setdefault("artifacts", {})["manifest_json"] = str(manifest_path.relative_to(paths.root))
    except Exception:
        pass

    try:
        _zip_dir(str(out_dir), str(paths.artifacts_zip))
    except Exception:
        pass

    try:
        job_store.persist_tree(out_dir)
        job_store.persist_file(paths.artifacts_zip)
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = f"Persist failed: {exc}"
        job["updated_at"] = _utc_now_iso()
        job_repository.write_job(job_id, tenant_id, job)
        return "failed", str(job["error"]), time.perf_counter() - t0

    job["status"] = "succeeded"
    job["error"] = None
    job["updated_at"] = _utc_now_iso()
    job_repository.write_job(job_id, tenant_id, job)
    return "succeeded", None, time.perf_counter() - t0
