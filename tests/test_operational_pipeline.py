import io
import json
import time
import zipfile

import pytest

from zocr.service.executor import execute_pipeline
from zocr.validate import validate_path


def test_real_pipeline_subprocess_writes_valid_artifacts(tmp_path, table_image):
    out = tmp_path / "out"
    summary = execute_pipeline(
        timeout_sec=60, inputs=[str(table_image)], outdir=str(out),
        do_tune=False, toy_lite=True, sweeps_fixed=1, max_cells=12, k=1,
    )
    assert summary["quality_status"] in {"pass", "warn", "fail", "unknown"}
    assert summary["publication_status"] in {"published", "review_required", "blocked"}
    assert (out / "doc.zocr.json").is_file()
    assert (out / "zocr.manifest.json").is_file()
    saved = json.loads((out / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert saved["quality_status"] == summary["quality_status"]
    assert validate_path(out, strict=True, max_jsonl_lines=10000, write_manifest_flag=False) == 0
    assert not summary.get("toy_self_correction", {}).get("executed")
    from zocr.diff.review import _crop
    cells_path = out / "rag/cells.jsonl"
    cells = [json.loads(line) for line in cells_path.read_text(encoding="utf-8").splitlines()]
    assert cells and any(_crop(cell, cells_path, out) for cell in cells)
    assert (out / "final_export.quality.json").is_file()


def test_real_inline_api_pipeline_and_download(tmp_path, table_image, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from zocr.service.app import create_app

    for key, value in {
        "ZOCR_API_STORAGE_DIR": str(tmp_path / "store"),
        "ZOCR_API_QUEUE_BACKEND": "inline", "ZOCR_API_METADATA_BACKEND": "local",
        "ZOCR_API_STORAGE_BACKEND": "local", "ZOCR_API_JOBS_CLEANUP_ON_STARTUP": "0",
        "ZOCR_API_JOBS_RESUME_ON_STARTUP": "0", "ZOCR_API_RUN_TIMEOUT_SEC": "60",
        "ZOCR_TOY_SWEEPS": "1", "ZOCR_TOY_SWEEP_LIMIT": "1",
    }.items():
        monkeypatch.setenv(key, value)
    with TestClient(create_app()) as client:
        response = client.post("/v1/jobs?toy_lite=true&k=1", files={
            "file": ("page.png", table_image.read_bytes(), "image/png"),
        })
        assert response.status_code == 200, response.text
        job_id = response.json()["job"]["id"]
        deadline = time.monotonic() + 65
        while time.monotonic() < deadline:
            job = client.get(f"/v1/jobs/{job_id}").json()["job"]
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(.05)
        assert job["status"] == "succeeded", job
        assert job["quality_status"] in {"pass", "warn", "fail", "unknown"}
        archive = client.get(f"/v1/jobs/{job_id}/artifacts.zip")
        assert archive.status_code == 200
        with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
            assert bundle.testzip() is None
            assert "zocr.manifest.json" in bundle.namelist()
