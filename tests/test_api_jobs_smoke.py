from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


def test_jobs_lifecycle_smoke(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    import zocr.orchestrator.zocr_pipeline as pipeline

    def fake_run_full_pipeline(*, inputs, outdir: str, **_kwargs):
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "doc.zocr.json").write_text(
            json.dumps({"doc_id": "doc", "pages": [], "metrics": {}}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary = {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "inputs": inputs,
        }
        (out / "pipeline_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary

    monkeypatch.setattr(pipeline, "run_full_pipeline", fake_run_full_pipeline, raising=True)
    storage_dir = tmp_path / "store"
    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_CREATE", "0")

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["storage"]["dir"] == str(storage_dir)

    resp = client.post(
        "/v1/jobs?domain=invoice",
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert resp.status_code == 200, resp.text
    job = resp.json()["job"]
    job_id = job["id"]

    final = None
    for _ in range(100):
        polled = client.get(f"/v1/jobs/{job_id}")
        assert polled.status_code == 200
        status = polled.json()["job"]["status"]
        if status in {"succeeded", "failed"}:
            final = polled.json()["job"]
            break
        time.sleep(0.02)
    assert final is not None
    assert final["status"] == "succeeded"

    summary_resp = client.get(f"/v1/jobs/{job_id}/artifacts/pipeline_summary")
    assert summary_resp.status_code == 200
    summary_payload = summary_resp.json()
    assert summary_payload["inputs"]

    zip_resp = client.get(f"/v1/jobs/{job_id}/artifacts.zip")
    assert zip_resp.status_code == 200
    assert zip_resp.headers.get("content-type") == "application/zip"
