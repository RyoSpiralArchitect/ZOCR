import json
import uuid

import pytest

from zocr.artifacts.manifest import write_manifest
from zocr.service.metadata import LocalJobRepository
from zocr.service.storage import LocalJobStore


@pytest.fixture
def review_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from zocr.service.app import create_app

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_KEYS_JSON", json.dumps({
        "alice": {"subject": "alice", "tenant_id": "tenant-a", "roles": ["admin"]},
        "operator": {"tenant_id": "tenant-a", "roles": ["operator"]},
        "other": {"tenant_id": "tenant-b", "roles": ["admin"]},
    }))
    store = LocalJobStore(tmp_path)
    repository = LocalJobRepository(job_store=store)
    ids = []
    for text in ("100", "120"):
        job_id = uuid.uuid4().hex
        paths = store.job_paths(job_id)
        (paths.out_dir / "rag").mkdir(parents=True)
        (paths.out_dir / "rag" / "cells.jsonl").write_text(json.dumps({
            "doc_id": "doc", "page": 1, "table_index": 0, "row": 1, "col": 0,
            "header": "Amount", "text": text, "bbox": [0, 0, 50, 30], "confidence": .9,
        }) + "\n", encoding="utf-8")
        summary = {"inputs": []}
        (paths.out_dir / "pipeline_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        write_manifest(paths.out_dir, summary=summary)
        repository.write_job(job_id, "tenant-a", {"status": "succeeded"})
        ids.append(job_id)
    with TestClient(create_app()) as client:
        yield client, ids, store


def test_review_api_binds_actor_sources_and_scopes(review_api):
    client, ids, store = review_api
    alice = {"X-API-Key": "alice"}
    created = client.post("/v1/reviews", json={"job_a": ids[0], "job_b": ids[1]}, headers=alice)
    assert created.status_code == 200, created.text
    review = created.json()["review"]
    url = f"/v1/reviews/{review['review_id']}"
    request = {"expected_revision": 0, "submission_id": "approval-1", "reviewer": "forged name",
               "source_hashes": review["result"]["review"]["source_hashes"],
               "decisions": [{"event_id": review["result"]["events"][0]["event_id"],
                              "status": "approved", "reason": "Verified against both sources"}]}
    assert client.get(url, headers={"X-API-Key": "other"}).status_code == 404
    assert client.post(url + "/decisions", json=request, headers={"X-API-Key": "operator"}).status_code == 403
    approved = client.post(url + "/decisions", json=request, headers=alice)
    assert approved.status_code == 200, approved.text
    assert approved.json()["review"]["history"][0]["actor"]["subject"] == "alice"
    history = client.get(url + "/history", headers=alice).json()
    assert history["revision"] == 1
    assert history["history"][0]["hash"]
    retried = client.post(url + "/decisions", json=request, headers=alice)
    assert retried.json() == approved.json()
    path = store.job_paths(ids[1]).out_dir / "rag" / "cells.jsonl"
    path.write_text('{"text":"tampered"}\n', encoding="utf-8")
    assert client.post(url + "/decisions", json=dict(request, expected_revision=1, submission_id="approval-2"),
                       headers=alice).status_code == 409


def test_unpublished_attempts_cannot_be_downloaded(review_api):
    client, ids, store = review_api
    repository = LocalJobRepository(job_store=store)
    repository.write_job(ids[0], "tenant-a", {"status": "running", "execution_token": "a" * 32})
    assert client.get(f"/v1/jobs/{ids[0]}/artifacts.zip", headers={"X-API-Key": "alice"}).status_code == 409
    assert client.get(f"/v1/jobs/{ids[0]}/artifacts/rag_cells", headers={"X-API-Key": "alice"}).status_code == 409
