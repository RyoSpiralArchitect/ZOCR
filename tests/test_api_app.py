# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from zocr.api_app import create_app


def _ingest_response(payload):
    job_id = "job-123"
    tenant = payload["tenant_id"]
    response = {
        "job_id": job_id,
        "tenant_id": tenant,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": {"files": len(payload["files"]), "out_root": "episodes"},
    }
    return {"job_id": job_id, "tenant_id": tenant}, response


def _query_response(payload):
    response = {
        "answer": {
            "data_summary": "売上は前年同期比で増加しています。",
            "business_commentary": "一般的には成長に伴い原価率管理が重要です。",
        },
        "artifacts": {"tables": [], "charts": []},
        "provenance": [],
        "flags": {"facts_insufficient": False},
    }
    return {"job_id": payload["job_id"]}, response


def test_app_exposes_ingest_and_query():
    app = create_app(ingest_runner=_ingest_response, query_runner=_query_response)
    client = TestClient(app)

    ingest_payload = {
        "tenant_id": "acme",
        "files": [{"id": "f1", "uri": "s3://bucket/doc.pdf"}],
    }
    ingest_resp = client.post("/ingest", json=ingest_payload)
    assert ingest_resp.status_code == 200
    assert ingest_resp.json()["job_id"] == "job-123"

    query_payload = {
        "tenant_id": "acme",
        "job_id": "job-123",
        "query": "売上の概要を教えて",
    }
    query_resp = client.post("/query", json=query_payload)
    assert query_resp.status_code == 200
    assert "data_summary" in query_resp.json()["answer"]


def test_invalid_ingest_payload_returns_400():
    app = create_app(ingest_runner=_ingest_response, query_runner=_query_response)
    client = TestClient(app)

    resp = client.post("/ingest", json={"tenant_id": "missing_files"})
    assert resp.status_code == 400
    assert "'files'" in resp.json()["detail"]
