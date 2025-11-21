from pathlib import Path

import jsonschema

from zocr import api_http
from zocr.api_spec import (
    INGEST_REQUEST_SCHEMA_V0,
    INGEST_RESPONSE_SCHEMA_V0,
    QUERY_REQUEST_SCHEMA_V0,
    QUERY_RESPONSE_SCHEMA_V0,
)


def test_ingest_request_round_trip(tmp_path):
    payload = {
        "tenant_id": "t-http",
        "files": [
            {
                "id": "f1",
                "uri": "https://example.com/demo.pdf",
                "kind": "pdf",
                "tags": ["demo"],
            }
        ],
        "domain_hint": "invoice",
        "options": {"snapshot": False, "priority": "high"},
    }

    jsonschema.Draft202012Validator(INGEST_REQUEST_SCHEMA_V0).validate(payload)

    result, public_payload = api_http.ingest_from_payload(
        payload, out_root=str(tmp_path), dry_run=True
    )

    assert result["tenant_id"] == "t-http"
    assert result["status"] == "queued"
    jsonschema.Draft202012Validator(INGEST_RESPONSE_SCHEMA_V0).validate(public_payload)
    assert "job_id" in public_payload
    assert public_payload["status"] == "queued"


def test_query_request_round_trip(tmp_path):
    job_dir = Path(tmp_path) / "episode_ready"
    manifest_dir = job_dir / "rag"
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(
        """{
        "trace_schema": {"label": "doc=1,page=1,row=2"},
        "fact_tag_example": "[fact trace=doc=1,page=1,row=2] 12345",
        "table_sections": 2,
        "page_sections": 3
    }""",
        encoding="utf-8",
    )

    payload = {
        "tenant_id": "t-http",
        "job_id": "episode_ready",
        "query": "売上を教えて",
        "mode": "analysis",
    }

    jsonschema.Draft202012Validator(QUERY_REQUEST_SCHEMA_V0).validate(payload)

    result, public_payload = api_http.query_from_payload(payload, base_dir=str(tmp_path))

    assert result["status"] == "ready"
    assert result["answer"]["data_summary"].startswith("Artifacts for episode_ready")
    jsonschema.Draft202012Validator(QUERY_RESPONSE_SCHEMA_V0).validate(public_payload)
    assert public_payload["flags"]["facts_insufficient"] is False
    assert public_payload["provenance"][0]["trace"] == "doc=1,page=1,row=2"
