from datetime import datetime
from typing import Any, Dict

from zocr import api
from zocr.api_spec import (
    INGEST_REQUEST_SCHEMA_V0,
    INGEST_RESPONSE_SCHEMA_V0,
    QUERY_REQUEST_SCHEMA_V0,
    QUERY_RESPONSE_SCHEMA_V0,
    SYSTEM_PROMPT_ANALYSIS_V0,
    USER_PROMPT_ANALYSIS_TEMPLATE_V0,
    get_api_schemas_v0,
    get_prompt_templates_v0,
)


def test_ingest_schema_v0_matches_contract():
    assert INGEST_REQUEST_SCHEMA_V0["title"] == "IngestRequest"
    props = INGEST_REQUEST_SCHEMA_V0["properties"]
    assert "tenant_id" in props and "files" in props
    file_props: Dict[str, Any] = props["files"]["items"]["properties"]
    assert file_props["uri"]["format"] == "uri"
    assert "priority" in props["options"]["properties"]

    resp_props = INGEST_RESPONSE_SCHEMA_V0["properties"]
    assert set(resp_props["status"]["enum"]) >= {"queued", "completed", "failed"}
    assert resp_props["created_at"]["format"] == "date-time"


def test_query_schema_v0_matches_contract():
    assert QUERY_REQUEST_SCHEMA_V0["properties"]["language"]["default"] == "ja"
    assert "analysis" in QUERY_REQUEST_SCHEMA_V0["properties"]["mode"]["enum"]

    resp_props = QUERY_RESPONSE_SCHEMA_V0["properties"]
    assert "answer" in resp_props
    answer_props = resp_props["answer"]["properties"]
    assert "data_summary" in answer_props and "business_commentary" in answer_props
    flags = resp_props["flags"]["properties"]
    assert flags["facts_insufficient"]["type"] == "boolean"


def test_prompt_templates_v0_present():
    assert "Data-grounded summary" in SYSTEM_PROMPT_ANALYSIS_V0
    assert "FACTS:" in USER_PROMPT_ANALYSIS_TEMPLATE_V0
    templates = get_prompt_templates_v0()
    assert templates["system_analysis"].startswith("You are a data analyst")


def test_payload_helpers_follow_schema(tmp_path):
    ingest_req = api.IngestRequest(
        tenant_id="t1",
        files=["demo.pdf"],
        job_id="episode_schema",
        out_root=str(tmp_path),
        dry_run=True,
    )
    ingest_result = api.ingest_job(ingest_req)
    ingest_payload = api.ingest_response_payload_v0(ingest_result)

    assert ingest_payload["job_id"] == "episode_schema"
    datetime.fromisoformat(ingest_payload["created_at"])

    query_result = api.query_job(
        job_id="episode_schema",
        tenant_id="t1",
        base_dir=str(tmp_path),
        query="demo?",
    )
    public_query = api.query_response_payload_v0(query_result)
    assert set(public_query.keys()) == {"answer", "artifacts", "provenance", "flags"}
    assert public_query["flags"]["facts_insufficient"] is True
    assert public_query["artifacts"]["tables"] == []

    schemas = get_api_schemas_v0()
    assert "ingest_request" in schemas and "query_response" in schemas
