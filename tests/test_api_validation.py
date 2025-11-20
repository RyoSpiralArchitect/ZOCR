from jsonschema import ValidationError

from zocr import api
from zocr.api_spec import render_user_prompt_analysis_v0


def test_ingest_validation_accepts_v0_shape():
    payload = {
        "tenant_id": "t1",
        "files": [
            {"id": "f1", "uri": "s3://bucket/demo.pdf", "tags": ["invoice"]},
            {"id": "f2", "uri": "https://example.com/local.pdf"},
        ],
    }

    api.validate_ingest_request_payload(payload)


def test_ingest_validation_rejects_missing_fields():
    payload = {"files": []}
    try:
        api.validate_ingest_request_payload(payload)
    except ValidationError:
        return
    assert False, "expected validation error"


def test_query_validation_and_prompt_rendering():
    query_payload = {"tenant_id": "t1", "job_id": "episode1", "query": "売上を教えて"}
    api.validate_query_request_payload(query_payload)

    prompt = render_user_prompt_analysis_v0(
        query="売上を教えて",
        facts=[{"trace": "doc=1;page=1;row=2", "text": "2024-01 revenue = 100"}],
        tables=[{"id": "t1", "columns": ["month", "revenue"], "rows": [["2024-01", 100]]}],
        flags={"facts_insufficient": False},
    )

    assert "QUESTION:" in prompt
    assert "FACTS:" in prompt and "trace" in prompt
    assert "TABLES:" in prompt and "2024-01" in prompt
    assert "facts_insufficient" in prompt
