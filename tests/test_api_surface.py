import json
from pathlib import Path

from zocr.api import IngestRequest, ingest_job, query_job


def test_ingest_job_dry_run_creates_structure(tmp_path):
    req = IngestRequest(
        tenant_id="tenantA",
        files=["demo.pdf"],
        job_id="episode_demo",
        out_root=str(tmp_path),
        dry_run=True,
    )

    result = ingest_job(req)

    assert result.job_id == "episode_demo"
    assert result.status == "queued"
    assert Path(result.outdir).exists()
    assert result.artifacts["pipeline_summary"].endswith("pipeline_summary.json")
    assert result.artifacts["rag_manifest"].endswith("rag/manifest.json")


def test_query_job_reads_manifest_and_provenance(tmp_path):
    job_id = "episode_query"
    job_dir = tmp_path / job_id
    rag_dir = job_dir / "rag"
    rag_dir.mkdir(parents=True)
    manifest = {
        "cell_count": 5,
        "table_sections": 2,
        "page_sections": 3,
        "trace_schema": {"label": "doc=demo;page=1;row=1"},
        "fact_tag_example": "<fact trace=\"doc=demo;page=1;row=1\">100</fact>",
    }
    with open(rag_dir / "manifest.json", "w", encoding="utf-8") as fw:
        json.dump(manifest, fw)

    result = query_job(
        job_id=job_id,
        tenant_id="tenantB",
        base_dir=str(tmp_path),
        query="売上推移を教えて",
        mode="analysis",
    )

    assert result.status == "ready"
    assert "cells=5" in result.answer["data_summary"]
    assert result.provenance
    assert result.provenance[0]["trace"].startswith("doc=")
    assert result.artifacts["sources"]["manifest"].endswith("manifest.json")
