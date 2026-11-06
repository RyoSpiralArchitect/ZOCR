import json
from unittest.mock import MagicMock

import pytest

from zocr.core.augmenter import augment
from zocr.core.query_engine import _confidence_penalty
from zocr.embeddings.multimodal import EmbeddingBackends, MultimodalEmbeddingService
from zocr.service.jobs import process_job


def test_augmentation_preserves_observed_cells(tmp_path):
    record = {
        "doc_id": "technical", "page": 1, "table_index": 0,
        "row": 0, "col": 0, "text": "Pressure",
        "meta": {"confidence": 0.95},
    }
    source = tmp_path / "source.jsonl"
    target = tmp_path / "target.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert augment(str(source), str(target)) == 1
    output = json.loads(target.read_text(encoding="utf-8"))
    assert output["text"] == "Pressure"
    assert output["meta"]["confidence"] == 0.95
    assert output["meta"]["observed_text"] == "Pressure"
    assert output["meta"]["evidence_kind"] == "observed"


def test_inferred_metadata_survives_sql_export(tmp_path):
    import csv
    from zocr.core.exporters import sql_export

    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({
        "text": "Total", "meta": {"synthetic": True, "confidence": 0.0},
    }) + "\n", encoding="utf-8")
    result = sql_export(str(source), str(tmp_path / "sql"))
    with open(result["csv"], encoding="utf-8-sig") as stream:
        row = next(csv.DictReader(stream))
    assert row["evidence_kind"] == "inferred"
    assert row["observed_text"] == ""
    assert row["synthetic"] == "True"


def test_embedding_output_is_consumable_by_hybrid_search(tmp_path):
    from zocr.core.indexer import build_index
    from zocr.core.query_engine import hybrid_query

    service = MultimodalEmbeddingService(backends=EmbeddingBackends(
        text_encoder=lambda batch: [[1.0, 0.0] for _ in batch],
    ))
    entry = service.embed_objects([{"text": "pressure"}]).entries[0]
    assert entry["embeddings"]["hybrid"] == entry["embeddings"]["fused"]
    source = tmp_path / "cells.jsonl"
    source.write_text(json.dumps({
        "text": "pressure", "confidence": 0.0, "embeddings": entry["embeddings"],
    }) + "\n", encoding="utf-8")
    index = tmp_path / "index.pkl"
    build_index(str(source), str(index))
    results = hybrid_query(str(index), str(source), hybrid_embedding=entry["vector"])
    scores = results[0][1]["meta"]["retrieval_scores"]
    assert scores["hybrid"] == pytest.approx(1.0)
    assert scores["penalty"] == pytest.approx(1.0)


@pytest.mark.parametrize("required", [False, True])
def test_empty_vision_output_is_rejected(required):
    service = MultimodalEmbeddingService(backends=EmbeddingBackends(
        text_encoder=lambda batch: [[1.0] for _ in batch],
        vision_encoder=lambda batch: [],
        vision_required=required,
    ))
    with pytest.raises(ValueError, match="mismatched"):
        service.embed_objects([{"text": "pressure"}])


@pytest.mark.parametrize("confidence, expected", [(0.0, 1.0), (0.2, 0.8), (1.0, 0.0)])
def test_scalar_confidence_penalty(confidence, expected):
    assert _confidence_penalty(confidence) == pytest.approx(expected)


@pytest.mark.parametrize("failure", ["manifest", "zip"])
def test_job_fails_when_required_artifact_cannot_be_created(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    import zocr.service.jobs as jobs

    root = tmp_path / "job"
    out = root / "out"
    out.mkdir(parents=True)
    store = MagicMock()
    store.storage_dir = tmp_path
    store.job_paths.return_value = SimpleNamespace(
        root=root, input_dir=root / "input", out_dir=out,
        artifacts_zip=root / "artifacts.zip",
    )
    repository = MagicMock()
    repository.read_job.return_value = {"input": {"filename": "page.png"}}
    def fail(*args, **kwargs):
        raise OSError("artifact unavailable")
    monkeypatch.setattr(jobs, "write_manifest" if failure == "manifest" else "_zip_dir", fail)
    status, error, _ = process_job(
        job_id="job", tenant_id="default", job_store=store,
        job_repository=repository, run_full_pipeline=lambda **kwargs: {},
    )
    assert status == "failed"
    assert "artifact" in error.lower()
    assert repository.write_job.call_args.args[2]["status"] == "failed"
