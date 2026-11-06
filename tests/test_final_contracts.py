import csv
import json
import zipfile
from unittest.mock import MagicMock

import pytest

from zocr.artifacts.manifest import write_manifest
from zocr.core.confidence import normalize_confidence
from zocr.core.exporters import sql_export
from zocr.embeddings.multimodal import EmbeddingBackends, MultimodalEmbeddingService
from zocr.service.jobs import process_job, verify_artifacts
from zocr.service.storage import LocalJobStore


@pytest.mark.parametrize("value, expected", [
    (0.0, {"ocr": 0.0}), (None, {}), (True, {}), (float("nan"), {}),
    (float("inf"), {}), ({"ocr": "0.8", "structure": 1.5}, {"ocr": .8, "structure": 1.0}),
    ([], {}), ({"ocr": -1, "structure": "bad"}, {"ocr": 0.0}),
])
def test_confidence_normalization(value, expected):
    assert normalize_confidence(value) == expected


def test_structured_confidence_survives_sql_export(tmp_path):
    path = tmp_path / "cells.jsonl"
    path.write_text(json.dumps({"text": "value", "confidence": {"ocr": .8, "structure": .7},
                                "meta": {"synthetic": True, "synthetic_role": "header"}}) + "\n", encoding="utf-8")
    result = sql_export(str(path), str(tmp_path / "sql"))
    with open(result["csv"], encoding="utf-8-sig") as stream:
        record = next(csv.DictReader(stream))
    assert record["confidence"] == "0.8"
    assert record["structure_confidence"] == "0.7"
    assert record["synthetic_role"] == "header"


@pytest.mark.parametrize("vision", [[], None, [None, None], [[float("nan")], [1.0]]])
def test_optional_invalid_vision_preserves_all_records(vision):
    service = MultimodalEmbeddingService(backends=EmbeddingBackends(
        text_encoder=lambda batch: [[1.0] for _ in batch], vision_encoder=lambda batch: vision,
    ))
    output = service.embed_objects([{"text": "a", "id": 1}, {"text": "b", "id": 2}])
    assert len(output.entries) == 2
    assert [entry["meta"]["id"] for entry in output.entries] == [1, 2]
    assert all(entry["embeddings"]["vision"] is None for entry in output.entries)


def test_mixed_optional_vision_keeps_fusion_dimensions():
    service = MultimodalEmbeddingService(backends=EmbeddingBackends(
        text_encoder=lambda batch: [[1.0] for _ in batch],
        vision_encoder=lambda batch: [[1.0, 0.0], None],
    ))
    entries = service.embed_objects([{"text": "a"}, {"text": "b"}]).entries
    assert len(entries[0]["vector"]) == len(entries[1]["vector"])
    assert entries[1]["embeddings"]["vision"] is None


def test_record_confidence_falls_back_from_explicit_null():
    from zocr.core.confidence import record_confidence
    from zocr.diff.review import source_evidence

    record = {"confidence": None, "meta": {"confidence": 0.0, "synthetic": True}}
    assert record_confidence(record) == {"ocr": 0.0}
    evidence = source_evidence(record)
    assert evidence["confidence"] == {"ocr": 0.0}
    assert evidence["observed_text"] is None
    assert evidence["evidence_kind"] == "inferred"


def test_unlimited_jsonl_validation_rejects_non_objects(tmp_path):
    from zocr.validate import _validate_jsonl

    path = tmp_path / "cells.jsonl"
    path.write_text('{"text":"ok"}\n[]\n', encoding="utf-8")
    errors, _ = _validate_jsonl(path, max_lines=0)
    assert errors and "not an object" in errors[0]


def test_zero_numpy_vision_vector_is_not_missing():
    import numpy as np

    service = MultimodalEmbeddingService(backends=EmbeddingBackends(
        text_encoder=lambda batch: np.array([[1.0, 0.0]]),
        vision_encoder=lambda batch: np.array([[0.0, 0.0]]),
    ))
    entry = service.embed_objects([{"text": "a"}]).entries[0]
    assert entry["embeddings"]["vision"] == [0.0, 0.0]


def test_invalid_required_text_vector_is_rejected():
    service = MultimodalEmbeddingService(backends=EmbeddingBackends(text_encoder=lambda batch: [[float("inf")]]))
    with pytest.raises(ValueError, match="non-finite"):
        service.embed_objects([{"text": "a"}])


def test_required_artifact_validation_catches_post_manifest_modification(tmp_path):
    summary = {"inputs": []}
    path = tmp_path / "pipeline_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    write_manifest(tmp_path, summary=summary)
    archive = tmp_path / "artifacts.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(path, "pipeline_summary.json")
        bundle.write(tmp_path / "zocr.manifest.json", "zocr.manifest.json")
    verify_artifacts(tmp_path, archive)
    path.write_text(json.dumps({**summary, "modified": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest validation"):
        verify_artifacts(tmp_path, archive)


def test_persistence_failure_is_durably_failed(tmp_path, monkeypatch):
    store = LocalJobStore(tmp_path)
    repository = MagicMock()
    repository.backend = "local"
    repository.read_job.return_value = {"status": "queued", "input": {"filename": "page.png"}}
    paths = store.job_paths("job")
    paths.input_dir.mkdir(parents=True)
    (paths.input_dir / "page.png").write_bytes(b"image")

    def runner(**kwargs):
        summary = {"inputs": kwargs["inputs"]}
        from pathlib import Path
        (Path(kwargs["outdir"]) / "pipeline_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    def fail(*args):
        raise OSError("storage unavailable")

    monkeypatch.setattr(store, "persist_tree", fail)
    status, error, _ = process_job(job_id="job", tenant_id="default", job_store=store,
                                   job_repository=repository, run_full_pipeline=runner)
    assert status == "failed"
    assert "storage unavailable" in error
    assert repository.write_job.call_args.args[2]["status"] == "failed"
