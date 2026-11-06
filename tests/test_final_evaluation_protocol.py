import base64
import hashlib
import io

from PIL import Image
import pytest

from zocr.service.learning import LearningRegistry
from zocr.service.metadata import LocalJobRepository
from zocr.service.review_ledger import ReviewConflict
from zocr.service.storage import LocalJobStore


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    import zocr.service.learning as implementation

    registry = LearningRegistry(LocalJobStore(tmp_path), LocalJobRepository(job_store=LocalJobStore(tmp_path)), None, None, None)
    model = {"sample_ids": ["training"], "memory": {"fixture": 1}, "report": {"allowed_chars": "2358"}}
    first = registry._put("tenant-a", "model-" + "1" * 64, model)
    second = registry._put("tenant-a", "model-" + "2" * 64, {**model, "memory": {"fixture": 2}})
    buffer = io.BytesIO()
    Image.new("L", (64, 32), "white").save(buffer, format="PNG")
    document = buffer.getvalue()
    sample = {"object_id": "sample-" + "a" * 64, "integrity_sha256": "b" * 64,
              "review_revision": 1, "role": "test", "document_group_id": "independent-family",
              "png": base64.b64encode(document).decode(), "expected": "2358", "source_kind": "fixture",
              "consent_actor": {"subject": "fixture-reviewer"}, "reason": "Protocol fixture, not real OCR",
              "evaluation_protocol": "one-shot-family-v1"}
    monkeypatch.setattr(registry, "_samples", lambda tenant, ids: [sample] if ids == [sample["object_id"]] else [])
    monkeypatch.setattr(registry, "_current", lambda tenant, sample: (tmp_path, {}, document, hashlib.sha256(document).hexdigest()))
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        return {"fixture": "metrics omitted: this tests reservations, not recognition"}

    monkeypatch.setattr(implementation, "evaluate_cells", evaluate)
    monkeypatch.setattr(implementation, "compare_cell_evaluations", lambda *args: {"status": "fixture"})
    return registry, first, second, sample, calls


def test_final_family_is_single_use_and_identical_requests_only_read_the_saved_result(protocol):
    registry, first, second, sample, calls = protocol
    actor = {"subject": "fixture-reviewer"}
    result = registry.evaluate("tenant-a", first["object_id"], [sample["object_id"]], actor)
    assert len(calls) == 2
    assert registry.evaluate("tenant-a", first["object_id"], [sample["object_id"]], actor) == result
    assert len(calls) == 2
    with pytest.raises(ReviewConflict, match="exposed|consumed"):
        registry.evaluate("tenant-a", second["object_id"], [sample["object_id"]], actor)
    assert len(calls) == 2
    assert result["evaluation_protocol"]["final_family_reuse"] == "forbidden"
    assert result["automatic_activation"] is False


def test_legacy_final_samples_cannot_be_promoted_to_fresh_unseen_evaluation(protocol):
    registry, first, second, sample, calls = protocol
    sample.pop("evaluation_protocol")
    with pytest.raises(ReviewConflict, match="Legacy|exposed"):
        registry.evaluate("tenant-a", first["object_id"], [sample["object_id"]], {})
    assert calls == []


def test_failed_final_evaluation_is_conservatively_consumed(protocol, monkeypatch):
    import zocr.service.learning as implementation

    registry, first, second, sample, calls = protocol

    def fail(*args, **kwargs):
        calls.append("failed")
        raise TimeoutError("Fixture interruption after final-set exposure")

    monkeypatch.setattr(implementation, "evaluate_cells", fail)
    with pytest.raises(TimeoutError):
        registry.evaluate("tenant-a", first["object_id"], [sample["object_id"]], {})
    with pytest.raises(ReviewConflict, match="exposed|consumed"):
        registry.evaluate("tenant-a", first["object_id"], [sample["object_id"]], {})
    with pytest.raises(ReviewConflict, match="exposed|consumed"):
        registry.evaluate("tenant-a", second["object_id"], [sample["object_id"]], {})
    assert calls == ["failed"]
