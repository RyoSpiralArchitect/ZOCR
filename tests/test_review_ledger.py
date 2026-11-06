import pytest

from zocr.service.metadata import LocalJobRepository
from zocr.service.storage import LocalJobStore


@pytest.fixture
def ledger(tmp_path):
    from zocr.service.review_ledger import ReviewLedger

    store = LocalJobStore(tmp_path)
    return ReviewLedger(store, LocalJobRepository(job_store=store))


def new_review(ledger):
    return ledger.create("tenant-a", {"events": [{"event_id": "event", "type": "cell_updated"}],
                                      "review": {"source_hashes": {"a": "old", "b": "new"}}},
                         {"a": {"job_id": "a"}, "b": {"job_id": "b"}}, {"subject": "creator"})


def bundle(review, revision=0, status="approved"):
    return {"submission_id": f"submission-{revision}", "expected_revision": revision,
            "source_hashes": review["result"]["review"]["source_hashes"],
            "decisions": [{"event_id": "event", "status": status, "reason": "Checked both source images"}]}


def test_ledger_keeps_history_revisions_and_idempotency(ledger):
    review = new_review(ledger)
    first = ledger.append(review["review_id"], "tenant-a", bundle(review), {"subject": "alice"})
    assert first["revision"] == 1
    assert first["review_status"] == "approved"
    assert first["history"][0]["actor"]["subject"] == "alice"
    retry = ledger.append(review["review_id"], "tenant-a", bundle(review), {"subject": "alice"})
    assert retry == first
    second = ledger.append(review["review_id"], "tenant-a", bundle(review, 1, "needs_review"), {"subject": "bob"})
    assert second["revision"] == 2
    assert second["history"][0] == first["history"][0]
    assert second["history"][1]["previous_hash"] == first["history"][0]["hash"]
    assert second["review_status"] == "review_required"


def test_ledger_rejects_stale_foreign_or_unknown_decisions(ledger):
    from zocr.service.review_ledger import ReviewConflict

    review = new_review(ledger)
    assert ledger.read(review["review_id"], "tenant-b") is None
    request = bundle(review)
    request["source_hashes"] = {"a": "other", "b": "new"}
    with pytest.raises(ReviewConflict):
        ledger.append(review["review_id"], "tenant-a", request, {"subject": "alice"})
    request = bundle(review)
    request["decisions"][0]["event_id"] = "unknown"
    with pytest.raises(ValueError):
        ledger.append(review["review_id"], "tenant-a", request, {"subject": "alice"})
    ledger.append(review["review_id"], "tenant-a", bundle(review), {"subject": "alice"})
    request = dict(bundle(review), submission_id="different")
    with pytest.raises(ReviewConflict):
        ledger.append(review["review_id"], "tenant-a", request, {"subject": "alice"})


def test_history_tampering_is_detected(ledger):
    review = new_review(ledger)
    ledger.append(review["review_id"], "tenant-a", bundle(review), {"subject": "alice"})
    path = ledger._path(review["review_id"], "tenant-a")
    record = ledger.store.read_json(path)
    record["history"][0]["decisions"][0]["reason"] = "forged"
    ledger.store.atomic_write_json(path, record)
    with pytest.raises(RuntimeError, match="integrity"):
        ledger.read(review["review_id"], "tenant-a")


def test_postgres_review_append_is_serialized(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from test_service_integrations import local_service
    from zocr.service.metadata import PostgresJobRepository
    from zocr.service.review_ledger import ReviewConflict, ReviewLedger

    store = LocalJobStore(tmp_path)
    repository = PostgresJobRepository(dsn=local_service("ZOCR_TEST_POSTGRES_URL"))
    ledger = ReviewLedger(store, repository)
    review = new_review(ledger)

    def approve(name):
        try:
            request = dict(bundle(review), submission_id=name)
            return ledger.append(review["review_id"], "tenant-a", request, {"subject": name})["revision"]
        except ReviewConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(approve, ["alice", "bob"]))
    assert results.count(1) == 1 and results.count("conflict") == 1
    restored = ReviewLedger(LocalJobStore(tmp_path / "another-process"), repository).read(review["review_id"], "tenant-a")
    assert restored["revision"] == 1 and len(restored["history"]) == 1
    assert ledger.read(review["review_id"], "tenant-b") is None


def test_corrections_are_separate_from_source_result(ledger):
    review = new_review(ledger)
    request = bundle(review, status="corrected")
    request["decisions"][0]["correction"] = {"side": "b", "text": "123"}
    result = ledger.append(review["review_id"], "tenant-a", request, {"subject": "alice"})
    assert result["result"] == review["result"]
    assert result["review_status"] == "correction_required"
    assert result["decisions"]["event"]["correction"] == {"side": "b", "text": "123"}
