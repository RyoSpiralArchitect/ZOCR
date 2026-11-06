from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from .execution_lock import execution_lock
from .metadata import normalize_tenant_id


class ReviewConflict(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def snapshot_hash(record):
    return digest({key: record[key] for key in (
        "schema", "review_id", "tenant_id", "created_at", "created_by", "sources", "result",
    )})


def review_status(events, decisions):
    statuses = {decisions.get(event["event_id"], {}).get("status", "needs_review") for event in events}
    return ("no_changes" if not statuses else "review_required" if "needs_review" in statuses else
            "correction_required" if "corrected" in statuses else
            "changes_rejected" if "rejected" in statuses else "approved")


def verify_record(record):
    previous = snapshot_hash(record)
    if previous != record["snapshot_hash"]:
        raise RuntimeError("Review snapshot integrity check failed")
    decisions = {}
    for revision, entry in enumerate(record["history"], 1):
        body = {key: value for key, value in entry.items() if key != "hash"}
        if entry["revision"] != revision or entry["previous_hash"] != previous or digest(body) != entry["hash"]:
            raise RuntimeError("Review history integrity check failed")
        previous = entry["hash"]
        decisions.update({decision["event_id"]: decision for decision in entry["decisions"]})
    if (record["revision"] != len(record["history"]) or record["decisions"] != decisions
            or record["review_status"] != review_status(record["result"]["events"], decisions)):
        raise RuntimeError("Review head integrity check failed")


class ReviewLedger:
    def __init__(self, store, repository):
        self.store = store
        self.repository = repository

    def _path(self, review_id, tenant_id):
        if not isinstance(review_id, str) or not re.fullmatch(r"[0-9a-f]{32}", review_id):
            raise ValueError("Invalid review id")
        return self.store.storage_dir / "reviews" / normalize_tenant_id(tenant_id) / f"{review_id}.json"

    def create(self, tenant_id, result, sources, actor):
        tenant_id = normalize_tenant_id(tenant_id)
        record = {
            "schema": "zocr.document_review.v1", "review_id": uuid.uuid4().hex,
            "tenant_id": tenant_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": actor, "sources": sources, "result": result,
            "revision": 0, "history": [], "decisions": {},
            "review_status": "review_required" if result["events"] else "no_changes",
        }
        record["snapshot_hash"] = snapshot_hash(record)
        if self.repository.backend == "postgres":
            self.repository._maybe_init_schema()
            with self.repository._connect() as connection:
                connection.execute("INSERT INTO zocr_document_reviews (tenant_id, review_id, payload) VALUES (%s,%s,%s)",
                                   (tenant_id, record["review_id"], self.repository._jsonb(record)))
        else:
            self.store.atomic_write_json(self._path(record["review_id"], tenant_id), record)
        return copy.deepcopy(record)

    def read(self, review_id, tenant_id):
        path = self._path(review_id, tenant_id)
        if self.repository.backend == "postgres":
            self.repository._maybe_init_schema()
            with self.repository._connect() as connection:
                row = connection.execute("SELECT payload FROM zocr_document_reviews WHERE tenant_id=%s AND review_id=%s",
                                         (tenant_id, review_id)).fetchone()
            record = self.repository._decode_payload(row[0]) if row else None
        else:
            record = self.store.read_json(path)
        if record is not None:
            verify_record(record)
        return copy.deepcopy(record)

    @contextmanager
    def _locked(self, review_id, tenant_id):
        path = self._path(review_id, tenant_id)
        if self.repository.backend == "postgres":
            self.repository._maybe_init_schema()
            with self.repository._connect() as connection:
                row = connection.execute(
                    "SELECT payload FROM zocr_document_reviews WHERE tenant_id=%s AND review_id=%s FOR UPDATE",
                    (tenant_id, review_id),
                ).fetchone()
                record = self.repository._decode_payload(row[0]) if row else None
                yield record, lambda updated: connection.execute(
                    "UPDATE zocr_document_reviews SET payload=%s WHERE tenant_id=%s AND review_id=%s",
                    (self.repository._jsonb(updated), tenant_id, review_id),
                )
        else:
            with execution_lock(self.store, self.repository, "review-" + review_id, tenant_id) as acquired:
                if not acquired:
                    raise ReviewConflict("Review is being updated; retry with its latest revision")
                yield self.store.read_json(path), lambda updated: self.store.atomic_write_json(path, updated)

    def append(self, review_id, tenant_id, request, actor):
        if not isinstance(request, dict):
            raise ValueError("Decision bundle must be an object")
        submission = request.get("submission_id")
        if not isinstance(submission, str) or not 1 <= len(submission) <= 128:
            raise ValueError("A bounded submission_id is required")
        if type(request.get("expected_revision")) is not int or request["expected_revision"] < 0:
            raise ValueError("expected_revision must be a non-negative integer")
        supplied = request.get("decisions")
        if not isinstance(supplied, list) or not 1 <= len(supplied) <= 1000:
            raise ValueError("Provide 1 to 1000 decisions")
        with self._locked(review_id, tenant_id) as (record, save):
            if record is None:
                raise KeyError("Unknown review")
            verify_record(record)
            request_hash = digest({"request": request, "actor": actor})
            for entry in record["history"]:
                if entry["submission_id"] == submission:
                    if entry["request_hash"] != request_hash:
                        raise ReviewConflict("submission_id was already used for a different request")
                    return copy.deepcopy(record)
            if record["revision"] != request["expected_revision"]:
                raise ReviewConflict("Review revision changed")
            if request.get("source_hashes") != record["result"]["review"]["source_hashes"]:
                raise ReviewConflict("Source hashes do not match this review")
            events = {event["event_id"]: event for event in record["result"]["events"]}
            decisions = []
            seen = set()
            for supplied_decision in supplied:
                if not isinstance(supplied_decision, dict):
                    raise ValueError("Each decision must be an object")
                event_id = supplied_decision.get("event_id")
                if not isinstance(event_id, str) or event_id not in events or event_id in seen:
                    raise ValueError("Unknown or duplicate event_id")
                seen.add(event_id)
                status, reason = supplied_decision.get("status"), supplied_decision.get("reason")
                if status not in {"approved", "rejected", "needs_review", "corrected"}:
                    raise ValueError("Unknown decision status")
                if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 4000:
                    raise ValueError("Each decision requires a bounded reason")
                decision = {"event_id": event_id, "status": status, "reason": reason.strip()}
                if status == "corrected":
                    correction = supplied_decision.get("correction")
                    if (events[event_id].get("type") != "cell_updated" or not isinstance(correction, dict)
                            or correction.get("side") not in {"a", "b"}
                            or not isinstance(correction.get("text"), str) or len(correction["text"]) > 20000):
                        raise ValueError("Cell correction requires side and bounded text")
                    decision["correction"] = {"side": correction["side"], "text": correction["text"]}
                decisions.append(decision)
            entry = {
                "revision": record["revision"] + 1, "submission_id": submission, "request_hash": request_hash,
                "actor": actor, "recorded_at": datetime.now(timezone.utc).isoformat(),
                "decisions": decisions,
                "previous_hash": record["history"][-1]["hash"] if record["history"] else record["snapshot_hash"],
            }
            entry["hash"] = digest(entry)
            record["history"].append(entry)
            record["revision"] = entry["revision"]
            record["decisions"].update({decision["event_id"]: decision for decision in decisions})
            record["review_status"] = review_status(record["result"]["events"], record["decisions"])
            save(record)
            return copy.deepcopy(record)
