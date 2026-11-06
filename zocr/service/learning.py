from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tempfile

from PIL import Image

from ..artifacts.manifest import MANIFEST_FILENAME
from ..consensus.learning import _glyph_image, _read_bounded, run_learning_trial
from ..consensus.cell_evaluation import evaluate_cells
from .execution_lock import execution_lock
from .metadata import normalize_tenant_id
from .review_ledger import ReviewConflict, digest
from .storage import result_paths


class LearningRegistry:
    def __init__(self, store, repository, ledger, source, read_job):
        self.store, self.repository, self.ledger = store, repository, ledger
        self.source, self.read_job = source, read_job

    def _path(self, tenant, key):
        if not isinstance(key, str) or not re.fullmatch(r"(?:sample|model|document|evaluation)-[0-9a-f]{64}", key):
            raise ValueError("Invalid learning object id")
        return self.store.storage_dir / "reviews" / normalize_tenant_id(tenant) / "learning" / (key + ".json")

    def get(self, tenant, key):
        path = self._path(tenant, key)
        if self.repository.backend == "postgres":
            self.repository._maybe_init_schema()
            with self.repository._connect() as connection:
                row = connection.execute("SELECT payload FROM zocr_learning_objects WHERE tenant_id=%s AND object_id=%s",
                                         (tenant, key)).fetchone()
            value = self.repository._decode_payload(row[0]) if row else None
        else:
            value = self.store.read_json(path)
            if value is None and self.store.has_file(path):
                raise RuntimeError("Unreadable learning object")
        if value is not None and (value.get("tenant_id") != tenant or value.get("object_id") != key or
                                  value.get("integrity_sha256") != digest({k: v for k, v in value.items() if k != "integrity_sha256"})):
            raise RuntimeError("Learning object integrity check failed")
        return value

    def _put(self, tenant, key, body):
        payload = {**body, "tenant_id": tenant, "object_id": key}
        payload["integrity_sha256"] = digest(payload)
        path = self._path(tenant, key)
        if self.repository.backend == "postgres":
            self.repository._maybe_init_schema()
            with self.repository._connect() as connection:
                connection.execute("INSERT INTO zocr_learning_objects (tenant_id,object_id,payload) VALUES (%s,%s,%s) "
                                   "ON CONFLICT DO NOTHING", (tenant, key, self.repository._jsonb(payload)))
        else:
            with execution_lock(self.store, self.repository, "learning-object:" + key, tenant) as acquired:
                if not acquired:
                    raise ReviewConflict("Learning object is busy; retry")
                existing = self.get(tenant, key)
                if existing is None:
                    self.store.atomic_write_json(path, payload)
        stored = self.get(tenant, key)
        if stored != payload:
            raise ReviewConflict("Immutable learning object already has a different assignment")
        return stored

    def _bundle(self, tenant, reference):
        cells, actual = self.source(reference["job_id"], tenant)
        if actual != reference:
            raise ReviewConflict("Learning source version changed")
        job = self.read_job(reference["job_id"], tenant)
        paths = result_paths(self.store, reference["job_id"], job)
        manifest = json.loads((paths.out_dir / MANIFEST_FILENAME).read_bytes())
        filename = job.get("input", {}).get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("Verified original input is required for document-level separation")
        original = paths.input_dir / filename
        self.store.ensure_local_file(original, force=True)
        document = _read_bounded(original, 64 * 1024 * 1024)
        document_hash = hashlib.sha256(document).hexdigest()
        if not any(entry.get("sha256") == document_hash for entry in manifest.get("inputs", [])):
            raise ReviewConflict("Original input is not hash-bound to the source manifest; rerun the source job")
        return paths.out_dir, manifest, document, document_hash

    def _current(self, tenant, sample):
        review = self.ledger.read(sample["review_id"], tenant)
        if review is None or review["revision"] != sample["review_revision"]:
            raise ReviewConflict("Training consent became stale; register a new sample")
        if digest(review["decisions"].get(sample["event_id"])) != sample["decision_sha256"]:
            raise ReviewConflict("Correction or consent changed")
        return self._bundle(tenant, sample["source"])

    def register(self, tenant, review_id, request, actor):
        review = self.ledger.read(review_id, tenant)
        if review is None:
            raise KeyError("Unknown review")
        if request.get("expected_revision") != review["revision"] or isinstance(request.get("expected_revision"), bool):
            raise ReviewConflict("Review revision changed")
        if request.get("consent") is not True or request.get("role") not in ("train", "selection", "test"):
            raise ValueError("Explicit consent and train/selection/test role are required")
        if request.get("source_kind") not in ("fixture", "real_document"):
            raise ValueError("Declare fixture or real_document provenance")
        group = request.get("document_group_id")
        if not isinstance(group, str) or not 1 <= len(group.strip()) <= 256:
            raise ValueError("An explicit document-family group id is required")
        event_id = request.get("event_id")
        if not isinstance(event_id, str):
            raise ValueError("Invalid event id")
        decision = review["decisions"].get(event_id, {})
        if decision.get("status") != "corrected":
            raise ValueError("Only an authenticated, recorded cell correction can supply learning labels")
        correction = decision["correction"]
        if len(correction["text"]) > 1000:
            raise ValueError("Learning supports corrected cells up to 1000 characters")
        event = next(event for event in review["result"]["events"] if event["event_id"] == event_id)
        evidence = event.get("evidence", {}).get(correction["side"], [])
        if len(evidence) != 1 or evidence[0].get("synthetic") or evidence[0].get("evidence_kind") != "observed":
            raise ValueError("One observed source cell is required")
        cell = evidence[0]
        reference = review["sources"][correction["side"]]
        root, manifest, document, document_hash = self._bundle(tenant, reference)
        image_path = (root / (cell.get("image_path") or "")).resolve()
        if not image_path.is_relative_to(root.resolve()):
            raise ValueError("Evidence image escaped its attempt")
        relative = image_path.relative_to(root.resolve()).as_posix()
        entry = next((entry for entry in manifest["artifacts"].values() if entry.get("path") == relative), None)
        self.store.ensure_local_file(image_path, force=True)
        raw = _read_bounded(image_path, 32 * 1024 * 1024)
        if entry is None or entry.get("sha256") != hashlib.sha256(raw).hexdigest():
            raise ReviewConflict("Evidence pixels failed their manifest check")
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 40_000_000:
                raise ValueError("Evidence image is too large")
            box = self._box(cell.get("bbox"), image.size)
            crop = image.crop(box).convert("L")
        if crop.width * crop.height > 1_000_000:
            raise ValueError("Cell crop is too large")
        glyphs = request.get("glyphs", [])
        if not isinstance(glyphs, list) or len(glyphs) > 64:
            raise ValueError("Provide at most 64 explicitly labelled glyph boxes")
        labels = [g.get("text") for g in glyphs if isinstance(g, dict)]
        if any(not isinstance(label, str) or len(label) != 1 or label.isspace() for label in labels):
            raise ValueError("Glyph labels must be single visible characters")
        if request["role"] != "test" and (not glyphs or "".join(labels) != "".join(c for c in correction["text"] if not c.isspace())):
            raise ValueError("Explicit glyph boxes must spell the corrected non-whitespace characters")
        encoded = []
        for glyph in glyphs:
            if not isinstance(glyph, dict) or not isinstance(glyph.get("text"), str) or len(glyph["text"]) != 1 or glyph["text"].isspace():
                raise ValueError("Each glyph requires one visible, human-labelled character")
            region = crop.crop(self._box(glyph.get("bbox"), crop.size))
            png = self._png(region)
            _glyph_image(base64.b64decode(png))
            encoded.append({"text": glyph["text"], "png": png})
        sample = {"schema": "zocr.approved_learning_sample.v1", "review_id": review_id,
                  "review_revision": review["revision"], "event_id": event_id, "decision_sha256": digest(decision),
                  "source": reference, "document_sha256": document_hash, "role": request["role"],
                  "document_group_id": group, "source_kind": request["source_kind"],
                  "consent_actor": actor,
                  "correction_actor": next(entry["actor"] for entry in reversed(review["history"])
                                           if any(d["event_id"] == event_id for d in entry["decisions"])),
                  "expected": correction["text"], "reason": decision["reason"],
                  "png": self._png(crop), "glyphs": encoded, "automatic_activation": False}
        self._current(tenant, sample)
        self._put(tenant, "document-" + digest({"group": group}), {"document_group_id": group, "role": request["role"]})
        self._put(tenant, "document-" + document_hash, {"document_sha256": document_hash,
                                                    "document_group_id": group, "role": request["role"]})
        return self._put(tenant, "sample-" + digest(sample), sample)

    @staticmethod
    def _box(box, size):
        if not isinstance(box, (list, tuple)) or len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            raise ValueError("Invalid source/glyph bounding box")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]
                and round(x1) > round(x0) and round(y1) > round(y0)):
            raise ValueError("Bounding box is outside its source")
        return tuple(box)

    @staticmethod
    def _png(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _samples(self, tenant, ids):
        if not isinstance(ids, list) or not 1 <= len(ids) <= 32 or any(not isinstance(key, str) for key in ids) or len(set(ids)) != len(ids):
            raise ValueError("Provide 1..32 distinct learning sample ids")
        samples = []
        for key in ids:
            sample = self.get(tenant, key)
            if sample is None or not key.startswith("sample-"):
                raise KeyError("Unknown learning sample")
            self._current(tenant, sample)
            samples.append(sample)
        return samples

    def train(self, tenant, ids, alphabet, actor):
        samples = self._samples(tenant, ids)
        if any(sample["role"] not in {"train", "selection"} for sample in samples):
            raise ValueError("Independent test data cannot enter training or model selection")
        with execution_lock(self.store, self.repository, "learning-trial", tenant) as acquired:
            if not acquired:
                raise ReviewConflict("Another tenant learning trial is running")
            with tempfile.TemporaryDirectory(prefix="zocr-approved-trial-") as directory:
                root = Path(directory)
                rows = []
                for sample in samples:
                    _, _, document, document_hash = self._current(tenant, sample)
                    (root / document_hash).write_bytes(document)
                    for index, glyph in enumerate(sample["glyphs"]):
                        key = sample["object_id"] + f"-{index}"
                        (root / (key + ".png")).write_bytes(base64.b64decode(glyph["png"]))
                        rows.append({"id": key, "image": key + ".png", "text": glyph["text"],
                                     "document": document_hash, "group_id": sample["document_group_id"],
                                     "split": "train" if sample["role"] == "train" else "holdout", "approved": True,
                                     "reviewer": sample["consent_actor"]["subject"], "reason": sample["reason"]})
                dataset = root / "glyphs.jsonl"
                dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
                report = run_learning_trial(dataset, root / "trial", allowed_chars=alphabet)
                model_path = root / "trial" / "candidate_memory.json"
                memory = json.loads(model_path.read_bytes()) if model_path.exists() else None
            for sample in samples:
                self._current(tenant, sample)
            record = {"schema": "zocr.tenant_learning_candidate.v1", "actor": actor, "sample_ids": ids,
                      "sample_revisions": {sample["object_id"]: sample["review_revision"] for sample in samples},
                      "report": report, "memory": memory, "automatic_activation": False}
            return self._put(tenant, "model-" + digest(record), record)

    def evaluate(self, tenant, model_id, ids, actor):
        model = self.get(tenant, model_id)
        if model is None or not model_id.startswith("model-"):
            raise KeyError("Unknown tenant model")
        self._samples(tenant, model["sample_ids"])
        if model["memory"] is None:
            raise ReviewConflict("Rejected candidates cannot be evaluated as deployable models")
        samples = self._samples(tenant, ids)
        if any(sample["role"] != "test" for sample in samples):
            raise ValueError("Final evaluation requires separately assigned test documents")
        with tempfile.TemporaryDirectory(prefix="zocr-independent-cells-") as directory:
            root = Path(directory)
            rows = []
            for sample in samples:
                _, _, document, document_hash = self._current(tenant, sample)
                (root / document_hash).write_bytes(document)
                name = sample["object_id"] + ".png"
                (root / name).write_bytes(base64.b64decode(sample["png"]))
                rows.append({"id": sample["object_id"], "image": name, "document": document_hash, "split": "test",
                             "text": sample["expected"], "source_kind": sample["source_kind"], "approved": True,
                             "group_id": sample["document_group_id"],
                             "reviewer": sample["consent_actor"]["subject"], "reason": sample["reason"]})
            path = root / "test.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            baseline = evaluate_cells(path, root / "baseline", allowed_chars=model["report"]["allowed_chars"])
            candidate = evaluate_cells(path, root / "candidate", memory=model["memory"], allowed_chars=model["report"]["allowed_chars"])
        for sample in samples:
            self._current(tenant, sample)
        self._samples(tenant, model["sample_ids"])
        result = {"schema": "zocr.tenant_model_evaluation.v1", "model_id": model_id, "sample_ids": ids,
                  "actor": actor, "baseline": baseline, "candidate": candidate,
                  "automatic_activation": False, "test_results_must_not_be_used_as_training_labels": True}
        return self._put(tenant, "evaluation-" + digest(result), result)
