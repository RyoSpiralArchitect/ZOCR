import hashlib
import json

from ..artifacts.manifest import MANIFEST_FILENAME
from ..diff.differ import SemanticDiffer
from .review_ledger import ReviewConflict, ReviewLedger, digest
from .storage import result_paths


def install_review_routes(app, store, repository, require_context, read_job, audit):
    from fastapi import Depends, HTTPException, Request

    from .learning import LearningRegistry

    ledger = ReviewLedger(store, repository)

    def actor(context):
        return {"subject": context.principal.subject, "auth_mode": context.principal.auth_mode}

    def source(job_id, tenant_id):
        if not isinstance(job_id, str):
            raise HTTPException(status_code=422, detail="Source job ids must be strings")
        job = read_job(job_id, tenant_id)
        if job.get("status") != "succeeded":
            raise HTTPException(status_code=409, detail="Review requires completed source jobs")
        out = result_paths(store, job_id, job).out_dir
        manifest_path, cells = out / MANIFEST_FILENAME, out / "rag" / "cells.jsonl"
        try:
            store.ensure_local_file(manifest_path, force=True)
            store.ensure_local_file(cells, force=True)
            raw_manifest = manifest_path.read_bytes()
            manifest = json.loads(raw_manifest)
            entry = manifest["artifacts"]["rag_cells"]
            cells_hash = hashlib.sha256(cells.read_bytes()).hexdigest()
            if entry.get("path") != "rag/cells.jsonl" or entry.get("sha256") != cells_hash:
                raise ValueError("Source cells failed their manifest hash check")
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail="Verified source cells are unavailable") from exc
        return cells, {"job_id": job_id, "execution_token": job.get("execution_token"),
                       "cells_sha256": cells_hash, "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest()}

    def get_review(review_id, tenant_id):
        try:
            record = ledger.read(review_id, tenant_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail="Unknown review")
        return record

    learning = LearningRegistry(store, repository, ledger, source, read_job)

    def learning_call(operation):
        import subprocess
        try:
            return operation()
        except ReviewConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown tenant learning object") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (subprocess.TimeoutExpired, TimeoutError) as exc:
            raise HTTPException(status_code=504, detail="Isolated learning/evaluation deadline exceeded") from exc
        except OSError as exc:
            raise HTTPException(status_code=409, detail="Verified learning source is unavailable") from exc

    @app.post("/v1/reviews/{review_id}/learning-samples")
    def register_learning_sample(request: Request, review_id: str, payload: dict,
                                 context=Depends(require_context("zocr.reviews.approve"))):
        sample = learning_call(lambda: learning.register(context.tenant_id, review_id, payload, actor(context)))
        audit("learning.sample.registered", {"object_id": sample["object_id"], "role": sample["role"]},
              request=request, principal=context.principal, tenant_id=context.tenant_id)
        return {"sample": sample}

    @app.get("/v1/learning/{object_id}")
    def read_learning_object(object_id: str, context=Depends(require_context("zocr.jobs.read"))):
        def read():
            value = learning.get(context.tenant_id, object_id)
            if value is None or object_id.startswith(("document-", "exposure-")):
                raise KeyError(object_id)
            if object_id.startswith("sample-"):
                learning._current(context.tenant_id, value)
            else:
                learning._samples(context.tenant_id, value["sample_ids"])
                if object_id.startswith("evaluation-"):
                    model = learning.get(context.tenant_id, value["model_id"])
                    if model is None:
                        raise KeyError(value["model_id"])
                    learning._samples(context.tenant_id, model["sample_ids"])
            return value
        return {"object": learning_call(read)}

    @app.post("/v1/learning/models")
    def train_learning_candidate(request: Request, payload: dict,
                                 context=Depends(require_context("zocr.jobs.run")),
                                 approval=Depends(require_context("zocr.reviews.approve"))):
        candidate = learning_call(lambda: learning.train(context.tenant_id, payload.get("sample_ids"),
                                                         payload.get("allowed_chars"), actor(context)))
        audit("learning.candidate.created", {"object_id": candidate["object_id"]}, request=request,
              principal=context.principal, tenant_id=context.tenant_id)
        return {"candidate": candidate}

    @app.post("/v1/learning/models/{model_id}/evaluations")
    def evaluate_learning_candidate(request: Request, model_id: str, payload: dict,
                                    context=Depends(require_context("zocr.jobs.run")),
                                    approval=Depends(require_context("zocr.reviews.approve"))):
        result = learning_call(lambda: learning.evaluate(context.tenant_id, model_id, payload.get("sample_ids"), actor(context)))
        audit("learning.candidate.evaluated", {"model_id": model_id}, request=request,
              principal=context.principal, tenant_id=context.tenant_id)
        return {"evaluation": result}

    @app.get("/review")
    def source_review_ui():
        from fastapi.responses import HTMLResponse
        from .source_review_ui import HTML

        return HTMLResponse(HTML, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src blob: data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"})

    @app.get("/v1/reviews/{review_id}/events/{event_id}/image")
    def source_review_image(review_id: str, event_id: str, bbox: str | None = None,
                             context=Depends(require_context("zocr.jobs.read"))):
        import io
        from pathlib import Path
        from PIL import Image
        from fastapi.responses import Response
        from ..consensus.learning import _read_bounded

        record = get_review(review_id, context.tenant_id)
        event = next((value for value in record["result"]["events"] if value["event_id"] == event_id), None)
        if event is None or record["result"].get("mode") != "source_observations":
            raise HTTPException(status_code=404, detail="Unknown source observation")
        reference = record["sources"]["a"]
        _, actual = source(reference["job_id"], context.tenant_id)
        if actual != reference:
            raise HTTPException(status_code=409, detail="Source version changed")
        try:
            root, manifest, _, _ = learning._bundle(context.tenant_id, reference)
            evidence = event["evidence"]["a"][0]
            path = (root / evidence["image_path"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Source image escaped its attempt")
            relative = Path(path).relative_to(root.resolve()).as_posix()
            entry = next((entry for entry in manifest["artifacts"].values() if entry.get("path") == relative), None)
            store.ensure_local_file(path, force=True)
            raw = _read_bounded(path, 32 * 1024 * 1024)
            if entry is None or entry.get("sha256") != hashlib.sha256(raw).hexdigest():
                raise ValueError("Evidence pixels failed their manifest check")
            with Image.open(io.BytesIO(raw)) as image:
                if image.width * image.height > 40_000_000:
                    raise ValueError("Source image exceeds the pixel budget")
                original = learning._box(evidence["bbox"], image.size)
                selected = learning._box(json.loads(bbox) if bbox else evidence["bbox"], image.size)
                if max(original[0], selected[0]) >= min(original[2], selected[2]) or max(original[1], selected[1]) >= min(original[3], selected[3]):
                    raise ValueError("Selected image must overlap its source observation")
                if (selected[2] - selected[0]) * (selected[3] - selected[1]) > 1_000_000:
                    raise ValueError("Selected image exceeds the crop budget")
                buffer = io.BytesIO()
                image.crop(selected).convert("RGB").save(buffer, format="PNG")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=409, detail="Verified source image or selected bounds are unavailable") from exc
        return Response(buffer.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/v1/source-reviews")
    def create_source_review(request: Request, payload: dict,
                             context=Depends(require_context("zocr.jobs.write"))):
        from ..diff.review import source_evidence

        cells, reference = source(payload.get("job_id"), context.tenant_id)
        if cells.stat().st_size > 32 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Source review exceeds the artifact budget")
        raw = cells.read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["cells_sha256"]:
            raise HTTPException(status_code=409, detail="Source changed during review creation")
        try:
            rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
            if not 1 <= len(rows) <= 10000 or any(not isinstance(row, dict) or not isinstance(row.get("text"), str) for row in rows):
                raise ValueError("Invalid source observations")
            events = [{"type": "source_observation", "event_id": digest({"source": reference, "ordinal": index}),
                       "review_required": True, "evidence": {"a": [source_evidence(row)], "b": []}}
                      for index, row in enumerate(rows)]
        except (ValueError, UnicodeError, TypeError) as exc:
            raise HTTPException(status_code=409, detail="Source observations could not be validated") from exc
        result = {"schema": "zocr.source_review.v1", "mode": "source_observations", "events": events,
                  "review": {"source_hashes": {"a": reference["cells_sha256"]},
                             "policy": "Source observations require human transcription and separate explicit learning consent."}}
        record = ledger.create(context.tenant_id, result, {"a": reference}, actor(context))
        audit("review.source.created", {"review_id": record["review_id"]}, request=request,
              principal=context.principal, tenant_id=context.tenant_id)
        return {"review": record}

    @app.post("/v1/reviews")
    def create_review(request: Request, payload: dict,
                      context=Depends(require_context("zocr.jobs.write"))):
        a, source_a = source(payload.get("job_a"), context.tenant_id)
        b, source_b = source(payload.get("job_b"), context.tenant_id)
        try:
            result = SemanticDiffer().compare_bundle(a, b)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=409, detail="Source comparison could not be validated") from exc
        if result["review"]["source_hashes"] != {"a": source_a["cells_sha256"], "b": source_b["cells_sha256"]}:
            raise HTTPException(status_code=409, detail="Source changed during comparison")
        record = ledger.create(context.tenant_id, result, {"a": source_a, "b": source_b}, actor(context))
        audit("review.created", {"review_id": record["review_id"]}, request=request,
              principal=context.principal, tenant_id=context.tenant_id)
        return {"review": record}

    @app.get("/v1/reviews/{review_id}")
    def read_review(review_id: str, context=Depends(require_context("zocr.jobs.read"))):
        return {"review": get_review(review_id, context.tenant_id)}

    @app.get("/v1/reviews/{review_id}/history")
    def read_history(review_id: str, context=Depends(require_context("zocr.jobs.read"))):
        record = get_review(review_id, context.tenant_id)
        return {"review_id": review_id, "revision": record["revision"],
                "snapshot_hash": record["snapshot_hash"], "history": record["history"]}

    @app.post("/v1/reviews/{review_id}/decisions")
    def append_decisions(request: Request, review_id: str, payload: dict,
                         context=Depends(require_context("zocr.reviews.approve"))):
        record = get_review(review_id, context.tenant_id)
        for expected in record["sources"].values():
            _, actual = source(expected["job_id"], context.tenant_id)
            if actual != expected:
                raise HTTPException(status_code=409, detail="Source version changed; create a new review")
        try:
            record = ledger.append(review_id, context.tenant_id, payload, actor(context))
        except ReviewConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown review") from exc
        audit("review.decisions.recorded", {"review_id": review_id, "revision": record["revision"]},
              request=request, principal=context.principal, tenant_id=context.tenant_id)
        return {"review": record}
