import io
import json
import subprocess
import sys
import uuid

import pytest
from PIL import Image, ImageDraw, ImageFont

from test_service_integrations import local_service


def learning_page(scale, text):
    image = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(image)
    for y in (20, 210, 440, 680):
        draw.line((20, y, 1180, y), fill="black", width=3)
    for x in (20, 600, 1180):
        draw.line((x, 20, x, 680), fill="black", width=3)
    for x, y, label in ((50, 80, "Item"), (650, 80, "Qty"), (50, 280, "A"), (50, 550, "B"), (650, 550, "2")):
        draw.text((x, y), label, fill="black", font=ImageFont.load_default())
    strip = Image.new("RGB", (96, 24), "white")
    for index, char in enumerate(text):
        ImageDraw.Draw(strip).text((index * 24 + 4, 3), char, fill="black", font=ImageFont.load_default())
    image.paste(strip.resize((96 * scale, 24 * scale), Image.Resampling.LANCZOS), (650, 250))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("metadata_backend", ["postgres", "local"])
def test_real_worker_correction_learning_and_independent_evaluation(tmp_path, monkeypatch, metadata_backend):
    redis_url = local_service("ZOCR_TEST_REDIS_URL")
    postgres_url = local_service("ZOCR_TEST_POSTGRES_URL")
    endpoint = local_service("ZOCR_TEST_MINIO_URL")
    import boto3
    from fastapi.testclient import TestClient
    from zocr.service.app import create_app

    token = uuid.uuid4().hex
    bucket = "learning-" + token
    settings = {
        "AWS_ACCESS_KEY_ID": "zocr_test", "AWS_SECRET_ACCESS_KEY": "zocr_test_local_only",
        "AWS_DEFAULT_REGION": "us-east-1", "AWS_EC2_METADATA_DISABLED": "true",
        "ZOCR_API_STORAGE_DIR": str(tmp_path / "cache"), "ZOCR_API_STORAGE_BACKEND": "s3",
        "ZOCR_API_S3_ENDPOINT_URL": endpoint, "ZOCR_API_S3_BUCKET": bucket,
        "ZOCR_API_S3_PREFIX": "zocr", "ZOCR_API_S3_FORCE_PATH_STYLE": "1", "ZOCR_API_S3_PRESIGN_ENABLED": "0",
        "ZOCR_API_METADATA_BACKEND": metadata_backend, "ZOCR_API_DATABASE_URL": postgres_url,
        "ZOCR_API_DB_AUTO_INIT": "1", "ZOCR_API_QUEUE_BACKEND": "redis", "ZOCR_API_REDIS_URL": redis_url,
        "ZOCR_API_REDIS_STREAM": "learning-" + token, "ZOCR_API_REDIS_GROUP": "workers",
        "ZOCR_API_JOBS_CLEANUP_ON_STARTUP": "0", "ZOCR_API_JOBS_CLEANUP_ON_CREATE": "0",
        "ZOCR_API_JOBS_RESUME_ON_STARTUP": "0", "ZOCR_API_RUN_TIMEOUT_SEC": "90",
        "ZOCR_TOY_SWEEPS": "1", "ZOCR_TOY_SWEEP_LIMIT": "1",
        "ZOCR_API_KEYS_JSON": json.dumps({
            "learner": {"subject": "reviewer", "tenant_id": "learning-" + token, "roles": ["admin"]},
            "other": {"tenant_id": "other-" + token, "roles": ["admin"]},
        }),
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    boto3.client("s3", endpoint_url=endpoint).create_bucket(Bucket=bucket)
    headers = {"X-API-Key": "learner"}
    worker = [sys.executable, "-m", "zocr.service.worker", "--consumer", "learning-operational", "--once"]
    registered = []
    final_review = None
    with TestClient(create_app()) as client:
        for scale, role in ((2, "train"), (3, "selection"), (4, "selection"), (5, "test")):
            jobs = []
            for text in ("1111", "2358"):
                response = client.post("/v1/jobs?toy_lite=true&k=1", headers=headers,
                                       files={"file": ("page.png", learning_page(scale, text), "image/png")})
                assert response.status_code == 200, response.text
                job_id = response.json()["job"]["id"]
                completed = subprocess.run(worker, capture_output=True, text=True, timeout=100)
                assert completed.returncode == 0, completed.stderr
                job = client.get(f"/v1/jobs/{job_id}", headers=headers).json()["job"]
                assert job["status"] == "succeeded", (job, completed.stderr)
                jobs.append(job_id)
            response = client.post("/v1/reviews", headers=headers, json={"job_a": jobs[0], "job_b": jobs[1]})
            assert response.status_code == 200, response.text
            review = response.json()["review"]
            eligible = []
            for event in review["result"]["events"]:
                evidence = event.get("evidence", {}).get("b", [])
                if event.get("type") == "cell_updated" and len(evidence) == 1:
                    box = evidence[0].get("bbox") or []
                    if len(box) == 4 and box[2] > 650 and box[0] < 650 + 96 * scale and box[3] > 250 and box[1] < 250 + 24 * scale:
                        eligible.append((event, box))
            assert eligible, [(event.get("type"), event.get("evidence", {}).get("b")) for event in review["result"]["events"]]
            event, original_box = eligible[0]
            box = [640, 250, 650 + 96 * scale, 250 + 24 * scale]
            decision = {"expected_revision": 0, "submission_id": "correct-" + token,
                        "source_hashes": review["result"]["review"]["source_hashes"],
                        "decisions": [{"event_id": event["event_id"], "status": "corrected", "reason": "Known rendered fixture labels",
                                       "correction": {"side": "b", "text": "2358", "bbox": box}}]}
            response = client.post(f"/v1/reviews/{review['review_id']}/decisions", headers=headers, json=decision)
            assert response.status_code == 200, response.text
            glyphs = [{"text": char, "bbox": [650 + index * 24 * scale - box[0], 250 - box[1],
                                                650 + (index + 1) * 24 * scale - box[0], 250 + 24 * scale - box[1]]}
                      for index, char in enumerate("2358")]
            request = {"expected_revision": 1, "event_id": event["event_id"], "role": role, "consent": True,
                       "document_group_id": "fixture-family-" + str(scale), "source_kind": "fixture",
                       "glyphs": glyphs if role != "test" else []}
            response = client.post(f"/v1/reviews/{review['review_id']}/learning-samples", headers=headers, json=request)
            assert response.status_code == 200, response.text
            sample = response.json()["sample"]
            assert sample["consent_actor"]["subject"] == "reviewer"
            assert sample["source_cell_bbox"] == original_box
            assert sample["annotation_bbox"] == box
            registered.append(sample)
            if role == "test":
                final_review = (review, event)
        response = client.post("/v1/learning/models", headers=headers,
                               json={"sample_ids": [sample["object_id"] for sample in registered[:-1]], "allowed_chars": "2358"})
        assert response.status_code == 200, response.text
        model = response.json()["candidate"]
        assert model["report"]["baseline"]["samples"] == 8
        assert model["memory"] is not None, model["report"]["selection"]
        response = client.post(f"/v1/learning/models/{model['object_id']}/evaluations", headers=headers,
                               json={"sample_ids": [registered[-1]["object_id"]]})
        assert response.status_code == 200, response.text
        evaluation = response.json()["evaluation"]
        assert evaluation["candidate"]["real_document_samples"] == 0
        assert evaluation["candidate"]["metrics"]["samples"] == 1
        assert not evaluation["automatic_activation"]
        assert model["production_eligible"] is False
        assert evaluation["decision"]["activation_supported"] is False
        if evaluation["candidate"]["metrics"]["character_edits"] > evaluation["baseline"]["metrics"]["character_edits"]:
            assert evaluation["decision"]["recommended_action"] == "reject"
    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "fresh"))
    with TestClient(create_app()) as fresh:
        stored = fresh.get(f"/v1/learning/{evaluation['object_id']}", headers=headers)
        assert stored.status_code == 200, stored.text
        assert stored.json()["object"] == evaluation
        assert fresh.get(f"/v1/learning/{model['object_id']}", headers={"X-API-Key": "other"}).status_code == 404
        review, event = final_review
        revoked = {"expected_revision": 1, "submission_id": "revoke-" + token,
                   "source_hashes": review["result"]["review"]["source_hashes"],
                   "decisions": [{"event_id": event["event_id"], "status": "needs_review", "reason": "Withdraw consent"}]}
        assert fresh.post(f"/v1/reviews/{review['review_id']}/decisions", headers=headers, json=revoked).status_code == 200
        assert fresh.get(f"/v1/learning/{evaluation['object_id']}", headers=headers).status_code == 409
    print(json.dumps({"backend": metadata_backend, "glyph_selection": model["report"]["selection"],
                      "baseline_cells": evaluation["baseline"]["metrics"], "candidate_cells": evaluation["candidate"]["metrics"]}))
