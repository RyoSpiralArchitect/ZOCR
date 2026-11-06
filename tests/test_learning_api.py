import json
import uuid
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from zocr.artifacts.evidence import archive_page_image
from zocr.artifacts.manifest import write_manifest
from zocr.service.metadata import LocalJobRepository
from zocr.service.storage import LocalJobStore


@pytest.fixture
def learning_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from zocr.service.app import create_app

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_KEYS_JSON", json.dumps({
        "alice": {"subject": "alice", "tenant_id": "tenant-a", "roles": ["admin"]},
        "operator": {"tenant_id": "tenant-a", "roles": ["operator"]},
        "other": {"tenant_id": "tenant-b", "roles": ["admin"]},
    }))
    store = LocalJobStore(tmp_path)
    repository = LocalJobRepository(job_store=store)
    auth = {"X-API-Key": "alice"}
    with TestClient(create_app()) as client:
        def make(scale, role="train", group=None):
            image = Image.new("L", (96, 24), 255)
            draw = ImageDraw.Draw(image)
            for index, char in enumerate("2358"):
                draw.text((index * 24 + 4, 3), char, fill=0, font=ImageFont.load_default())
            image = image.resize((96 * scale, 24 * scale), Image.Resampling.LANCZOS)
            jobs = []
            for text in ("0000", "0001"):
                job_id = uuid.uuid4().hex
                paths = store.job_paths(job_id)
                paths.input_dir.mkdir(parents=True)
                original = paths.input_dir / "source.png"
                image.save(original)
                (paths.out_dir / "rag").mkdir(parents=True)
                evidence = archive_page_image(image, paths.out_dir)
                (paths.out_dir / "rag" / "cells.jsonl").write_text(json.dumps({
                    "doc_id": "doc", "page": 1, "table_index": 0, "row": 1, "col": 0,
                    "header": "Amount", "text": text, "bbox": [0, 0, *image.size], "confidence": .4,
                    "image_path": str(Path(evidence).relative_to(paths.out_dir)),
                }) + "\n", encoding="utf-8")
                summary = {"inputs": [str(original)]}
                (paths.out_dir / "pipeline_summary.json").write_text(json.dumps(summary), encoding="utf-8")
                write_manifest(paths.out_dir, summary=summary, inputs=[str(original)], hash_inputs=True)
                repository.write_job(job_id, "tenant-a", {"status": "succeeded", "input": {"filename": "source.png"}})
                jobs.append(job_id)
            response = client.post("/v1/reviews", headers=auth, json={"job_a": jobs[0], "job_b": jobs[1]})
            assert response.status_code == 200, response.text
            review = response.json()["review"]
            event = review["result"]["events"][0]["event_id"]
            decision = {"expected_revision": 0, "submission_id": uuid.uuid4().hex,
                        "source_hashes": review["result"]["review"]["source_hashes"],
                        "decisions": [{"event_id": event, "status": "corrected", "reason": "Verified fixture pixels",
                                       "correction": {"side": "b", "text": "2358"}}]}
            response = client.post(f"/v1/reviews/{review['review_id']}/decisions", headers=auth, json=decision)
            assert response.status_code == 200, response.text
            request = {"expected_revision": 1, "event_id": event, "role": role, "consent": True,
                       "document_group_id": group or f"fixture-document-{scale}", "source_kind": "fixture",
                       "glyphs": [{"text": char, "bbox": [i * 24 * scale, 0, (i + 1) * 24 * scale, 24 * scale]}
                                  for i, char in enumerate("2358")] if role != "test" else []}
            return review, request, jobs
        yield client, make, auth, store


def register(client, make, auth, scale, role="train", group=None):
    review, request, jobs = make(scale, role, group)
    response = client.post(f"/v1/reviews/{review['review_id']}/learning-samples", headers=auth, json=request)
    assert response.status_code == 200, response.text
    return response.json()["sample"], review, request, jobs


def test_consent_is_actor_bound_and_tenant_scoped(learning_api):
    client, make, auth, store = learning_api
    sample, review, request, jobs = register(client, make, auth, 2)
    assert sample["consent_actor"]["subject"] == "alice"
    assert sample["automatic_activation"] is False
    url = f"/v1/learning/{sample['object_id']}"
    assert client.get(url, headers={"X-API-Key": "other"}).status_code == 404
    response = client.post(f"/v1/reviews/{review['review_id']}/learning-samples", headers={"X-API-Key": "operator"}, json=request)
    assert response.status_code == 403
    assert client.post(f"/v1/reviews/{review['review_id']}/learning-samples", headers=auth, json=dict(request, consent=False)).status_code == 422
    assert client.post(f"/v1/reviews/{review['review_id']}/learning-samples", headers=auth, json=request).json()["sample"] == sample
    assert (store.job_paths(jobs[1]).out_dir / "rag" / "cells.jsonl").read_text().find("0001") >= 0


def test_single_source_review_requires_human_correction_and_consent_before_learning(learning_api):
    client, make, auth, store = learning_api
    _, request, jobs = make(2)
    response = client.post("/v1/source-reviews", headers=auth, json={"job_id": jobs[1]})
    assert response.status_code == 200, response.text
    review = response.json()["review"]
    assert review["result"]["mode"] == "source_observations"
    event = review["result"]["events"][0]
    assert event["type"] == "source_observation"
    assert event["evidence"]["a"][0]["text"] == "0001"
    url = f"/v1/reviews/{review['review_id']}/learning-samples"
    request.update(expected_revision=0, event_id=event["event_id"])
    assert client.post(url, headers=auth, json=request).status_code == 422
    decision = {"expected_revision": 0, "submission_id": uuid.uuid4().hex,
                "source_hashes": review["result"]["review"]["source_hashes"],
                "decisions": [{"event_id": event["event_id"], "status": "corrected", "reason": "Human checked fixture pixels",
                               "correction": {"side": "a", "text": "2358"}}]}
    assert client.post(f"/v1/reviews/{review['review_id']}/decisions", headers=auth, json=decision).status_code == 200
    request["expected_revision"] = 1
    assert client.post(url, headers=auth, json=dict(request, consent=False)).status_code == 422
    sample = client.post(url, headers=auth, json=request)
    assert sample.status_code == 200, sample.text
    assert sample.json()["sample"]["expected"] == "2358"
    assert client.post("/v1/source-reviews", headers={"X-API-Key": "other"}, json={"job_id": jobs[1]}).status_code == 404
    assert "0001" in (store.job_paths(jobs[1]).out_dir / "rag/cells.jsonl").read_text()


def test_source_review_pixels_are_authenticated_hash_verified_and_bounded(learning_api):
    import io

    client, make, auth, store = learning_api
    _, _, jobs = make(2)
    shell = client.get("/review")
    assert shell.status_code == 200
    assert "connect-src 'self'" in shell.headers["content-security-policy"]
    review = client.post("/v1/source-reviews", headers=auth, json={"job_id": jobs[1]}).json()["review"]
    event = review["result"]["events"][0]
    url = f"/v1/reviews/{review['review_id']}/events/{event['event_id']}/image"
    assert client.get(url).status_code == 401
    assert client.get(url, headers={"X-API-Key": "other"}).status_code == 404
    response = client.get(url, headers=auth)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.size == (192, 48)
    assert client.get(url, headers=auth, params={"bbox": "[0,0,1000,1000]"}).status_code == 409
    evidence = next((store.job_paths(jobs[1]).out_dir / "evidence").glob("*.png"))
    evidence.write_bytes(evidence.read_bytes() + b"tampered")
    assert client.get(url, headers=auth).status_code == 409


def test_source_review_ui_manual_labels_and_consent_in_a_real_browser(learning_api):
    import os
    from urllib.parse import urlsplit

    if os.environ.get("ZOCR_TEST_BROWSER") != "1":
        pytest.skip("Set ZOCR_TEST_BROWSER=1 after installing Playwright Chromium")
    playwright = pytest.importorskip("playwright.sync_api")
    client, make, auth, store = learning_api
    _, _, jobs = make(2)
    errors = []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        page = browser.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))

        def dispatch(route):
            request = route.request
            parsed = urlsplit(request.url)
            response = client.request(request.method, parsed.path + ("?" + parsed.query if parsed.query else ""),
                                      headers=request.headers, content=request.post_data_buffer)
            headers = {key: value for key, value in response.headers.items() if key not in {"content-encoding", "content-length"}}
            route.fulfill(status=response.status_code, headers=headers, body=response.content)

        page.route("http://127.0.0.1:8765/**", dispatch)
        page.goto("http://127.0.0.1:8765/review")
        page.locator("#key").fill("alice")
        page.locator("#job").fill(jobs[1])
        page.locator("#create").click()
        playwright.expect(page.locator("#canvas")).to_have_attribute("width", "192")
        assert not page.locator("#confirmed").is_checked()
        assert not page.locator("#consent").is_checked()
        assert page.locator("#register").is_disabled()
        assert page.locator("#glyphs").inner_text() == "[]"
        for index, char in enumerate("2358"):
            page.locator("#canvas").scroll_into_view_if_needed()
            canvas = page.locator("#canvas").bounding_box()
            page.mouse.move(canvas["x"] + index * 96 + 2, canvas["y"] + 2)
            page.mouse.down()
            page.mouse.move(canvas["x"] + (index + 1) * 96 - 2, canvas["y"] + 94)
            page.mouse.up()
            page.locator("#glyph").fill(char)
            page.locator("#addGlyph").click()
        page.locator("#text").fill("2358")
        page.locator("#reason").fill("Browser fixture: explicit manual glyph boxes")
        page.locator("#confirmed").check()
        page.locator("#consent").check()
        page.locator("#group").fill("manual-browser-fixture")
        assert page.locator("#role").evaluate("el => [...el.options].map(option => option.value)") == ["", "train", "selection", "test"]
        page.locator("#role").select_option(value="train")
        page.locator("#kind").select_option("fixture")
        page.locator("#save").click()
        playwright.expect(page.locator("#register")).to_be_enabled()
        page.locator("#register").click()
        playwright.expect(page.locator("#status")).to_contain_text("登録済み")
        key = page.locator("#status").inner_text().splitlines()[-1]
        sample = client.get(f"/v1/learning/{key}", headers=auth).json()["object"]
        assert sample["expected"] == "2358"
        assert [glyph["text"] for glyph in sample["glyphs"]] == list("2358")
        assert sample["consent_actor"]["subject"] == "alice"
        assert not sample["automatic_activation"]
        assert page.evaluate("localStorage.length") == 0
        assert not errors
        browser.close()
    assert "0001" in (store.job_paths(jobs[1]).out_dir / "rag/cells.jsonl").read_text()


def test_document_bytes_and_families_cannot_cross_splits(learning_api):
    client, make, auth, _ = learning_api
    register(client, make, auth, 2, "train", "document-family")
    for scale, group in ((2, "renamed-document"), (3, "document-family"), (4, " document-family ")):
        review, request, _ = make(scale, "test", group)
        response = client.post(f"/v1/reviews/{review['review_id']}/learning-samples", headers=auth, json=request)
        assert response.status_code == 409, response.text


@pytest.mark.parametrize("part", ["input", "evidence"])
def test_tampered_original_or_pixels_cannot_supply_training_labels(learning_api, part):
    client, make, auth, store = learning_api
    review, request, jobs = make(2)
    paths = store.job_paths(jobs[1])
    target = paths.input_dir / "source.png" if part == "input" else next((paths.out_dir / "evidence").glob("*.png"))
    target.write_bytes(target.read_bytes() + b"tampered")
    response = client.post(f"/v1/reviews/{review['review_id']}/learning-samples", headers=auth, json=request)
    assert response.status_code == 409, response.text


def test_changed_review_revokes_existing_learning_sample(learning_api):
    client, make, auth, _ = learning_api
    sample, review, request, _ = register(client, make, auth, 2)
    revoke = {"expected_revision": 1, "submission_id": "withdraw-consent",
              "source_hashes": review["result"]["review"]["source_hashes"],
              "decisions": [{"event_id": request["event_id"], "status": "needs_review", "reason": "Withdraw learning consent"}]}
    assert client.post(f"/v1/reviews/{review['review_id']}/decisions", headers=auth, json=revoke).status_code == 200
    assert client.get(f"/v1/learning/{sample['object_id']}", headers=auth).status_code == 409


def test_training_creates_immutable_candidate_and_excludes_final_test(learning_api):
    client, make, auth, _ = learning_api
    samples = [register(client, make, auth, scale, role)[0] for scale, role in ((2, "train"), (3, "selection"), (4, "selection"))]
    final = register(client, make, auth, 5, "test")[0]
    ids = [sample["object_id"] for sample in samples]
    response = client.post("/v1/learning/models", headers=auth, json={"sample_ids": ids, "allowed_chars": "2358"})
    assert response.status_code == 200, response.text
    model = response.json()["candidate"]
    assert model["report"]["baseline"]["samples"] == 8
    assert model["automatic_activation"] is False
    assert client.get(f"/v1/learning/{model['object_id']}", headers={"X-API-Key": "other"}).status_code == 404
    rejected = client.post("/v1/learning/models", headers=auth, json={"sample_ids": ids + [final["object_id"]], "allowed_chars": "2358"})
    assert rejected.status_code == 422
    if model["memory"] is not None:
        response = client.post(f"/v1/learning/models/{model['object_id']}/evaluations", headers=auth,
                               json={"sample_ids": [final["object_id"]]})
        assert response.status_code == 200, response.text
        evaluation = response.json()["evaluation"]
        assert evaluation["baseline"]["metrics"]["samples"] == 1
        assert evaluation["candidate"]["real_document_samples"] == 0
        assert not evaluation["automatic_activation"]
        stored = client.get(f"/v1/learning/{evaluation['object_id']}", headers=auth)
        assert stored.status_code == 200
        assert stored.json()["object"] == evaluation


@pytest.mark.parametrize("bbox, expected", [([2, 2, 190, 46], 200), ([0, 0, 1000, 1000], 422),
                                            ([0.1, 0.1, 0.2, 0.2], 422)])
def test_explicit_region_correction_preserves_original_source(learning_api, bbox, expected):
    client, make, auth, store = learning_api
    review, request, jobs = make(2, "test")
    source = store.job_paths(jobs[1]).out_dir / "rag" / "cells.jsonl"
    before = source.read_bytes()
    response = client.post(f"/v1/reviews/{review['review_id']}/decisions", headers=auth, json={
        "expected_revision": 1, "submission_id": "region-correction",
        "source_hashes": review["result"]["review"]["source_hashes"],
        "decisions": [{"event_id": request["event_id"], "status": "corrected", "reason": "Checked segmentation region",
                       "correction": {"side": "b", "text": "2358", "bbox": bbox}}],
    })
    assert response.status_code == 200, response.text
    assert response.json()["review"]["decisions"][request["event_id"]]["correction"]["bbox"] == bbox
    response = client.post(f"/v1/reviews/{review['review_id']}/learning-samples", headers=auth,
                           json=dict(request, expected_revision=2))
    assert response.status_code == expected, response.text
    assert source.read_bytes() == before
    if expected == 200:
        assert response.json()["sample"]["source_cell_bbox"] == [0, 0, 192, 48]
        assert response.json()["sample"]["annotation_bbox"] == bbox
