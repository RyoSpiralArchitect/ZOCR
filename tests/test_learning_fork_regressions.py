import json

import pytest

from test_toy_learning import dataset, save_rows
from test_cell_evaluation import cell_dataset
from zocr.consensus.learning import load_glyph_dataset
from zocr.consensus.cell_evaluation import load_cell_dataset


def test_repeated_training_glyphs_are_allowed_but_cross_split_reuse_is_not(tmp_path):
    path, rows = dataset(tmp_path)
    rows.append(dict(rows[0], id="another-training-occurrence"))
    save_rows(path, rows)
    assert len(load_glyph_dataset(path, "2358")["samples"]) == 13
    rows[-1]["split"] = "holdout"
    rows[-1]["group_id"] = "different-document"
    save_rows(path, rows)
    with pytest.raises(ValueError, match="contaminate"):
        load_glyph_dataset(path, "2358")


def test_repeated_test_cells_keep_occurrences_but_not_conflicting_labels(tmp_path):
    path, row = cell_dataset(tmp_path)
    other = dict(row, id="second-occurrence")
    path.write_text(json.dumps(row) + "\n" + json.dumps(other), encoding="utf-8")
    assert len(load_cell_dataset(path)["samples"]) == 2
    other["text"] = "wrong-label"
    path.write_text(json.dumps(row) + "\n" + json.dumps(other), encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting"):
        load_cell_dataset(path)


def test_explicit_readonly_unbound_key_does_not_become_admin(monkeypatch):
    from zocr.service.auth import AuthBackend, AuthError

    monkeypatch.setenv("ZOCR_API_KEYS_JSON", json.dumps({"readonly": {"scopes": ["zocr.jobs.read"]}}))
    backend = AuthBackend(multi_tenant_enabled=True, default_tenant="default")
    principal = backend.authenticate_principal(x_api_key="readonly", authorization=None)
    assert not principal.is_admin
    assert not backend.can_override_tenant(principal)
    with pytest.raises(AuthError):
        backend.require_scopes(principal, "zocr.reviews.approve")


def test_tenant_admin_cannot_use_platform_governance_routes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from zocr.service.app import create_app

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_KEYS_JSON", json.dumps({
        "tenant": {"tenant_id": "tenant-a", "roles": ["admin"]},
        "platform": {"admin": True},
    }))
    with TestClient(create_app()) as client:
        assert client.get("/v1/admin/tenant-policies", headers={"X-API-Key": "tenant"}).status_code == 403
        assert client.put("/v1/admin/tenant-policies/tenant-b", headers={"X-API-Key": "tenant"}, json={"max_active_jobs": 0}).status_code == 403
        assert client.get("/v1/admin/tenant-policies", headers={"X-API-Key": "platform"}).status_code == 200


@pytest.mark.parametrize("date", ["soon", {}, True, float("nan"), float("inf")])
def test_invalid_signed_jwt_dates_are_authentication_failures(monkeypatch, date):
    from test_api_jobs_smoke import _jwt_hs256
    from zocr.service.auth import AuthBackend, AuthError

    monkeypatch.setenv("ZOCR_API_AUTH_MODE", "jwt")
    monkeypatch.setenv("ZOCR_API_JWT_SECRET", "synthetic-test-secret")
    backend = AuthBackend(multi_tenant_enabled=True, default_tenant="default")
    token = _jwt_hs256({"sub": "fixture", "tenant_id": "tenant-a", "exp": date}, secret="synthetic-test-secret")
    with pytest.raises(AuthError) as error:
        backend.authenticate_principal(x_api_key=None, authorization="Bearer " + token)
    assert error.value.status_code == 401


def test_evaluation_shares_training_concurrency_lock(tmp_path):
    from zocr.service.execution_lock import execution_lock
    from zocr.service.learning import LearningRegistry
    from zocr.service.metadata import LocalJobRepository
    from zocr.service.review_ledger import ReviewConflict
    from zocr.service.storage import LocalJobStore

    store = LocalJobStore(tmp_path)
    repository = LocalJobRepository(job_store=store)
    registry = LearningRegistry(store, repository, None, None, None)
    with execution_lock(store, repository, "learning-trial", "tenant-a") as acquired:
        assert acquired
        with pytest.raises(ReviewConflict, match="trial is running"):
            registry.evaluate("tenant-a", "unused", [], {})
