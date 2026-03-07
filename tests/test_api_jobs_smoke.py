from __future__ import annotations

import json
import base64
import hashlib
import hmac
import sys
import time
import types
from pathlib import Path

import pytest


def _jwt_hs256(payload: dict[str, object], *, secret: str, kid: str | None = None) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    if kid:
        header["kid"] = kid
    header_b64 = base64.urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode("utf-8")).decode("ascii").rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii").rstrip("=")
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode("ascii").rstrip("=")
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def _install_fake_psycopg(monkeypatch, *, tenant_policy_rows: dict[str, tuple] | None = None) -> dict[str, object]:
    tenant_policy_rows = dict(tenant_policy_rows or {})
    state: dict[str, object] = {"events": [], "sql": [], "change_requests": [], "tenant_policies": tenant_policy_rows}

    class FakeJsonb:
        def __init__(self, value: object):
            self.value = value

    class FakeCursor:
        def __init__(self) -> None:
            self._row = None
            self._rows = []
            self.rowcount = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def execute(self, sql: str, params=None) -> None:
            sql_text = " ".join(str(sql).split()).lower()
            state["sql"].append((sql_text, params))
            self._row = None
            self._rows = []
            self.rowcount = 0
            if "insert into zocr_audit_events" in sql_text:
                payload = params[5]
                if hasattr(payload, "value"):
                    payload = payload.value
                state["events"].append(
                    {
                        "occurred_at": params[0],
                        "event": params[1],
                        "tenant_id": params[2],
                        "subject": params[3],
                        "request_id": params[4],
                        "payload": payload,
                    }
                )
                self.rowcount = 1
            elif "insert into zocr_tenant_change_requests" in sql_text:
                approval_records = params[6]
                payload = params[12]
                if hasattr(approval_records, "value"):
                    approval_records = approval_records.value
                if hasattr(payload, "value"):
                    payload = payload.value
                state["change_requests"].append(
                    {
                        "request_id": params[0],
                        "target_type": params[1],
                        "target_id": params[2],
                        "action": params[3],
                        "status": params[4],
                        "approvals_required": params[5],
                        "approval_records": approval_records,
                        "requested_by": params[7],
                        "requested_at": params[8],
                        "reviewed_by": params[9],
                        "reviewed_at": params[10],
                        "review_reason": params[11],
                        "payload": payload,
                    }
                )
                self.rowcount = 1
            elif "from zocr_tenant_change_requests where request_id = %s" in sql_text:
                request_id = params[0]
                item = next(
                    (
                        request
                        for request in state["change_requests"]
                        if request["request_id"] == request_id
                    ),
                    None,
                )
                if item is not None:
                    self._row = (
                        item["request_id"],
                        item["target_type"],
                        item["target_id"],
                        item["action"],
                        item["status"],
                        item["approvals_required"],
                        item["approval_records"],
                        item["requested_by"],
                        item["requested_at"],
                        item["reviewed_by"],
                        item["reviewed_at"],
                        item["review_reason"],
                        item["payload"],
                    )
            elif "from zocr_tenant_change_requests" in sql_text:
                params_list = list(params or [])
                idx = 0
                status_filter = None
                target_type_filter = None
                if "status = %s" in sql_text:
                    status_filter = params_list[idx]
                    idx += 1
                if "target_type = %s" in sql_text:
                    target_type_filter = params_list[idx]
                    idx += 1
                limit = int(params_list[idx]) if idx < len(params_list) else len(state["change_requests"])
                for item in reversed(state["change_requests"]):
                    if status_filter and item["status"] != status_filter:
                        continue
                    if target_type_filter and item["target_type"] != target_type_filter:
                        continue
                    self._rows.append(
                        (
                            item["request_id"],
                            item["target_type"],
                            item["target_id"],
                            item["action"],
                            item["status"],
                            item["approvals_required"],
                            item["approval_records"],
                            item["requested_by"],
                            item["requested_at"],
                            item["reviewed_by"],
                            item["reviewed_at"],
                            item["review_reason"],
                            item["payload"],
                        )
                    )
                    if len(self._rows) >= limit:
                        break
                self.rowcount = len(self._rows)
            elif "update zocr_tenant_change_requests set status = %s, approval_records = %s, reviewed_by = %s, reviewed_at = %s, review_reason = %s where request_id = %s" in sql_text:
                approval_records = params[1]
                if hasattr(approval_records, "value"):
                    approval_records = approval_records.value
                request_id = params[5]
                for item in state["change_requests"]:
                    if item["request_id"] != request_id:
                        continue
                    item["status"] = params[0]
                    item["approval_records"] = approval_records
                    item["reviewed_by"] = params[2]
                    item["reviewed_at"] = params[3]
                    item["review_reason"] = params[4]
                    self.rowcount = 1
                    break
            elif "update zocr_tenant_change_requests set status = %s, reviewed_by = %s, reviewed_at = now(), review_reason = %s, approval_records = %s where request_id = %s" in sql_text:
                approval_records = params[3]
                if hasattr(approval_records, "value"):
                    approval_records = approval_records.value
                request_id = params[4]
                for item in state["change_requests"]:
                    if item["request_id"] != request_id:
                        continue
                    item["status"] = params[0]
                    item["reviewed_by"] = params[1]
                    item["reviewed_at"] = "2026-01-01T00:00:00+00:00"
                    item["review_reason"] = params[2]
                    item["approval_records"] = approval_records
                    self.rowcount = 1
                    break
            elif "insert into zocr_tenant_policies" in sql_text:
                tenant_policy_rows[str(params[0])] = (params[1], params[2], params[3], params[4])
                self.rowcount = 1
            elif "delete from zocr_tenant_policies where tenant_id = %s" in sql_text:
                self.rowcount = 1 if tenant_policy_rows.pop(str(params[0]), None) is not None else 0
            elif "from zocr_audit_events" in sql_text:
                params_list = list(params or [])
                idx = 0
                tenant_filter = None
                event_filter = None
                subject_filter = None
                request_filter = None
                since_filter = None
                until_filter = None
                contains_filter = None
                if "tenant_id = %s" in sql_text:
                    tenant_filter = params_list[idx]
                    idx += 1
                if "event = %s" in sql_text:
                    event_filter = params_list[idx]
                    idx += 1
                if "subject = %s" in sql_text:
                    subject_filter = params_list[idx]
                    idx += 1
                if "request_id = %s" in sql_text:
                    request_filter = params_list[idx]
                    idx += 1
                if "occurred_at >= %s" in sql_text:
                    since_filter = params_list[idx]
                    idx += 1
                if "occurred_at <= %s" in sql_text:
                    until_filter = params_list[idx]
                    idx += 1
                if "payload::text ilike %s" in sql_text:
                    contains_filter = str(params_list[idx]).strip("%").lower()
                    idx += 1
                limit = int(params_list[idx]) if idx < len(params_list) else len(state["events"])
                for item in reversed(state["events"]):
                    payload = item["payload"]
                    payload_json = json.dumps(payload, ensure_ascii=False).lower()
                    if tenant_filter and item["tenant_id"] != tenant_filter:
                        continue
                    if event_filter and item["event"] != event_filter:
                        continue
                    if subject_filter and item["subject"] != subject_filter:
                        continue
                    if request_filter and item["request_id"] != request_filter:
                        continue
                    if since_filter and str(item["occurred_at"]) < str(since_filter):
                        continue
                    if until_filter and str(item["occurred_at"]) > str(until_filter):
                        continue
                    if contains_filter and contains_filter not in payload_json:
                        continue
                    self._rows.append(
                        (
                            item["occurred_at"],
                            item["event"],
                            item["tenant_id"],
                            item["subject"],
                            item["request_id"],
                            payload,
                        )
                    )
                    if len(self._rows) >= limit:
                        break
                self.rowcount = len(self._rows)
            elif "from zocr_tenant_policies tp" in sql_text:
                tenant_id = params[0]
                self._row = tenant_policy_rows.get(str(tenant_id))

        def fetchone(self):
            return self._row

        def fetchall(self):
            return list(self._rows)

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def cursor(self):
            return FakeCursor()

        def commit(self) -> None:
            return

    fake_psycopg = types.ModuleType("psycopg")
    fake_types = types.ModuleType("psycopg.types")
    fake_types_json = types.ModuleType("psycopg.types.json")
    fake_psycopg.connect = lambda _dsn: FakeConnection()
    fake_types_json.Jsonb = FakeJsonb
    fake_types.json = fake_types_json
    fake_psycopg.types = fake_types
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.types", fake_types)
    monkeypatch.setitem(sys.modules, "psycopg.types.json", fake_types_json)
    return state


def test_jobs_lifecycle_smoke(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    import zocr.orchestrator.zocr_pipeline as pipeline

    def fake_run_full_pipeline(*, inputs, outdir: str, **_kwargs):
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "doc.zocr.json").write_text(
            json.dumps({"doc_id": "doc", "pages": [], "metrics": {}}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary = {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "inputs": inputs,
        }
        (out / "pipeline_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary

    monkeypatch.setattr(pipeline, "run_full_pipeline", fake_run_full_pipeline, raising=True)
    storage_dir = tmp_path / "store"
    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_CREATE", "0")

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["storage"]["dir"] == str(storage_dir)

    resp = client.post(
        "/v1/jobs?domain=invoice",
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert resp.status_code == 200, resp.text
    job = resp.json()["job"]
    job_id = job["id"]

    final = None
    for _ in range(100):
        polled = client.get(f"/v1/jobs/{job_id}")
        assert polled.status_code == 200
        status = polled.json()["job"]["status"]
        if status in {"succeeded", "failed"}:
            final = polled.json()["job"]
            break
        time.sleep(0.02)
    assert final is not None
    assert final["status"] == "succeeded"

    summary_resp = client.get(f"/v1/jobs/{job_id}/artifacts/pipeline_summary")
    assert summary_resp.status_code == 200
    summary_payload = summary_resp.json()
    assert summary_payload["inputs"]

    zip_resp = client.get(f"/v1/jobs/{job_id}/artifacts.zip")
    assert zip_resp.status_code == 200
    assert zip_resp.headers.get("content-type") == "application/zip"


def test_jobs_queue_dispatch_smoke(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    class FakeQueue:
        def __init__(self) -> None:
            self.groups: list[str] = []
            self.enqueued: list[tuple[str, str]] = []

        def ensure_consumer_group(self, group: str) -> None:
            self.groups.append(group)

        def enqueue(self, job_id: str, tenant_id: str) -> str:
            self.enqueued.append((job_id, tenant_id))
            return "1-0"

    fake_queue = FakeQueue()
    storage_dir = tmp_path / "store"
    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("ZOCR_API_QUEUE_BACKEND", "redis")
    monkeypatch.setenv("ZOCR_API_REDIS_STREAM", "zocr_jobs_test")
    monkeypatch.setenv("ZOCR_API_REDIS_GROUP", "zocr_workers_test")
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_CREATE", "0")

    import zocr.service.queue as queue_module

    monkeypatch.setattr(queue_module, "build_redis_queue", lambda: fake_queue, raising=True)

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["queue"]["backend"] == "redis"
    assert health.json()["queue"]["stream"] == "zocr_jobs_test"
    assert health.json()["queue"]["group"] == "zocr_workers_test"
    assert fake_queue.groups == ["zocr_workers_test"]

    resp = client.post(
        "/v1/jobs?domain=invoice",
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dispatch"]["backend"] == "redis"
    assert body["dispatch"]["message_id"] == "1-0"

    job_id = body["job"]["id"]
    assert body["job"]["tenant_id"] == "default"
    assert fake_queue.enqueued == [(job_id, "default")]

    polled = client.get(f"/v1/jobs/{job_id}")
    assert polled.status_code == 200
    assert polled.json()["job"]["status"] == "queued"


def test_jobs_tenant_isolation_smoke(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    import zocr.orchestrator.zocr_pipeline as pipeline

    def fake_run_full_pipeline(*, inputs, outdir: str, **_kwargs):
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "pipeline_summary.json").write_text(
            json.dumps({"generated_at": "2026-01-01T00:00:00+00:00", "inputs": inputs}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {"inputs": inputs}

    monkeypatch.setattr(pipeline, "run_full_pipeline", fake_run_full_pipeline, raising=True)
    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_MULTI_TENANT_ENABLED", "1")
    monkeypatch.delenv("ZOCR_API_DEFAULT_TENANT", raising=False)
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_CREATE", "0")

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    missing_tenant = client.post(
        "/v1/jobs?domain=invoice",
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert missing_tenant.status_code == 400

    resp = client.post(
        "/v1/jobs?domain=invoice",
        headers={"X-Tenant-ID": "tenant-a"},
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job"]["id"]

    ok = client.get(f"/v1/jobs/{job_id}", headers={"X-Tenant-ID": "tenant-a"})
    assert ok.status_code == 200
    assert ok.json()["job"]["tenant_id"] == "tenant-a"

    hidden = client.get(f"/v1/jobs/{job_id}", headers={"X-Tenant-ID": "tenant-b"})
    assert hidden.status_code == 404

    listed = client.get("/v1/jobs", headers={"X-Tenant-ID": "tenant-b"})
    assert listed.status_code == 200
    assert listed.json()["jobs"] == []


def test_jobs_api_key_tenant_binding(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    class FakeQueue:
        def ensure_consumer_group(self, group: str) -> None:
            return

        def enqueue(self, job_id: str, tenant_id: str) -> str:
            return f"{tenant_id}:{job_id}"

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_QUEUE_BACKEND", "redis")
    monkeypatch.setenv(
        "ZOCR_API_KEYS_JSON",
        json.dumps(
            {
                "tenant-a-key": {"tenant_id": "tenant-a", "subject": "svc-a"},
                "tenant-b-key": {"tenant_id": "tenant-b", "subject": "svc-b"},
            }
        ),
    )
    monkeypatch.delenv("ZOCR_API_KEY", raising=False)
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_CREATE", "0")

    import zocr.service.queue as queue_module

    monkeypatch.setattr(queue_module, "build_redis_queue", lambda: FakeQueue(), raising=True)

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    resp = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "tenant-a-key"},
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job"]["id"]
    assert resp.json()["job"]["tenant_id"] == "tenant-a"

    same = client.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": "tenant-a-key"})
    assert same.status_code == 200

    mismatch = client.get(
        f"/v1/jobs/{job_id}",
        headers={"X-API-Key": "tenant-a-key", "X-Tenant-ID": "tenant-b"},
    )
    assert mismatch.status_code == 403

    hidden = client.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": "tenant-b-key"})
    assert hidden.status_code == 404


def test_jobs_jwt_tenant_binding(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    class FakeQueue:
        def ensure_consumer_group(self, group: str) -> None:
            return

        def enqueue(self, job_id: str, tenant_id: str) -> str:
            return f"{tenant_id}:{job_id}"

    secret = "jwt-secret"
    now = int(time.time())
    tenant_a_token = _jwt_hs256(
        {"sub": "svc-a", "tenant_id": "tenant-a", "exp": now + 3600},
        secret=secret,
    )
    tenant_b_token = _jwt_hs256(
        {"sub": "svc-b", "tenant_id": "tenant-b", "exp": now + 3600},
        secret=secret,
    )

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_QUEUE_BACKEND", "redis")
    monkeypatch.setenv("ZOCR_API_AUTH_MODE", "jwt")
    monkeypatch.setenv("ZOCR_API_JWT_SECRET", secret)
    monkeypatch.delenv("ZOCR_API_KEY", raising=False)
    monkeypatch.delenv("ZOCR_API_KEYS_JSON", raising=False)
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_CREATE", "0")

    import zocr.service.queue as queue_module

    monkeypatch.setattr(queue_module, "build_redis_queue", lambda: FakeQueue(), raising=True)

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    resp = client.post(
        "/v1/jobs",
        headers={"Authorization": f"Bearer {tenant_a_token}"},
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job"]["id"]
    assert resp.json()["job"]["tenant_id"] == "tenant-a"

    same = client.get(
        f"/v1/jobs/{job_id}",
        headers={"Authorization": f"Bearer {tenant_a_token}"},
    )
    assert same.status_code == 200

    mismatch = client.get(
        f"/v1/jobs/{job_id}",
        headers={
            "Authorization": f"Bearer {tenant_a_token}",
            "X-Tenant-ID": "tenant-b",
        },
    )
    assert mismatch.status_code == 403

    hidden = client.get(
        f"/v1/jobs/{job_id}",
        headers={"Authorization": f"Bearer {tenant_b_token}"},
    )
    assert hidden.status_code == 404


def test_jobs_role_authorization_api_key(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    class FakeQueue:
        def ensure_consumer_group(self, group: str) -> None:
            return

        def enqueue(self, job_id: str, tenant_id: str) -> str:
            return f"{tenant_id}:{job_id}"

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_QUEUE_BACKEND", "redis")
    monkeypatch.setenv(
        "ZOCR_API_KEYS_JSON",
        json.dumps(
            {
                "viewer-key": {"tenant_id": "tenant-a", "roles": ["viewer"]},
                "operator-key": {"tenant_id": "tenant-a", "roles": ["operator"]},
                "admin-key": {"tenant_id": "*", "roles": ["admin"], "admin": True},
            }
        ),
    )
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_CREATE", "0")

    import zocr.service.queue as queue_module

    monkeypatch.setattr(queue_module, "build_redis_queue", lambda: FakeQueue(), raising=True)

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    viewer_create = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "viewer-key"},
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert viewer_create.status_code == 403

    resp = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "operator-key"},
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job"]["id"]

    viewer_read = client.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": "viewer-key"})
    assert viewer_read.status_code == 200

    viewer_delete = client.delete(f"/v1/jobs/{job_id}", headers={"X-API-Key": "viewer-key"})
    assert viewer_delete.status_code == 403

    admin_delete = client.delete(
        f"/v1/jobs/{job_id}",
        headers={"X-API-Key": "admin-key", "X-Tenant-ID": "tenant-a"},
    )
    assert admin_delete.status_code == 200


def test_metrics_scope_authorization_jwt(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    secret = "jwt-secret"
    now = int(time.time())
    metrics_token = _jwt_hs256(
        {"sub": "viewer", "tenant_id": "tenant-a", "scope": "zocr.metrics.read", "exp": now + 3600},
        secret=secret,
    )
    limited_token = _jwt_hs256(
        {"sub": "limited", "tenant_id": "tenant-a", "scope": "zocr.jobs.read", "exp": now + 3600},
        secret=secret,
    )

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_AUTH_MODE", "jwt")
    monkeypatch.setenv("ZOCR_API_JWT_SECRET", secret)

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    allowed = client.get("/metrics", headers={"Authorization": f"Bearer {metrics_token}"})
    assert allowed.status_code == 200

    denied = client.get("/metrics", headers={"Authorization": f"Bearer {limited_token}"})
    assert denied.status_code == 403


def test_metrics_oidc_discovery_jwks(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    secret = "oidc-secret"
    kid = "oidc-key"
    issuer = "https://issuer.example"
    jwks_path = tmp_path / "jwks.json"
    discovery_path = tmp_path / "openid-configuration.json"
    jwks_path.write_text(
        json.dumps(
            {
                "keys": [
                    {
                        "kty": "oct",
                        "kid": kid,
                        "alg": "HS256",
                        "use": "sig",
                        "k": base64.urlsafe_b64encode(secret.encode("utf-8")).decode("ascii").rstrip("="),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    discovery_path.write_text(
        json.dumps({"issuer": issuer, "jwks_uri": jwks_path.as_uri()}),
        encoding="utf-8",
    )

    token = _jwt_hs256(
        {
            "sub": "oidc-user",
            "tenant_id": "tenant-a",
            "scope": "zocr.metrics.read",
            "iss": issuer,
            "exp": int(time.time()) + 3600,
        },
        secret=secret,
        kid=kid,
    )

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_AUTH_MODE", "jwt")
    monkeypatch.setenv("ZOCR_API_OIDC_DISCOVERY_URL", discovery_path.as_uri())
    monkeypatch.delenv("ZOCR_API_JWT_SECRET", raising=False)

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.get("/metrics", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200


def test_strict_authz_denies_unscoped_api_key(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    class FakeQueue:
        def ensure_consumer_group(self, group: str) -> None:
            return

        def enqueue(self, job_id: str, tenant_id: str) -> str:
            return f"{tenant_id}:{job_id}"

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_QUEUE_BACKEND", "redis")
    monkeypatch.setenv("ZOCR_API_AUTHZ_STRICT", "1")
    monkeypatch.setenv(
        "ZOCR_API_KEYS_JSON",
        json.dumps({"plain-key": {"tenant_id": "tenant-a", "subject": "plain"}}),
    )
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_CREATE", "0")

    import zocr.service.queue as queue_module

    monkeypatch.setattr(queue_module, "build_redis_queue", lambda: FakeQueue(), raising=True)

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    create_resp = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "plain-key"},
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert create_resp.status_code == 403

    metrics_resp = client.get("/metrics", headers={"X-API-Key": "plain-key"})
    assert metrics_resp.status_code == 403


def test_strict_authz_allows_legacy_key_role(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    class FakeQueue:
        def ensure_consumer_group(self, group: str) -> None:
            return

        def enqueue(self, job_id: str, tenant_id: str) -> str:
            return f"{tenant_id}:{job_id}"

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_QUEUE_BACKEND", "redis")
    monkeypatch.setenv("ZOCR_API_AUTHZ_STRICT", "1")
    monkeypatch.setenv("ZOCR_API_KEY", "legacy-key")
    monkeypatch.setenv("ZOCR_API_KEY_TENANT", "tenant-a")
    monkeypatch.setenv("ZOCR_API_KEY_ROLES", "operator")
    monkeypatch.delenv("ZOCR_API_KEYS_JSON", raising=False)
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_CREATE", "0")

    import zocr.service.queue as queue_module

    monkeypatch.setattr(queue_module, "build_redis_queue", lambda: FakeQueue(), raising=True)

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    metrics_resp = client.get("/metrics", headers={"X-API-Key": "legacy-key"})
    assert metrics_resp.status_code == 200

    create_resp = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "legacy-key"},
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert create_resp.status_code == 200, create_resp.text


def test_strict_authz_denies_unscoped_jwt(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    class FakeQueue:
        def ensure_consumer_group(self, group: str) -> None:
            return

        def enqueue(self, job_id: str, tenant_id: str) -> str:
            return f"{tenant_id}:{job_id}"

    secret = "jwt-secret"
    now = int(time.time())
    plain_token = _jwt_hs256(
        {"sub": "plain", "tenant_id": "tenant-a", "exp": now + 3600},
        secret=secret,
    )

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_QUEUE_BACKEND", "redis")
    monkeypatch.setenv("ZOCR_API_AUTH_MODE", "jwt")
    monkeypatch.setenv("ZOCR_API_AUTHZ_STRICT", "1")
    monkeypatch.setenv("ZOCR_API_JWT_SECRET", secret)
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_CREATE", "0")

    import zocr.service.queue as queue_module

    monkeypatch.setattr(queue_module, "build_redis_queue", lambda: FakeQueue(), raising=True)

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    create_resp = client.post(
        "/v1/jobs",
        headers={"Authorization": f"Bearer {plain_token}"},
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert create_resp.status_code == 403


def test_metrics_rate_limit_api_key(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_RATE_LIMIT_PER_MIN", "1")
    monkeypatch.setenv(
        "ZOCR_API_KEYS_JSON",
        json.dumps({"viewer-key": {"tenant_id": "tenant-a", "roles": ["viewer"], "subject": "viewer-a"}}),
    )

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    allowed = client.get("/metrics", headers={"X-API-Key": "viewer-key"})
    assert allowed.status_code == 200
    assert allowed.headers.get("x-ratelimit-limit") == "1"
    assert allowed.headers.get("x-ratelimit-remaining") == "0"

    denied = client.get("/metrics", headers={"X-API-Key": "viewer-key"})
    assert denied.status_code == 429
    assert 1 <= int(denied.headers["retry-after"]) <= 60
    assert denied.headers.get("x-ratelimit-limit") == "1"


def test_jobs_tenant_active_quota(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    class FakeQueue:
        def ensure_consumer_group(self, group: str) -> None:
            return

        def enqueue(self, job_id: str, tenant_id: str) -> str:
            return f"{tenant_id}:{job_id}"

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_QUEUE_BACKEND", "redis")
    monkeypatch.setenv("ZOCR_API_TENANT_MAX_ACTIVE_JOBS", "1")
    monkeypatch.setenv(
        "ZOCR_API_KEYS_JSON",
        json.dumps({"operator-key": {"tenant_id": "tenant-a", "roles": ["operator"], "subject": "operator-a"}}),
    )
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_CREATE", "0")

    import zocr.service.queue as queue_module

    monkeypatch.setattr(queue_module, "build_redis_queue", lambda: FakeQueue(), raising=True)

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["policies"]["tenant_max_active_jobs"] == 1

    first = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "operator-key"},
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert first.status_code == 200, first.text

    second = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "operator-key"},
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert second.status_code == 429
    assert "active job quota exceeded" in second.text


def test_metrics_rate_limit_redis_shared_across_apps(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    class FakeRedisClient:
        def __init__(self) -> None:
            self.values: dict[str, int] = {}
            self.expiry: dict[str, float] = {}

        def _cleanup(self, key: str) -> None:
            expires_at = self.expiry.get(key)
            if expires_at is not None and expires_at <= time.time():
                self.expiry.pop(key, None)
                self.values.pop(key, None)

        def incr(self, key: str) -> int:
            self._cleanup(key)
            value = int(self.values.get(key, 0)) + 1
            self.values[key] = value
            return value

        def expire(self, key: str, ttl: int) -> bool:
            self.expiry[key] = time.time() + int(ttl)
            return True

    shared_client = FakeRedisClient()
    fake_redis = types.SimpleNamespace(
        Redis=types.SimpleNamespace(from_url=lambda _url: shared_client),
    )
    monkeypatch.setitem(sys.modules, "redis", fake_redis)
    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_RATE_LIMIT_PER_MIN", "1")
    monkeypatch.setenv("ZOCR_API_RATE_LIMIT_BACKEND", "redis")
    monkeypatch.setenv("ZOCR_API_RATE_LIMIT_REDIS_PREFIX", "zocr_test_rate")
    monkeypatch.setenv(
        "ZOCR_API_KEYS_JSON",
        json.dumps({"viewer-key": {"tenant_id": "tenant-a", "roles": ["viewer"], "subject": "viewer-a"}}),
    )

    from zocr.service.app import create_app

    app_a = create_app()
    app_b = create_app()

    from fastapi.testclient import TestClient

    client_a = TestClient(app_a)
    client_b = TestClient(app_b)

    health = client_a.get("/healthz")
    assert health.status_code == 200
    assert health.json()["policies"]["rate_limit_backend"] == "redis"

    first = client_a.get("/metrics", headers={"X-API-Key": "viewer-key"})
    assert first.status_code == 200

    second = client_b.get("/metrics", headers={"X-API-Key": "viewer-key"})
    assert second.status_code == 429
    assert second.headers.get("x-ratelimit-limit") == "1"
    assert second.headers.get("x-ratelimit-remaining") == "0"


def test_tenant_plan_policy_json(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    class FakeQueue:
        def ensure_consumer_group(self, group: str) -> None:
            return

        def enqueue(self, job_id: str, tenant_id: str) -> str:
            return f"{tenant_id}:{job_id}"

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_QUEUE_BACKEND", "redis")
    monkeypatch.setenv(
        "ZOCR_API_KEYS_JSON",
        json.dumps(
            {
                "viewer-read-key": {"tenant_id": "tenant-read", "roles": ["viewer"], "subject": "viewer-read"},
                "operator-a-key": {"tenant_id": "tenant-a", "roles": ["operator"], "subject": "operator-a"},
                "viewer-b-key": {"tenant_id": "tenant-b", "roles": ["viewer"], "subject": "viewer-b"},
            }
        ),
    )
    monkeypatch.setenv(
        "ZOCR_API_TENANT_PLANS_JSON",
        json.dumps(
            {
                "limited-read": {"rate_limit_per_min": 1},
                "starter": {"max_active_jobs": 1},
                "pro": {"rate_limit_per_min": 5, "max_active_jobs": 3},
            }
        ),
    )
    monkeypatch.setenv(
        "ZOCR_API_TENANT_POLICIES_JSON",
        json.dumps(
            {
                "tenant-read": {"plan": "limited-read"},
                "tenant-a": {"plan": "starter"},
                "tenant-b": {"plan": "pro"},
            }
        ),
    )
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_CREATE", "0")

    import zocr.service.queue as queue_module

    monkeypatch.setattr(queue_module, "build_redis_queue", lambda: FakeQueue(), raising=True)

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["policies"]["tenant_plans_configured"] is True
    assert health.json()["policies"]["tenant_policies_configured"] is True

    limited_first = client.get("/metrics", headers={"X-API-Key": "viewer-read-key"})
    assert limited_first.status_code == 200

    limited_second = client.get("/metrics", headers={"X-API-Key": "viewer-read-key"})
    assert limited_second.status_code == 429
    assert limited_second.headers.get("x-ratelimit-limit") == "1"

    relaxed_first = client.get("/metrics", headers={"X-API-Key": "viewer-b-key"})
    assert relaxed_first.status_code == 200
    relaxed_second = client.get("/metrics", headers={"X-API-Key": "viewer-b-key"})
    assert relaxed_second.status_code == 200

    create_first = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "operator-a-key"},
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert create_first.status_code == 200, create_first.text

    create_second = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "operator-a-key"},
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert create_second.status_code == 429
    assert "active job quota exceeded" in create_second.text


def test_audit_log_records_job_events(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    class FakeQueue:
        def ensure_consumer_group(self, group: str) -> None:
            return

        def enqueue(self, job_id: str, tenant_id: str) -> str:
            return f"{tenant_id}:{job_id}"

    audit_path = tmp_path / "audit" / "zocr_audit.jsonl"
    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_QUEUE_BACKEND", "redis")
    monkeypatch.setenv("ZOCR_API_MULTI_TENANT_ENABLED", "1")
    monkeypatch.setenv("ZOCR_API_AUDIT_LOG_PATH", str(audit_path))
    monkeypatch.setenv(
        "ZOCR_API_KEYS_JSON",
        json.dumps({"admin-key": {"tenant_id": "*", "roles": ["admin"], "admin": True, "subject": "admin"}}),
    )
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_CREATE", "0")

    import zocr.service.queue as queue_module

    monkeypatch.setattr(queue_module, "build_redis_queue", lambda: FakeQueue(), raising=True)

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    create_resp = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "admin-key", "X-Tenant-ID": "tenant-a", "X-Request-ID": "req-create"},
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert create_resp.status_code == 200, create_resp.text
    job_id = create_resp.json()["job"]["id"]

    get_resp = client.get(
        f"/v1/jobs/{job_id}",
        headers={"X-API-Key": "admin-key", "X-Tenant-ID": "tenant-a", "X-Request-ID": "req-get"},
    )
    assert get_resp.status_code == 200

    delete_resp = client.delete(
        f"/v1/jobs/{job_id}",
        headers={"X-API-Key": "admin-key", "X-Tenant-ID": "tenant-a", "X-Request-ID": "req-delete"},
    )
    assert delete_resp.status_code == 200

    assert audit_path.exists()
    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [record.get("event") for record in records]
    assert "http.request" in events
    assert "job.created" in events
    assert "job.deleted" in events
    created_record = next(record for record in records if record.get("event") == "job.created")
    assert created_record["job_id"] == job_id
    assert created_record["tenant_id"] == "tenant-a"
    assert created_record["subject"] == "admin"
    deleted_record = next(record for record in records if record.get("event") == "job.deleted")
    assert deleted_record["request_id"] == "req-delete"


def test_audit_events_api_file_search(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    class FakeQueue:
        def ensure_consumer_group(self, group: str) -> None:
            return

        def enqueue(self, job_id: str, tenant_id: str) -> str:
            return f"{tenant_id}:{job_id}"

    audit_path = tmp_path / "audit" / "zocr_audit.jsonl"
    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_QUEUE_BACKEND", "redis")
    monkeypatch.setenv("ZOCR_API_MULTI_TENANT_ENABLED", "1")
    monkeypatch.setenv("ZOCR_API_AUDIT_SINKS", "file")
    monkeypatch.setenv("ZOCR_API_AUDIT_READ_BACKEND", "file")
    monkeypatch.setenv("ZOCR_API_AUDIT_LOG_PATH", str(audit_path))
    monkeypatch.setenv(
        "ZOCR_API_KEYS_JSON",
        json.dumps(
            {
                "admin-key": {"tenant_id": "*", "roles": ["admin"], "admin": True, "subject": "admin"},
                "audit-read-key": {
                    "tenant_id": "tenant-a",
                    "scopes": ["zocr.audit.read"],
                    "subject": "tenant-a-auditor",
                },
            }
        ),
    )
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_CREATE", "0")

    import zocr.service.queue as queue_module

    monkeypatch.setattr(queue_module, "build_redis_queue", lambda: FakeQueue(), raising=True)

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    create_resp = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "admin-key", "X-Tenant-ID": "tenant-a", "X-Request-ID": "req-audit-create"},
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert create_resp.status_code == 200, create_resp.text
    job_id = create_resp.json()["job"]["id"]

    delete_resp = client.delete(
        f"/v1/jobs/{job_id}",
        headers={"X-API-Key": "admin-key", "X-Tenant-ID": "tenant-a", "X-Request-ID": "req-audit-delete"},
    )
    assert delete_resp.status_code == 200

    admin_search = client.get(
        "/v1/admin/audit-events?tenant_id=tenant-a&event=job.deleted&limit=5",
        headers={"X-API-Key": "admin-key"},
    )
    assert admin_search.status_code == 200, admin_search.text
    admin_payload = admin_search.json()
    assert admin_payload["backend"] == "file"
    assert admin_payload["count"] == 1
    assert admin_payload["events"][0]["event"] == "job.deleted"
    assert admin_payload["events"][0]["tenant_id"] == "tenant-a"

    scoped_search = client.get(
        "/v1/admin/audit-events?event=job.deleted",
        headers={"X-API-Key": "audit-read-key"},
    )
    assert scoped_search.status_code == 200, scoped_search.text
    scoped_payload = scoped_search.json()
    assert scoped_payload["filters"]["tenant_id"] == "tenant-a"
    assert scoped_payload["events"][0]["tenant_id"] == "tenant-a"

    denied = client.get(
        "/v1/admin/audit-events?tenant_id=tenant-b",
        headers={"X-API-Key": "audit-read-key"},
    )
    assert denied.status_code == 403


def test_audit_sink_postgres(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    state = _install_fake_psycopg(monkeypatch)
    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_AUDIT_SINKS", "postgres")
    monkeypatch.setenv("ZOCR_API_AUDIT_DATABASE_URL", "postgresql://audit")
    monkeypatch.setenv(
        "ZOCR_API_KEYS_JSON",
        json.dumps(
            {
                "viewer-key": {"tenant_id": "tenant-a", "roles": ["viewer"], "subject": "viewer-a"},
                "audit-key": {"tenant_id": "tenant-a", "scopes": ["zocr.audit.read"], "subject": "auditor-a"},
            }
        ),
    )

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["policies"]["audit_backends"] == ["postgres"]
    assert health.json()["policies"]["audit_read_backend"] == "postgres"

    resp = client.get("/metrics", headers={"X-API-Key": "viewer-key", "X-Request-ID": "req-pg-audit"})
    assert resp.status_code == 200

    search_resp = client.get(
        "/v1/admin/audit-events?request_id=req-pg-audit",
        headers={"X-API-Key": "audit-key"},
    )
    assert search_resp.status_code == 200, search_resp.text
    search_payload = search_resp.json()
    assert search_payload["backend"] == "postgres"
    assert search_payload["count"] == 1
    assert search_payload["events"][0]["request_id"] == "req-pg-audit"

    events = state["events"]
    assert isinstance(events, list)
    assert any(event["event"] == "http.request" for event in events)
    http_event = next(
        event
        for event in events
        if event["event"] == "http.request" and event["request_id"] == "req-pg-audit"
    )
    assert http_event["tenant_id"] == "tenant-a"
    assert http_event["request_id"] == "req-pg-audit"
    assert http_event["payload"]["subject"] == "viewer-a"


def test_postgres_tenant_policy_resolution(monkeypatch) -> None:
    _install_fake_psycopg(
        monkeypatch,
        tenant_policy_rows={"tenant-a": ("starter", 2, 25, 90)},
    )

    from zocr.service.metadata import PostgresJobRepository

    repo = PostgresJobRepository(dsn="postgresql://metadata", auto_init=True)
    policy = repo.resolve_tenant_policy("tenant-a")

    assert policy is not None
    assert policy.plan_name == "starter"
    assert policy.max_active_jobs == 2
    assert policy.max_stored_jobs == 25
    assert policy.rate_limit_per_min == 90
    assert policy.source == "postgres"


def test_postgres_tenant_change_request_approve(monkeypatch) -> None:
    state = _install_fake_psycopg(monkeypatch)

    from zocr.service.metadata import PostgresJobRepository

    repo = PostgresJobRepository(dsn="postgresql://metadata", auto_init=True)
    change_request = repo.create_tenant_change_request(
        {
            "target_type": "policy",
            "target_id": "tenant-a",
            "action": "upsert",
            "payload": {"max_active_jobs": 2, "rate_limit_per_min": 30},
            "requested_by": "writer-a",
        }
    )

    assert change_request.status == "pending"
    assert change_request.requested_by == "writer-a"

    listed = repo.list_tenant_change_requests(status="pending")
    assert len(listed) == 1
    assert listed[0].request_id == change_request.request_id

    with pytest.raises(PermissionError):
        repo.approve_tenant_change_request(
            change_request.request_id,
            reviewed_by="writer-a",
        )

    approved = repo.approve_tenant_change_request(
        change_request.request_id,
        reviewed_by="approver-a",
        review_reason="looks-good",
    )
    assert approved is not None
    assert approved.status == "approved"
    assert approved.reviewed_by == "approver-a"
    assert approved.review_reason == "looks-good"

    policy = repo.resolve_tenant_policy("tenant-a")
    assert policy is not None
    assert policy.max_active_jobs == 2
    assert policy.rate_limit_per_min == 30
    assert state["tenant_policies"]["tenant-a"] == (None, 2, None, 30)


def test_postgres_tenant_change_request_dual_approval(monkeypatch) -> None:
    state = _install_fake_psycopg(monkeypatch)
    monkeypatch.setenv("ZOCR_API_TENANT_APPROVAL_MIN_APPROVERS", "2")

    from zocr.service.metadata import PostgresJobRepository

    repo = PostgresJobRepository(dsn="postgresql://metadata", auto_init=True)
    change_request = repo.create_tenant_change_request(
        {
            "target_type": "policy",
            "target_id": "tenant-a",
            "action": "upsert",
            "payload": {"max_active_jobs": 3},
            "requested_by": "writer-a",
        }
    )

    first = repo.approve_tenant_change_request(
        change_request.request_id,
        reviewed_by="approver-a",
        review_reason="first-pass",
    )
    assert first is not None
    assert first.status == "pending"
    assert first.approvals_required == 2
    assert first.approvals_received == 1
    assert first.reviewed_by is None
    assert "tenant-a" not in state["tenant_policies"]

    second = repo.approve_tenant_change_request(
        change_request.request_id,
        reviewed_by="approver-b",
        review_reason="second-pass",
    )
    assert second is not None
    assert second.status == "approved"
    assert second.approvals_received == 2
    assert second.reviewed_by == "approver-b"
    assert state["tenant_policies"]["tenant-a"] == (None, 3, None, None)


def test_admin_tenant_policy_api(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    class FakeQueue:
        def ensure_consumer_group(self, group: str) -> None:
            return

        def enqueue(self, job_id: str, tenant_id: str) -> str:
            return f"{tenant_id}:{job_id}"

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_QUEUE_BACKEND", "redis")
    monkeypatch.setenv(
        "ZOCR_API_KEYS_JSON",
        json.dumps(
            {
                "admin-key": {"tenant_id": "*", "roles": ["admin"], "admin": True, "subject": "admin"},
                "operator-key": {"tenant_id": "tenant-a", "roles": ["operator"], "subject": "operator-a"},
            }
        ),
    )
    monkeypatch.setenv("ZOCR_API_JOBS_RESUME_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_STARTUP", "0")
    monkeypatch.setenv("ZOCR_API_JOBS_CLEANUP_ON_CREATE", "0")

    import zocr.service.queue as queue_module

    monkeypatch.setattr(queue_module, "build_redis_queue", lambda: FakeQueue(), raising=True)

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    put_plan = client.put(
        "/v1/admin/tenant-plans/starter",
        headers={"X-API-Key": "admin-key"},
        json={"max_active_jobs": 1, "rate_limit_per_min": 10},
    )
    assert put_plan.status_code == 200, put_plan.text

    put_policy = client.put(
        "/v1/admin/tenant-policies/tenant-a",
        headers={"X-API-Key": "admin-key"},
        json={"plan_name": "starter"},
    )
    assert put_policy.status_code == 200, put_policy.text
    assert put_policy.json()["policy"]["plan_name"] == "starter"

    list_plans = client.get("/v1/admin/tenant-plans", headers={"X-API-Key": "admin-key"})
    assert list_plans.status_code == 200
    assert list_plans.json()["count"] == 1

    list_policies = client.get("/v1/admin/tenant-policies", headers={"X-API-Key": "admin-key"})
    assert list_policies.status_code == 200
    assert list_policies.json()["count"] == 1

    create_first = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "operator-key"},
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert create_first.status_code == 200, create_first.text

    create_second = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "operator-key"},
        files={"file": ("test.png", b"fake", "image/png")},
    )
    assert create_second.status_code == 429
    assert "active job quota exceeded" in create_second.text

    delete_policy = client.delete("/v1/admin/tenant-policies/tenant-a", headers={"X-API-Key": "admin-key"})
    assert delete_policy.status_code == 200

    delete_plan = client.delete("/v1/admin/tenant-plans/starter", headers={"X-API-Key": "admin-key"})
    assert delete_plan.status_code == 200


def test_admin_tenant_approval_workflow_api(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    audit_path = tmp_path / "audit" / "tenant_approvals.jsonl"
    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_TENANT_APPROVAL_REQUIRED", "1")
    monkeypatch.setenv("ZOCR_API_AUDIT_SINKS", "file")
    monkeypatch.setenv("ZOCR_API_AUDIT_LOG_PATH", str(audit_path))
    monkeypatch.setenv("ZOCR_API_TENANT_APPROVAL_NOTIFY_URL", "https://notify.example/tenant-approvals")
    monkeypatch.setenv(
        "ZOCR_API_KEYS_JSON",
        json.dumps(
            {
                "writer-key": {
                    "tenant_id": "tenant-a",
                    "scopes": ["zocr.tenants.write"],
                    "subject": "writer-a",
                },
                "approver-key": {
                    "tenant_id": "tenant-a",
                    "scopes": ["zocr.tenants.read", "zocr.tenants.approve"],
                    "subject": "approver-a",
                },
            }
        ),
    )

    notifications: list[dict[str, object]] = []

    class FakeNotifyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def read(self, _size: int = -1) -> bytes:
            return b"ok"

    import zocr.service.policy as policy_module

    def fake_urlopen(request, timeout=0):
        notifications.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "payload": json.loads(request.data.decode("utf-8")),
            }
        )
        return FakeNotifyResponse()

    monkeypatch.setattr(policy_module, "urlopen", fake_urlopen, raising=True)

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["policies"]["tenant_approval_required"] is True
    assert health.json()["policies"]["tenant_approval_allow_self"] is False
    assert health.json()["policies"]["tenant_approval_notify_enabled"] is True

    create_request = client.put(
        "/v1/admin/tenant-plans/starter",
        headers={"X-API-Key": "writer-key", "X-Request-ID": "req-tenant-request"},
        json={"max_active_jobs": 2, "rate_limit_per_min": 30},
    )
    assert create_request.status_code == 202, create_request.text
    change_request = create_request.json()["change_request"]
    assert change_request["status"] == "pending"
    assert change_request["target_type"] == "plan"
    assert change_request["requested_by"] == "writer-a"

    plans_before = client.get("/v1/admin/tenant-plans", headers={"X-API-Key": "approver-key"})
    assert plans_before.status_code == 200
    assert plans_before.json()["count"] == 0

    pending = client.get(
        "/v1/admin/tenant-change-requests?status=pending",
        headers={"X-API-Key": "approver-key"},
    )
    assert pending.status_code == 200, pending.text
    assert pending.json()["count"] == 1
    assert pending.json()["change_requests"][0]["request_id"] == change_request["request_id"]

    denied = client.post(
        f"/v1/admin/tenant-change-requests/{change_request['request_id']}/approve",
        headers={"X-API-Key": "writer-key"},
    )
    assert denied.status_code == 403

    approved = client.post(
        f"/v1/admin/tenant-change-requests/{change_request['request_id']}/approve",
        headers={"X-API-Key": "approver-key", "X-Request-ID": "req-tenant-approve"},
        json={"reason": "approved"},
    )
    assert approved.status_code == 200, approved.text
    approved_payload = approved.json()["change_request"]
    assert approved_payload["status"] == "approved"
    assert approved_payload["reviewed_by"] == "approver-a"
    assert approved_payload["review_reason"] == "approved"

    plans_after = client.get("/v1/admin/tenant-plans", headers={"X-API-Key": "approver-key"})
    assert plans_after.status_code == 200
    assert plans_after.json()["count"] == 1
    assert plans_after.json()["plans"][0]["plan_name"] == "starter"

    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [record["event"] for record in records]
    assert "tenant_change_request.created" in events
    assert "tenant_change_request.approved" in events
    approved_record = next(record for record in records if record["event"] == "tenant_change_request.approved")
    assert approved_record["request_id"] == "req-tenant-approve"
    assert approved_record["change_request_id"] == change_request["request_id"]

    assert [item["payload"]["event"] for item in notifications] == [
        "tenant_change_request.created",
        "tenant_change_request.approved",
    ]
    created_notification = notifications[0]["payload"]
    approved_notification = notifications[1]["payload"]
    assert created_notification["actor"]["subject"] == "writer-a"
    assert created_notification["change_request"]["request_id"] == change_request["request_id"]
    assert approved_notification["actor"]["subject"] == "approver-a"
    assert approved_notification["change_request"]["status"] == "approved"


def test_admin_tenant_self_approval_denied_api(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_TENANT_APPROVAL_REQUIRED", "1")
    monkeypatch.setenv(
        "ZOCR_API_KEYS_JSON",
        json.dumps(
            {
                "self-key": {
                    "tenant_id": "tenant-a",
                    "scopes": ["zocr.tenants.read", "zocr.tenants.write", "zocr.tenants.approve"],
                    "subject": "self-a",
                },
            }
        ),
    )

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    create_request = client.put(
        "/v1/admin/tenant-plans/starter",
        headers={"X-API-Key": "self-key"},
        json={"max_active_jobs": 1},
    )
    assert create_request.status_code == 202, create_request.text
    request_id = create_request.json()["change_request"]["request_id"]

    approve = client.post(
        f"/v1/admin/tenant-change-requests/{request_id}/approve",
        headers={"X-API-Key": "self-key"},
    )
    assert approve.status_code == 403
    assert "Self-approval is not allowed" in approve.text


def test_admin_tenant_dual_approval_api(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_TENANT_APPROVAL_REQUIRED", "1")
    monkeypatch.setenv("ZOCR_API_TENANT_APPROVAL_MIN_APPROVERS", "2")
    monkeypatch.setenv(
        "ZOCR_API_KEYS_JSON",
        json.dumps(
            {
                "writer-key": {
                    "tenant_id": "tenant-a",
                    "scopes": ["zocr.tenants.write"],
                    "subject": "writer-a",
                },
                "approver-a-key": {
                    "tenant_id": "tenant-a",
                    "scopes": ["zocr.tenants.read", "zocr.tenants.approve"],
                    "subject": "approver-a",
                },
                "approver-b-key": {
                    "tenant_id": "tenant-a",
                    "scopes": ["zocr.tenants.read", "zocr.tenants.approve"],
                    "subject": "approver-b",
                },
            }
        ),
    )

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    create_request = client.put(
        "/v1/admin/tenant-plans/starter",
        headers={"X-API-Key": "writer-key"},
        json={"max_active_jobs": 2},
    )
    assert create_request.status_code == 202, create_request.text
    request_id = create_request.json()["change_request"]["request_id"]

    first = client.post(
        f"/v1/admin/tenant-change-requests/{request_id}/approve",
        headers={"X-API-Key": "approver-a-key"},
        json={"reason": "first-pass"},
    )
    assert first.status_code == 200, first.text
    first_payload = first.json()["change_request"]
    assert first_payload["status"] == "pending"
    assert first_payload["approvals_required"] == 2
    assert first_payload["approvals_received"] == 1

    plans_before = client.get("/v1/admin/tenant-plans", headers={"X-API-Key": "approver-a-key"})
    assert plans_before.status_code == 200
    assert plans_before.json()["count"] == 0

    second = client.post(
        f"/v1/admin/tenant-change-requests/{request_id}/approve",
        headers={"X-API-Key": "approver-b-key"},
        json={"reason": "second-pass"},
    )
    assert second.status_code == 200, second.text
    second_payload = second.json()["change_request"]
    assert second_payload["status"] == "approved"
    assert second_payload["approvals_received"] == 2
    assert second_payload["reviewed_by"] == "approver-b"

    plans_after = client.get("/v1/admin/tenant-plans", headers={"X-API-Key": "approver-b-key"})
    assert plans_after.status_code == 200
    assert plans_after.json()["count"] == 1


def test_admin_tenant_scope_split_api_key(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    pytest.importorskip("python_multipart")

    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv(
        "ZOCR_API_KEYS_JSON",
        json.dumps(
            {
                "tenant-read-key": {
                    "tenant_id": "tenant-a",
                    "scopes": ["zocr.tenants.read"],
                    "subject": "tenant-read",
                },
                "tenant-write-key": {
                    "tenant_id": "tenant-a",
                    "scopes": ["zocr.tenants.write"],
                    "subject": "tenant-write",
                },
            }
        ),
    )

    from zocr.service.app import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    read_ok = client.get("/v1/admin/tenant-plans", headers={"X-API-Key": "tenant-read-key"})
    assert read_ok.status_code == 200

    read_denied = client.put(
        "/v1/admin/tenant-plans/starter",
        headers={"X-API-Key": "tenant-read-key"},
        json={"max_active_jobs": 1},
    )
    assert read_denied.status_code == 403

    write_ok = client.put(
        "/v1/admin/tenant-plans/starter",
        headers={"X-API-Key": "tenant-write-key"},
        json={"max_active_jobs": 1},
    )
    assert write_ok.status_code == 200

    write_denied = client.get("/v1/admin/tenant-plans", headers={"X-API-Key": "tenant-write-key"})
    assert write_denied.status_code == 403


def test_admin_cli_persists_local_tenant_config(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.delenv("ZOCR_API_TENANT_PLANS_JSON", raising=False)
    monkeypatch.delenv("ZOCR_API_TENANT_POLICIES_JSON", raising=False)
    audit_path = tmp_path / "audit" / "admin_cli.jsonl"
    monkeypatch.setenv("ZOCR_API_AUDIT_SINKS", "file")
    monkeypatch.setenv("ZOCR_API_AUDIT_LOG_PATH", str(audit_path))

    from zocr.service.admin_cli import main as admin_main

    admin_main(["tenant-plan", "put", "starter", "--max-active-jobs", "2", "--rate-limit-per-min", "30"])
    plan_put = json.loads(capsys.readouterr().out)
    assert plan_put["plan"]["plan_name"] == "starter"
    assert plan_put["plan"]["max_active_jobs"] == 2

    admin_main(["tenant-policy", "put", "tenant-a", "--plan", "starter"])
    policy_put = json.loads(capsys.readouterr().out)
    assert policy_put["policy"]["tenant_id"] == "tenant-a"
    assert policy_put["policy"]["plan_name"] == "starter"

    admin_main(["tenant-plan", "list"])
    listed_plans = json.loads(capsys.readouterr().out)
    assert listed_plans["count"] == 1
    assert listed_plans["plans"][0]["plan_name"] == "starter"

    admin_main(["tenant-policy", "list"])
    listed_policies = json.loads(capsys.readouterr().out)
    assert listed_policies["count"] == 1
    assert listed_policies["policies"][0]["tenant_id"] == "tenant-a"

    admin_main(["audit", "list", "--event", "tenant_policy.upserted", "--limit", "5"])
    listed_audit = json.loads(capsys.readouterr().out)
    assert listed_audit["backend"] == "file"
    assert listed_audit["count"] >= 1
    assert listed_audit["events"][0]["event"] == "tenant_policy.upserted"

    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [record["event"] for record in records]
    assert "tenant_plan.upserted" in events
    assert "tenant_policy.upserted" in events


def test_admin_cli_tenant_approval_workflow(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("ZOCR_API_TENANT_APPROVAL_REQUIRED", "1")
    monkeypatch.setenv("ZOCR_API_TENANT_APPROVAL_MIN_APPROVERS", "2")
    monkeypatch.setenv("ZOCR_API_TENANT_APPROVAL_NOTIFY_URL", "https://notify.example/tenant-approvals")
    audit_path = tmp_path / "audit" / "admin_cli_approvals.jsonl"
    monkeypatch.setenv("ZOCR_API_AUDIT_SINKS", "file")
    monkeypatch.setenv("ZOCR_API_AUDIT_LOG_PATH", str(audit_path))

    notifications: list[dict[str, object]] = []

    class FakeNotifyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def read(self, _size: int = -1) -> bytes:
            return b"ok"

    import zocr.service.policy as policy_module

    def fake_urlopen(request, timeout=0):
        notifications.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "payload": json.loads(request.data.decode("utf-8")),
            }
        )
        return FakeNotifyResponse()

    monkeypatch.setattr(policy_module, "urlopen", fake_urlopen, raising=True)

    from zocr.service.admin_cli import main as admin_main

    monkeypatch.setenv("ZOCR_ADMIN_SUBJECT", "writer-a")
    admin_main(["tenant-policy", "put", "tenant-a", "--max-active-jobs", "2"])
    created = json.loads(capsys.readouterr().out)
    change_request = created["change_request"]
    assert change_request["status"] == "pending"
    assert change_request["target_type"] == "policy"

    admin_main(["tenant-policy", "list"])
    policies_before = json.loads(capsys.readouterr().out)
    assert policies_before["count"] == 0

    admin_main(["tenant-request", "list", "--status", "pending"])
    pending = json.loads(capsys.readouterr().out)
    assert pending["count"] == 1
    assert pending["change_requests"][0]["request_id"] == change_request["request_id"]

    monkeypatch.setenv("ZOCR_ADMIN_SUBJECT", "approver-a")
    admin_main(["tenant-request", "approve", change_request["request_id"], "--reason", "ok"])
    first_review = json.loads(capsys.readouterr().out)
    assert first_review["change_request"]["status"] == "pending"
    assert first_review["change_request"]["approvals_required"] == 2
    assert first_review["change_request"]["approvals_received"] == 1

    admin_main(["tenant-policy", "list"])
    policies_mid = json.loads(capsys.readouterr().out)
    assert policies_mid["count"] == 0

    monkeypatch.setenv("ZOCR_ADMIN_SUBJECT", "approver-b")
    admin_main(["tenant-request", "approve", change_request["request_id"], "--reason", "final-ok"])
    approved = json.loads(capsys.readouterr().out)
    assert approved["change_request"]["status"] == "approved"
    assert approved["change_request"]["review_reason"] == "final-ok"
    assert approved["change_request"]["approvals_received"] == 2

    admin_main(["tenant-policy", "list"])
    policies_after = json.loads(capsys.readouterr().out)
    assert policies_after["count"] == 1
    assert policies_after["policies"][0]["tenant_id"] == "tenant-a"
    assert policies_after["policies"][0]["max_active_jobs"] == 2

    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [record["event"] for record in records]
    assert "tenant_change_request.created" in events
    assert "tenant_change_request.listed" in events
    assert "tenant_change_request.reviewed" in events
    assert "tenant_change_request.approved" in events
    assert [item["payload"]["event"] for item in notifications] == [
        "tenant_change_request.created",
        "tenant_change_request.reviewed",
        "tenant_change_request.approved",
    ]
    assert notifications[0]["payload"]["actor"]["subject"] == "writer-a"
    assert notifications[1]["payload"]["actor"]["subject"] == "approver-a"
    assert notifications[2]["payload"]["actor"]["subject"] == "approver-b"
