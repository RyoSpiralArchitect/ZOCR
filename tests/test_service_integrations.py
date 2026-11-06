import io
import os
import subprocess
import sys
import time
import uuid
import zipfile
from urllib.parse import urlparse

import pytest


def local_service(name):
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"Set {name} to a dedicated local test service")
    if urlparse(value).hostname not in {"127.0.0.1", "localhost", "::1"}:
        pytest.fail(f"{name} must refer to a dedicated loopback test service")
    return value


def test_real_postgres_execution_exclusion(tmp_path):
    from zocr.service.execution_lock import execution_lock
    from zocr.service.metadata import PostgresJobRepository
    from zocr.service.storage import LocalJobStore

    repository = PostgresJobRepository(dsn=local_service("ZOCR_TEST_POSTGRES_URL"))
    store = LocalJobStore(tmp_path)
    job_id = uuid.uuid4().hex
    with execution_lock(store, repository, job_id, "test") as first:
        assert first
        with execution_lock(store, repository, job_id, "test") as duplicate:
            assert not duplicate
    with execution_lock(store, repository, job_id, "test") as released:
        assert released


def test_real_redis_heartbeat_reclaim_and_ack():
    from zocr.service.queue import RedisStreamQueue

    queue = RedisStreamQueue(url=local_service("ZOCR_TEST_REDIS_URL"), stream="test-" + uuid.uuid4().hex)
    queue.ensure_consumer_group("workers")
    message_id = queue.enqueue(job_id="job", tenant_id="test")
    messages = queue.read(group="workers", consumer="one", block_ms=100)
    assert messages[0][0] == message_id
    with queue.keep_alive(group="workers", consumer="one", message_id=message_id, interval=.01):
        time.sleep(.15)
        _, claimed = queue.claim_stale(group="workers", consumer="two", min_idle_ms=100)
        assert not claimed
    time.sleep(.15)
    _, claimed = queue.claim_stale(group="workers", consumer="two", min_idle_ms=100)
    assert claimed[0][0] == message_id
    assert queue.ack(group="workers", message_id=message_id) == 1


@pytest.mark.parametrize(("s3_environment", "metadata_backend"), [
    ("ZOCR_TEST_S3_URL", "postgres"), ("ZOCR_TEST_MINIO_URL", "postgres"), ("ZOCR_TEST_MINIO_URL", "local"),
])
def test_real_api_redis_worker_postgres_and_s3(tmp_path, table_image, monkeypatch, s3_environment, metadata_backend):
    redis_url = local_service("ZOCR_TEST_REDIS_URL")
    postgres_url = local_service("ZOCR_TEST_POSTGRES_URL")
    s3_url = local_service(s3_environment)
    import boto3
    from fastapi.testclient import TestClient
    from zocr.service.app import create_app
    from zocr.service.storage import S3JobStore, result_paths
    from zocr.service.metadata import PostgresJobRepository, LocalJobRepository
    from zocr.service.execution_lock import execution_lock
    from zocr.service.queue import RedisStreamQueue

    token = uuid.uuid4().hex
    bucket = "zocr-test-" + token
    settings = {
        "AWS_ACCESS_KEY_ID": "local_test", "AWS_SECRET_ACCESS_KEY": "local_test",
        "AWS_DEFAULT_REGION": "us-east-1", "AWS_EC2_METADATA_DISABLED": "true",
        "ZOCR_API_STORAGE_DIR": str(tmp_path / "store"),
        "ZOCR_API_STORAGE_BACKEND": "s3", "ZOCR_API_S3_BUCKET": bucket,
        "ZOCR_API_S3_ENDPOINT_URL": s3_url, "ZOCR_API_S3_FORCE_PATH_STYLE": "1",
        "ZOCR_API_S3_PRESIGN_ENABLED": "0", "ZOCR_API_METADATA_BACKEND": metadata_backend,
        "ZOCR_API_DATABASE_URL": postgres_url, "ZOCR_API_DB_AUTO_INIT": "1",
        "ZOCR_API_QUEUE_BACKEND": "redis", "ZOCR_API_REDIS_URL": redis_url,
        "ZOCR_API_REDIS_STREAM": "test-" + token, "ZOCR_API_REDIS_GROUP": "workers",
        "ZOCR_API_JOBS_CLEANUP_ON_STARTUP": "0", "ZOCR_API_JOBS_CLEANUP_ON_CREATE": "0",
        "ZOCR_API_JOBS_RESUME_ON_STARTUP": "0", "ZOCR_API_RUN_TIMEOUT_SEC": "60",
        "ZOCR_TOY_SWEEPS": "1", "ZOCR_TOY_SWEEP_LIMIT": "1",
    }
    if s3_environment == "ZOCR_TEST_MINIO_URL":
        settings.update(AWS_ACCESS_KEY_ID="zocr_test", AWS_SECRET_ACCESS_KEY="zocr_test_local_only")
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    boto3.client("s3", endpoint_url=s3_url).create_bucket(Bucket=bucket)
    with TestClient(create_app()) as client:
        response = client.post("/v1/jobs?toy_lite=true&k=1", files={
            "file": ("page.png", table_image.read_bytes(), "image/png"),
        })
        assert response.status_code == 200, response.text
        job_id = response.json()["job"]["id"]
        store = S3JobStore(storage_dir=tmp_path / "store", bucket=bucket, endpoint_url=s3_url, force_path_style=True)
        repository = (PostgresJobRepository(dsn=postgres_url) if metadata_backend == "postgres"
                      else LocalJobRepository(job_store=store))
        queue = RedisStreamQueue(url=redis_url, stream=settings["ZOCR_API_REDIS_STREAM"])
        command = [sys.executable, "-m", "zocr.service.worker", "--consumer", "pr4", "--once"]
        with execution_lock(store, repository, job_id, "default") as acquired:
            assert acquired
            busy = subprocess.run(command, capture_output=True, text=True, timeout=10)
            assert busy.returncode == 0, busy.stderr
            assert repository.read_job(job_id, "default")["status"] == "queued"
            assert queue._client.xpending(queue.stream, "workers")["pending"] == 1
        completed = subprocess.run(command + ["--drain-pending"],
                                   capture_output=True, text=True, timeout=75)
        assert completed.returncode == 0, completed.stderr
        job = client.get(f"/v1/jobs/{job_id}").json()["job"]
        assert job["status"] == "succeeded", (job, completed.stderr)
        archive = client.get(f"/v1/jobs/{job_id}/artifacts.zip")
        assert archive.status_code == 200
        with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
            assert bundle.testzip() is None
            assert "zocr.manifest.json" in bundle.namelist()
        review = client.post("/v1/reviews", json={"job_a": job_id, "job_b": job_id})
        assert review.status_code == 200, review.text
        assert review.json()["review"]["review_status"] == "no_changes"
        review_id = review.json()["review"]["review_id"]
        assert client.get(f"/v1/reviews/{review_id}/history").json()["revision"] == 0
        updated_at = job["updated_at"]
        queue.enqueue(job_id, "default")
        replay = subprocess.run(command, capture_output=True, text=True, timeout=10)
        assert replay.returncode == 0, replay.stderr
        assert repository.read_job(job_id, "default")["updated_at"] == updated_at
        assert queue._client.xpending(queue.stream, "workers")["pending"] == 0
    fresh = S3JobStore(storage_dir=tmp_path / "fresh", bucket=bucket,
                       endpoint_url=s3_url, force_path_style=True)
    downloaded = fresh.ensure_local_file(result_paths(fresh, job_id, job).artifacts_zip)
    assert downloaded.read_bytes() == archive.content
