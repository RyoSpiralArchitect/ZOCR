import hashlib
import time
import uuid
from urllib.request import urlopen

import pytest

from test_service_integrations import local_service
from zocr.service.storage import S3JobStore


@pytest.fixture
def minio_store(tmp_path, monkeypatch):
    endpoint = local_service("ZOCR_TEST_MINIO_URL")
    import boto3

    for key, value in {"AWS_ACCESS_KEY_ID": "zocr_test", "AWS_SECRET_ACCESS_KEY": "zocr_test_local_only",
                       "AWS_DEFAULT_REGION": "us-east-1", "AWS_EC2_METADATA_DISABLED": "true"}.items():
        monkeypatch.setenv(key, value)
    deadline = time.monotonic() + 30
    while True:
        try:
            with urlopen(endpoint + "/minio/health/live", timeout=1) as response:
                assert response.status == 200
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.2)
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    bucket = "zocr-test-" + uuid.uuid4().hex
    boto3.client("s3", endpoint_url=endpoint).create_bucket(Bucket=bucket)
    return S3JobStore(storage_dir=tmp_path, bucket=bucket, endpoint_url=endpoint, force_path_style=True)


def test_minio_multipart_presign_and_fresh_cache(minio_store, tmp_path):
    store = minio_store
    path = store.job_paths(uuid.uuid4().hex).artifacts_zip
    path.parent.mkdir(parents=True)
    content = b"zocr-evidence-" * 800000
    path.write_bytes(content)
    store.persist_file(path)
    with urlopen(store.presigned_get_url(path), timeout=15) as response:
        assert hashlib.sha256(response.read()).digest() == hashlib.sha256(content).digest()
    fresh = S3JobStore(storage_dir=tmp_path / "fresh", bucket=store.bucket,
                       endpoint_url=store._client.meta.endpoint_url, force_path_style=True)
    local = fresh.job_paths(path.parent.name).artifacts_zip
    assert fresh.ensure_local_file(local).read_bytes() == content
    assert store._client.head_object(Bucket=store.bucket, Key=store._key_for(path))["ContentLength"] == len(content)


def test_minio_local_review_history_survives_fresh_cache(minio_store, tmp_path):
    from test_review_ledger import new_review, bundle
    from zocr.service.metadata import LocalJobRepository
    from zocr.service.review_ledger import ReviewLedger

    ledger = ReviewLedger(minio_store, LocalJobRepository(job_store=minio_store))
    review = new_review(ledger)
    approved = ledger.append(review["review_id"], "tenant-a", bundle(review), {"subject": "alice"})
    fresh = S3JobStore(storage_dir=tmp_path / "fresh", bucket=minio_store.bucket,
                       endpoint_url=minio_store._client.meta.endpoint_url, force_path_style=True)
    restored = ReviewLedger(fresh, LocalJobRepository(job_store=fresh)).read(review["review_id"], "tenant-a")
    assert restored == approved
    assert restored["revision"] == 1
    with pytest.raises(ValueError):
        fresh._key_for(tmp_path / "outside.json")


def test_minio_missing_and_authentication_errors_are_distinct(minio_store, tmp_path, monkeypatch):
    from botocore.exceptions import ClientError

    missing = minio_store.job_paths(uuid.uuid4().hex).artifacts_zip
    with pytest.raises(FileNotFoundError):
        minio_store.ensure_local_file(missing, force=True)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "invalid-test-identity")
    monkeypatch.setattr("boto3.DEFAULT_SESSION", None)
    denied = S3JobStore(storage_dir=tmp_path / "denied", bucket=minio_store.bucket,
                        endpoint_url=minio_store._client.meta.endpoint_url, force_path_style=True)
    with pytest.raises(ClientError) as failure:
        denied.ensure_local_file(denied.job_paths("job").artifacts_zip, force=True)
    assert failure.value.response["ResponseMetadata"]["HTTPStatusCode"] == 403
