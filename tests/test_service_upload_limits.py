import pytest

pytest.importorskip("fastapi")


def test_run_rejects_oversize_upload(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ZOCR_API_MAX_UPLOAD_MB", "1")
    monkeypatch.setenv("ZOCR_API_STORAGE_DIR", str(tmp_path))
    monkeypatch.delenv("ZOCR_API_KEY", raising=False)

    from fastapi.testclient import TestClient

    from zocr.service.app import create_app

    app = create_app()
    client = TestClient(app)

    payload = b"x" * (1024 * 1024 + 1)
    resp = client.post("/v1/run", files={"file": ("big.png", payload, "image/png")})

    assert resp.status_code == 413
