from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


@dataclass(frozen=True)
class JobPaths:
    root: Path
    job_json: Path
    input_dir: Path
    out_dir: Path
    artifacts_zip: Path


@dataclass(frozen=True)
class JobMeta:
    job_id: str
    last_modified: float


class JobStore(Protocol):
    backend: str
    storage_dir: Path
    jobs_dir: Path

    def job_paths(self, job_id: str) -> JobPaths: ...
    def list_job_metas(self) -> list[JobMeta]: ...
    def delete_tree(self, path: Path) -> None: ...
    def delete_job(self, job_id: str) -> None: ...
    def read_json(self, path: Path) -> Optional[Dict[str, Any]]: ...
    def atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None: ...
    def write_job(self, job_id: str, payload: Dict[str, Any]) -> None: ...
    def read_job(self, job_id: str) -> Optional[Dict[str, Any]]: ...
    def has_file(self, path: Path) -> bool: ...
    def ensure_local_file(self, path: Path, *, force: bool = False) -> Path: ...
    def persist_file(self, path: Path) -> None: ...
    def persist_tree(self, root: Path) -> None: ...
    def presigned_get_url(self, path: Path, *, ttl_sec: Optional[int] = None) -> Optional[str]: ...


class LocalJobStore:
    """Filesystem-backed job store for single-node / on-prem deployments."""

    backend = "local"

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.jobs_dir = self.storage_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def job_paths(self, job_id: str) -> JobPaths:
        job_root = self.jobs_dir / job_id
        return JobPaths(
            root=job_root,
            job_json=job_root / "job.json",
            input_dir=job_root / "input",
            out_dir=job_root / "out",
            artifacts_zip=job_root / "artifacts.zip",
        )

    def list_job_roots(self) -> list[Path]:
        try:
            return [path for path in self.jobs_dir.iterdir() if path.is_dir()]
        except Exception:
            return []

    def list_job_metas(self) -> list[JobMeta]:
        items: list[JobMeta] = []
        for path in self.list_job_roots():
            try:
                last_modified = float(path.stat().st_mtime)
            except Exception:
                last_modified = 0.0
            items.append(JobMeta(job_id=path.name, last_modified=last_modified))
        items.sort(key=lambda item: item.last_modified, reverse=True)
        return items

    def delete_tree(self, path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)

    def delete_job(self, job_id: str) -> None:
        self.delete_tree(self.job_paths(job_id).root)

    def read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return loaded if isinstance(loaded, dict) else None

    def atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    def write_job(self, job_id: str, payload: Dict[str, Any]) -> None:
        self.atomic_write_json(self.job_paths(job_id).job_json, payload)

    def read_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.read_json(self.job_paths(job_id).job_json)

    def has_file(self, path: Path) -> bool:
        return path.exists() and path.is_file()

    def ensure_local_file(self, path: Path, *, force: bool = False) -> Path:
        return path

    def persist_file(self, path: Path) -> None:
        return

    def persist_tree(self, root: Path) -> None:
        return

    def presigned_get_url(self, path: Path, *, ttl_sec: Optional[int] = None) -> Optional[str]:
        return None


class S3JobStore:
    """S3-backed job store with a local staging cache."""

    backend = "s3"

    def __init__(
        self,
        *,
        storage_dir: Path,
        bucket: str,
        prefix: str = "zocr",
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        force_path_style: bool = False,
        presign_enabled: bool = True,
        presign_ttl_sec: int = 3600,
    ):
        try:
            import boto3  # type: ignore
            from botocore.config import Config  # type: ignore
            from botocore.exceptions import ClientError  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "S3 storage requires `boto3`. Install with `pip install -e '.[api_s3]'`."
            ) from exc

        bucket = (bucket or "").strip()
        if not bucket:
            raise RuntimeError("ZOCR_API_S3_BUCKET is required when ZOCR_API_STORAGE_BACKEND=s3.")

        self.bucket = bucket
        self._base_prefix = ((prefix or "zocr").strip().strip("/") or "zocr") + "/jobs"
        self._client_error = ClientError
        self._presign_enabled = bool(presign_enabled)
        self._presign_ttl_sec = max(1, int(presign_ttl_sec))

        self._local = LocalJobStore(storage_dir)
        self.storage_dir = self._local.storage_dir
        self.jobs_dir = self._local.jobs_dir

        config = Config(s3={"addressing_style": "path"}) if force_path_style else None
        self._client = boto3.client(
            "s3",
            region_name=(region or None),
            endpoint_url=(endpoint_url or None),
            config=config,
        )

    def job_paths(self, job_id: str) -> JobPaths:
        return self._local.job_paths(job_id)

    def _key_for(self, path: Path) -> str:
        rel = path.relative_to(self.jobs_dir)
        return f"{self._base_prefix}/{rel.as_posix()}"

    def _is_missing_error(self, exc: Exception) -> bool:
        if not isinstance(exc, self._client_error):
            return False
        code = str(exc.response.get("Error", {}).get("Code", ""))
        return code in {"404", "NoSuchKey", "NotFound"}

    def list_job_metas(self) -> list[JobMeta]:
        items: list[JobMeta] = []
        paginator = self._client.get_paginator("list_objects_v2")
        prefix = f"{self._base_prefix}/"
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                key = obj.get("Key")
                if not isinstance(key, str) or not key.endswith("/job.json"):
                    continue
                rel = key[len(prefix) :]
                job_id = rel.split("/", 1)[0]
                raw_last_modified = obj.get("LastModified")
                if isinstance(raw_last_modified, datetime):
                    if raw_last_modified.tzinfo is None:
                        raw_last_modified = raw_last_modified.replace(tzinfo=timezone.utc)
                    last_modified = float(raw_last_modified.timestamp())
                else:
                    last_modified = 0.0
                items.append(JobMeta(job_id=job_id, last_modified=last_modified))
        items.sort(key=lambda item: item.last_modified, reverse=True)
        return items

    def delete_tree(self, path: Path) -> None:
        try:
            rel = path.relative_to(self.jobs_dir)
        except Exception:
            rel = None
        if rel is not None:
            prefix = f"{self._base_prefix}/{rel.as_posix().rstrip('/')}/"
            paginator = self._client.get_paginator("list_objects_v2")
            batch: list[dict[str, str]] = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []) or []:
                    key = obj.get("Key")
                    if not isinstance(key, str) or not key:
                        continue
                    batch.append({"Key": key})
                    if len(batch) >= 1000:
                        self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})
                        batch = []
            if batch:
                self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})
        shutil.rmtree(path, ignore_errors=True)

    def delete_job(self, job_id: str) -> None:
        self.delete_tree(self.job_paths(job_id).root)

    def read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            self.ensure_local_file(path, force=True)
        except FileNotFoundError:
            return None
        return self._local.read_json(path)

    def atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        self._local.atomic_write_json(path, payload)
        self.persist_file(path)

    def write_job(self, job_id: str, payload: Dict[str, Any]) -> None:
        self.atomic_write_json(self.job_paths(job_id).job_json, payload)

    def read_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.read_json(self.job_paths(job_id).job_json)

    def has_file(self, path: Path) -> bool:
        if path.exists() and path.is_file():
            return True
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._key_for(path))
        except Exception as exc:
            if self._is_missing_error(exc):
                return False
            raise
        return True

    def ensure_local_file(self, path: Path, *, force: bool = False) -> Path:
        if path.exists() and not force:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        key = self._key_for(path)
        try:
            self._client.download_file(self.bucket, key, str(path))
        except Exception as exc:
            if self._is_missing_error(exc):
                raise FileNotFoundError(str(path)) from exc
            raise
        return path

    def persist_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            return
        self._client.upload_file(str(path), self.bucket, self._key_for(path))

    def persist_tree(self, root: Path) -> None:
        if not root.exists():
            return
        for path in root.rglob("*"):
            if path.is_file():
                self.persist_file(path)

    def presigned_get_url(self, path: Path, *, ttl_sec: Optional[int] = None) -> Optional[str]:
        if not self._presign_enabled:
            return None
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": self._key_for(path)},
                ExpiresIn=int(ttl_sec or self._presign_ttl_sec),
            )
        except Exception:
            return None


def _pick_local_storage_dir() -> Path:
    env = os.environ.get("ZOCR_API_STORAGE_DIR")
    if env:
        return Path(env)
    candidates = [
        Path("/data"),
        Path(tempfile.gettempdir()) / "zocr_api_store",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except Exception:
            continue
    fallback = Path(tempfile.gettempdir()) / "zocr_api_store"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def build_job_store() -> tuple[str, JobStore]:
    backend = (os.environ.get("ZOCR_API_STORAGE_BACKEND") or "local").strip().lower()
    storage_dir = _pick_local_storage_dir()
    if backend in {"local", "filesystem", "fs"}:
        return "local", LocalJobStore(storage_dir)
    if backend == "s3":
        raw_ttl = os.environ.get("ZOCR_API_S3_PRESIGN_TTL_SEC") or "3600"
        try:
            presign_ttl_sec = int(raw_ttl)
        except ValueError:
            presign_ttl_sec = 3600
        return "s3", S3JobStore(
            storage_dir=storage_dir,
            bucket=os.environ.get("ZOCR_API_S3_BUCKET") or "",
            prefix=os.environ.get("ZOCR_API_S3_PREFIX") or "zocr",
            region=os.environ.get("ZOCR_API_S3_REGION") or None,
            endpoint_url=os.environ.get("ZOCR_API_S3_ENDPOINT_URL") or None,
            force_path_style=_env_truthy("ZOCR_API_S3_FORCE_PATH_STYLE", False),
            presign_enabled=_env_truthy("ZOCR_API_S3_PRESIGN_ENABLED", True),
            presign_ttl_sec=presign_ttl_sec,
        )
    raise RuntimeError(
        f"Unsupported ZOCR_API_STORAGE_BACKEND={backend!r}. Supported backends: local (filesystem), s3."
    )
