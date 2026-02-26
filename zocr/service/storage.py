from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class JobPaths:
    root: Path
    job_json: Path
    input_dir: Path
    out_dir: Path
    artifacts_zip: Path


class LocalJobStore:
    """Filesystem-backed job store.

    This is the default backend and works for on-prem deployments and single-node
    services. For SaaS deployments, the intent is to replace this backend with a
    shared/object-storage implementation.
    """

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
            return [p for p in self.jobs_dir.iterdir() if p.is_dir()]
        except Exception:
            return []

    def delete_tree(self, path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)

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


def build_job_store() -> tuple[str, LocalJobStore]:
    backend = (os.environ.get("ZOCR_API_STORAGE_BACKEND") or "local").strip().lower()
    if backend in {"local", "filesystem", "fs"}:
        return backend, LocalJobStore(_pick_local_storage_dir())
    raise RuntimeError(
        f"Unsupported ZOCR_API_STORAGE_BACKEND={backend!r}. "
        "Supported backends: local (filesystem)."
    )

