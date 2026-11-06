from __future__ import annotations

import hashlib
from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def execution_lock(store: Any, repository: Any, job_id: str, tenant_id: str) -> Iterator[bool]:
    key = hashlib.sha256(f"{tenant_id}:{job_id}".encode()).digest()
    if repository.backend == "postgres":
        with repository._connect() as connection:
            with connection.cursor() as cursor:
                lock_id = int.from_bytes(key[:8], "big", signed=True)
                cursor.execute("SELECT pg_try_advisory_lock(%s)", (lock_id,))
                acquired = bool(cursor.fetchone()[0])
                try:
                    yield acquired
                finally:
                    if acquired:
                        cursor.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
        return
    import fcntl

    locks_dir = store.storage_dir / "execution-locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    with (locks_dir / f"{key.hex()}.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
