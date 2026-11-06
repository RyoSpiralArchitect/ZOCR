from __future__ import annotations

import hashlib
import os
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
    locks_dir = store.storage_dir / "execution-locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    with (locks_dir / f"{key.hex()}.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                yield False
                return
            unlock = lambda: (stream.seek(0), msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            yield True
        finally:
            unlock()
